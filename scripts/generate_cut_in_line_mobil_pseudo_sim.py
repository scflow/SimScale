#!/usr/bin/env python3
"""
Generate cut-in-line pseudo-sim traces with:
- ego controlled by native NAVSIM PDM-Closed;
- stage1 background traffic induced by IDM + MOBIL;
- stage2 rescue driven by centerline-only PDM with calmer reactive traffic.

This script differs from proposal-mining approaches:
1) it runs a stepwise closed loop for stage1 until ego first leaves its original lane;
2) it uses that boundary-onset state as the split point;
3) it then runs a second closed loop that must recover ego back into the original lane.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import Point2D, StateSE2, StateVector2D, TimeDuration
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusData, TrafficLightStatusType
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely.geometry import Point

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.behavior.idm import IDM, IDMParams
from navsim.behavior.lane_change_trajectory import propagate_actor, rollout_ego_trajectory
from navsim.behavior.mobil import KEEP, LEFT, RIGHT, MobilModel, MobilParams, decide_with_mobil
from navsim.behavior.scene_adapter import (
    ActorState,
    build_actor_from_detection_track,
    build_ego_actor_from_ego_state,
    build_neighbor_set,
    lane_heading_at_point,
)
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.observation.navsim_idm_agents import NavsimIDMAgents
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from navsim.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner
from navsim.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_states_to_state_array,
    state_array_to_ego_state,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from navsim.traffic_agents_policies.mobil_traffic_agents import MobilTrafficAgentsPolicy
from navsim.traffic_agents_policies.navsim_IDM_traffic_agents import NavsimIDMTrafficAgents
try:
    from scripts.mine_cut_in_line_boundary_states import BoundaryMiningPDMPlanner
except ModuleNotFoundError:
    from mine_cut_in_line_boundary_states import BoundaryMiningPDMPlanner


EPS = 1e-6
WorkerContext = Dict[str, Any]
_CTX: Optional[WorkerContext] = None


@dataclass
class InducementSpec:
    actor_token: str
    mode: str
    start_time_s: float
    preferred_action: Optional[str]
    idm_params: IDMParams
    mobil_params: MobilParams
    score_hint: float


@dataclass
class RunConfig:
    output_dir: str
    save_format: str
    top_k_per_token: int
    interval: float
    horizon_sec: float
    map_radius: float
    stage1_max_steps: int
    stage2_max_steps: int
    min_stage1_steps: int
    recovery_hold_steps: int
    boundary_min_center_dev_m: float
    recovery_heading_error_deg: float
    recovery_max_center_dev_m: float
    min_stage2_progress_m: float
    top_k_specs: int
    candidate_radius_m: float
    parallel_backend: str
    num_workers: int
    map_root_override: Optional[str]


@dataclass
class StepStatus:
    in_original_lane: bool
    in_drivable_area: bool
    heading_error_deg: float
    center_deviation_m: float
    min_vehicle_distance_m: float
    collision: bool


@dataclass
class StageSummary:
    steps: int
    elapsed_s: float
    center_deviation_m: float
    heading_error_deg: float
    min_vehicle_distance_m: float
    in_original_lane: bool
    in_drivable_area: bool
    recovered: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cut-in-line pseudo-sim traces with IDM+MOBIL.")
    parser.add_argument("--metric-cache-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("generated_cut_in_line_mobil_data"))
    parser.add_argument("--manifest-name", type=str, default="manifest.jsonl")
    parser.add_argument("--save-format", type=str, default="both", choices=["json", "pkl", "both", "none"])
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--tokens-file", type=Path, default=None)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--horizon-sec", type=float, default=4.0)
    parser.add_argument("--map-radius", type=float, default=100.0)
    parser.add_argument("--stage1-max-steps", type=int, default=16)
    parser.add_argument("--stage2-max-steps", type=int, default=24)
    parser.add_argument("--min-stage1-steps", type=int, default=4)
    parser.add_argument("--recovery-hold-steps", type=int, default=3)
    parser.add_argument("--boundary-min-center-dev-m", type=float, default=0.6)
    parser.add_argument("--recovery-heading-error-deg", type=float, default=8.0)
    parser.add_argument("--recovery-max-center-dev-m", type=float, default=0.35)
    parser.add_argument("--min-stage2-progress-m", type=float, default=3.0)
    parser.add_argument("--candidate-radius-m", type=float, default=35.0)
    parser.add_argument("--top-k-specs", type=int, default=6)
    parser.add_argument("--top-k-per-token", type=int, default=2)
    parser.add_argument("--map-root-override", type=str, default=None)
    parser.add_argument("--parallel-backend", type=str, default="process", choices=["process", "none"])
    parser.add_argument("--num-workers", type=int, default=0, help="0 means auto, capped to 32.")
    return parser.parse_args()


def _auto_workers(num_workers: int) -> int:
    if num_workers > 0:
        return num_workers
    count = os.cpu_count() or 1
    return max(1, min(count, 32))


def _has_metric_cache_metadata(root: Path) -> bool:
    metadata_dir = root / "metadata"
    return metadata_dir.exists() and any(p.suffix.lower() == ".csv" for p in metadata_dir.iterdir())


def resolve_metric_cache_root(input_path: Path) -> Path:
    p = input_path.expanduser().resolve()
    if _has_metric_cache_metadata(p):
        return p
    if p.exists() and p.is_dir():
        matched = [child for child in p.iterdir() if child.is_dir() and _has_metric_cache_metadata(child)]
        if len(matched) == 1:
            return matched[0]
    raise RuntimeError(f"Invalid metric cache root: {input_path}")


def _load_metric_cache_loader(cache_root: Path) -> MetricCacheLoader:
    loader = MetricCacheLoader(cache_root)
    missing = [token for token, path in loader.metric_cache_paths.items() if not Path(path).exists()]
    if not missing:
        return loader

    local_index: Dict[str, str] = {}
    for metric_file in cache_root.rglob("metric_cache.pkl"):
        local_index[metric_file.parent.name] = str(metric_file)
    for token in missing:
        local_path = local_index.get(token)
        if local_path is not None:
            loader.metric_cache_paths[token] = local_path
    return loader


def _load_tokens(loader: MetricCacheLoader, tokens_file: Optional[Path], max_scenes: Optional[int]) -> List[str]:
    if tokens_file is None:
        tokens = loader.tokens
    else:
        requested = [line.strip() for line in tokens_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        available = set(loader.tokens)
        tokens = [token for token in requested if token in available]
    if max_scenes is not None:
        tokens = tokens[:max_scenes]
    return tokens


def _build_planner_initialization(metric_cache: MetricCache, map_root_override: Optional[str]) -> PlannerInitialization:
    map_params = metric_cache.map_parameters
    map_root = map_root_override or map_params.map_root
    map_api = get_maps_api(map_root, map_params.map_version, map_params.map_name)

    roadblock_ids: List[str] = []
    seen: set[str] = set()
    for lane_id in metric_cache.route_lane_ids:
        lane = map_api.get_map_object(lane_id, SemanticMapLayer.LANE)
        if lane is None:
            lane = map_api.get_map_object(lane_id, SemanticMapLayer.LANE_CONNECTOR)
        if lane is None:
            continue
        roadblock_id = lane.get_roadblock_id()
        if roadblock_id in seen:
            continue
        seen.add(roadblock_id)
        roadblock_ids.append(roadblock_id)
    if not roadblock_ids:
        raise RuntimeError("Failed to infer route roadblocks from metric cache route lanes.")

    return PlannerInitialization(
        route_roadblock_ids=roadblock_ids,
        mission_goal=metric_cache.ego_state.center,
        map_api=map_api,
    )


def _build_planner_input(
    ego_state: Any,
    current_tracks: DetectionsTracks,
    traffic_light_data: Sequence[Any],
) -> PlannerInput:
    return PlannerInput(
        iteration=SimulationIteration(index=0, time_point=ego_state.time_point),
        history=SimulationHistoryBuffer.initialize_from_list(
            buffer_size=1,
            ego_states=[ego_state],
            observations=[current_tracks],
        ),
        traffic_light_data=list(traffic_light_data),
    )


class PreferredLanePDMPlanner(PDMClosedPlanner):
    """Force planner to start from a specific route lane when rescuing."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        proposal_sampling: TrajectorySampling,
        idm_policies: BatchIDMPolicy,
        lateral_offsets: Optional[List[float]],
        map_radius: float,
        preferred_start_lane_id: Optional[str],
    ):
        super().__init__(
            trajectory_sampling=trajectory_sampling,
            proposal_sampling=proposal_sampling,
            idm_policies=idm_policies,
            lateral_offsets=lateral_offsets,
            map_radius=map_radius,
        )
        self._preferred_start_lane_id = preferred_start_lane_id

    def _get_starting_lane(self, ego_state):
        if (
            self._preferred_start_lane_id
            and self._route_lane_dict
            and self._preferred_start_lane_id in self._route_lane_dict
        ):
            return self._route_lane_dict[self._preferred_start_lane_id]
        return super()._get_starting_lane(ego_state)


def _build_pdm_planner(
    proposal_sampling: TrajectorySampling,
    map_radius: float,
    lateral_offsets: Optional[List[float]],
    preferred_start_lane_id: Optional[str],
) -> PDMClosedPlanner:
    future_poses = proposal_sampling.num_poses + int(1.0 / proposal_sampling.interval_length)
    trajectory_sampling = TrajectorySampling(
        num_poses=future_poses,
        interval_length=proposal_sampling.interval_length,
    )
    planner_kwargs = dict(
        trajectory_sampling=trajectory_sampling,
        proposal_sampling=proposal_sampling,
        idm_policies=BatchIDMPolicy(
            speed_limit_fraction=[0.2, 0.4, 0.6, 0.8, 1.0],
            fallback_target_velocity=15.0,
            min_gap_to_lead_agent=1.0,
            headway_time=1.5,
            accel_max=1.5,
            decel_max=3.0,
        ),
        lateral_offsets=lateral_offsets,
        map_radius=map_radius,
    )
    if preferred_start_lane_id is not None:
        return PreferredLanePDMPlanner(
            preferred_start_lane_id=preferred_start_lane_id,
            **planner_kwargs,
        )
    return PDMClosedPlanner(**planner_kwargs)


def _build_stage2_idm_policy(
    proposal_sampling: TrajectorySampling,
    map_root_override: Optional[str],
) -> NavsimIDMTrafficAgents:
    idm_agents_observation = NavsimIDMAgents(
        target_velocity=10.0,
        min_gap_to_lead_agent=1.0,
        headway_time=1.5,
        accel_max=1.0,
        decel_max=2.0,
        open_loop_detections_types=[],
        minimum_path_length=20.0,
        planned_trajectory_samples=None,
        planned_trajectory_sample_interval=None,
        radius=100.0,
        add_open_loop_parked_vehicles=True,
        idm_snap_threshold=3.0,
    )
    return NavsimIDMTrafficAgents(
        future_trajectory_sampling=proposal_sampling,
        idm_agents_observation=idm_agents_observation,
        map_root_override=map_root_override,
    )


def _rotate_to_actor_frame(dx: float, dy: float, heading: float) -> Tuple[float, float]:
    c = math.cos(heading)
    s = math.sin(heading)
    lon = c * dx + s * dy
    lat = -s * dx + c * dy
    return lon, lat


def _actor_distance(actor: ActorState, ego_actor: ActorState) -> float:
    return math.hypot(actor.x - ego_actor.x, actor.y - ego_actor.y)


def _same_lane(actor: ActorState, ego_actor: ActorState) -> bool:
    return (
        actor.lane_object is not None
        and ego_actor.lane_object is not None
        and actor.lane_object.id == ego_actor.lane_object.id
    )


def _adjacent_action_to_ego(actor: ActorState, ego_actor: ActorState) -> Optional[str]:
    if actor.lane_object is None or ego_actor.lane_object is None:
        return None
    left_lane, right_lane = actor.lane_object.adjacent_edges
    if left_lane is not None and left_lane.id == ego_actor.lane_object.id:
        return LEFT
    if right_lane is not None and right_lane.id == ego_actor.lane_object.id:
        return RIGHT
    return None


def _action_to_route_command(action: Optional[str], mobil_params: MobilParams) -> Optional[np.ndarray]:
    if action not in (LEFT, RIGHT, KEEP):
        return None
    command = np.zeros(4, dtype=np.float32)
    if action == LEFT:
        command[mobil_params.command_left_idx] = 1.0
    elif action == RIGHT:
        command[mobil_params.command_right_idx] = 1.0
    else:
        command[mobil_params.command_straight_idx] = 1.0
    return command


def _cut_in_specs(actor: ActorState, distance: float, preferred_action: str) -> List[InducementSpec]:
    score_hint = 3.0 + max(0.0, (30.0 - distance) / 15.0)
    return [
        InducementSpec(
            actor_token=actor.token,
            mode="cut_in",
            start_time_s=start_time_s,
            preferred_action=preferred_action,
            idm_params=IDMParams(
                v0=max(actor.speed + 3.0, 10.0),
                T=0.9,
                s0=1.0,
                a=2.3,
                b=3.0,
                delta=4,
            ),
            mobil_params=MobilParams(
                politeness=0.0,
                accel_threshold=0.0,
                safe_decel=4.0,
                cooldown_s=2.5,
                route_bias=1.4,
            ),
            score_hint=score_hint - 0.1 * idx,
        )
        for idx, start_time_s in enumerate((0.1, 0.4, 0.7))
    ]


def _brake_check_specs(actor: ActorState, longitudinal_gap: float) -> List[InducementSpec]:
    score_hint = 1.5 + max(0.0, (25.0 - longitudinal_gap) / 15.0)
    return [
        InducementSpec(
            actor_token=actor.token,
            mode="brake_check",
            start_time_s=start_time_s,
            preferred_action=None,
            idm_params=IDMParams(
                v0=max(1.0, actor.speed * speed_scale),
                T=2.1,
                s0=2.5,
                a=1.0,
                b=3.5,
                delta=4,
            ),
            mobil_params=MobilParams(
                politeness=0.3,
                accel_threshold=0.2,
                safe_decel=4.0,
                cooldown_s=3.0,
                route_bias=0.0,
            ),
            score_hint=score_hint - 0.1 * idx,
        )
        for idx, (start_time_s, speed_scale) in enumerate(((0.2, 0.40), (0.5, 0.30)))
    ]


def enumerate_inducement_specs(
    metric_cache: MetricCache,
    candidate_radius_m: float,
    top_k_specs: int,
    map_root_override: Optional[str],
) -> List[InducementSpec]:
    map_params = metric_cache.map_parameters
    map_api = get_maps_api(
        map_root_override or map_params.map_root,
        map_params.map_version,
        map_params.map_name,
    )
    ego_actor = build_ego_actor_from_ego_state(metric_cache.ego_state, map_api)
    current_tracks = metric_cache.current_tracked_objects[0].tracked_objects.tracked_objects
    actors = [
        build_actor_from_detection_track(track, map_api)
        for track in current_tracks
        if str(track.tracked_object_type).lower().endswith("vehicle")
    ]

    specs: List[InducementSpec] = []
    for actor in actors:
        distance = _actor_distance(actor, ego_actor)
        if distance > candidate_radius_m:
            continue

        dx = actor.x - ego_actor.x
        dy = actor.y - ego_actor.y
        rel_lon, rel_lat = _rotate_to_actor_frame(dx, dy, ego_actor.heading)

        preferred_action = _adjacent_action_to_ego(actor, ego_actor)
        if preferred_action is not None and -15.0 <= rel_lon <= 25.0 and 0.5 <= abs(rel_lat) <= 12.0:
            specs.extend(_cut_in_specs(actor, distance, preferred_action))
            continue

        if (_same_lane(actor, ego_actor) or abs(rel_lat) <= 4.5) and 2.0 <= rel_lon <= 30.0:
            specs.extend(_brake_check_specs(actor, rel_lon))

    if not specs:
        fallback_candidates: List[Tuple[float, ActorState, float, float]] = []
        for actor in actors:
            distance = _actor_distance(actor, ego_actor)
            if distance > candidate_radius_m:
                continue
            dx = actor.x - ego_actor.x
            dy = actor.y - ego_actor.y
            rel_lon, rel_lat = _rotate_to_actor_frame(dx, dy, ego_actor.heading)
            if rel_lon > 2.0 and abs(rel_lat) <= 6.0:
                fallback_candidates.append((distance, actor, rel_lon, rel_lat))
        fallback_candidates.sort(key=lambda item: item[0])
        for _, actor, rel_lon, _ in fallback_candidates[:2]:
            specs.extend(_brake_check_specs(actor, rel_lon))

    specs.sort(key=lambda spec: spec.score_hint, reverse=True)
    return specs[:top_k_specs]


class InducedMobilTrafficAgentsPolicy(MobilTrafficAgentsPolicy):
    """Target a single actor with stronger IDM/MOBIL parameters after an offset time."""

    def __init__(self, inducement_spec: InducementSpec, elapsed_time_offset_s: float, **kwargs: Any):
        super().__init__(**kwargs)
        self._inducement_spec = inducement_spec
        self._elapsed_time_offset_s = elapsed_time_offset_s

    def _active(self, token: str, timestep: int) -> bool:
        if token != self._inducement_spec.actor_token:
            return False
        current_time_s = self._elapsed_time_offset_s + max(
            0.0, (timestep - 1) * self.future_trajectory_sampling.interval_length
        )
        return current_time_s >= self._inducement_spec.start_time_s - EPS

    def simulate_traffic_agents(self, simulated_ego_states: np.ndarray, metric_cache: MetricCache) -> List[DetectionsTracks]:
        map_root = self._map_root_override or metric_cache.map_parameters.map_root
        map_api = get_maps_api(
            map_root,
            metric_cache.map_parameters.map_version,
            metric_cache.map_parameters.map_name,
        )
        runtime_actors = self._build_runtime_actors(metric_cache.current_tracked_objects[0], map_api)
        self._infer_agent_routes(runtime_actors, metric_cache, map_api)

        dt = self.future_trajectory_sampling.interval_length
        timestamp_us = metric_cache.timepoint.time_us
        traffic_light_status = getattr(metric_cache, "traffic_light_status", None)
        future_tracks: List[DetectionsTracks] = []

        for timestep in range(1, self.future_trajectory_sampling.num_poses + 1):
            ego_actor = self._build_ego_actor_from_simulated_state(
                simulated_ego_states[min(timestep - 1, len(simulated_ego_states) - 1)],
                metric_cache,
                map_api,
            )
            open_loop_tracks = []
            if self._open_loop_detections_types and timestep - 1 < len(metric_cache.future_tracked_objects):
                open_loop_tracks = metric_cache.future_tracked_objects[timestep - 1].tracked_objects.get_tracked_objects_of_types(
                    self._open_loop_detections_types
                )

            step_tl = {}
            if traffic_light_status is not None and timestep < len(traffic_light_status):
                step_tl = self._normalize_traffic_light_status(traffic_light_status[timestep])

            snapshot_actors = [runtime.actor for runtime in runtime_actors.values()] + [ego_actor]
            updated_runtime: Dict[str, Any] = {}

            for token, runtime_actor in runtime_actors.items():
                actor = runtime_actor.actor
                neighbors = build_neighbor_set(actor, snapshot_actors)
                acceleration_caps = self._compute_action_acceleration_caps(
                    actor=actor,
                    runtime_actor=runtime_actor,
                    neighbors=neighbors,
                    open_loop_tracks=open_loop_tracks,
                    traffic_light_status=step_tl,
                    map_api=map_api,
                )
                lane_change_allowed = runtime_actor.cooldown_remaining <= 0.0 and runtime_actor.lane_change_elapsed <= 0.0
                idm = self._idm
                mobil = self._mobil
                route_command = None

                if self._active(token, timestep):
                    idm = IDM(self._inducement_spec.idm_params)
                    mobil = MobilModel(self._inducement_spec.mobil_params, idm)
                    route_command = _action_to_route_command(
                        self._inducement_spec.preferred_action,
                        self._inducement_spec.mobil_params,
                    )
                    if self._inducement_spec.mode == "brake_check":
                        lane_change_allowed = False

                action, acceleration, target_lane = decide_with_mobil(
                    actor=actor,
                    neighbors=neighbors,
                    route_command=route_command,
                    idm=idm,
                    mobil=mobil,
                    lane_change_allowed=lane_change_allowed,
                    acceleration_caps=acceleration_caps,
                )

                if self._active(token, timestep) and self._inducement_spec.mode == "brake_check":
                    acceleration = min(acceleration, -2.0)
                    action = KEEP
                    target_lane = actor.lane_object

                selected_target = runtime_actor.target_lane
                elapsed = runtime_actor.lane_change_elapsed
                cooldown_remaining = max(0.0, runtime_actor.cooldown_remaining - dt)
                if runtime_actor.lane_change_elapsed > 0.0:
                    selected_target = runtime_actor.target_lane
                elif action in (LEFT, RIGHT) and target_lane is not None:
                    selected_target = target_lane
                    elapsed = 1e-3
                    cooldown_remaining = mobil.p.cooldown_s
                else:
                    selected_target = actor.lane_object
                    elapsed = 0.0

                next_actor, next_elapsed = propagate_actor(
                    actor=actor,
                    map_api=map_api,
                    current_lane=actor.lane_object,
                    target_lane=selected_target,
                    acceleration=acceleration,
                    dt=dt,
                    route_lane_ids=runtime_actor.route_lane_ids or [],
                    route_roadblock_ids=runtime_actor.route_roadblock_ids or [],
                    lane_change_duration=self._lane_change_duration,
                    lane_change_elapsed=elapsed,
                )
                if next_elapsed >= self._lane_change_duration:
                    next_elapsed = 0.0
                updated_runtime[token] = type(runtime_actor)(
                    actor=next_actor,
                    target_lane=selected_target if next_elapsed > 0.0 else next_actor.lane_object,
                    lane_change_elapsed=next_elapsed,
                    cooldown_remaining=cooldown_remaining,
                    route_lane_ids=runtime_actor.route_lane_ids,
                    route_roadblock_ids=runtime_actor.route_roadblock_ids,
                )

            runtime_actors = updated_runtime
            current_timestamp = timestamp_us + int(round(timestep * dt * 1e6))
            future_tracks.append(self._build_detection_tracks(runtime_actors, current_timestamp))

        return future_tracks


def _build_stage1_policy(
    proposal_sampling: TrajectorySampling,
    inducement_spec: InducementSpec,
    elapsed_time_offset_s: float,
    map_root_override: Optional[str],
) -> MobilTrafficAgentsPolicy:
    return InducedMobilTrafficAgentsPolicy(
        inducement_spec=inducement_spec,
        elapsed_time_offset_s=elapsed_time_offset_s,
        future_trajectory_sampling=proposal_sampling,
        lane_change_duration=3.0,
        open_loop_detections_types=[
            "PEDESTRIAN",
            "BICYCLE",
            "BARRIER",
            "CZONE_SIGN",
            "TRAFFIC_CONE",
            "GENERIC_OBJECT",
        ],
        stop_line_min_gap=3.0,
        open_loop_min_gap=2.0,
        open_loop_path_buffer=2.5,
        max_stop_brake=6.0,
        map_root_override=map_root_override,
    )


def _traffic_light_slice(
    metric_cache: MetricCache,
    start_index: int,
    required_length: int,
) -> Optional[List[List[TrafficLightStatusData]]]:
    status = getattr(metric_cache, "traffic_light_status", None)
    if status is None:
        return None
    if not status:
        return []
    start = min(start_index, len(status) - 1)
    sliced = list(status[start : start + required_length])
    while len(sliced) < required_length:
        sliced.append(sliced[-1])
    return sliced


def _tracks_slice(
    metric_cache: MetricCache,
    current_tracks: DetectionsTracks,
    start_index: int,
    required_length: int,
) -> List[DetectionsTracks]:
    future_tracks = metric_cache.future_tracked_objects
    if future_tracks:
        start = min(start_index, len(future_tracks) - 1)
        sliced = [copy.deepcopy(track) for track in future_tracks[start : start + required_length]]
        while len(sliced) < required_length:
            sliced.append(copy.deepcopy(sliced[-1]))
        return sliced
    return [copy.deepcopy(current_tracks) for _ in range(required_length)]


def _build_step_metric_cache(
    metric_cache: MetricCache,
    ego_state: Any,
    current_tracks: DetectionsTracks,
    step_offset: int,
    proposal_sampling: TrajectorySampling,
) -> MetricCache:
    step_cache = copy.copy(metric_cache)
    step_cache.timepoint = ego_state.time_point
    step_cache.ego_state = ego_state
    step_cache.current_tracked_objects = [current_tracks]
    step_cache.future_tracked_objects = _tracks_slice(metric_cache, current_tracks, step_offset, proposal_sampling.num_poses)
    step_cache.traffic_light_status = _traffic_light_slice(metric_cache, step_offset, proposal_sampling.num_poses + 1)
    return step_cache


def _find_lane_object(
    map_api: Any,
    lane_id: str,
) -> Optional[LaneGraphEdgeMapObject]:
    lane = map_api.get_map_object(lane_id, SemanticMapLayer.LANE)
    if lane is None:
        lane = map_api.get_map_object(lane_id, SemanticMapLayer.LANE_CONNECTOR)
    return lane


def _find_boundary_split_index(
    states: np.ndarray,
    original_lane: LaneGraphEdgeMapObject,
    min_stage1_steps: int,
    min_center_dev_m: float,
) -> Optional[int]:
    onset_dev_threshold = min(min_center_dev_m, 0.3)
    for idx in range(max(1, min_stage1_steps), len(states)):
        x = float(states[idx, StateIndex.X])
        y = float(states[idx, StateIndex.Y])
        point2d = Point2D(x, y)
        center_dev = float(original_lane.baseline_path.linestring.distance(Point(x, y)))
        if (not original_lane.contains_point(point2d)) and center_dev >= onset_dev_threshold:
            return idx
    return None


def _stage1_candidate_rank(
    states: np.ndarray,
    split_idx: int,
    original_lane: LaneGraphEdgeMapObject,
) -> float:
    split_dev = float(
        original_lane.baseline_path.linestring.distance(
            Point(float(states[split_idx, StateIndex.X]), float(states[split_idx, StateIndex.Y]))
        )
    )
    end_speed = float(
        np.hypot(states[split_idx, StateIndex.VELOCITY_X], states[split_idx, StateIndex.VELOCITY_Y])
    )
    return 0.7 * split_dev + 0.3 * min(end_speed, 12.0)


def _infer_route_command_to_original_lane(
    current_lane: Optional[LaneGraphEdgeMapObject],
    original_lane: LaneGraphEdgeMapObject,
    mobil_params: MobilParams,
) -> Optional[np.ndarray]:
    if current_lane is None or current_lane.id == original_lane.id:
        return _action_to_route_command(KEEP, mobil_params)
    left_lane, right_lane = current_lane.adjacent_edges
    if left_lane is not None and left_lane.id == original_lane.id:
        return _action_to_route_command(LEFT, mobil_params)
    if right_lane is not None and right_lane.id == original_lane.id:
        return _action_to_route_command(RIGHT, mobil_params)
    return _action_to_route_command(KEEP, mobil_params)


def _adjacent_target_lane_to_original(
    current_lane: Optional[LaneGraphEdgeMapObject],
    original_lane: LaneGraphEdgeMapObject,
) -> Optional[LaneGraphEdgeMapObject]:
    if current_lane is None or current_lane.id == original_lane.id:
        return current_lane
    left_lane, right_lane = current_lane.adjacent_edges
    if left_lane is not None and left_lane.id == original_lane.id:
        return original_lane
    if right_lane is not None and right_lane.id == original_lane.id:
        return original_lane
    return None


def _lane_is_original_or_adjacent(
    current_lane: Optional[LaneGraphEdgeMapObject],
    original_lane: LaneGraphEdgeMapObject,
) -> bool:
    if current_lane is None:
        return False
    if current_lane.id == original_lane.id:
        return True
    return _adjacent_target_lane_to_original(current_lane, original_lane) is not None


def _ego_mobil_rescue_states(
    split_ego_state: Any,
    split_tracks: DetectionsTracks,
    metric_cache: MetricCache,
    map_api: Any,
    proposal_sampling: TrajectorySampling,
    route_lane_ids: List[str],
    route_roadblock_ids: List[str],
    original_lane: LaneGraphEdgeMapObject,
) -> np.ndarray:
    ego_actor = build_ego_actor_from_ego_state(split_ego_state, map_api)
    current_lane = ego_actor.lane_object or original_lane
    on_original_lane_geometry = original_lane.contains_point(Point2D(float(ego_actor.x), float(ego_actor.y)))
    actors = [
        build_actor_from_detection_track(track, map_api)
        for track in split_tracks.tracked_objects.tracked_objects
        if str(track.tracked_object_type).lower().endswith("vehicle")
    ]
    neighbors = build_neighbor_set(ego_actor, actors)
    idm_params = IDMParams(
        v0=max(ego_actor.speed + 2.0, 10.0),
        T=1.1,
        s0=1.5,
        a=2.0,
        b=3.0,
        delta=4,
    )
    mobil_params = MobilParams(
        politeness=0.0,
        accel_threshold=0.0,
        safe_decel=4.0,
        cooldown_s=2.5,
        route_bias=2.4,
    )
    idm = IDM(idm_params)
    mobil = MobilModel(mobil_params, idm)
    route_command = _infer_route_command_to_original_lane(current_lane, original_lane, mobil_params)
    action, acceleration, target_lane = decide_with_mobil(
        actor=ego_actor,
        neighbors=neighbors,
        route_command=route_command,
        idm=idm,
        mobil=mobil,
    )
    forced_target_lane = _adjacent_target_lane_to_original(current_lane, original_lane)
    if (
        forced_target_lane is not None
        and current_lane is not None
        and current_lane.id != original_lane.id
    ):
        left_lane, right_lane = current_lane.adjacent_edges
        if left_lane is not None and left_lane.id == original_lane.id:
            action = LEFT
        elif right_lane is not None and right_lane.id == original_lane.id:
            action = RIGHT
        target_lane = forced_target_lane
        acceleration = max(acceleration, idm.free_accel(ego_actor.speed))
    elif not on_original_lane_geometry:
        target_lane = original_lane
        acceleration = max(acceleration, idm.free_accel(ego_actor.speed))
    elif action == KEEP:
        target_lane = current_lane
    trajectory = rollout_ego_trajectory(
        ego=ego_actor,
        current_lane=current_lane,
        target_lane=target_lane,
        acceleration=acceleration,
        trajectory_sampling=proposal_sampling,
        route_lane_ids=route_lane_ids,
        route_roadblock_ids=route_roadblock_ids,
        lane_change_duration=min(1.5, max(0.8, proposal_sampling.time_horizon * 0.4)),
    )
    interpolated = transform_trajectory(trajectory, split_ego_state)
    return get_trajectory_as_array(interpolated, proposal_sampling, split_ego_state.time_point)


def _is_point_in_drivable_area(drivable_area_map: PDMDrivableMap, point: Point) -> bool:
    return len(drivable_area_map.query(point, predicate="within")) > 0


def _min_vehicle_distance(point: Point, detections: DetectionsTracks) -> float:
    d_min = float("inf")
    for obj in detections.tracked_objects.tracked_objects:
        if str(obj.tracked_object_type).lower().endswith("vehicle"):
            d_min = min(d_min, point.distance(obj.box.geometry))
    return d_min if d_min < float("inf") else float("nan")


def _step_status(
    ego_state: Any,
    detections: DetectionsTracks,
    drivable_area_map: PDMDrivableMap,
    original_lane: LaneGraphEdgeMapObject,
) -> StepStatus:
    rear_point = Point(float(ego_state.rear_axle.x), float(ego_state.rear_axle.y))
    in_original_lane = original_lane.contains_point(Point2D(float(ego_state.rear_axle.x), float(ego_state.rear_axle.y)))
    in_drivable_area = _is_point_in_drivable_area(drivable_area_map, rear_point)
    original_heading = lane_heading_at_point(original_lane, float(ego_state.rear_axle.x), float(ego_state.rear_axle.y))
    heading_error_deg = abs(math.degrees(((float(ego_state.rear_axle.heading - original_heading) + math.pi) % (2 * math.pi)) - math.pi))
    center_deviation_m = float(original_lane.baseline_path.linestring.distance(rear_point))
    collision = any(
        ego_state.car_footprint.geometry.intersects(obj.box.geometry)
        for obj in detections.tracked_objects.tracked_objects
        if str(obj.tracked_object_type).lower().endswith("vehicle")
    )
    return StepStatus(
        in_original_lane=bool(in_original_lane),
        in_drivable_area=bool(in_drivable_area),
        heading_error_deg=float(heading_error_deg),
        center_deviation_m=float(center_deviation_m),
        min_vehicle_distance_m=float(_min_vehicle_distance(rear_point, detections)),
        collision=bool(collision),
    )


def _serialize_vehicles(detections: DetectionsTracks) -> List[Dict[str, Any]]:
    vehicles: List[Dict[str, Any]] = []
    for obj in detections.tracked_objects.tracked_objects:
        if str(obj.tracked_object_type).lower().endswith("vehicle"):
            velocity = getattr(obj, "velocity", None)
            vx = float(velocity.x) if velocity is not None else 0.0
            vy = float(velocity.y) if velocity is not None else 0.0
            vehicles.append(
                {
                    "id": str(obj.track_token),
                    "x": float(obj.center.x),
                    "y": float(obj.center.y),
                    "heading": float(obj.center.heading),
                    "velocity": float(np.hypot(vx, vy)),
                    "velocity_x": vx,
                    "velocity_y": vy,
                    "width": float(obj.box.width),
                    "length": float(obj.box.length),
                    "height": float(obj.box.height),
                }
            )
    return vehicles


def _make_frame_payload(t_idx: int, state: np.ndarray, detections: DetectionsTracks) -> Dict[str, Any]:
    vx = float(state[StateIndex.VELOCITY_X])
    vy = float(state[StateIndex.VELOCITY_Y])
    return {
        "t_idx": int(t_idx),
        "ego": {
            "x": float(state[StateIndex.X]),
            "y": float(state[StateIndex.Y]),
            "heading": float(state[StateIndex.HEADING]),
            "velocity": float(np.hypot(vx, vy)),
            "velocity_x": vx,
            "velocity_y": vy,
            "acceleration_x": float(state[StateIndex.ACCELERATION_X]),
            "acceleration_y": float(state[StateIndex.ACCELERATION_Y]),
            "steering_angle": float(state[StateIndex.STEERING_ANGLE]),
        },
        "vehicles": _serialize_vehicles(detections),
    }


def _build_trace_payload(
    token: str,
    inducement_spec: InducementSpec,
    stage1_states: Sequence[np.ndarray],
    stage1_tracks: Sequence[DetectionsTracks],
    stage2_states: Sequence[np.ndarray],
    stage2_tracks: Sequence[DetectionsTracks],
    stage1_score: Dict[str, Any],
    stage2_score: Dict[str, Any],
) -> Dict[str, Any]:
    frames: List[Dict[str, Any]] = []
    for idx, (state, detections) in enumerate(zip(stage1_states, stage1_tracks)):
        frames.append(_make_frame_payload(idx, np.asarray(state), detections))
    for idx, (state, detections) in enumerate(zip(stage2_states[1:], stage2_tracks[1:]), start=len(frames)):
        frames.append(_make_frame_payload(idx, np.asarray(state), detections))
    full_states = np.concatenate([np.asarray(stage1_states), np.asarray(stage2_states[1:])], axis=0)
    return {
        "token": token,
        "inducement_spec": {
            "actor_token": inducement_spec.actor_token,
            "mode": inducement_spec.mode,
            "start_time_s": inducement_spec.start_time_s,
            "preferred_action": inducement_spec.preferred_action,
            "idm_params": asdict(inducement_spec.idm_params),
            "mobil_params": asdict(inducement_spec.mobil_params),
            "score_hint": float(inducement_spec.score_hint),
        },
        "stage1_score": stage1_score,
        "stage2_score": stage2_score,
        "strong_longtail": {"split_idx": int(len(stage1_states) - 1)},
        "num_frames": len(frames),
        "frames": frames,
        "states": full_states.tolist(),
    }


def _candidate_stem(token: str, rank: int, inducement_spec: InducementSpec) -> str:
    return f"cut_in_line_mobil_{token}_r{rank:02d}_{inducement_spec.mode}_{inducement_spec.actor_token[:8]}"


def _save_payload(
    output_dir: Path,
    stem: str,
    payload: Dict[str, Any],
    save_format: str,
) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    if save_format in ("json", "both"):
        json_path = output_dir / f"{stem}.json"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        paths["json_path"] = str(json_path)
    if save_format in ("pkl", "both"):
        pkl_path = output_dir / f"{stem}.pkl"
        with pkl_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        paths["pkl_path"] = str(pkl_path)
    return paths


def _run_closed_loop_stage(
    base_metric_cache: MetricCache,
    proposal_sampling: TrajectorySampling,
    ego_state: Any,
    current_tracks: DetectionsTracks,
    start_step_offset: int,
    max_steps: int,
    map_radius: float,
    planner_lateral_offsets: Optional[List[float]],
    preferred_start_lane_id: Optional[str],
    original_lane: LaneGraphEdgeMapObject,
    stage_policy_builder: Any,
    recovery_mode: bool,
    cfg: RunConfig,
) -> Tuple[List[np.ndarray], List[DetectionsTracks], StageSummary, Optional[str]]:
    state_seq: List[np.ndarray] = [ego_states_to_state_array([ego_state])[0]]
    track_seq: List[DetectionsTracks] = [current_tracks]
    current_ego_state = ego_state
    current_tracks_step = current_tracks
    recovered_streak = 0

    for step_idx in range(max_steps):
        absolute_step = start_step_offset + step_idx
        step_cache = _build_step_metric_cache(
            metric_cache=base_metric_cache,
            ego_state=current_ego_state,
            current_tracks=current_tracks_step,
            step_offset=absolute_step,
            proposal_sampling=proposal_sampling,
        )
        planner = _build_pdm_planner(
            proposal_sampling=proposal_sampling,
            map_radius=map_radius,
            lateral_offsets=planner_lateral_offsets,
            preferred_start_lane_id=preferred_start_lane_id,
        )
        planner.initialize(_build_planner_initialization(step_cache, cfg.map_root_override))
        traffic_light_data = []
        if getattr(step_cache, "traffic_light_status", None):
            traffic_light_data = list(step_cache.traffic_light_status[0])
        planner_input = _build_planner_input(current_ego_state, current_tracks_step, traffic_light_data)
        trajectory = planner.compute_planner_trajectory(planner_input)
        planned_states = get_trajectory_as_array(trajectory, proposal_sampling, current_ego_state.time_point)
        reactive_policy = stage_policy_builder(elapsed_time_offset_s=absolute_step * cfg.interval)
        simulated_tracks = reactive_policy.simulate_environment(planned_states, step_cache)

        next_state_arr = planned_states[1]
        next_time = current_ego_state.time_point + TimeDuration.from_s(cfg.interval)
        next_ego_state = state_array_to_ego_state(
            next_state_arr,
            next_time,
            current_ego_state.car_footprint.vehicle_parameters,
        )
        next_tracks = simulated_tracks[1]
        step_status = _step_status(next_ego_state, next_tracks, planner._drivable_area_map, original_lane)

        state_seq.append(next_state_arr)
        track_seq.append(next_tracks)
        current_ego_state = next_ego_state
        current_tracks_step = next_tracks

        if step_status.collision:
            summary = StageSummary(
                steps=step_idx + 1,
                elapsed_s=(step_idx + 1) * cfg.interval,
                center_deviation_m=step_status.center_deviation_m,
                heading_error_deg=step_status.heading_error_deg,
                min_vehicle_distance_m=step_status.min_vehicle_distance_m,
                in_original_lane=step_status.in_original_lane,
                in_drivable_area=step_status.in_drivable_area,
                recovered=False,
            )
            return state_seq, track_seq, summary, "collision"

        if not step_status.in_drivable_area:
            summary = StageSummary(
                steps=step_idx + 1,
                elapsed_s=(step_idx + 1) * cfg.interval,
                center_deviation_m=step_status.center_deviation_m,
                heading_error_deg=step_status.heading_error_deg,
                min_vehicle_distance_m=step_status.min_vehicle_distance_m,
                in_original_lane=step_status.in_original_lane,
                in_drivable_area=step_status.in_drivable_area,
                recovered=False,
            )
            return state_seq, track_seq, summary, "offroad"

        if not recovery_mode:
            departed = (
                step_idx + 1 >= cfg.min_stage1_steps
                and not step_status.in_original_lane
                and step_status.in_drivable_area
                and step_status.center_deviation_m >= cfg.boundary_min_center_dev_m
            )
            if departed:
                summary = StageSummary(
                    steps=step_idx + 1,
                    elapsed_s=(step_idx + 1) * cfg.interval,
                    center_deviation_m=step_status.center_deviation_m,
                    heading_error_deg=step_status.heading_error_deg,
                    min_vehicle_distance_m=step_status.min_vehicle_distance_m,
                    in_original_lane=step_status.in_original_lane,
                    in_drivable_area=step_status.in_drivable_area,
                    recovered=False,
                )
                return state_seq, track_seq, summary, None
        else:
            if (
                step_status.in_original_lane
                and step_status.center_deviation_m <= cfg.recovery_max_center_dev_m
                and step_status.heading_error_deg <= cfg.recovery_heading_error_deg
            ):
                recovered_streak += 1
            else:
                recovered_streak = 0
            if recovered_streak >= cfg.recovery_hold_steps:
                summary = StageSummary(
                    steps=step_idx + 1,
                    elapsed_s=(step_idx + 1) * cfg.interval,
                    center_deviation_m=step_status.center_deviation_m,
                    heading_error_deg=step_status.heading_error_deg,
                    min_vehicle_distance_m=step_status.min_vehicle_distance_m,
                    in_original_lane=step_status.in_original_lane,
                    in_drivable_area=step_status.in_drivable_area,
                    recovered=True,
                )
                return state_seq, track_seq, summary, None

    final_status = _step_status(current_ego_state, current_tracks_step, planner._drivable_area_map, original_lane)
    summary = StageSummary(
        steps=max_steps,
        elapsed_s=max_steps * cfg.interval,
        center_deviation_m=final_status.center_deviation_m,
        heading_error_deg=final_status.heading_error_deg,
        min_vehicle_distance_m=final_status.min_vehicle_distance_m,
        in_original_lane=final_status.in_original_lane,
        in_drivable_area=final_status.in_drivable_area,
        recovered=bool(recovery_mode and recovered_streak >= cfg.recovery_hold_steps),
    )
    return state_seq, track_seq, summary, "no_boundary" if not recovery_mode else "no_recovery"


def _stage_score_from_summary(summary: StageSummary, stage_name: str) -> Dict[str, Any]:
    return {
        "stage": stage_name,
        "steps": int(summary.steps),
        "elapsed_s": float(summary.elapsed_s),
        "center_deviation_m": float(summary.center_deviation_m),
        "heading_error_deg": float(summary.heading_error_deg),
        "min_vehicle_distance_m": float(summary.min_vehicle_distance_m),
        "in_original_lane": bool(summary.in_original_lane),
        "in_drivable_area": bool(summary.in_drivable_area),
        "recovered": bool(summary.recovered),
    }


def _recovery_rank(stage1_summary: StageSummary, stage2_summary: StageSummary) -> float:
    improvement = max(0.0, stage1_summary.center_deviation_m - stage2_summary.center_deviation_m)
    centeredness = max(0.0, 1.0 - stage2_summary.center_deviation_m / max(stage1_summary.center_deviation_m, 1e-3))
    heading_term = max(0.0, 1.0 - stage2_summary.heading_error_deg / 15.0)
    return 0.45 * improvement + 0.35 * centeredness + 0.20 * heading_term


def _process_token(
    token: str,
    metric_cache: MetricCache,
    cfg: RunConfig,
    proposal_sampling: TrajectorySampling,
) -> Dict[str, Any]:
    output_dir = Path(cfg.output_dir)
    reject_counts: Dict[str, int] = {}

    def _bump(reason: str) -> None:
        reject_counts[reason] = reject_counts.get(reason, 0) + 1

    planner_init = _build_planner_initialization(metric_cache, cfg.map_root_override)
    map_api = planner_init.map_api
    ego_actor = build_ego_actor_from_ego_state(metric_cache.ego_state, map_api)
    original_lane = None
    if ego_actor.lane_object is not None and ego_actor.lane_object.id in set(metric_cache.route_lane_ids):
        original_lane = ego_actor.lane_object
    if original_lane is None and metric_cache.route_lane_ids:
        original_lane = _find_lane_object(map_api, metric_cache.route_lane_ids[0])
    if original_lane is None:
        _bump("no_original_lane")
        return {
            "token": token,
            "kept": 0,
            "attempted_specs": 0,
            "manifest_records": [
                json.dumps(
                    {
                        "token": token,
                        "result_status": "rejected",
                        "reject_reason": "no_original_lane",
                        "reject_counts": reject_counts,
                    },
                    ensure_ascii=False,
                )
            ],
            "reject_counts": reject_counts,
            "top_reject_reason": "no_original_lane",
        }

    boundary_planner = BoundaryMiningPDMPlanner(
        trajectory_sampling=TrajectorySampling(
            num_poses=proposal_sampling.num_poses + int(1.0 / proposal_sampling.interval_length),
            interval_length=proposal_sampling.interval_length,
        ),
        proposal_sampling=proposal_sampling,
        idm_policies=BatchIDMPolicy(
            speed_limit_fraction=[0.2, 0.4, 0.6, 0.8, 1.0],
            fallback_target_velocity=15.0,
            min_gap_to_lead_agent=1.0,
            headway_time=1.5,
            accel_max=1.5,
            decel_max=3.0,
        ),
        lateral_offsets=[-1.0, 1.0],
        map_radius=cfg.map_radius,
    )
    boundary_planner.initialize(planner_init)
    simulated_proposals, proposal_df = boundary_planner.compute_all_proposals(
        _build_planner_input(metric_cache.ego_state, metric_cache.current_tracked_objects[0], list(metric_cache.traffic_light_status[0]) if getattr(metric_cache, "traffic_light_status", None) else [])
    )

    stage1_candidates: List[Tuple[float, int, int]] = []
    for proposal_idx in range(len(proposal_df)):
        states = simulated_proposals[proposal_idx]
        split_idx = _find_boundary_split_index(
            states=states,
            original_lane=original_lane,
            min_stage1_steps=cfg.min_stage1_steps,
            min_center_dev_m=cfg.boundary_min_center_dev_m,
        )
        if split_idx is None:
            continue
        split_state = state_array_to_ego_state(
            np.asarray(states[split_idx]),
            metric_cache.ego_state.time_point + TimeDuration.from_s(split_idx * cfg.interval),
            metric_cache.ego_state.car_footprint.vehicle_parameters,
        )
        split_lane = build_ego_actor_from_ego_state(split_state, map_api).lane_object
        if not _lane_is_original_or_adjacent(split_lane, original_lane):
            continue
        stage1_candidates.append((_stage1_candidate_rank(states, split_idx, original_lane), proposal_idx, split_idx))

    stage1_candidates.sort(key=lambda item: item[0], reverse=True)
    stage1_candidates = stage1_candidates[: cfg.top_k_specs]
    if not stage1_candidates:
        _bump("stage1_no_boundary")

    kept_results: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for candidate_rank, (stage1_rank, proposal_idx, split_idx) in enumerate(stage1_candidates):
        full_states = simulated_proposals[proposal_idx]
        stage1_states = full_states[: split_idx + 1]
        stage1_sampling = TrajectorySampling(
            num_poses=max(1, len(stage1_states) - 1),
            interval_length=cfg.interval,
        )
        stage1_reactive_policy = _build_stage2_idm_policy(stage1_sampling, cfg.map_root_override)
        stage1_tracks_full = stage1_reactive_policy.simulate_environment(stage1_states, metric_cache)
        stage1_tracks = stage1_tracks_full[: len(stage1_states)]
        split_ego_state = state_array_to_ego_state(
            np.asarray(stage1_states[-1]),
            metric_cache.ego_state.time_point + TimeDuration.from_s(split_idx * cfg.interval),
            metric_cache.ego_state.car_footprint.vehicle_parameters,
        )
        split_tracks = stage1_tracks[-1]

        stage1_status = _step_status(
            split_ego_state,
            split_tracks,
            boundary_planner._drivable_area_map,
            original_lane,
        )
        if stage1_status.collision:
            _bump("stage1_collision")
            continue
        if not stage1_status.in_drivable_area:
            _bump("stage1_offroad")
            continue

        stage2_cache = _build_step_metric_cache(
            metric_cache=metric_cache,
            ego_state=split_ego_state,
            current_tracks=split_tracks,
            step_offset=split_idx,
            proposal_sampling=proposal_sampling,
        )
        stage2_states = _ego_mobil_rescue_states(
            split_ego_state=split_ego_state,
            split_tracks=split_tracks,
            metric_cache=metric_cache,
            map_api=map_api,
            proposal_sampling=proposal_sampling,
            route_lane_ids=metric_cache.route_lane_ids,
            route_roadblock_ids=list(planner_init.route_roadblock_ids),
            original_lane=original_lane,
        )
        stage2_reactive_policy = _build_stage2_idm_policy(proposal_sampling, cfg.map_root_override)
        stage2_tracks = stage2_reactive_policy.simulate_environment(stage2_states, stage2_cache)

        final_ego_state = state_array_to_ego_state(
            np.asarray(stage2_states[-1]),
            split_ego_state.time_point + TimeDuration.from_s(proposal_sampling.time_horizon),
            split_ego_state.car_footprint.vehicle_parameters,
        )
        stage2_map = PDMDrivableMap.from_simulation(map_api, split_ego_state, cfg.map_radius)
        stage2_status = _step_status(
            final_ego_state,
            stage2_tracks[-1],
            stage2_map,
            original_lane,
        )
        if stage2_status.collision:
            _bump("stage2_collision")
            continue
        if not stage2_status.in_drivable_area:
            _bump("stage2_offroad")
            continue
        if not stage2_status.in_original_lane:
            _bump("stage2_not_back_in_lane")
            continue
        if stage2_status.center_deviation_m > cfg.recovery_max_center_dev_m:
            _bump("stage2_center_dev")
            continue
        if stage2_status.heading_error_deg > cfg.recovery_heading_error_deg:
            _bump("stage2_heading")
            continue

        rear_start = Point(float(split_ego_state.rear_axle.x), float(split_ego_state.rear_axle.y))
        rear_end = Point(float(final_ego_state.rear_axle.x), float(final_ego_state.rear_axle.y))
        progress_m = rear_end.distance(rear_start)
        if progress_m < cfg.min_stage2_progress_m:
            _bump("stage2_low_progress")
            continue

        stage1_summary = StageSummary(
            steps=split_idx,
            elapsed_s=split_idx * cfg.interval,
            center_deviation_m=stage1_status.center_deviation_m,
            heading_error_deg=stage1_status.heading_error_deg,
            min_vehicle_distance_m=stage1_status.min_vehicle_distance_m,
            in_original_lane=stage1_status.in_original_lane,
            in_drivable_area=stage1_status.in_drivable_area,
            recovered=False,
        )
        stage2_summary = StageSummary(
            steps=proposal_sampling.num_poses,
            elapsed_s=proposal_sampling.time_horizon,
            center_deviation_m=stage2_status.center_deviation_m,
            heading_error_deg=stage2_status.heading_error_deg,
            min_vehicle_distance_m=stage2_status.min_vehicle_distance_m,
            in_original_lane=stage2_status.in_original_lane,
            in_drivable_area=stage2_status.in_drivable_area,
            recovered=True,
        )
        stage1_score = _stage_score_from_summary(stage1_summary, "stage1_boundary_onset")
        stage1_score["proposal_idx"] = int(proposal_idx)
        stage1_score["proposal_rank"] = float(stage1_rank)
        stage1_score["split_idx"] = int(split_idx)
        stage2_score = _stage_score_from_summary(stage2_summary, "stage2_mobil_recovery")
        stage2_score["progress_m"] = float(progress_m)

        dummy_spec = InducementSpec(
            actor_token="ego",
            mode="proposal_boundary",
            start_time_s=0.0,
            preferred_action=None,
            idm_params=IDMParams(),
            mobil_params=MobilParams(),
            score_hint=float(stage1_rank),
        )
        record = {
            "token": token,
            "result_status": "success",
            "stage1_score": stage1_score,
            "stage2_score": stage2_score,
            "stage1_candidate_rank": int(candidate_rank),
        }
        payload = _build_trace_payload(
            token=token,
            inducement_spec=dummy_spec,
            stage1_states=stage1_states,
            stage1_tracks=stage1_tracks,
            stage2_states=stage2_states,
            stage2_tracks=stage2_tracks,
            stage1_score=stage1_score,
            stage2_score=stage2_score,
        )
        kept_results.append((_recovery_rank(stage1_summary, stage2_summary), record, payload))

    kept_results.sort(key=lambda item: item[0], reverse=True)
    kept_results = kept_results[: cfg.top_k_per_token]

    manifest_records: List[str] = []
    for final_rank, (rank_score, record, payload) in enumerate(kept_results):
        record["final_rank"] = int(final_rank)
        record["recovery_rank"] = float(rank_score)
        if cfg.save_format != "none":
            stem = _candidate_stem(token, final_rank, inducement_specs[record["spec_rank"]])
            record.update(_save_payload(output_dir, stem, payload, cfg.save_format))
        manifest_records.append(json.dumps(record, ensure_ascii=False))

    top_reject_reason = None
    if reject_counts:
        top_reject_reason = max(reject_counts.items(), key=lambda item: item[1])[0]
    if not manifest_records:
        manifest_records.append(
            json.dumps(
                {
                    "token": token,
                    "result_status": "rejected",
                    "reject_reason": top_reject_reason or "unknown",
                    "reject_counts": reject_counts,
                    "attempted_specs": len(stage1_candidates),
                },
                ensure_ascii=False,
            )
        )

    return {
        "token": token,
        "kept": len(kept_results),
        "attempted_specs": len(stage1_candidates),
        "manifest_records": manifest_records,
        "reject_counts": reject_counts,
        "top_reject_reason": top_reject_reason,
    }


def _build_run_config(args: argparse.Namespace, output_dir: Path) -> RunConfig:
    return RunConfig(
        output_dir=str(output_dir),
        save_format=args.save_format,
        top_k_per_token=int(args.top_k_per_token),
        interval=float(args.interval),
        horizon_sec=float(args.horizon_sec),
        map_radius=float(args.map_radius),
        stage1_max_steps=int(args.stage1_max_steps),
        stage2_max_steps=int(args.stage2_max_steps),
        min_stage1_steps=int(args.min_stage1_steps),
        recovery_hold_steps=int(args.recovery_hold_steps),
        boundary_min_center_dev_m=float(args.boundary_min_center_dev_m),
        recovery_heading_error_deg=float(args.recovery_heading_error_deg),
        recovery_max_center_dev_m=float(args.recovery_max_center_dev_m),
        min_stage2_progress_m=float(args.min_stage2_progress_m),
        top_k_specs=int(args.top_k_specs),
        candidate_radius_m=float(args.candidate_radius_m),
        parallel_backend=str(args.parallel_backend),
        num_workers=int(args.num_workers),
        map_root_override=args.map_root_override,
    )


def _init_worker(metric_cache_path: str, run_cfg_dict: Dict[str, Any]) -> None:
    global _CTX
    cfg = RunConfig(**run_cfg_dict)
    proposal_sampling = TrajectorySampling(
        num_poses=int(round(cfg.horizon_sec / cfg.interval)),
        interval_length=cfg.interval,
    )
    loader = _load_metric_cache_loader(Path(metric_cache_path))
    _CTX = {
        "loader": loader,
        "cfg": cfg,
        "proposal_sampling": proposal_sampling,
    }


def _run_token_worker(token: str) -> Dict[str, Any]:
    global _CTX
    assert _CTX is not None
    metric_cache = _CTX["loader"].get_from_token(token)
    return _process_token(
        token=token,
        metric_cache=metric_cache,
        cfg=_CTX["cfg"],
        proposal_sampling=_CTX["proposal_sampling"],
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_root = resolve_metric_cache_root(args.metric_cache_path)
    loader = _load_metric_cache_loader(cache_root)
    tokens = _load_tokens(loader, args.tokens_file, args.max_scenes)
    run_cfg = _build_run_config(args, output_dir)

    manifest_lines: List[str] = []
    stats: Dict[str, Any] = {
        "num_tokens": len(tokens),
        "num_candidates": 0,
        "tokens_with_candidates": 0,
        "attempted_specs": 0,
        "reject_counts": {},
    }

    if args.parallel_backend == "none":
        _init_worker(str(cache_root), asdict(run_cfg))
        for idx, token in enumerate(tokens, start=1):
            result = _run_token_worker(token)
            manifest_lines.extend(result["manifest_records"])
            stats["num_candidates"] += int(result["kept"])
            stats["attempted_specs"] += int(result["attempted_specs"])
            if int(result["kept"]) > 0:
                stats["tokens_with_candidates"] += 1
            for reason, count in result.get("reject_counts", {}).items():
                stats["reject_counts"][reason] = stats["reject_counts"].get(reason, 0) + int(count)
            print(
                f"[{idx}/{len(tokens)}] {token}: kept={int(result['kept'])}, "
                f"specs={int(result['attempted_specs'])}, "
                f"top_reject={result.get('top_reject_reason')}"
            )
    else:
        num_workers = _auto_workers(args.num_workers)
        ordered_results: Dict[int, Dict[str, Any]] = {}
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(str(cache_root), asdict(run_cfg)),
        ) as ex:
            future_map = {ex.submit(_run_token_worker, token): (idx, token) for idx, token in enumerate(tokens, start=1)}
            for future in as_completed(future_map):
                idx, token = future_map[future]
                result = future.result()
                ordered_results[idx] = result
                print(
                    f"[{idx}/{len(tokens)}] {token}: kept={int(result['kept'])}, "
                    f"specs={int(result['attempted_specs'])}, "
                    f"top_reject={result.get('top_reject_reason')}"
                )
        for idx in range(1, len(tokens) + 1):
            result = ordered_results[idx]
            manifest_lines.extend(result["manifest_records"])
            stats["num_candidates"] += int(result["kept"])
            stats["attempted_specs"] += int(result["attempted_specs"])
            if int(result["kept"]) > 0:
                stats["tokens_with_candidates"] += 1
            for reason, count in result.get("reject_counts", {}).items():
                stats["reject_counts"][reason] = stats["reject_counts"].get(reason, 0) + int(count)

    manifest_path = output_dir / args.manifest_name
    manifest_path.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")
    (output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
