#!/usr/bin/env python3
"""
Generate collision-oriented counterfactual pseudo-simulation traces on NAVSIM.

Design goals for V1:
- Keep ego planning one-shot, matching NAVSIM pseudo-sim assumptions.
- Search over targeted background-vehicle attacks instead of ego perturbations.
- Support three attack modes: cut-in, brake-check, intersection-seize.
- Reuse metric cache + PDM scorer + reactive traffic rollout from the existing stack.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import generate_ood_mini as base
except ModuleNotFoundError:
    from scripts import generate_ood_mini as base
from navsim.behavior.idm import IDM, IDMParams
from navsim.behavior.lane_change_trajectory import propagate_actor
from navsim.behavior.mobil import KEEP, LEFT, RIGHT, MobilModel, MobilParams, decide_with_mobil
from navsim.behavior.scene_adapter import (
    ActorState,
    build_actor_from_detection_track,
    build_ego_actor_from_ego_state,
    build_neighbor_set,
)
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from navsim.traffic_agents_policies.mobil_traffic_agents import MobilTrafficAgentsPolicy


EPS = 1e-6
WorkerContext = Dict[str, Any]
_CTX: Optional[WorkerContext] = None


@dataclass
class AttackSpec:
    actor_token: str
    mode: str
    start_time_s: float
    preferred_action: Optional[str]
    idm_params: IDMParams
    mobil_params: MobilParams
    score_hint: float


@dataclass
class SearchConfig:
    candidate_radius_m: float = 35.0
    top_k_actors: int = 5
    top_k_attacks: int = 3
    min_risk_gain: float = 0.15
    keep_drivable_area: bool = True
    keep_driving_direction: bool = True
    keep_traffic_light: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate collision counterfactual pseudo-sim traces.")
    parser.add_argument("--metric-cache-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("generated_collision_pseudo_data"))
    parser.add_argument("--save-format", type=str, default="both", choices=["json", "pkl", "both"])
    parser.add_argument("--manifest-name", type=str, default="manifest.jsonl")
    parser.add_argument("--map-root-override", type=str, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--horizon-sec", type=float, default=4.0)
    parser.add_argument("--candidate-radius-m", type=float, default=35.0)
    parser.add_argument("--top-k-actors", type=int, default=5)
    parser.add_argument("--top-k-attacks", type=int, default=3)
    parser.add_argument("--min-risk-gain", type=float, default=0.15)
    parser.add_argument("--allow-dac-drop", action="store_true")
    parser.add_argument("--allow-ddc-drop", action="store_true")
    parser.add_argument("--allow-tlc-drop", action="store_true")
    parser.add_argument("--parallel-backend", type=str, default="process", choices=["process", "none"])
    parser.add_argument("--num-workers", type=int, default=0, help="0 means auto, capped to 32.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _auto_workers(num_workers: int) -> int:
    if num_workers > 0:
        return num_workers
    try:
        import os

        count = os.cpu_count() or 1
    except Exception:
        count = 1
    return max(1, min(count, 32))


def _token_seed(base_seed: int, token: str) -> int:
    h = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)
    return (base_seed + h) % (2**32 - 1)


def _sanitize_fragment(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    token = "".join(out).strip("_")
    return token or "unknown"


def _vehicle_min_dist(states: np.ndarray, tracks: Sequence[Any]) -> float:
    d_min = float("inf")
    for i in range(min(len(states), len(tracks))):
        ex = float(states[i, StateIndex.X])
        ey = float(states[i, StateIndex.Y])
        for obj in tracks[i].tracked_objects.tracked_objects:
            if str(obj.tracked_object_type).lower().endswith("vehicle"):
                d = math.hypot(float(obj.center.x) - ex, float(obj.center.y) - ey)
                if d < d_min:
                    d_min = d
    return d_min if d_min < float("inf") else float("nan")


def _rotate_to_actor_frame(dx: float, dy: float, heading: float) -> Tuple[float, float]:
    c = math.cos(heading)
    s = math.sin(heading)
    lon = c * dx + s * dy
    lat = -s * dx + c * dy
    return lon, lat


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


def _is_in_intersection(actor: ActorState, map_api) -> bool:
    return map_api.is_in_layer(Point2D(actor.x, actor.y), SemanticMapLayer.INTERSECTION)


def _actor_distance(actor: ActorState, ego_actor: ActorState) -> float:
    return math.hypot(actor.x - ego_actor.x, actor.y - ego_actor.y)


def _cut_in_specs(actor: ActorState, ego_actor: ActorState, preferred_action: str, distance: float) -> List[AttackSpec]:
    score_hint = 2.5 + max(0.0, (35.0 - distance) / 20.0)
    return [
        AttackSpec(
            actor_token=actor.token,
            mode="cut_in",
            start_time_s=start_time_s,
            preferred_action=preferred_action,
            idm_params=IDMParams(
                v0=max(actor.speed + 4.0, 12.0),
                T=0.9,
                s0=1.0,
                a=2.5,
                b=3.0,
                delta=4,
            ),
            mobil_params=MobilParams(
                politeness=0.0,
                accel_threshold=0.0,
                safe_decel=4.0,
                cooldown_s=3.0,
                route_bias=1.5,
            ),
            score_hint=score_hint - 0.1 * idx,
        )
        for idx, start_time_s in enumerate((0.2, 0.6, 1.0))
    ]


def _brake_check_specs(actor: ActorState, ego_actor: ActorState, longitudinal_gap: float) -> List[AttackSpec]:
    score_hint = 2.0 + max(0.0, (25.0 - longitudinal_gap) / 15.0)
    return [
        AttackSpec(
            actor_token=actor.token,
            mode="brake_check",
            start_time_s=start_time_s,
            preferred_action=None,
            idm_params=IDMParams(
                v0=max(1.5, actor.speed * speed_scale),
                T=2.2,
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
        for idx, (start_time_s, speed_scale) in enumerate(((0.3, 0.35), (0.7, 0.25), (1.1, 0.20)))
    ]


def _intersection_specs(actor: ActorState, ego_actor: ActorState, distance: float) -> List[AttackSpec]:
    score_hint = 1.8 + max(0.0, (35.0 - distance) / 20.0)
    return [
        AttackSpec(
            actor_token=actor.token,
            mode="intersection_seize",
            start_time_s=start_time_s,
            preferred_action=KEEP,
            idm_params=IDMParams(
                v0=max(actor.speed + 5.0, 13.0),
                T=0.8,
                s0=1.0,
                a=2.8,
                b=3.0,
                delta=4,
            ),
            mobil_params=MobilParams(
                politeness=0.1,
                accel_threshold=0.0,
                safe_decel=4.0,
                cooldown_s=3.0,
                route_bias=0.4,
            ),
            score_hint=score_hint - 0.1 * idx,
        )
        for idx, start_time_s in enumerate((0.1, 0.5, 0.9))
    ]


def enumerate_attack_specs(metric_cache: MetricCache, search_cfg: SearchConfig) -> List[AttackSpec]:
    map_params = metric_cache.map_parameters
    map_api = get_maps_api(map_params.map_root, map_params.map_version, map_params.map_name)
    ego_actor = build_ego_actor_from_ego_state(metric_cache.ego_state, map_api)
    current_tracks = metric_cache.current_tracked_objects[0].tracked_objects.tracked_objects
    actors = [build_actor_from_detection_track(track, map_api) for track in current_tracks if str(track.tracked_object_type).lower().endswith("vehicle")]

    specs: List[AttackSpec] = []
    for actor in actors:
        distance = _actor_distance(actor, ego_actor)
        if distance > search_cfg.candidate_radius_m:
            continue

        dx = actor.x - ego_actor.x
        dy = actor.y - ego_actor.y
        rel_lon, rel_lat = _rotate_to_actor_frame(dx, dy, ego_actor.heading)

        if _same_lane(actor, ego_actor) and rel_lon > 2.0:
            specs.extend(_brake_check_specs(actor, ego_actor, rel_lon))
            continue

        preferred_action = _adjacent_action_to_ego(actor, ego_actor)
        if preferred_action is not None and -12.0 <= rel_lon <= 18.0 and 1.0 <= abs(rel_lat) <= 10.0:
            specs.extend(_cut_in_specs(actor, ego_actor, preferred_action, distance))
            continue

        if _is_in_intersection(actor, map_api) or _is_in_intersection(ego_actor, map_api):
            specs.extend(_intersection_specs(actor, ego_actor, distance))

    specs.sort(key=lambda spec: spec.score_hint, reverse=True)
    actor_budget: Dict[str, int] = {}
    selected: List[AttackSpec] = []
    for spec in specs:
        actor_count = actor_budget.get(spec.actor_token, 0)
        if actor_count >= search_cfg.top_k_actors:
            continue
        selected.append(spec)
        actor_budget[spec.actor_token] = actor_count + 1
        if len(selected) >= max(search_cfg.top_k_attacks * 4, search_cfg.top_k_attacks):
            break
    return selected


class CounterfactualMobilTrafficAgentsPolicy(MobilTrafficAgentsPolicy):
    """Targeted attack wrapper around the MOBIL reactive traffic policy."""

    def __init__(self, attack_spec: AttackSpec, **kwargs: Any):
        super().__init__(**kwargs)
        self._attack_spec = attack_spec

    def _attack_active(self, token: str, timestep: int) -> bool:
        if token != self._attack_spec.actor_token:
            return False
        current_time_s = max(0.0, (timestep - 1) * self.future_trajectory_sampling.interval_length)
        return current_time_s >= self._attack_spec.start_time_s - EPS

    def simulate_traffic_agents(self, simulated_ego_states: np.ndarray, metric_cache: MetricCache) -> List[Any]:
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
        future_tracks: List[Any] = []

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

            timestep_traffic_light_status = {}
            if traffic_light_status is not None and timestep < len(traffic_light_status):
                timestep_traffic_light_status = self._normalize_traffic_light_status(traffic_light_status[timestep])

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
                    traffic_light_status=timestep_traffic_light_status,
                    map_api=map_api,
                )

                lane_change_allowed = runtime_actor.cooldown_remaining <= 0.0 and runtime_actor.lane_change_elapsed <= 0.0
                route_command = None
                idm = self._idm
                mobil = self._mobil

                if self._attack_active(token, timestep):
                    idm = IDM(self._attack_spec.idm_params)
                    mobil = MobilModel(self._attack_spec.mobil_params, idm)
                    route_command = _action_to_route_command(self._attack_spec.preferred_action, self._attack_spec.mobil_params)
                    if self._attack_spec.mode in ("brake_check", "intersection_seize"):
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

                if self._attack_active(token, timestep):
                    if self._attack_spec.mode == "brake_check":
                        acceleration = min(acceleration, -2.0)
                        action = KEEP
                        target_lane = actor.lane_object
                    elif self._attack_spec.mode == "intersection_seize":
                        accel_cap = acceleration_caps.get(KEEP, np.inf)
                        acceleration = min(max(acceleration, 0.5), accel_cap)
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


def build_attack_policy(
    proposal_sampling: TrajectorySampling,
    attack_spec: Optional[AttackSpec],
    map_root_override: Optional[str] = None,
):
    kwargs: Dict[str, Any] = dict(
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
    if attack_spec is None:
        return MobilTrafficAgentsPolicy(**kwargs)
    return CounterfactualMobilTrafficAgentsPolicy(attack_spec=attack_spec, **kwargs)


def compute_ego_trajectory(
    metric_cache: MetricCache,
    proposal_sampling: TrajectorySampling,
    map_root_override: Optional[str] = None,
) -> Tuple[np.ndarray, Any]:
    planner = base.build_expert_planner(proposal_sampling=proposal_sampling)
    planner_init = base.build_planner_initialization(metric_cache, map_root_override=map_root_override)
    planner.initialize(planner_init)

    traffic_light_data = []
    if getattr(metric_cache, "traffic_light_status", None):
        traffic_light_data = list(metric_cache.traffic_light_status[0])

    planner_input = PlannerInput(
        iteration=SimulationIteration(index=0, time_point=metric_cache.ego_state.time_point),
        history=SimulationHistoryBuffer.initialize_from_list(
            buffer_size=1,
            ego_states=[metric_cache.ego_state],
            observations=[metric_cache.current_tracked_objects[0]],
        ),
        traffic_light_data=traffic_light_data,
    )
    trajectory = planner.compute_planner_trajectory(planner_input)
    states = base.get_trajectory_as_array(trajectory, proposal_sampling, metric_cache.ego_state.time_point)
    return states, trajectory


def score_rollout(
    scorer: PDMScorer,
    metric_cache: MetricCache,
    ego_states: np.ndarray,
    tracks: Sequence[Any],
) -> Dict[str, float]:
    observation = copy.deepcopy(metric_cache.observation)
    pdm_row = scorer.score_proposals(
        states=ego_states[None, ...],
        observation=observation,
        centerline=metric_cache.centerline,
        route_lane_ids=metric_cache.route_lane_ids,
        drivable_area_map=metric_cache.drivable_area_map,
        map_parameters=metric_cache.map_parameters,
        simulated_agent_detections_tracks=tracks,
        human_past_trajectory=metric_cache.past_human_trajectory,
    )[0].iloc[0]

    return {
        "no_at_fault_collisions": float(pdm_row.get("no_at_fault_collisions", float("nan"))),
        "drivable_area_compliance": float(pdm_row.get("drivable_area_compliance", float("nan"))),
        "driving_direction_compliance": float(pdm_row.get("driving_direction_compliance", float("nan"))),
        "traffic_light_compliance": float(pdm_row.get("traffic_light_compliance", float("nan"))),
        "ego_progress": float(pdm_row.get("ego_progress", float("nan"))),
        "time_to_collision_within_bound": float(pdm_row.get("time_to_collision_within_bound", 1.0)),
        "pdm_score": float(pdm_row.get("pdm_score", float("nan"))),
    }


def attack_risk_score(metrics: Dict[str, float], min_dist_m: float) -> float:
    ttc_term = 1.0 - float(metrics.get("time_to_collision_within_bound", 1.0))
    collision_term = 1.0 - float(metrics.get("no_at_fault_collisions", 1.0))
    pdm_term = 1.0 - float(metrics.get("pdm_score", 1.0))
    dist_term = 0.0 if math.isnan(min_dist_m) else max(0.0, min(1.0, (4.0 - min_dist_m) / 4.0))
    return 2.5 * collision_term + 1.5 * ttc_term + 0.8 * pdm_term + 0.7 * dist_term


def admissible_attack(metrics: Dict[str, float], search_cfg: SearchConfig) -> bool:
    if search_cfg.keep_drivable_area and float(metrics["drivable_area_compliance"]) < 1.0 - EPS:
        return False
    if search_cfg.keep_driving_direction and float(metrics["driving_direction_compliance"]) < 1.0 - EPS:
        return False
    if search_cfg.keep_traffic_light and float(metrics["traffic_light_compliance"]) < 1.0 - EPS:
        return False
    return True


def serialize_result(
    token: str,
    attack_spec: AttackSpec,
    ego_states: np.ndarray,
    tracks: Sequence[Any],
    nominal_metrics: Dict[str, float],
    attacked_metrics: Dict[str, float],
    nominal_min_dist: float,
    attacked_min_dist: float,
    risk_gain: float,
) -> Dict[str, Any]:
    frames: List[Dict[str, Any]] = []
    for t in range(min(len(ego_states), len(tracks))):
        ego = ego_states[t]
        vehicles = []
        for obj in tracks[t].tracked_objects.tracked_objects:
            if str(obj.tracked_object_type).lower().endswith("vehicle"):
                vehicles.append(
                    {
                        "id": str(obj.track_token),
                        "x": float(obj.center.x),
                        "y": float(obj.center.y),
                        "heading": float(obj.center.heading),
                        "length": float(obj.box.length),
                        "width": float(obj.box.width),
                    }
                )
        frames.append(
            {
                "t_idx": t,
                "ego": {
                    "x": float(ego[StateIndex.X]),
                    "y": float(ego[StateIndex.Y]),
                    "heading": float(ego[StateIndex.HEADING]),
                    "speed": float(math.hypot(float(ego[StateIndex.VELOCITY_X]), float(ego[StateIndex.VELOCITY_Y]))),
                },
                "vehicles": vehicles,
            }
        )

    return {
        "token": token,
        "attack_spec": {
            "actor_token": attack_spec.actor_token,
            "mode": attack_spec.mode,
            "start_time_s": attack_spec.start_time_s,
            "preferred_action": attack_spec.preferred_action,
            "idm_params": asdict(attack_spec.idm_params),
            "mobil_params": asdict(attack_spec.mobil_params),
            "score_hint": attack_spec.score_hint,
        },
        "nominal_metrics": nominal_metrics,
        "attacked_metrics": attacked_metrics,
        "nominal_min_dist_m": nominal_min_dist,
        "attacked_min_dist_m": attacked_min_dist,
        "risk_gain": risk_gain,
        "num_frames": len(frames),
        "frames": frames,
    }


def save_result(result: Dict[str, Any], output_dir: Path, fmt: str) -> None:
    token = _sanitize_fragment(str(result["token"]))
    actor = _sanitize_fragment(str(result["attack_spec"]["actor_token"]))
    mode = _sanitize_fragment(str(result["attack_spec"]["mode"]))
    start_time_s = float(result["attack_spec"].get("start_time_s", 0.0))
    start_tag = _sanitize_fragment(f"{start_time_s:.2f}s")
    stem = f"collision_cf_{token}_{actor}_{mode}_{start_tag}"
    if fmt in ("json", "both"):
        with (output_dir / f"{stem}.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    if fmt in ("pkl", "both"):
        with (output_dir / f"{stem}.pkl").open("wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)


def _init_worker(
    metric_cache_path: str,
    interval: float,
    horizon_sec: float,
    search_cfg_dict: Dict[str, Any],
    map_root_override: Optional[str],
    verbose: bool,
) -> None:
    global _CTX
    cache_root = base.resolve_metric_cache_root(Path(metric_cache_path))
    loader = base.build_metric_cache_loader(cache_root)
    proposal_sampling = TrajectorySampling(
        num_poses=int(round(horizon_sec / interval)),
        interval_length=interval,
    )
    _CTX = {
        "loader": loader,
        "proposal_sampling": proposal_sampling,
        "search_cfg": SearchConfig(**search_cfg_dict),
        "map_root_override": map_root_override,
        "verbose": verbose,
    }


def _run_token_worker(token: str, seed: int) -> Dict[str, Any]:
    global _CTX
    assert _CTX is not None
    np.random.seed(seed)
    try:
        metric_cache = _CTX["loader"].get_from_token(token)
        results = run_for_token(
            token=token,
            metric_cache=metric_cache,
            proposal_sampling=_CTX["proposal_sampling"],
            map_root_override=_CTX["map_root_override"],
            search_cfg=_CTX["search_cfg"],
            verbose=bool(_CTX["verbose"]),
        )
        return {"token": token, "results": results, "error": None}
    except Exception as exc:
        return {"token": token, "results": [], "error": str(exc)}


def run_for_token(
    token: str,
    metric_cache: MetricCache,
    proposal_sampling: TrajectorySampling,
    map_root_override: Optional[str],
    search_cfg: SearchConfig,
    verbose: bool,
) -> List[Dict[str, Any]]:
    if map_root_override is not None:
        metric_cache.map_parameters.map_root = map_root_override

    ego_states, _ = compute_ego_trajectory(metric_cache, proposal_sampling, map_root_override)
    scorer = PDMScorer(proposal_sampling=proposal_sampling, config=PDMScorerConfig(human_penalty_filter=False))

    nominal_policy = build_attack_policy(proposal_sampling, attack_spec=None, map_root_override=map_root_override)
    nominal_tracks = nominal_policy.simulate_environment(ego_states, metric_cache)
    nominal_metrics = score_rollout(scorer, metric_cache, ego_states, nominal_tracks)
    nominal_min_dist = _vehicle_min_dist(ego_states, nominal_tracks)
    nominal_risk = attack_risk_score(nominal_metrics, nominal_min_dist)

    attack_specs = enumerate_attack_specs(metric_cache, search_cfg)
    if verbose:
        print(f"[{token}] enumerated {len(attack_specs)} attack specs")

    results: List[Dict[str, Any]] = []
    for spec in attack_specs:
        attacked_policy = build_attack_policy(proposal_sampling, attack_spec=spec, map_root_override=map_root_override)
        attacked_tracks = attacked_policy.simulate_environment(ego_states, metric_cache)
        attacked_metrics = score_rollout(scorer, metric_cache, ego_states, attacked_tracks)
        attacked_min_dist = _vehicle_min_dist(ego_states, attacked_tracks)
        attacked_risk = attack_risk_score(attacked_metrics, attacked_min_dist)
        risk_gain = attacked_risk - nominal_risk

        if not admissible_attack(attacked_metrics, search_cfg):
            continue
        if risk_gain < search_cfg.min_risk_gain:
            continue

        results.append(
            serialize_result(
                token=token,
                attack_spec=spec,
                ego_states=ego_states,
                tracks=attacked_tracks,
                nominal_metrics=nominal_metrics,
                attacked_metrics=attacked_metrics,
                nominal_min_dist=nominal_min_dist,
                attacked_min_dist=attacked_min_dist,
                risk_gain=risk_gain,
            )
        )

    results.sort(key=lambda item: float(item["risk_gain"]), reverse=True)
    return results[: search_cfg.top_k_attacks]


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    cache_root = base.resolve_metric_cache_root(args.metric_cache_path)
    metric_cache_loader = base.build_metric_cache_loader(cache_root)
    tokens = list(metric_cache_loader.tokens)
    if args.max_scenes is not None:
        tokens = tokens[: args.max_scenes]

    search_cfg = SearchConfig(
        candidate_radius_m=float(args.candidate_radius_m),
        top_k_actors=int(args.top_k_actors),
        top_k_attacks=int(args.top_k_attacks),
        min_risk_gain=float(args.min_risk_gain),
        keep_drivable_area=not bool(args.allow_dac_drop),
        keep_driving_direction=not bool(args.allow_ddc_drop),
        keep_traffic_light=not bool(args.allow_tlc_drop),
    )
    search_cfg_dict = asdict(search_cfg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / args.manifest_name
    saved = 0

    print(f"Processing {len(tokens)} scene tokens from {cache_root}")
    with manifest_path.open("w", encoding="utf-8") as manifest_fp:
        if args.parallel_backend == "none":
            _init_worker(
                str(cache_root),
                float(args.interval),
                float(args.horizon_sec),
                search_cfg_dict,
                args.map_root_override,
                bool(args.verbose),
            )
            iterator = ((_run_token_worker(token, _token_seed(args.seed, token)), idx, token) for idx, token in enumerate(tokens))
            for worker_result, idx, token in iterator:
                if args.verbose:
                    print(f"[{idx + 1}/{len(tokens)}] token={token}")
                if worker_result["error"] is not None:
                    print(f"[warn] token={token} failed: {worker_result['error']}")
                    continue
                for result in worker_result["results"]:
                    save_result(result, args.output_dir, args.save_format)
                    manifest_fp.write(
                        json.dumps(
                            {
                                "token": result["token"],
                                "attack_spec": result["attack_spec"],
                                "risk_gain": result["risk_gain"],
                                "nominal_metrics": result["nominal_metrics"],
                                "attacked_metrics": result["attacked_metrics"],
                                "nominal_min_dist_m": result["nominal_min_dist_m"],
                                "attacked_min_dist_m": result["attacked_min_dist_m"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    saved += 1
        else:
            num_workers = _auto_workers(int(args.num_workers))
            print(f"Running process pool with workers={num_workers}")
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_init_worker,
                initargs=(
                    str(cache_root),
                    float(args.interval),
                    float(args.horizon_sec),
                    search_cfg_dict,
                    args.map_root_override,
                    bool(args.verbose),
                ),
            ) as ex:
                futs = {
                    ex.submit(_run_token_worker, token, _token_seed(args.seed, token)): (idx, token)
                    for idx, token in enumerate(tokens)
                }
                completed = 0
                for fut in as_completed(futs):
                    completed += 1
                    idx, token = futs[fut]
                    if args.verbose:
                        print(f"[{completed}/{len(tokens)}] token={token}")
                    worker_result = fut.result()
                    if worker_result["error"] is not None:
                        print(f"[warn] token={token} failed: {worker_result['error']}")
                        continue
                    for result in worker_result["results"]:
                        save_result(result, args.output_dir, args.save_format)
                        manifest_fp.write(
                            json.dumps(
                                {
                                    "token": result["token"],
                                    "attack_spec": result["attack_spec"],
                                    "risk_gain": result["risk_gain"],
                                    "nominal_metrics": result["nominal_metrics"],
                                    "attacked_metrics": result["attacked_metrics"],
                                    "nominal_min_dist_m": result["nominal_min_dist_m"],
                                    "attacked_min_dist_m": result["attacked_min_dist_m"],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        saved += 1

    print(f"Done. saved={saved}, output_dir={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
