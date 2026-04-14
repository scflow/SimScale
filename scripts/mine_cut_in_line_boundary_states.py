#!/usr/bin/env python3
"""
Mine recoverable cut-in-line boundary states from standard NAVSIM metric caches.

The script stays close to NAVSIM's native PDM planning flow:
1) load a standard MetricCache;
2) instantiate a standard PDMClosedPlanner-style proposal stack;
3) score all proposals with the native PDMScorer;
4) keep proposals whose endpoint is near a lane boundary while still admissible.

The output is a manifest of stage-one boundary-state candidates that can later be
turned into stage-two follow-up scenes.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from nuplan.common.actor_state.state_representation import TimeDuration
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from shapely.geometry import Point

from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import get_trajectory_as_array
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.observation.navsim_idm_agents import NavsimIDMAgents
from navsim.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    state_array_to_coords_array,
    state_array_to_ego_state,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex, StateIndex
from navsim.traffic_agents_policies.navsim_IDM_traffic_agents import NavsimIDMTrafficAgents

WorkerContext = Dict[str, Any]
_CTX: Optional[WorkerContext] = None


@dataclass
class CandidateSummary:
    token: str
    proposal_idx: int
    lateral_idx: int
    longitudinal_idx: int
    lateral_offset_m: float
    pdm_score: float
    no_at_fault_collisions: float
    drivable_area_compliance: float
    driving_direction_compliance: float
    traffic_light_compliance: float
    ego_progress: float
    time_to_collision_within_bound: float
    lane_keeping: float
    history_comfort: float
    max_lateral_deviation_m: float
    end_lateral_deviation_m: float
    end_heading_error_deg: float
    end_speed_mps: float
    end_x: float
    end_y: float


@dataclass
class RunConfig:
    output_dir: str
    save_format: str
    top_k_per_token: int
    stage1_eval_top_k: int
    interval: float
    horizon_sec: float
    map_radius: float
    lateral_offsets: List[float]
    speed_limit_fractions: List[float]
    fallback_target_velocity: float
    min_gap_to_lead_agent: float
    headway_time: float
    accel_max: float
    decel_max: float
    min_end_lateral_dev_m: float
    max_end_lateral_dev_m: float
    min_max_lateral_dev_m: float
    max_max_lateral_dev_m: float
    max_end_heading_error_deg: float
    min_end_speed_mps: float
    max_end_speed_mps: float
    min_pdm_score: float
    min_ttc_within_bound: float
    skip_intersection_candidates: bool
    max_join_translation_m: float
    max_join_heading_deg: float
    max_join_speed_delta_mps: float
    recovery_max_end_lateral_dev_m: float
    recovery_max_end_heading_error_deg: float
    recovery_min_lateral_improvement_m: float
    recovery_min_progress: float


def _score_row_to_dict(row: pd.Series) -> Dict[str, float]:
    return {
        "pdm_score": float(row.get("pdm_score", float("nan"))),
        "no_at_fault_collisions": float(row.get("no_at_fault_collisions", float("nan"))),
        "drivable_area_compliance": float(row.get("drivable_area_compliance", float("nan"))),
        "driving_direction_compliance": float(row.get("driving_direction_compliance", float("nan"))),
        "traffic_light_compliance": float(row.get("traffic_light_compliance", float("nan"))),
        "ego_progress": float(row.get("ego_progress", float("nan"))),
        "time_to_collision_within_bound": float(row.get("time_to_collision_within_bound", float("nan"))),
        "lane_keeping": float(row.get("lane_keeping", float("nan"))),
        "history_comfort": float(row.get("history_comfort", float("nan"))),
    }


class BoundaryMiningPDMPlanner(PDMClosedPlanner):
    """Expose native PDM proposal batches before argmax selection."""

    def compute_all_proposals(self, current_input: PlannerInput) -> Tuple[np.ndarray, pd.DataFrame]:
        ego_state, observation = current_input.history.current_state

        if self._iteration == 0:
            self._route_roadblock_correction(ego_state)

        self._drivable_area_map = PDMDrivableMap.from_simulation(self._map_api, ego_state, self._map_radius)

        self._observation.update(
            ego_state,
            observation,
            current_input.traffic_light_data,
            self._route_lane_dict,
        )
        self._update_proposal_manager(ego_state)

        proposals_array = self._generator.generate_proposals(ego_state, self._observation, self._proposal_manager)
        simulated_proposals = self._simulator.simulate_proposals(proposals_array, ego_state)
        pdm_results = self._scorer.score_proposals(
            simulated_proposals,
            self._observation,
            self._centerline,
            list(self._route_lane_dict.keys()),
            self._drivable_area_map,
        )
        proposal_df = pd.concat(pdm_results, ignore_index=True)
        self._iteration += 1
        return simulated_proposals, proposal_df


class PreferredLanePDMPlanner(BoundaryMiningPDMPlanner):
    """Keep rescue planning anchored to a specific on-route starting lane."""

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
        if self._preferred_start_lane_id and self._route_lane_dict and self._preferred_start_lane_id in self._route_lane_dict:
            return self._route_lane_dict[self._preferred_start_lane_id]
        return super()._get_starting_lane(ego_state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine standard NAVSIM cut-in-line boundary states.")
    parser.add_argument("--metric-cache-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/cut_in_line_boundary_states"))
    parser.add_argument("--manifest-name", type=str, default="manifest.jsonl")
    parser.add_argument("--save-format", type=str, default="both", choices=["json", "pkl", "both", "none"])
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--tokens-file", type=Path, default=None)
    parser.add_argument("--top-k-per-token", type=int, default=3)
    parser.add_argument("--stage1-eval-top-k", type=int, default=12)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--horizon-sec", type=float, default=4.0)
    parser.add_argument("--map-radius", type=float, default=100.0)
    parser.add_argument("--lateral-offsets", type=str, default="-1.0,1.0")
    parser.add_argument("--speed-limit-fractions", type=str, default="0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--fallback-target-velocity", type=float, default=15.0)
    parser.add_argument("--min-gap-to-lead-agent", type=float, default=1.0)
    parser.add_argument("--headway-time", type=float, default=1.5)
    parser.add_argument("--accel-max", type=float, default=1.5)
    parser.add_argument("--decel-max", type=float, default=3.0)
    parser.add_argument("--min-end-lateral-dev-m", type=float, default=0.4)
    parser.add_argument("--max-end-lateral-dev-m", type=float, default=1.2)
    parser.add_argument("--min-max-lateral-dev-m", type=float, default=0.5)
    parser.add_argument("--max-max-lateral-dev-m", type=float, default=2.0)
    parser.add_argument("--max-end-heading-error-deg", type=float, default=15.0)
    parser.add_argument("--min-end-speed-mps", type=float, default=1.0)
    parser.add_argument("--max-end-speed-mps", type=float, default=15.0)
    parser.add_argument("--min-pdm-score", type=float, default=0.2)
    parser.add_argument("--min-ttc-within-bound", type=float, default=1.0)
    parser.add_argument("--skip-intersection-candidates", action="store_true")
    parser.add_argument("--max-join-translation-m", type=float, default=1.0)
    parser.add_argument("--max-join-heading-deg", type=float, default=8.0)
    parser.add_argument("--max-join-speed-delta-mps", type=float, default=2.0)
    parser.add_argument("--recovery-max-end-lateral-dev-m", type=float, default=0.35)
    parser.add_argument("--recovery-max-end-heading-error-deg", type=float, default=8.0)
    parser.add_argument("--recovery-min-lateral-improvement-m", type=float, default=0.2)
    parser.add_argument("--recovery-min-progress", type=float, default=0.2)
    parser.add_argument("--parallel-backend", type=str, default="process", choices=["process", "none"])
    parser.add_argument("--num-workers", type=int, default=0, help="0 means auto, capped to 32.")
    return parser.parse_args()


def _auto_workers(num_workers: int) -> int:
    if num_workers > 0:
        return num_workers
    count = os.cpu_count() or 1
    return max(1, min(count, 32))


def _parse_float_list(raw: str) -> List[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return [float(item) for item in values]


def _load_metric_cache_loader(cache_root: Path) -> MetricCacheLoader:
    loader = MetricCacheLoader(cache_root)
    missing_tokens = [token for token, path in loader.metric_cache_paths.items() if not Path(path).exists()]
    if not missing_tokens:
        return loader

    local_index: Dict[str, str] = {}
    for metric_file in cache_root.rglob("metric_cache.pkl"):
        local_index[metric_file.parent.name] = str(metric_file)

    for token in missing_tokens:
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


def _build_planner_initialization(metric_cache: MetricCache) -> PlannerInitialization:
    map_api = get_maps_api(
        metric_cache.map_parameters.map_root,
        metric_cache.map_parameters.map_version,
        metric_cache.map_parameters.map_name,
    )

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


def _build_planner_input(metric_cache: MetricCache) -> PlannerInput:
    traffic_light_data = []
    traffic_light_status = getattr(metric_cache, "traffic_light_status", None)
    if traffic_light_status:
        traffic_light_data = list(traffic_light_status[0])

    history = SimulationHistoryBuffer.initialize_from_list(
        buffer_size=1,
        ego_states=[metric_cache.ego_state],
        observations=[metric_cache.current_tracked_objects[0]],
    )
    return PlannerInput(
        iteration=SimulationIteration(index=0, time_point=metric_cache.ego_state.time_point),
        history=history,
        traffic_light_data=traffic_light_data,
    )


def _proposal_lateral_offset(lateral_idx: int, lateral_offsets: Sequence[float]) -> float:
    if lateral_idx == 0:
        return 0.0
    offset_idx = lateral_idx - 1
    if 0 <= offset_idx < len(lateral_offsets):
        return float(lateral_offsets[offset_idx])
    return float("nan")


def _trajectory_features(
    centerline: Any,
    drivable_area_map: Any,
    states: np.ndarray,
    vehicle_parameters: Any,
    skip_intersection_candidates: bool,
) -> Dict[str, float]:
    coords = state_array_to_coords_array(states[None, ...], vehicle_parameters)[0, :, BBCoordsIndex.CENTER]

    deviations: List[float] = []
    for x, y in coords:
        point = Point(float(x), float(y))
        if skip_intersection_candidates and drivable_area_map.is_in_layer(point, SemanticMapLayer.INTERSECTION):
            continue
        deviations.append(float(centerline.linestring.distance(point)))

    if not deviations:
        deviations = [0.0]

    end_x = float(states[-1, StateIndex.X])
    end_y = float(states[-1, StateIndex.Y])
    end_heading = float(states[-1, StateIndex.HEADING])
    end_speed = float(np.hypot(states[-1, StateIndex.VELOCITY_X], states[-1, StateIndex.VELOCITY_Y]))

    progress = float(centerline.project(Point(float(coords[-1, 0]), float(coords[-1, 1]))))
    nearest = centerline.interpolate([progress], as_array=True)[0]
    end_heading_error = abs(np.degrees(((end_heading - float(nearest[2]) + np.pi) % (2 * np.pi)) - np.pi))

    return {
        "max_lateral_deviation_m": float(max(deviations)),
        "end_lateral_deviation_m": float(deviations[-1]),
        "end_heading_error_deg": float(end_heading_error),
        "end_speed_mps": float(end_speed),
        "end_x": end_x,
        "end_y": end_y,
    }


def _lateral_deviation_series(
    centerline: Any,
    states: np.ndarray,
    vehicle_parameters: Any,
) -> List[float]:
    coords = state_array_to_coords_array(states[None, ...], vehicle_parameters)[0, :, BBCoordsIndex.CENTER]
    deviations: List[float] = []
    for x, y in coords:
        deviations.append(float(centerline.linestring.distance(Point(float(x), float(y)))))
    return deviations


def _select_stage1_split_index(
    summary: CandidateSummary,
    states: np.ndarray,
    centerline: Any,
    vehicle_parameters: Any,
    interval_s: float,
    min_boundary_dev_m: float,
) -> int:
    deviations = _lateral_deviation_series(centerline, states, vehicle_parameters)
    if not deviations:
        return len(states) - 1

    min_idx = max(1, int(round(1.0 / interval_s)))
    trigger = max(min_boundary_dev_m, min(summary.end_lateral_deviation_m, 0.8 * summary.max_lateral_deviation_m))
    for idx in range(min_idx, len(deviations)):
        if deviations[idx] >= trigger:
            return idx
    return len(states) - 1


def _pass_join_continuity_gate(
    stage1_states: np.ndarray,
    stage2_states: np.ndarray,
    cfg: RunConfig,
) -> bool:
    if len(stage1_states) == 0 or len(stage2_states) < 2:
        return False

    prev = stage1_states[-1]
    nxt = stage2_states[1]

    dx = float(nxt[StateIndex.X] - prev[StateIndex.X])
    dy = float(nxt[StateIndex.Y] - prev[StateIndex.Y])
    translation = float(np.hypot(dx, dy))

    heading_delta = abs(np.degrees(((float(nxt[StateIndex.HEADING] - prev[StateIndex.HEADING]) + np.pi) % (2 * np.pi)) - np.pi))

    prev_speed = float(np.hypot(prev[StateIndex.VELOCITY_X], prev[StateIndex.VELOCITY_Y]))
    next_speed = float(np.hypot(nxt[StateIndex.VELOCITY_X], nxt[StateIndex.VELOCITY_Y]))
    speed_delta = abs(next_speed - prev_speed)

    return (
        translation <= cfg.max_join_translation_m
        and heading_delta <= cfg.max_join_heading_deg
        and speed_delta <= cfg.max_join_speed_delta_mps
    )


def _pass_recovery_gate(
    stage1_features: Dict[str, float],
    stage2_features: Dict[str, float],
    stage2_score: Dict[str, float],
    cfg: RunConfig,
) -> bool:
    eps = 1e-6
    if float(stage2_score["no_at_fault_collisions"]) < 1.0 - eps:
        return False
    if float(stage2_score["drivable_area_compliance"]) < 1.0 - eps:
        return False
    if float(stage2_score["driving_direction_compliance"]) < 1.0 - eps:
        return False
    if float(stage2_score["traffic_light_compliance"]) < 1.0 - eps:
        return False
    if float(stage2_score["ego_progress"]) < cfg.recovery_min_progress:
        return False
    if stage2_features["end_lateral_deviation_m"] > cfg.recovery_max_end_lateral_dev_m:
        return False
    if stage2_features["end_heading_error_deg"] > cfg.recovery_max_end_heading_error_deg:
        return False
    improvement = stage1_features["end_lateral_deviation_m"] - stage2_features["end_lateral_deviation_m"]
    if improvement < cfg.recovery_min_lateral_improvement_m:
        return False
    return True


def _build_reactive_policy(proposal_sampling: TrajectorySampling) -> NavsimIDMTrafficAgents:
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
    )


def _score_segment(
    scorer: PDMScorer,
    states: np.ndarray,
    metric_cache: MetricCache,
    simulated_tracks: List[DetectionsTracks],
    centerline=None,
    route_lane_ids=None,
    drivable_area_map=None,
) -> Dict[str, float]:
    observation = copy.deepcopy(metric_cache.observation)
    row = scorer.score_proposals(
        states=states[None, ...],
        observation=observation,
        centerline=centerline if centerline is not None else metric_cache.centerline,
        route_lane_ids=route_lane_ids if route_lane_ids is not None else metric_cache.route_lane_ids,
        drivable_area_map=drivable_area_map if drivable_area_map is not None else metric_cache.drivable_area_map,
        map_parameters=metric_cache.map_parameters,
        simulated_agent_detections_tracks=simulated_tracks,
        human_past_trajectory=metric_cache.past_human_trajectory,
    )[0].iloc[0]
    return _score_row_to_dict(row)


def _build_stage2_metric_cache(
    metric_cache: MetricCache,
    ego_state_ood: Any,
    current_tracks: DetectionsTracks,
    proposal_sampling: TrajectorySampling,
) -> MetricCache:
    stage2_cache = copy.copy(metric_cache)
    stage2_cache.ego_state = ego_state_ood
    stage2_cache.current_tracked_objects = [current_tracks]
    stage2_cache.future_tracked_objects = [copy.deepcopy(current_tracks) for _ in range(proposal_sampling.num_poses)]

    traffic_light_status = getattr(metric_cache, "traffic_light_status", None)
    if traffic_light_status is not None and len(traffic_light_status) > 0:
        last_status = traffic_light_status[-1]
        stage2_cache.traffic_light_status = [last_status for _ in range(proposal_sampling.num_poses + 1)]
    return stage2_cache


def _build_rescue_input(ego_state: Any, current_tracks: DetectionsTracks, traffic_light_data: Sequence[Any]) -> PlannerInput:
    return PlannerInput(
        iteration=SimulationIteration(index=0, time_point=ego_state.time_point),
        history=SimulationHistoryBuffer.initialize_from_list(
            buffer_size=1,
            ego_states=[ego_state],
            observations=[current_tracks],
        ),
        traffic_light_data=list(traffic_light_data),
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


def _build_two_stage_trace(
    summary: CandidateSummary,
    stage1_states: np.ndarray,
    stage1_tracks: List[DetectionsTracks],
    stage1_score: Dict[str, float],
    stage2_states: np.ndarray,
    stage2_tracks: List[DetectionsTracks],
    stage2_score: Dict[str, float],
) -> Dict[str, Any]:
    frames: List[Dict[str, Any]] = []
    for t_idx, (state, detections) in enumerate(zip(stage1_states, stage1_tracks)):
        frames.append(_make_frame_payload(t_idx, state, detections))

    for local_idx, (state, detections) in enumerate(zip(stage2_states[1:], stage2_tracks[1:]), start=len(frames)):
        frames.append(_make_frame_payload(local_idx, state, detections))

    split_idx = len(stage1_states) - 1
    full_states = np.concatenate([stage1_states, stage2_states[1:]], axis=0)
    return {
        "token": summary.token,
        "proposal_idx": int(summary.proposal_idx),
        "lateral_idx": int(summary.lateral_idx),
        "longitudinal_idx": int(summary.longitudinal_idx),
        "lateral_offset_m": float(summary.lateral_offset_m),
        "boundary_summary": asdict(summary),
        "stage1_score": stage1_score,
        "stage2_score": stage2_score,
        "strong_longtail": {"split_idx": int(split_idx)},
        "num_frames": len(frames),
        "frames": frames,
        "states": full_states.tolist(),
    }


def _is_admissible_candidate(
    row: pd.Series,
    features: Dict[str, float],
    args: argparse.Namespace,
) -> bool:
    eps = 1e-6
    if float(row["no_at_fault_collisions"]) < 1.0 - eps:
        return False
    if float(row["drivable_area_compliance"]) < 1.0 - eps:
        return False
    if float(row["driving_direction_compliance"]) < 1.0 - eps:
        return False
    if float(row["traffic_light_compliance"]) < 1.0 - eps:
        return False
    if float(row["pdm_score"]) < args.min_pdm_score:
        return False
    if float(row["time_to_collision_within_bound"]) < args.min_ttc_within_bound - eps:
        return False

    if features["end_lateral_deviation_m"] < args.min_end_lateral_dev_m:
        return False
    if features["end_lateral_deviation_m"] > args.max_end_lateral_dev_m:
        return False
    if features["max_lateral_deviation_m"] < args.min_max_lateral_dev_m:
        return False
    if features["max_lateral_deviation_m"] > args.max_max_lateral_dev_m:
        return False
    if features["end_heading_error_deg"] > args.max_end_heading_error_deg:
        return False
    if features["end_speed_mps"] < args.min_end_speed_mps:
        return False
    if features["end_speed_mps"] > args.max_end_speed_mps:
        return False
    return True


def _candidate_rank(row: pd.Series, features: Dict[str, float], args: argparse.Namespace) -> float:
    target_dev = 0.5 * (args.min_end_lateral_dev_m + args.max_end_lateral_dev_m)
    dev_term = max(0.0, 1.0 - abs(features["end_lateral_deviation_m"] - target_dev) / max(target_dev, 1e-3))
    comfort_term = float(row.get("history_comfort", 0.0))
    lk_term = 1.0 - float(row.get("lane_keeping", 1.0))
    return 0.55 * float(row["pdm_score"]) + 0.25 * dev_term + 0.10 * comfort_term + 0.10 * lk_term


def _recovery_rank(
    stage1_summary: CandidateSummary,
    stage2_score: Dict[str, float],
    stage2_features: Dict[str, float],
) -> float:
    improvement = max(0.0, float(stage1_summary.end_lateral_deviation_m) - float(stage2_features["end_lateral_deviation_m"]))
    centeredness = max(0.0, 1.0 - float(stage2_features["end_lateral_deviation_m"]) / max(stage1_summary.end_lateral_deviation_m, 1e-3))
    heading_term = max(0.0, 1.0 - float(stage2_features["end_heading_error_deg"]) / 15.0)
    lk_term = float(stage2_score.get("lane_keeping", 0.0))
    pdm_term = float(stage2_score.get("pdm_score", 0.0))
    progress_term = float(stage2_score.get("ego_progress", 0.0))
    return (
        0.30 * improvement
        + 0.25 * centeredness
        + 0.15 * heading_term
        + 0.15 * lk_term
        + 0.10 * pdm_term
        + 0.05 * progress_term
    )


def _summarize_candidates(
    token: str,
    metric_cache: MetricCache,
    planner: BoundaryMiningPDMPlanner,
    simulated_proposals: np.ndarray,
    proposal_df: pd.DataFrame,
    lateral_offsets: Sequence[float],
    args: argparse.Namespace,
) -> List[Tuple[CandidateSummary, np.ndarray]]:
    output: List[Tuple[CandidateSummary, np.ndarray]] = []

    for proposal_idx in range(len(proposal_df)):
        row = proposal_df.iloc[proposal_idx]
        proposal_meta = planner._proposal_manager[proposal_idx]
        states = simulated_proposals[proposal_idx]
        features = _trajectory_features(
            centerline=planner._centerline,
            drivable_area_map=planner._drivable_area_map,
            states=states,
            vehicle_parameters=metric_cache.ego_state.car_footprint.vehicle_parameters,
            skip_intersection_candidates=args.skip_intersection_candidates,
        )
        if not _is_admissible_candidate(row, features, args):
            continue

        summary = CandidateSummary(
            token=token,
            proposal_idx=int(proposal_idx),
            lateral_idx=int(proposal_meta.lateral_idx),
            longitudinal_idx=int(proposal_meta.longitudinal_idx),
            lateral_offset_m=_proposal_lateral_offset(proposal_meta.lateral_idx, lateral_offsets),
            pdm_score=float(row["pdm_score"]),
            no_at_fault_collisions=float(row["no_at_fault_collisions"]),
            drivable_area_compliance=float(row["drivable_area_compliance"]),
            driving_direction_compliance=float(row["driving_direction_compliance"]),
            traffic_light_compliance=float(row["traffic_light_compliance"]),
            ego_progress=float(row["ego_progress"]),
            time_to_collision_within_bound=float(row["time_to_collision_within_bound"]),
            lane_keeping=float(row["lane_keeping"]),
            history_comfort=float(row["history_comfort"]),
            max_lateral_deviation_m=features["max_lateral_deviation_m"],
            end_lateral_deviation_m=features["end_lateral_deviation_m"],
            end_heading_error_deg=features["end_heading_error_deg"],
            end_speed_mps=features["end_speed_mps"],
            end_x=features["end_x"],
            end_y=features["end_y"],
        )
        output.append((summary, states))

    output.sort(key=lambda item: _candidate_rank(pd.Series(asdict(item[0])), asdict(item[0]), args), reverse=True)
    return output[: args.stage1_eval_top_k]


def _candidate_stem(summary: CandidateSummary) -> str:
    return f"cut_in_line_boundary_{summary.token}_p{summary.proposal_idx:02d}"


def _save_candidate_trace(
    output_dir: Path,
    summary: CandidateSummary,
    payload: Dict[str, Any],
    save_format: str,
) -> Dict[str, str]:
    stem = _candidate_stem(summary)
    saved_paths: Dict[str, str] = {}

    if save_format in ("json", "both"):
        json_path = output_dir / f"{stem}.json"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        saved_paths["json_path"] = str(json_path)

    if save_format in ("pkl", "both"):
        pkl_path = output_dir / f"{stem}.pkl"
        with pkl_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        saved_paths["pkl_path"] = str(pkl_path)

    return saved_paths


def _build_run_config(args: argparse.Namespace, output_dir: Path) -> RunConfig:
    return RunConfig(
        output_dir=str(output_dir),
        save_format=args.save_format,
        top_k_per_token=int(args.top_k_per_token),
        stage1_eval_top_k=int(args.stage1_eval_top_k),
        interval=float(args.interval),
        horizon_sec=float(args.horizon_sec),
        map_radius=float(args.map_radius),
        lateral_offsets=_parse_float_list(args.lateral_offsets),
        speed_limit_fractions=_parse_float_list(args.speed_limit_fractions),
        fallback_target_velocity=float(args.fallback_target_velocity),
        min_gap_to_lead_agent=float(args.min_gap_to_lead_agent),
        headway_time=float(args.headway_time),
        accel_max=float(args.accel_max),
        decel_max=float(args.decel_max),
        min_end_lateral_dev_m=float(args.min_end_lateral_dev_m),
        max_end_lateral_dev_m=float(args.max_end_lateral_dev_m),
        min_max_lateral_dev_m=float(args.min_max_lateral_dev_m),
        max_max_lateral_dev_m=float(args.max_max_lateral_dev_m),
        max_end_heading_error_deg=float(args.max_end_heading_error_deg),
        min_end_speed_mps=float(args.min_end_speed_mps),
        max_end_speed_mps=float(args.max_end_speed_mps),
        min_pdm_score=float(args.min_pdm_score),
        min_ttc_within_bound=float(args.min_ttc_within_bound),
        skip_intersection_candidates=bool(args.skip_intersection_candidates),
        max_join_translation_m=float(args.max_join_translation_m),
        max_join_heading_deg=float(args.max_join_heading_deg),
        max_join_speed_delta_mps=float(args.max_join_speed_delta_mps),
        recovery_max_end_lateral_dev_m=float(args.recovery_max_end_lateral_dev_m),
        recovery_max_end_heading_error_deg=float(args.recovery_max_end_heading_error_deg),
        recovery_min_lateral_improvement_m=float(args.recovery_min_lateral_improvement_m),
        recovery_min_progress=float(args.recovery_min_progress),
    )


def _init_worker(metric_cache_path: str, run_cfg_dict: Dict[str, Any]) -> None:
    global _CTX
    cfg = RunConfig(**run_cfg_dict)
    proposal_sampling = TrajectorySampling(
        num_poses=int(round(cfg.horizon_sec / cfg.interval)),
        interval_length=cfg.interval,
    )
    future_poses = proposal_sampling.num_poses + int(1.0 / proposal_sampling.interval_length)
    trajectory_sampling = TrajectorySampling(
        num_poses=future_poses,
        interval_length=proposal_sampling.interval_length,
    )
    loader = _load_metric_cache_loader(Path(metric_cache_path))
    _CTX = {
        "loader": loader,
        "cfg": cfg,
        "proposal_sampling": proposal_sampling,
        "trajectory_sampling": trajectory_sampling,
        "reactive_policy": _build_reactive_policy(proposal_sampling),
        "scorer": PDMScorer(proposal_sampling=proposal_sampling),
    }


def _namespace_from_cfg(cfg: RunConfig) -> argparse.Namespace:
    return argparse.Namespace(**asdict(cfg))


def _process_token(token: str, metric_cache: MetricCache, cfg: RunConfig, proposal_sampling: TrajectorySampling, trajectory_sampling: TrajectorySampling, reactive_policy: NavsimIDMTrafficAgents, scorer: PDMScorer) -> Dict[str, Any]:
    output_dir = Path(cfg.output_dir)
    planner = BoundaryMiningPDMPlanner(
        trajectory_sampling=trajectory_sampling,
        proposal_sampling=proposal_sampling,
        idm_policies=BatchIDMPolicy(
            speed_limit_fraction=cfg.speed_limit_fractions,
            fallback_target_velocity=cfg.fallback_target_velocity,
            min_gap_to_lead_agent=cfg.min_gap_to_lead_agent,
            headway_time=cfg.headway_time,
            accel_max=cfg.accel_max,
            decel_max=cfg.decel_max,
        ),
        lateral_offsets=cfg.lateral_offsets,
        map_radius=cfg.map_radius,
    )
    planner.initialize(_build_planner_initialization(metric_cache))
    planner._drivable_area_map = PDMDrivableMap.from_simulation(planner._map_api, metric_cache.ego_state, cfg.map_radius)
    preferred_start_lane_id = planner._get_starting_lane(metric_cache.ego_state).id
    simulated_proposals, proposal_df = planner.compute_all_proposals(_build_planner_input(metric_cache))

    candidates = _summarize_candidates(
        token=token,
        metric_cache=metric_cache,
        planner=planner,
        simulated_proposals=simulated_proposals,
        proposal_df=proposal_df,
        lateral_offsets=cfg.lateral_offsets,
        args=_namespace_from_cfg(cfg),
    )

    kept_results: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    stage2_continuity_rejects = 0
    stage2_recovery_rejects = 0
    stage1_without_rescue = 0
    for rank, (summary, states) in enumerate(candidates):
        record = asdict(summary)
        record["rank"] = int(rank)
        stage1_tracks_full = reactive_policy.simulate_environment(states, metric_cache)
        stage1_score = _score_segment(
            scorer=scorer,
            states=states,
            metric_cache=metric_cache,
            simulated_tracks=stage1_tracks_full,
        )
        split_idx = _select_stage1_split_index(
            summary=summary,
            states=states,
            centerline=planner._centerline,
            vehicle_parameters=metric_cache.ego_state.car_footprint.vehicle_parameters,
            interval_s=cfg.interval,
            min_boundary_dev_m=cfg.min_end_lateral_dev_m,
        )
        stage1_states = states[: split_idx + 1]
        stage1_tracks = stage1_tracks_full[: split_idx + 1]
        stage1_features = _trajectory_features(
            centerline=planner._centerline,
            drivable_area_map=planner._drivable_area_map,
            states=stage1_states,
            vehicle_parameters=metric_cache.ego_state.car_footprint.vehicle_parameters,
            skip_intersection_candidates=cfg.skip_intersection_candidates,
        )
        record["split_idx"] = int(split_idx)
        record["split_time_s"] = float(split_idx * cfg.interval)

        ood_time = metric_cache.ego_state.time_point + TimeDuration.from_s(split_idx * cfg.interval)
        ood_ego_state = state_array_to_ego_state(
            stage1_states[-1],
            ood_time,
            metric_cache.ego_state.car_footprint.vehicle_parameters,
        )
        ood_obs = stage1_tracks[-1]
        stage2_cache = _build_stage2_metric_cache(
            metric_cache=metric_cache,
            ego_state_ood=ood_ego_state,
            current_tracks=ood_obs,
            proposal_sampling=proposal_sampling,
        )
        traffic_light_data = []
        traffic_light_status = getattr(stage2_cache, "traffic_light_status", None)
        if traffic_light_status:
            traffic_light_data = list(traffic_light_status[0])

        rescue_planner = PreferredLanePDMPlanner(
            trajectory_sampling=trajectory_sampling,
            proposal_sampling=proposal_sampling,
            idm_policies=BatchIDMPolicy(
                speed_limit_fraction=cfg.speed_limit_fractions,
                fallback_target_velocity=cfg.fallback_target_velocity,
                min_gap_to_lead_agent=cfg.min_gap_to_lead_agent,
                headway_time=cfg.headway_time,
                accel_max=cfg.accel_max,
                decel_max=cfg.decel_max,
            ),
            lateral_offsets=None,
            map_radius=cfg.map_radius,
            preferred_start_lane_id=preferred_start_lane_id,
        )
        rescue_planner.initialize(_build_planner_initialization(metric_cache))
        rescue_input = _build_rescue_input(ood_ego_state, ood_obs, traffic_light_data)
        rescue_proposals, rescue_df = rescue_planner.compute_all_proposals(
            rescue_input
        )
        rescue_candidates: List[Tuple[float, np.ndarray, Dict[str, float], Dict[str, float]]] = []
        for rescue_idx in range(len(rescue_df)):
            rescue_states = rescue_proposals[rescue_idx]
            if not _pass_join_continuity_gate(stage1_states, rescue_states, cfg):
                stage2_continuity_rejects += 1
                continue

            stage2_tracks = reactive_policy.simulate_environment(rescue_states, stage2_cache)
            stage2_score = _score_segment(
                scorer=scorer,
                states=rescue_states,
                metric_cache=stage2_cache,
                simulated_tracks=stage2_tracks,
                centerline=rescue_planner._centerline,
                route_lane_ids=list(rescue_planner._route_lane_dict.keys()),
                drivable_area_map=rescue_planner._drivable_area_map,
            )
            stage2_features = _trajectory_features(
                centerline=rescue_planner._centerline,
                drivable_area_map=rescue_planner._drivable_area_map,
                states=rescue_states,
                vehicle_parameters=metric_cache.ego_state.car_footprint.vehicle_parameters,
                skip_intersection_candidates=cfg.skip_intersection_candidates,
            )
            if not _pass_recovery_gate(stage1_features, stage2_features, stage2_score, cfg):
                stage2_recovery_rejects += 1
                continue

            rescue_rank = _recovery_rank(summary, stage2_score, stage2_features)
            rescue_candidates.append((rescue_rank, rescue_states, stage2_score, stage2_features))

        if not rescue_candidates:
            stage1_without_rescue += 1
            continue

        rescue_candidates.sort(key=lambda item: item[0], reverse=True)
        rescue_rank, rescue_states, stage2_score, stage2_features = rescue_candidates[0]
        if not _pass_join_continuity_gate(stage1_states, rescue_states, cfg):
            continue

        stage2_tracks = reactive_policy.simulate_environment(rescue_states, stage2_cache)
        record["stage1_score"] = stage1_score
        record["stage2_score"] = stage2_score
        record["stage2_end_lateral_deviation_m"] = float(stage2_features["end_lateral_deviation_m"])
        record["stage2_end_heading_error_deg"] = float(stage2_features["end_heading_error_deg"])
        record["recovery_rank"] = float(rescue_rank)
        record["recovery_improvement_m"] = float(
            stage1_features["end_lateral_deviation_m"] - stage2_features["end_lateral_deviation_m"]
        )
        payload = _build_two_stage_trace(
            summary=summary,
            stage1_states=stage1_states,
            stage1_tracks=stage1_tracks,
            stage1_score=stage1_score,
            stage2_states=rescue_states,
            stage2_tracks=stage2_tracks,
            stage2_score=stage2_score,
        )
        kept_results.append((float(rescue_rank), record, payload))

    kept_results.sort(key=lambda item: item[0], reverse=True)
    kept_results = kept_results[: cfg.top_k_per_token]
    manifest_records: List[str] = []
    for final_rank, (_, record, payload) in enumerate(kept_results):
        record["final_rank"] = int(final_rank)
        if cfg.save_format != "none":
            summary = CandidateSummary(**{field: record[field] for field in CandidateSummary.__dataclass_fields__.keys()})
            record.update(_save_candidate_trace(output_dir, summary, payload, cfg.save_format))
        manifest_records.append(json.dumps(record, ensure_ascii=False))

    return {
        "token": token,
        "num_stage1_candidates": len(candidates),
        "num_candidates": len(manifest_records),
        "manifest_records": manifest_records,
        "stage2_continuity_rejects": int(stage2_continuity_rejects),
        "stage2_recovery_rejects": int(stage2_recovery_rejects),
        "stage1_without_rescue": int(stage1_without_rescue),
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
        trajectory_sampling=_CTX["trajectory_sampling"],
        reactive_policy=_CTX["reactive_policy"],
        scorer=_CTX["scorer"],
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = _build_run_config(args, output_dir)
    loader = _load_metric_cache_loader(args.metric_cache_path.expanduser().resolve())
    tokens = _load_tokens(loader, args.tokens_file, args.max_scenes)

    manifest_lines: List[str] = []
    stats: Dict[str, Any] = {
        "num_tokens": len(tokens),
        "num_candidates": 0,
        "tokens_with_candidates": 0,
        "num_stage1_candidates": 0,
        "stage1_without_rescue": 0,
        "stage2_continuity_rejects": 0,
        "stage2_recovery_rejects": 0,
    }

    if args.parallel_backend == "none":
        _init_worker(str(args.metric_cache_path.expanduser().resolve()), asdict(run_cfg))
        results_iter = ((_run_token_worker(token), idx, token) for idx, token in enumerate(tokens, start=1))
        for result, idx, token in results_iter:
            manifest_lines.extend(result["manifest_records"])
            stats["num_stage1_candidates"] += int(result["num_stage1_candidates"])
            stats["num_candidates"] += int(result["num_candidates"])
            if int(result["num_candidates"]) > 0:
                stats["tokens_with_candidates"] += 1
            stats["stage1_without_rescue"] += int(result["stage1_without_rescue"])
            stats["stage2_continuity_rejects"] += int(result["stage2_continuity_rejects"])
            stats["stage2_recovery_rejects"] += int(result["stage2_recovery_rejects"])
            print(
                f"[{idx}/{len(tokens)}] {token}: kept={int(result['num_candidates'])}, "
                f"stage1={int(result['num_stage1_candidates'])}, "
                f"no_rescue={int(result['stage1_without_rescue'])}, "
                f"join_rejects={int(result['stage2_continuity_rejects'])}, "
                f"recovery_rejects={int(result['stage2_recovery_rejects'])}"
            )
    else:
        num_workers = _auto_workers(args.num_workers)
        ordered_results: Dict[int, Dict[str, Any]] = {}
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(str(args.metric_cache_path.expanduser().resolve()), asdict(run_cfg)),
        ) as ex:
            future_map = {
                ex.submit(_run_token_worker, token): (idx, token) for idx, token in enumerate(tokens, start=1)
            }
            for future in as_completed(future_map):
                idx, token = future_map[future]
                ordered_results[idx] = future.result()
                result = ordered_results[idx]
                print(
                    f"[{idx}/{len(tokens)}] {token}: kept={int(result['num_candidates'])}, "
                    f"stage1={int(result['num_stage1_candidates'])}, "
                    f"no_rescue={int(result['stage1_without_rescue'])}, "
                    f"join_rejects={int(result['stage2_continuity_rejects'])}, "
                    f"recovery_rejects={int(result['stage2_recovery_rejects'])}"
                )

        for idx in range(1, len(tokens) + 1):
            result = ordered_results[idx]
            manifest_lines.extend(result["manifest_records"])
            stats["num_stage1_candidates"] += int(result["num_stage1_candidates"])
            stats["num_candidates"] += int(result["num_candidates"])
            if int(result["num_candidates"]) > 0:
                stats["tokens_with_candidates"] += 1
            stats["stage1_without_rescue"] += int(result["stage1_without_rescue"])
            stats["stage2_continuity_rejects"] += int(result["stage2_continuity_rejects"])
            stats["stage2_recovery_rejects"] += int(result["stage2_recovery_rejects"])

    manifest_path = output_dir / args.manifest_name
    manifest_path.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")
    (output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
