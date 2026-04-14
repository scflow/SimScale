#!/usr/bin/env python3
"""
Generate cut-in-line pseudo-sim traces with an explicitly scripted ego trajectory.

The generator keeps the environment simulation standard:
- ego longitudinal motion follows a native NAVSIM PDM baseline;
- ego lateral motion is scripted in two phases: encroach, then recenter;
- surrounding traffic remains reactive via IDM;
- candidates are filtered with native PDM metrics plus lane-center geometry checks.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from nuplan.common.actor_state.state_representation import Point2D, TimeDuration
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely.geometry import Point

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.mine_cut_in_line_boundary_states import (
        _build_planner_initialization,
        _build_planner_input,
        _build_stage2_metric_cache,
        _load_metric_cache_loader,
        _load_tokens,
        _score_segment,
    )
    from scripts.generate_cut_in_line_mobil_pseudo_sim import _build_pdm_planner, _build_stage2_idm_policy
except ModuleNotFoundError:
    from mine_cut_in_line_boundary_states import (
        _build_planner_initialization,
        _build_planner_input,
        _build_stage2_metric_cache,
        _load_metric_cache_loader,
        _load_tokens,
        _score_segment,
    )
    from generate_cut_in_line_mobil_pseudo_sim import _build_pdm_planner, _build_stage2_idm_policy

from navsim.behavior.scene_adapter import build_ego_actor_from_ego_state, lane_heading_at_point
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import get_trajectory_as_array
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_ego_state
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex


warnings.filterwarnings(
    "ignore",
    message=r"Expected length of detections_tracks .*",
    category=UserWarning,
)


WorkerContext = Dict[str, Any]
_CTX: Optional[WorkerContext] = None


@dataclass
class ScriptedSpec:
    sign: int
    d_peak_m: float
    peak_time_s: float


@dataclass
class LaneFeatures:
    max_lateral_deviation_m: float
    end_lateral_deviation_m: float
    end_heading_error_deg: float
    end_speed_mps: float


@dataclass
class RunConfig:
    output_dir: str
    save_format: str
    interval: float
    horizon_sec: float
    map_radius: float
    top_k_per_token: int
    d_peak_values: List[float]
    peak_time_values: List[float]
    directions: List[int]
    min_peak_deviation_m: float
    max_peak_deviation_m: float
    recovery_max_end_lateral_dev_m: float
    recovery_max_end_heading_error_deg: float
    recovery_min_improvement_m: float
    min_stage2_progress_m: float
    parallel_backend: str
    num_workers: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate scripted cut-in-line pseudo-sim traces.")
    parser.add_argument("--metric-cache-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/cut_in_line_scripted_v1"))
    parser.add_argument("--manifest-name", type=str, default="manifest.jsonl")
    parser.add_argument("--save-format", type=str, default="both", choices=["json", "pkl", "both", "none"])
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--tokens-file", type=Path, default=None)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--horizon-sec", type=float, default=4.0)
    parser.add_argument("--map-radius", type=float, default=100.0)
    parser.add_argument("--top-k-per-token", type=int, default=2)
    parser.add_argument("--d-peak-values", type=str, default="0.8,1.0,1.2")
    parser.add_argument("--peak-time-values", type=str, default="1.0,1.2,1.5")
    parser.add_argument("--directions", type=str, default="-1,1")
    parser.add_argument("--min-peak-deviation-m", type=float, default=0.55)
    parser.add_argument("--max-peak-deviation-m", type=float, default=1.5)
    parser.add_argument("--recovery-max-end-lateral-dev-m", type=float, default=0.2)
    parser.add_argument("--recovery-max-end-heading-error-deg", type=float, default=6.0)
    parser.add_argument("--recovery-min-improvement-m", type=float, default=0.35)
    parser.add_argument("--min-stage2-progress-m", type=float, default=3.0)
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
    return [float(value) for value in values]


def _parse_int_list(raw: str) -> List[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return [int(value) for value in values]


def _smoothstep01(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return 10.0 * clipped**3 - 15.0 * clipped**4 + 6.0 * clipped**5


def _find_lane_object(map_api: Any, lane_id: str) -> Optional[LaneGraphEdgeMapObject]:
    lane = map_api.get_map_object(lane_id, SemanticMapLayer.LANE)
    if lane is None:
        lane = map_api.get_map_object(lane_id, SemanticMapLayer.LANE_CONNECTOR)
    return lane


def _normalize_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _build_specs(cfg: RunConfig) -> List[ScriptedSpec]:
    specs: List[ScriptedSpec] = []
    for sign in cfg.directions:
        if sign == 0:
            continue
        for d_peak_m in cfg.d_peak_values:
            for peak_time_s in cfg.peak_time_values:
                if peak_time_s <= 0.0 or peak_time_s >= cfg.horizon_sec:
                    continue
                specs.append(ScriptedSpec(sign=int(math.copysign(1, sign)), d_peak_m=float(d_peak_m), peak_time_s=float(peak_time_s)))
    return specs


def _build_lateral_profile(num_states: int, interval_s: float, spec: ScriptedSpec) -> np.ndarray:
    horizon_s = interval_s * max(0, num_states - 1)
    times = np.arange(num_states, dtype=np.float64) * interval_s
    profile = np.zeros(num_states, dtype=np.float64)

    encroach_mask = times <= spec.peak_time_s
    if np.any(encroach_mask):
        profile[encroach_mask] = spec.sign * spec.d_peak_m * _smoothstep01(times[encroach_mask] / max(spec.peak_time_s, 1e-3))

    recover_duration = max(horizon_s - spec.peak_time_s, interval_s)
    recover_mask = times > spec.peak_time_s
    if np.any(recover_mask):
        alpha = (times[recover_mask] - spec.peak_time_s) / recover_duration
        profile[recover_mask] = spec.sign * spec.d_peak_m * (1.0 - _smoothstep01(alpha))

    profile[0] = 0.0
    profile[-1] = 0.0
    return profile


def _apply_lateral_profile(base_states: np.ndarray, lateral_profile_m: np.ndarray, interval_s: float) -> np.ndarray:
    states = np.array(base_states, copy=True)
    base_x = np.asarray(base_states[:, StateIndex.X], dtype=np.float64)
    base_y = np.asarray(base_states[:, StateIndex.Y], dtype=np.float64)
    base_heading = np.asarray(base_states[:, StateIndex.HEADING], dtype=np.float64)

    states[:, StateIndex.X] = base_x - np.sin(base_heading) * lateral_profile_m
    states[:, StateIndex.Y] = base_y + np.cos(base_heading) * lateral_profile_m

    dx = np.gradient(states[:, StateIndex.X], interval_s)
    dy = np.gradient(states[:, StateIndex.Y], interval_s)
    heading = np.unwrap(np.arctan2(dy, dx))
    states[:, StateIndex.HEADING] = _normalize_angle(heading)
    states[:, StateIndex.VELOCITY_X] = dx
    states[:, StateIndex.VELOCITY_Y] = dy
    states[:, StateIndex.ACCELERATION_X] = np.gradient(dx, interval_s)
    states[:, StateIndex.ACCELERATION_Y] = np.gradient(dy, interval_s)

    if StateIndex.ANGULAR_VELOCITY < states.shape[1]:
        angular_velocity = np.gradient(heading, interval_s)
        states[:, StateIndex.ANGULAR_VELOCITY] = angular_velocity
        if StateIndex.ANGULAR_ACCELERATION < states.shape[1]:
            states[:, StateIndex.ANGULAR_ACCELERATION] = np.gradient(angular_velocity, interval_s)
    return states


def _reference_features(states: np.ndarray, reference_states: np.ndarray) -> LaneFeatures:
    if len(states) != len(reference_states):
        raise ValueError(f"Expected equal trajectory lengths, got {len(states)} and {len(reference_states)}.")
    deviations: List[float] = []
    for state, ref_state in zip(states, reference_states):
        dx = float(state[StateIndex.X] - ref_state[StateIndex.X])
        dy = float(state[StateIndex.Y] - ref_state[StateIndex.Y])
        deviations.append(float(np.hypot(dx, dy)))

    end_heading = float(states[-1, StateIndex.HEADING])
    end_heading_ref = float(reference_states[-1, StateIndex.HEADING])
    end_heading_error = abs(np.degrees(((end_heading - end_heading_ref + np.pi) % (2 * np.pi)) - np.pi))
    end_speed = float(np.hypot(states[-1, StateIndex.VELOCITY_X], states[-1, StateIndex.VELOCITY_Y]))
    return LaneFeatures(
        max_lateral_deviation_m=float(max(deviations) if deviations else 0.0),
        end_lateral_deviation_m=float(deviations[-1] if deviations else 0.0),
        end_heading_error_deg=float(end_heading_error),
        end_speed_mps=float(end_speed),
    )


def _hard_pass(score: Dict[str, float], eps: float = 1e-6) -> bool:
    return (
        float(score.get("no_at_fault_collisions", 0.0)) >= 1.0 - eps
        and float(score.get("drivable_area_compliance", 0.0)) >= 1.0 - eps
        and float(score.get("driving_direction_compliance", 0.0)) >= 1.0 - eps
        and float(score.get("traffic_light_compliance", 0.0)) >= 1.0 - eps
    )


def _serialize_vehicles(detections) -> List[Dict[str, Any]]:
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


def _make_frame_payload(t_idx: int, state: np.ndarray, detections) -> Dict[str, Any]:
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
    spec: ScriptedSpec,
    stage1_states: np.ndarray,
    stage1_tracks: Sequence[Any],
    stage1_score: Dict[str, float],
    stage2_states: np.ndarray,
    stage2_tracks: Sequence[Any],
    stage2_score: Dict[str, float],
) -> Dict[str, Any]:
    frames: List[Dict[str, Any]] = []
    for idx, (state, detections) in enumerate(zip(stage1_states, stage1_tracks)):
        frames.append(_make_frame_payload(idx, state, detections))
    for idx, (state, detections) in enumerate(zip(stage2_states[1:], stage2_tracks[1:]), start=len(frames)):
        frames.append(_make_frame_payload(idx, state, detections))

    full_states = np.concatenate([stage1_states, stage2_states[1:]], axis=0)
    return {
        "token": token,
        "scripted_spec": asdict(spec),
        "stage1_score": stage1_score,
        "stage2_score": stage2_score,
        "strong_longtail": {"split_idx": int(len(stage1_states) - 1)},
        "num_frames": int(len(frames)),
        "frames": frames,
        "states": full_states.tolist(),
    }


def _save_payload(output_dir: Path, stem: str, payload: Dict[str, Any], save_format: str) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    if save_format in ("json", "both"):
        json_path = output_dir / f"{stem}.json"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        paths["json_path"] = str(json_path)
    if save_format in ("pkl", "both"):
        pkl_path = output_dir / f"{stem}.pkl"
        with pkl_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        paths["pkl_path"] = str(pkl_path)
    return paths


def _candidate_rank(stage1_features: LaneFeatures, stage2_features: LaneFeatures, stage2_score: Dict[str, float], cfg: RunConfig) -> float:
    target_peak = 0.5 * (cfg.min_peak_deviation_m + cfg.max_peak_deviation_m)
    peak_term = max(0.0, 1.0 - abs(stage1_features.end_lateral_deviation_m - target_peak) / max(target_peak, 1e-3))
    recovery_term = max(
        0.0,
        1.0 - stage2_features.end_lateral_deviation_m / max(stage1_features.end_lateral_deviation_m, 1e-3),
    )
    heading_term = max(0.0, 1.0 - stage2_features.end_heading_error_deg / 15.0)
    lane_keep_term = float(stage2_score.get("lane_keeping", 0.0))
    return 0.35 * peak_term + 0.30 * recovery_term + 0.20 * heading_term + 0.15 * lane_keep_term


def _build_run_config(args: argparse.Namespace, output_dir: Path) -> RunConfig:
    return RunConfig(
        output_dir=str(output_dir),
        save_format=str(args.save_format),
        interval=float(args.interval),
        horizon_sec=float(args.horizon_sec),
        map_radius=float(args.map_radius),
        top_k_per_token=int(args.top_k_per_token),
        d_peak_values=_parse_float_list(args.d_peak_values),
        peak_time_values=_parse_float_list(args.peak_time_values),
        directions=_parse_int_list(args.directions),
        min_peak_deviation_m=float(args.min_peak_deviation_m),
        max_peak_deviation_m=float(args.max_peak_deviation_m),
        recovery_max_end_lateral_dev_m=float(args.recovery_max_end_lateral_dev_m),
        recovery_max_end_heading_error_deg=float(args.recovery_max_end_heading_error_deg),
        recovery_min_improvement_m=float(args.recovery_min_improvement_m),
        min_stage2_progress_m=float(args.min_stage2_progress_m),
        parallel_backend=str(args.parallel_backend),
        num_workers=int(args.num_workers),
    )


def _process_token(token: str, metric_cache: MetricCache, cfg: RunConfig, proposal_sampling: TrajectorySampling) -> Dict[str, Any]:
    reject_counts: Dict[str, int] = {}

    def _bump(reason: str) -> None:
        reject_counts[reason] = reject_counts.get(reason, 0) + 1

    planner_init = _build_planner_initialization(metric_cache)
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

    planner = _build_pdm_planner(
        proposal_sampling=proposal_sampling,
        map_radius=cfg.map_radius,
        lateral_offsets=None,
        preferred_start_lane_id=None,
    )
    planner.initialize(planner_init)
    baseline_trajectory = planner.compute_planner_trajectory(_build_planner_input(metric_cache))
    baseline_states = get_trajectory_as_array(baseline_trajectory, proposal_sampling, metric_cache.ego_state.time_point)

    stage_specs = _build_specs(cfg)
    kept_results: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []

    for spec_rank, spec in enumerate(stage_specs):
        lateral_profile = _build_lateral_profile(len(baseline_states), cfg.interval, spec)
        scripted_states = _apply_lateral_profile(baseline_states, lateral_profile, cfg.interval)
        split_idx = int(np.argmax(np.abs(lateral_profile)))
        if split_idx <= 0 or split_idx >= len(scripted_states) - 1:
            _bump("bad_split")
            continue

        stage1_states = scripted_states[: split_idx + 1]
        stage1_sampling = TrajectorySampling(num_poses=len(stage1_states) - 1, interval_length=cfg.interval)
        stage1_reactive_policy = _build_stage2_idm_policy(stage1_sampling, None)
        stage1_tracks = stage1_reactive_policy.simulate_environment(stage1_states, metric_cache)
        stage1_scorer = PDMScorer(stage1_sampling)
        stage1_score = _score_segment(
            scorer=stage1_scorer,
            states=stage1_states,
            metric_cache=metric_cache,
            simulated_tracks=stage1_tracks,
            centerline=planner._centerline,
            route_lane_ids=list(planner._route_lane_dict.keys()),
            drivable_area_map=planner._drivable_area_map,
        )
        stage1_features = _reference_features(stage1_states, baseline_states[: split_idx + 1])
        if not _hard_pass(stage1_score):
            _bump("stage1_hard_fail")
            continue
        if stage1_features.end_lateral_deviation_m < cfg.min_peak_deviation_m:
            _bump("stage1_peak_too_small")
            continue
        if stage1_features.end_lateral_deviation_m > cfg.max_peak_deviation_m:
            _bump("stage1_peak_too_large")
            continue

        split_time = metric_cache.ego_state.time_point + TimeDuration.from_s(split_idx * cfg.interval)
        split_ego_state = state_array_to_ego_state(
            np.asarray(stage1_states[-1]),
            split_time,
            metric_cache.ego_state.car_footprint.vehicle_parameters,
        )
        split_tracks = stage1_tracks[-1]

        stage2_states = scripted_states[split_idx:]
        stage2_sampling = TrajectorySampling(num_poses=len(stage2_states) - 1, interval_length=cfg.interval)
        stage2_cache = _build_stage2_metric_cache(
            metric_cache=metric_cache,
            ego_state_ood=split_ego_state,
            current_tracks=split_tracks,
            proposal_sampling=stage2_sampling,
        )
        stage2_reactive_policy = _build_stage2_idm_policy(stage2_sampling, None)
        stage2_tracks = stage2_reactive_policy.simulate_environment(stage2_states, stage2_cache)
        stage2_map = PDMDrivableMap.from_simulation(map_api, split_ego_state, cfg.map_radius)
        stage2_scorer = PDMScorer(stage2_sampling)
        stage2_score = _score_segment(
            scorer=stage2_scorer,
            states=stage2_states,
            metric_cache=stage2_cache,
            simulated_tracks=stage2_tracks,
            centerline=planner._centerline,
            route_lane_ids=list(planner._route_lane_dict.keys()),
            drivable_area_map=stage2_map,
        )
        stage2_features = _reference_features(stage2_states, baseline_states[split_idx:])
        if not _hard_pass(stage2_score):
            _bump("stage2_hard_fail")
            continue
        if stage2_features.end_lateral_deviation_m > cfg.recovery_max_end_lateral_dev_m:
            _bump("stage2_center_dev")
            continue
        if stage2_features.end_heading_error_deg > cfg.recovery_max_end_heading_error_deg:
            _bump("stage2_heading")
            continue
        if stage1_features.end_lateral_deviation_m - stage2_features.end_lateral_deviation_m < cfg.recovery_min_improvement_m:
            _bump("stage2_low_improvement")
            continue
        stage2_progress_m = float(
            Point(float(stage2_states[-1, StateIndex.X]), float(stage2_states[-1, StateIndex.Y])).distance(
                Point(float(stage2_states[0, StateIndex.X]), float(stage2_states[0, StateIndex.Y]))
            )
        )
        if stage2_progress_m < cfg.min_stage2_progress_m:
            _bump("stage2_low_progress")
            continue

        stage1_score = dict(stage1_score)
        stage1_score.update(
            {
                "stage": "stage1_scripted_encroach",
                "split_idx": int(split_idx),
                "peak_lateral_deviation_m": float(stage1_features.end_lateral_deviation_m),
                "spec_rank": int(spec_rank),
            }
        )
        stage2_score = dict(stage2_score)
        stage2_score.update(
            {
                "stage": "stage2_scripted_recenter",
                "end_lateral_deviation_m": float(stage2_features.end_lateral_deviation_m),
                "end_heading_error_deg": float(stage2_features.end_heading_error_deg),
                "progress_m": float(stage2_progress_m),
            }
        )

        payload = _build_trace_payload(
            token=token,
            spec=spec,
            stage1_states=stage1_states,
            stage1_tracks=stage1_tracks,
            stage1_score=stage1_score,
            stage2_states=stage2_states,
            stage2_tracks=stage2_tracks,
            stage2_score=stage2_score,
        )
        record = {
            "token": token,
            "result_status": "success",
            "scripted_spec": asdict(spec),
            "spec_rank": int(spec_rank),
            "stage1_score": stage1_score,
            "stage2_score": stage2_score,
            "stage1_peak_lateral_deviation_m": float(stage1_features.end_lateral_deviation_m),
            "stage2_end_lateral_deviation_m": float(stage2_features.end_lateral_deviation_m),
            "stage2_end_heading_error_deg": float(stage2_features.end_heading_error_deg),
        }
        kept_results.append((_candidate_rank(stage1_features, stage2_features, stage2_score, cfg), record, payload))

    kept_results.sort(key=lambda item: item[0], reverse=True)
    kept_results = kept_results[: cfg.top_k_per_token]

    output_dir = Path(cfg.output_dir)
    manifest_records: List[str] = []
    for final_rank, (rank_score, record, payload) in enumerate(kept_results):
        record["final_rank"] = int(final_rank)
        record["candidate_rank"] = float(rank_score)
        if cfg.save_format != "none":
            stem = f"cut_in_line_scripted_{token}_r{final_rank:02d}"
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
                    "attempted_specs": len(stage_specs),
                },
                ensure_ascii=False,
            )
        )

    return {
        "token": token,
        "kept": len(kept_results),
        "attempted_specs": len(stage_specs),
        "manifest_records": manifest_records,
        "reject_counts": reject_counts,
        "top_reject_reason": top_reject_reason,
    }


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

    loader: MetricCacheLoader = _load_metric_cache_loader(args.metric_cache_path)
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

    if run_cfg.parallel_backend == "none":
        _init_worker(str(args.metric_cache_path), asdict(run_cfg))
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
                f"[{idx}/{len(tokens)}] {token}: kept={result['kept']}, "
                f"specs={result['attempted_specs']}, top_reject={result.get('top_reject_reason')}"
            )
    else:
        max_workers = _auto_workers(run_cfg.num_workers)
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(str(args.metric_cache_path), asdict(run_cfg)),
        ) as executor:
            future_to_meta = {executor.submit(_run_token_worker, token): (idx, token) for idx, token in enumerate(tokens, start=1)}
            for future in as_completed(future_to_meta):
                idx, token = future_to_meta[future]
                result = future.result()
                manifest_lines.extend(result["manifest_records"])
                stats["num_candidates"] += int(result["kept"])
                stats["attempted_specs"] += int(result["attempted_specs"])
                if int(result["kept"]) > 0:
                    stats["tokens_with_candidates"] += 1
                for reason, count in result.get("reject_counts", {}).items():
                    stats["reject_counts"][reason] = stats["reject_counts"].get(reason, 0) + int(count)
                print(
                    f"[{idx}/{len(tokens)}] {token}: kept={result['kept']}, "
                    f"specs={result['attempted_specs']}, top_reject={result.get('top_reject_reason')}"
                )

    manifest_path = output_dir / args.manifest_name
    manifest_path.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")
    (output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
