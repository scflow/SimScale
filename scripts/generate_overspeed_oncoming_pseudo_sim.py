#!/usr/bin/env python3
"""
Generate single-stage overspeed pseudo-sim traces on NAVSIM.

Design:
1) Pick a single target oncoming lane near the ego start pose.
2) Start the generated ego trajectory on that target lane and keep a 4s,
   single-stage overspeed rollout toward the target speed.
3) Continue along the oncoming lane's outgoing geometry instead of the original
   route centerline.
4) Remove only vehicles that lie on that single target oncoming lane from both
   current and future tracked objects.
5) Roll out reactive IDM traffic once and score the single segment with PDM.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely.geometry import Point

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import generate_ood_mini as base
except ModuleNotFoundError:
    from scripts import generate_ood_mini as base

from navsim.behavior.scene_adapter import build_actor_from_detection_track, lane_heading_at_point
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_states_to_state_array,
    state_array_to_ego_state,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath


@dataclass
class OverspeedConfig:
    target_speed_mps: float = 50.0 / 3.6
    max_accel_mps2: float = 6.0
    front_min_lon_m: float = 0.0
    keep_strict_gate: bool = False


@dataclass
class OncomingLaneInfo:
    lane_id: str
    roadblock_id: str
    layer_name: str
    search_radius_m: float
    distance_to_ego_m: float
    heading_delta_deg: float


def _normalize_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _heading_delta_abs(a: float, b: float) -> float:
    return float(abs(math.atan2(math.sin(a - b), math.cos(a - b))))


def _rotate_to_local(dx: float, dy: float, heading: float) -> Tuple[float, float]:
    c = math.cos(heading)
    s = math.sin(heading)
    lon = c * dx + s * dy
    lat = -s * dx + c * dy
    return lon, lat


def _state_xyh(state_or_arr: Any) -> Tuple[float, float, float]:
    if hasattr(state_or_arr, "rear_axle"):
        rear = state_or_arr.rear_axle
        return float(rear.x), float(rear.y), float(rear.heading)
    return (
        float(state_or_arr[StateIndex.X]),
        float(state_or_arr[StateIndex.Y]),
        float(state_or_arr[StateIndex.HEADING]),
    )


def _select_target_oncoming_lane(metric_cache: MetricCache, map_api: Any) -> Tuple[Any, OncomingLaneInfo]:
    ego = metric_cache.ego_state.rear_axle
    ego_point = Point(float(ego.x), float(ego.y))
    ego_point2d = Point2D(float(ego.x), float(ego.y))
    route_lane_ids = {str(lane_id) for lane_id in metric_cache.route_lane_ids}

    def collect_candidates(radius: float, layers: List[SemanticMapLayer]) -> List[Tuple[float, float, float, Any, str]]:
        proximal = map_api.get_proximal_map_objects(ego_point2d, radius, layers)
        candidates: List[Tuple[float, float, float, Any, str]] = []
        for layer, objects in proximal.items():
            for lane_object in objects:
                lane_id = str(lane_object.id)
                if lane_id in route_lane_ids:
                    continue
                heading = float(lane_heading_at_point(lane_object, float(ego.x), float(ego.y)))
                heading_delta = _heading_delta_abs(heading, float(ego.heading))
                if heading_delta < math.radians(150.0):
                    continue
                distance = float(lane_object.polygon.distance(ego_point))
                score = distance + 8.0 * abs(math.pi - heading_delta)
                candidates.append((score, distance, heading_delta, lane_object, layer.name))
        candidates.sort(key=lambda item: (item[0], item[1], -item[2], str(item[3].id)))
        return candidates

    for radius in (12.0, 20.0, 30.0, 40.0):
        lane_candidates = collect_candidates(radius, [SemanticMapLayer.LANE])
        if lane_candidates:
            _, distance, heading_delta, lane_object, layer_name = lane_candidates[0]
            info = OncomingLaneInfo(
                lane_id=str(lane_object.id),
                roadblock_id=str(lane_object.get_roadblock_id()),
                layer_name=str(layer_name),
                search_radius_m=float(radius),
                distance_to_ego_m=float(distance),
                heading_delta_deg=float(math.degrees(heading_delta)),
            )
            return lane_object, info

    for radius in (12.0, 20.0, 30.0, 40.0):
        connector_candidates = collect_candidates(radius, [SemanticMapLayer.LANE_CONNECTOR])
        if connector_candidates:
            _, distance, heading_delta, lane_object, layer_name = connector_candidates[0]
            info = OncomingLaneInfo(
                lane_id=str(lane_object.id),
                roadblock_id=str(lane_object.get_roadblock_id()),
                layer_name=str(layer_name),
                search_radius_m=float(radius),
                distance_to_ego_m=float(distance),
                heading_delta_deg=float(math.degrees(heading_delta)),
            )
            return lane_object, info

    raise RuntimeError("Failed to find a target oncoming lane near the ego start pose.")


def _build_heading_continuous_path(start_lane: Any, min_length: float) -> PDMPath:
    discrete_path = []
    current_lane = start_lane
    visited_lane_ids = set()

    while current_lane is not None and str(current_lane.id) not in visited_lane_ids:
        visited_lane_ids.add(str(current_lane.id))
        lane_path = list(current_lane.baseline_path.discrete_path)
        if discrete_path and lane_path:
            first_state = lane_path[0]
            last_state = discrete_path[-1]
            if math.hypot(first_state.x - last_state.x, first_state.y - last_state.y) < 1e-3:
                lane_path = lane_path[1:]
        discrete_path.extend(lane_path)
        if len(discrete_path) >= 2 and PDMPath(discrete_path).length >= float(min_length):
            break

        outgoing_edges = [lane for lane in current_lane.outgoing_edges if str(lane.id) not in visited_lane_ids]
        if not outgoing_edges:
            break

        current_heading = float(discrete_path[-1].heading)
        current_lane = min(
            outgoing_edges,
            key=lambda lane: _heading_delta_abs(current_heading, float(lane.baseline_path.discrete_path[0].heading)),
        )

    if len(discrete_path) < 2:
        raise RuntimeError("Failed to build a continuous path for the selected oncoming lane.")

    return PDMPath(discrete_path)


def build_overspeed_states(
    metric_cache: MetricCache,
    proposal_sampling: TrajectorySampling,
    cfg: OverspeedConfig,
    map_api: Any,
) -> Tuple[np.ndarray, OncomingLaneInfo]:
    target_lane, target_info = _select_target_oncoming_lane(metric_cache, map_api)
    min_path_length = max(float(cfg.target_speed_mps) * float(proposal_sampling.time_horizon) + 40.0, 60.0)
    target_path = _build_heading_continuous_path(target_lane, min_length=min_path_length)

    init_state_arr = ego_states_to_state_array([metric_cache.ego_state])[0]
    init_speed = float(metric_cache.ego_state.dynamic_car_state.speed)
    dt = float(proposal_sampling.interval_length)

    speeds = np.zeros(proposal_sampling.num_poses + 1, dtype=np.float64)
    distances = np.zeros(proposal_sampling.num_poses + 1, dtype=np.float64)
    speeds[0] = max(0.0, init_speed)
    for idx in range(1, proposal_sampling.num_poses + 1):
        prev = speeds[idx - 1]
        nxt = min(prev + float(cfg.max_accel_mps2) * dt, float(cfg.target_speed_mps))
        speeds[idx] = nxt
        distances[idx] = distances[idx - 1] + 0.5 * (prev + nxt) * dt

    start_progress = float(target_path.project(Point(float(metric_cache.ego_state.rear_axle.x), float(metric_cache.ego_state.rear_axle.y))))
    start_progress = max(0.0, min(start_progress, float(target_path.length)))

    sampled_progress = start_progress + distances
    clipped_progress = np.minimum(sampled_progress, float(target_path.length))
    sampled_states = target_path.interpolate(clipped_progress, as_array=True)

    x_all = sampled_states[:, 0].astype(np.float64)
    y_all = sampled_states[:, 1].astype(np.float64)
    target_heading = sampled_states[:, 2].astype(np.float64)

    if float(target_path.length) > 0.0:
        overrun_mask = sampled_progress > float(target_path.length)
        if np.any(overrun_mask):
            end_state = target_path.interpolate([float(target_path.length)], as_array=True)[0]
            end_heading = float(end_state[2])
            remain = sampled_progress[overrun_mask] - float(target_path.length)
            x_all[overrun_mask] = float(end_state[0]) + remain * math.cos(end_heading)
            y_all[overrun_mask] = float(end_state[1]) + remain * math.sin(end_heading)
            target_heading[overrun_mask] = end_heading

    vx = np.gradient(x_all, dt)
    vy = np.gradient(y_all, dt)
    heading = np.unwrap(np.arctan2(vy, vx))
    heading[0] = target_heading[0]
    heading = np.unwrap(heading)
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    angular_velocity = np.gradient(heading, dt)
    angular_acceleration = np.gradient(angular_velocity, dt)

    states = np.zeros((proposal_sampling.num_poses + 1, StateIndex.size()), dtype=np.float64)
    states[:, StateIndex.X] = x_all
    states[:, StateIndex.Y] = y_all
    states[:, StateIndex.HEADING] = _normalize_angle(heading)
    states[:, StateIndex.VELOCITY_X] = vx
    states[:, StateIndex.VELOCITY_Y] = vy
    states[:, StateIndex.ACCELERATION_X] = ax
    states[:, StateIndex.ACCELERATION_Y] = ay
    states[:, StateIndex.STEERING_ANGLE] = 0.0
    states[:, StateIndex.STEERING_RATE] = 0.0
    states[:, StateIndex.ANGULAR_VELOCITY] = angular_velocity
    states[:, StateIndex.ANGULAR_ACCELERATION] = angular_acceleration

    states[0, StateIndex.STEERING_ANGLE] = float(init_state_arr[StateIndex.STEERING_ANGLE])
    states[0, StateIndex.STEERING_RATE] = float(init_state_arr[StateIndex.STEERING_RATE])

    return states, target_info


def _filter_detections_tracks(
    detections: DetectionsTracks,
    ego_state_or_arr: Any,
    target_lane_id: str,
    map_api: Any,
    front_min_lon_m: float,
) -> Tuple[DetectionsTracks, List[str]]:
    ex, ey, eh = _state_xyh(ego_state_or_arr)
    filtered: List[Any] = []
    removed_tokens: List[str] = []

    for track in detections.tracked_objects.tracked_objects:
        track_type = str(getattr(track, "tracked_object_type", "")).lower()
        if not track_type.endswith("vehicle"):
            filtered.append(track)
            continue

        actor = build_actor_from_detection_track(track, map_api)
        lane_object = getattr(actor, "lane_object", None)
        if lane_object is None or str(lane_object.id) != str(target_lane_id):
            filtered.append(track)
            continue

        dx = float(track.center.x) - ex
        dy = float(track.center.y) - ey
        rel_lon, _ = _rotate_to_local(dx, dy, eh)
        if rel_lon > float(front_min_lon_m):
            removed_tokens.append(str(track.track_token))
            continue

        filtered.append(track)

    return DetectionsTracks(TrackedObjects(filtered)), removed_tokens


def clone_metric_cache_with_filtered_tracks(
    metric_cache: MetricCache,
    ego_states: np.ndarray,
    map_root_override: Optional[str],
    cfg: OverspeedConfig,
    target_lane_info: OncomingLaneInfo,
    map_api: Any,
) -> Tuple[MetricCache, Dict[str, Any]]:
    filtered_cache = copy.copy(metric_cache)
    filtered_cache.map_parameters = copy.copy(metric_cache.map_parameters)
    if map_root_override is not None:
        filtered_cache.map_parameters.map_root = map_root_override

    filtered_cache.ego_state = state_array_to_ego_state(
        ego_states[0],
        metric_cache.ego_state.time_point,
        metric_cache.ego_state.car_footprint.vehicle_parameters,
    )

    current_tracks = metric_cache.current_tracked_objects[0]
    filtered_current, removed_current = _filter_detections_tracks(
        current_tracks,
        ego_states[0],
        target_lane_info.lane_id,
        map_api,
        cfg.front_min_lon_m,
    )

    filtered_future: List[DetectionsTracks] = []
    removed_future_tokens: List[str] = []
    for idx, detections in enumerate(metric_cache.future_tracked_objects):
        state_idx = min(idx + 1, len(ego_states) - 1)
        filtered_frame, removed_frame = _filter_detections_tracks(
            detections,
            ego_states[state_idx],
            target_lane_info.lane_id,
            map_api,
            cfg.front_min_lon_m,
        )
        filtered_future.append(filtered_frame)
        removed_future_tokens.extend(removed_frame)

    filtered_cache.current_tracked_objects = [filtered_current]
    filtered_cache.future_tracked_objects = filtered_future

    removed_all = removed_current + removed_future_tokens
    stats = {
        "target_lane_id": str(target_lane_info.lane_id),
        "current_removed_count": int(len(removed_current)),
        "future_removed_count": int(len(removed_future_tokens)),
        "removed_count_total": int(len(removed_all)),
        "removed_track_tokens": sorted(set(removed_all)),
    }
    return filtered_cache, stats


def _nan_stage2_score() -> Dict[str, float]:
    return {
        "no_at_fault_collisions": float("nan"),
        "drivable_area_compliance": float("nan"),
        "driving_direction_compliance": float("nan"),
        "traffic_light_compliance": float("nan"),
        "ego_progress": float("nan"),
        "pdm_score": float("nan"),
    }


def run_for_token(
    token: str,
    metric_cache: MetricCache,
    proposal_sampling: TrajectorySampling,
    reactive_policy: Any,
    thresholds: base.Thresholds,
    overspeed_cfg: OverspeedConfig,
    verbose: bool,
    map_root_override: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    map_params = metric_cache.map_parameters
    map_root = map_root_override or map_params.map_root
    map_api = get_maps_api(map_root, map_params.map_version, map_params.map_name)

    try:
        ego_states, target_lane_info = build_overspeed_states(metric_cache, proposal_sampling, overspeed_cfg, map_api)
    except Exception as exc:
        if verbose:
            print(f"[{token}] skip: oncoming-lane build failed: {exc}")
        return None

    if not base.pass_physics_filter(ego_states, thresholds):
        if verbose:
            print(f"[{token}] skip: physics filter failed.")
        return None

    filtered_cache, removal_stats = clone_metric_cache_with_filtered_tracks(
        metric_cache=metric_cache,
        ego_states=ego_states,
        map_root_override=map_root_override,
        cfg=overspeed_cfg,
        target_lane_info=target_lane_info,
        map_api=map_api,
    )

    scorer = PDMScorer(
        proposal_sampling=proposal_sampling,
        config=PDMScorerConfig(human_penalty_filter=False),
    )
    phase1_tracks = reactive_policy.simulate_environment(ego_states, filtered_cache)
    stage1_score = base.score_segment(
        scorer=scorer,
        states=ego_states,
        metric_cache=filtered_cache,
        simulated_tracks=phase1_tracks,
    )

    if overspeed_cfg.keep_strict_gate and not base.pass_strict_gate(stage1_score, thresholds):
        if verbose:
            print(f"[{token}] skip: strict gate failed with score={stage1_score}")
        return None

    trace = base.serialize_trace(
        token=token,
        vocab_index=-1,
        ego_states=ego_states,
        tracks=phase1_tracks,
        stage1_score=stage1_score,
        stage2_score=_nan_stage2_score(),
    )
    ego_speed = np.hypot(ego_states[:, StateIndex.VELOCITY_X], ego_states[:, StateIndex.VELOCITY_Y])
    trace["scenario_type"] = "overspeed_oncoming_single_stage"
    trace["overspeed"] = {
        "target_speed_kmh": float(overspeed_cfg.target_speed_mps * 3.6),
        "target_speed_mps": float(overspeed_cfg.target_speed_mps),
        "max_accel_mps2": float(overspeed_cfg.max_accel_mps2),
        "front_min_lon_m": float(overspeed_cfg.front_min_lon_m),
        "route_lane_ids": [str(lane_id) for lane_id in metric_cache.route_lane_ids],
        "target_oncoming_lane": {
            "lane_id": str(target_lane_info.lane_id),
            "roadblock_id": str(target_lane_info.roadblock_id),
            "layer_name": str(target_lane_info.layer_name),
            "search_radius_m": float(target_lane_info.search_radius_m),
            "distance_to_ego_m": float(target_lane_info.distance_to_ego_m),
            "heading_delta_deg": float(target_lane_info.heading_delta_deg),
        },
        "single_stage_horizon_s": float(proposal_sampling.time_horizon),
        "filtered_traffic": removal_stats,
        "mean_ego_speed_mps": float(np.mean(ego_speed)),
        "max_ego_speed_mps": float(np.max(ego_speed)),
    }
    return trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate single-stage oncoming overspeed pseudo-sim traces.")
    parser.add_argument("--metric-cache-path", type=Path, required=True, help="Path to metric cache root.")
    parser.add_argument("--output-dir", type=Path, default=Path("generated_overspeed_data"))
    parser.add_argument("--save-format", type=str, default="both", choices=["json", "pkl", "both"])
    parser.add_argument("--max-scenes", type=int, default=None, help="Limit number of tokens to process.")
    parser.add_argument("--seed", type=int, default=0, help="Reserved for parity with other scripts.")
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--horizon-sec", type=float, default=4.0)
    parser.add_argument("--target-speed-kmh", type=float, default=50.0)
    parser.add_argument("--overspeed-max-accel-mps2", type=float, default=6.0)
    parser.add_argument("--front-min-lon-m", type=float, default=0.0)
    parser.add_argument("--max-abs-accel-mps2", type=float, default=10.0)
    parser.add_argument("--max-abs-steer-deg", type=float, default=60.0)
    parser.add_argument("--min-progress", type=float, default=0.5)
    parser.add_argument("--map-root-override", type=str, default=None, help="Optional map root override for IDM.")
    parser.add_argument("--enable-strict-gate", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ = args.seed
    cache_root = base.resolve_metric_cache_root(args.metric_cache_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    proposal_sampling = TrajectorySampling(
        num_poses=int(round(args.horizon_sec / args.interval)),
        interval_length=args.interval,
    )
    thresholds = base.Thresholds(
        max_abs_accel_mps2=float(args.max_abs_accel_mps2),
        max_abs_steer_deg=float(args.max_abs_steer_deg),
        min_progress=float(args.min_progress),
        max_join_translation_m=float("inf"),
        max_join_heading_deg=float("inf"),
        max_join_speed_delta_mps=float("inf"),
    )
    overspeed_cfg = OverspeedConfig(
        target_speed_mps=float(args.target_speed_kmh) / 3.6,
        max_accel_mps2=float(args.overspeed_max_accel_mps2),
        front_min_lon_m=float(args.front_min_lon_m),
        keep_strict_gate=bool(args.enable_strict_gate),
    )

    metric_cache_loader = base.build_metric_cache_loader(cache_root)
    tokens = list(metric_cache_loader.tokens)
    if args.max_scenes is not None:
        tokens = tokens[: args.max_scenes]

    reactive_policy = base.build_reactive_policy(
        proposal_sampling=proposal_sampling,
        map_root_override=args.map_root_override,
    )

    print(f"Processing {len(tokens)} scene tokens from {cache_root}")
    print(
        f"Overspeed target={overspeed_cfg.target_speed_mps:.2f} m/s ({overspeed_cfg.target_speed_mps * 3.6:.1f} km/h), "
        f"horizon={proposal_sampling.time_horizon:.1f}s"
    )

    saved = 0
    failed = 0
    for idx, token in enumerate(tokens):
        if args.verbose:
            print(f"[{idx + 1}/{len(tokens)}] token={token}")
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            trace = run_for_token(
                token=token,
                metric_cache=metric_cache,
                proposal_sampling=proposal_sampling,
                reactive_policy=reactive_policy,
                thresholds=thresholds,
                overspeed_cfg=overspeed_cfg,
                verbose=args.verbose,
                map_root_override=args.map_root_override,
            )
            if trace is None:
                failed += 1
                continue
            base.save_trace(trace, args.output_dir, args.save_format)
            saved += 1
            if args.verbose:
                print(f"  saved token={token}")
        except Exception as exc:
            failed += 1
            print(f"  token={token} failed: {exc}")

    print(f"Done. saved={saved}, failed_or_skipped={failed}, output_dir={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
