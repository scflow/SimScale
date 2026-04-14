#!/usr/bin/env python3
"""
Generate single-stage overspeed pseudo-sim traces on NAVSIM.

Design:
1) Use the original human future trajectory as the geometric path reference.
2) Re-parameterize that path with an acceleration-limited speed profile toward a
   target speed (default: 50 km/h) over a single 4s stage.
3) Remove route-approximate front vehicles from current + future tracked objects.
4) When overspeed distance exceeds the original human path length, continue along
   the route centerline instead of straight-line extrapolation.
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
from nuplan.common.actor_state.tracked_objects import TrackedObjects
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

from navsim.behavior.scene_adapter import build_actor_from_detection_track
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_states_to_state_array
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex


@dataclass
class OverspeedConfig:
    target_speed_mps: float = 50.0 / 3.6
    max_accel_mps2: float = 6.0
    front_min_lon_m: float = 0.0
    keep_strict_gate: bool = False


def _resample_local_poses(local_poses: np.ndarray, target_len: int) -> np.ndarray:
    poses = np.asarray(local_poses, dtype=np.float32)
    n = len(poses)
    t = int(target_len)
    if t <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    if n == 0:
        return np.zeros((t, 3), dtype=np.float32)
    if n == t:
        return poses.copy()
    if n == 1:
        return np.repeat(poses, t, axis=0)

    src = np.linspace(0.0, 1.0, num=n, dtype=np.float64)
    dst = np.linspace(0.0, 1.0, num=t, dtype=np.float64)
    x = np.interp(dst, src, poses[:, 0].astype(np.float64))
    y = np.interp(dst, src, poses[:, 1].astype(np.float64))
    h_unwrap = np.unwrap(poses[:, 2].astype(np.float64))
    h = np.interp(dst, src, h_unwrap)
    h = np.arctan2(np.sin(h), np.cos(h))
    return np.stack([x, y, h], axis=1).astype(np.float32)


def _normalize_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _smoothstep01(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _sample_xy_along_path(
    path_xy: np.ndarray,
    path_heading: np.ndarray,
    cum_s: np.ndarray,
    query_s: float,
) -> Tuple[float, float, float]:
    total_s = float(cum_s[-1]) if len(cum_s) > 0 else 0.0
    if total_s <= 1e-6:
        heading = float(path_heading[-1]) if len(path_heading) else 0.0
        return 0.0, 0.0, heading

    if query_s >= total_s:
        end_xy = path_xy[-1]
        heading = float(path_heading[-1])
        return float(end_xy[0]), float(end_xy[1]), heading

    idx = int(np.searchsorted(cum_s, query_s, side="right") - 1)
    idx = max(0, min(idx, len(path_xy) - 2))
    ds = float(cum_s[idx + 1] - cum_s[idx])
    alpha = 0.0 if ds <= 1e-6 else float((query_s - cum_s[idx]) / ds)
    xy = (1.0 - alpha) * path_xy[idx] + alpha * path_xy[idx + 1]
    heading = float((1.0 - alpha) * path_heading[idx] + alpha * path_heading[idx + 1])
    return float(xy[0]), float(xy[1]), heading


def _human_reference_geometry(
    metric_cache: MetricCache,
    proposal_sampling: TrajectorySampling,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if metric_cache.human_trajectory is None:
        raise ValueError("metric_cache.human_trajectory is required for overspeed generation.")

    ref_local = _resample_local_poses(
        np.asarray(metric_cache.human_trajectory.poses, dtype=np.float32),
        proposal_sampling.num_poses,
    )
    path_xy = np.concatenate([np.zeros((1, 2), dtype=np.float64), ref_local[:, :2].astype(np.float64)], axis=0)
    seg = np.diff(path_xy, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum_s = np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(seg_len, dtype=np.float64)], axis=0)

    path_heading = np.zeros(len(path_xy), dtype=np.float64)
    if len(path_xy) >= 2:
        seg_heading = np.arctan2(seg[:, 1], seg[:, 0])
        if len(seg_heading) == 0:
            seg_heading = np.array([0.0], dtype=np.float64)
        seg_heading = np.unwrap(seg_heading)
        path_heading[0] = seg_heading[0]
        path_heading[1:] = seg_heading
        path_heading = _normalize_angle(path_heading)

    return path_xy, path_heading, cum_s


def build_overspeed_states(
    metric_cache: MetricCache,
    proposal_sampling: TrajectorySampling,
    cfg: OverspeedConfig,
) -> np.ndarray:
    path_xy_local, path_heading_local, human_cum_s = _human_reference_geometry(metric_cache, proposal_sampling)
    states = np.zeros((proposal_sampling.num_poses + 1, StateIndex.size()), dtype=np.float64)
    states[0] = ego_states_to_state_array([metric_cache.ego_state])[0]

    init = metric_cache.ego_state.rear_axle
    c = math.cos(float(init.heading))
    s = math.sin(float(init.heading))
    human_length = float(human_cum_s[-1]) if len(human_cum_s) > 0 else 0.0

    dt = float(proposal_sampling.interval_length)
    init_speed = float(metric_cache.ego_state.dynamic_car_state.speed)
    speeds = np.zeros(proposal_sampling.num_poses + 1, dtype=np.float64)
    distances = np.zeros(proposal_sampling.num_poses + 1, dtype=np.float64)
    speeds[0] = max(0.0, init_speed)
    for idx in range(1, proposal_sampling.num_poses + 1):
        prev = speeds[idx - 1]
        nxt = min(prev + float(cfg.max_accel_mps2) * dt, float(cfg.target_speed_mps))
        speeds[idx] = nxt
        distances[idx] = distances[idx - 1] + 0.5 * (prev + nxt) * dt

    human_end_x_local, human_end_y_local, human_end_heading_local = _sample_xy_along_path(
        path_xy_local,
        path_heading_local,
        human_cum_s,
        human_length,
    )
    human_end_heading_global = float(init.heading) + float(human_end_heading_local)
    human_end_x_global = float(init.x) + c * human_end_x_local - s * human_end_y_local
    human_end_y_global = float(init.y) + s * human_end_x_local + c * human_end_y_local

    centerline = metric_cache.centerline
    centerline_start_progress = float(centerline.project(Point(human_end_x_global, human_end_y_global)))
    blend_distance_m = 20.0

    x_all = np.zeros(proposal_sampling.num_poses + 1, dtype=np.float64)
    y_all = np.zeros(proposal_sampling.num_poses + 1, dtype=np.float64)
    x_all[0] = float(init.x)
    y_all[0] = float(init.y)

    for idx in range(1, proposal_sampling.num_poses + 1):
        query_s = float(distances[idx])
        if query_s <= human_length + 1e-6:
            lx, ly, _ = _sample_xy_along_path(path_xy_local, path_heading_local, human_cum_s, query_s)
            gx = float(init.x) + c * lx - s * ly
            gy = float(init.y) + s * lx + c * ly
        else:
            extra_s = query_s - human_length
            route_progress = centerline_start_progress + extra_s
            if route_progress <= centerline.length:
                center_state = centerline.interpolate([route_progress], as_array=True)[0]
                center_x = float(center_state[0])
                center_y = float(center_state[1])
            else:
                center_state = centerline.interpolate([centerline.length], as_array=True)[0]
                end_heading = float(center_state[2])
                remain = route_progress - float(centerline.length)
                center_x = float(center_state[0] + remain * math.cos(end_heading))
                center_y = float(center_state[1] + remain * math.sin(end_heading))

            tangent_x = float(human_end_x_global + extra_s * math.cos(human_end_heading_global))
            tangent_y = float(human_end_y_global + extra_s * math.sin(human_end_heading_global))
            alpha = _smoothstep01(extra_s / blend_distance_m) if blend_distance_m > 1e-6 else 1.0
            gx = (1.0 - alpha) * tangent_x + alpha * center_x
            gy = (1.0 - alpha) * tangent_y + alpha * center_y
        x_all[idx] = gx
        y_all[idx] = gy

    vx = np.gradient(x_all, dt)
    vy = np.gradient(y_all, dt)
    heading = np.unwrap(np.arctan2(vy, vx))
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    angular_velocity = np.gradient(heading, dt)
    angular_acceleration = np.gradient(angular_velocity, dt)

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
    states[0, StateIndex.STEERING_ANGLE] = float(metric_cache.ego_state.tire_steering_angle)
    states[0, StateIndex.STEERING_RATE] = float(metric_cache.ego_state.dynamic_car_state.tire_steering_rate)

    return states


def _state_xyh(state_or_arr: Any) -> Tuple[float, float, float]:
    if hasattr(state_or_arr, "rear_axle"):
        rear = state_or_arr.rear_axle
        return float(rear.x), float(rear.y), float(rear.heading)
    return (
        float(state_or_arr[StateIndex.X]),
        float(state_or_arr[StateIndex.Y]),
        float(state_or_arr[StateIndex.HEADING]),
    )


def _rotate_to_local(dx: float, dy: float, heading: float) -> Tuple[float, float]:
    c = math.cos(heading)
    s = math.sin(heading)
    lon = c * dx + s * dy
    lat = -s * dx + c * dy
    return lon, lat


def _filter_detections_tracks(
    detections: DetectionsTracks,
    ego_state_or_arr: Any,
    route_lane_ids: Sequence[str],
    map_api: Any,
    front_min_lon_m: float,
) -> Tuple[DetectionsTracks, List[str]]:
    ex, ey, eh = _state_xyh(ego_state_or_arr)
    route_lane_ids_set = {str(lane_id) for lane_id in route_lane_ids}
    filtered: List[Any] = []
    removed_tokens: List[str] = []

    for track in detections.tracked_objects.tracked_objects:
        track_type = str(getattr(track, "tracked_object_type", "")).lower()
        if not track_type.endswith("vehicle"):
            filtered.append(track)
            continue

        actor = build_actor_from_detection_track(track, map_api)
        lane_object = getattr(actor, "lane_object", None)
        if lane_object is None or str(lane_object.id) not in route_lane_ids_set:
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
) -> Tuple[MetricCache, Dict[str, Any]]:
    map_params = metric_cache.map_parameters
    map_root = map_root_override or map_params.map_root
    map_api = get_maps_api(map_root, map_params.map_version, map_params.map_name)

    filtered_cache = copy.copy(metric_cache)
    if map_root_override is not None:
        filtered_cache.map_parameters.map_root = map_root_override

    current_tracks = metric_cache.current_tracked_objects[0]
    filtered_current, removed_current = _filter_detections_tracks(
        current_tracks,
        metric_cache.ego_state,
        metric_cache.route_lane_ids,
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
            metric_cache.route_lane_ids,
            map_api,
            cfg.front_min_lon_m,
        )
        filtered_future.append(filtered_frame)
        removed_future_tokens.extend(removed_frame)

    filtered_cache.current_tracked_objects = [filtered_current]
    filtered_cache.future_tracked_objects = filtered_future

    removed_all = removed_current + removed_future_tokens
    stats = {
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
    if metric_cache.human_trajectory is None:
        if verbose:
            print(f"[{token}] skip: human_trajectory is None.")
        return None

    ego_states = build_overspeed_states(metric_cache, proposal_sampling, overspeed_cfg)

    if not base.pass_physics_filter(ego_states, thresholds):
        if verbose:
            print(f"[{token}] skip: physics filter failed.")
        return None

    filtered_cache, removal_stats = clone_metric_cache_with_filtered_tracks(
        metric_cache=metric_cache,
        ego_states=ego_states,
        map_root_override=map_root_override,
        cfg=overspeed_cfg,
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
    trace["scenario_type"] = "overspeed_single_stage"
    trace["overspeed"] = {
        "target_speed_kmh": float(overspeed_cfg.target_speed_mps * 3.6),
        "target_speed_mps": float(overspeed_cfg.target_speed_mps),
        "max_accel_mps2": float(overspeed_cfg.max_accel_mps2),
        "front_min_lon_m": float(overspeed_cfg.front_min_lon_m),
        "route_lane_ids": [str(lane_id) for lane_id in metric_cache.route_lane_ids],
        "single_stage_horizon_s": float(proposal_sampling.time_horizon),
        "filtered_traffic": removal_stats,
        "mean_ego_speed_mps": float(np.mean(ego_speed)),
        "max_ego_speed_mps": float(np.max(ego_speed)),
    }
    return trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate single-stage overspeed pseudo-sim traces.")
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
