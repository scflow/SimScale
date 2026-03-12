#!/usr/bin/env python3
"""
Generate red-light pseudo-simulation traces on NAVSIM using a two-stage flow.

Design goals (V1):
- trainability-first (robust, not purely extreme conflicts)
- Stage-A: construct red-light violation boundary states
- Stage-B: pseudo-expert rescue rollout
- save success / failure with detailed event and traffic-light metadata
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from nuplan.common.actor_state.state_representation import Point2D, StateSE2, TimeDuration, TimePoint
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely.geometry import Point, Polygon

import generate_ood_mini as base
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_ego_state
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex


# -------------------------
# Dataclasses / Config
# -------------------------


@dataclass
class SeedEntry:
    log_name: str
    scene_token: str
    source_type: str  # waiting_red | passing_green


@dataclass
class TemplateWeights:
    late_brake_stopline: float = 0.30
    crosswalk_intrude_stop: float = 0.25
    commit_go: float = 0.45


@dataclass
class SourceWeights:
    waiting_red: float = 0.30
    passing_green: float = 0.70


@dataclass
class ParamRanges:
    t_boundary_min_s: float = 0.8
    t_boundary_max_s: float = 1.8
    reaction_delay_min_s: float = 0.2
    reaction_delay_max_s: float = 1.2
    brake_delay_min_s: float = 0.1
    brake_delay_max_s: float = 0.8
    a_brake_peak_min_mps2: float = 1.5
    a_brake_peak_max_mps2: float = 4.0
    phase_shift_min_s: float = 0.0
    phase_shift_max_s: float = 1.6


@dataclass
class IntrusionRanges:
    late_brake_stopline: Tuple[float, float] = (0.3, 1.5)
    crosswalk_intrude_stop: Tuple[float, float] = (1.5, 4.0)
    commit_go: Tuple[float, float] = (4.0, 8.0)


@dataclass
class EventMixConfig:
    template_weights: TemplateWeights
    source_weights: SourceWeights
    params: ParamRanges
    intrusion_ranges: IntrusionRanges


@dataclass
class RunConfig:
    sample_batch_size: int
    max_candidate_trials: int
    top_k_per_seed: int
    max_failure_traces_per_seed: int
    verbose: bool


@dataclass
class DynamicsCriteria:
    max_abs_accel_mps2: float = 6.0
    max_abs_steer_deg: float = 60.0
    max_abs_yaw_rate_deg_s: float = 45.0
    max_abs_jerk_mps3: float = 8.0
    max_abs_steer_rate_deg_s: float = 80.0
    join_window_k: int = 3


# -------------------------
# CLI
# -------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate red-light pseudo-simulation traces.")
    parser.add_argument("--metric-cache-path", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True, help="CSV/JSONL with log_name, scene_token, source_type.")
    parser.add_argument("--mini-log-root", type=Path, default=Path("mini_navsim_logs/mini"))
    parser.add_argument("--output-dir", type=Path, default=Path("generated_redlight_pseudo_data"))
    parser.add_argument("--save-format", type=str, default="both", choices=["json", "pkl", "both"])
    parser.add_argument("--manifest-name", type=str, default="manifest.jsonl")
    parser.add_argument("--event-mix-config", type=Path, default=None)
    parser.add_argument("--map-root-override", type=str, default=None)

    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--sample-batch-size", type=int, default=48)
    parser.add_argument("--max-candidate-trials", type=int, default=240)
    parser.add_argument("--top-k-per-seed", type=int, default=2)
    parser.add_argument("--max-failure-traces-per-seed", type=int, default=10)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--horizon-sec", type=float, default=4.0)

    # Basic gates (same style as generate_ood_mini)
    parser.add_argument("--max-abs-lon-m", type=float, default=20.0)
    parser.add_argument("--max-abs-lat-m", type=float, default=2.0)
    parser.add_argument("--max-abs-heading-deg", type=float, default=20.0)
    parser.add_argument("--max-abs-accel-mps2", type=float, default=6.0)
    parser.add_argument("--max-abs-steer-deg", type=float, default=60.0)
    parser.add_argument("--min-progress", type=float, default=0.5)
    parser.add_argument("--max-join-translation-m", type=float, default=1.0)
    parser.add_argument("--max-join-heading-deg", type=float, default=8.0)
    parser.add_argument("--max-join-speed-delta-mps", type=float, default=2.0)

    # Dynamics gates
    parser.add_argument("--dyn-max-abs-accel-mps2", type=float, default=6.0)
    parser.add_argument("--dyn-max-abs-steer-deg", type=float, default=60.0)
    parser.add_argument("--dyn-max-abs-yaw-rate-deg-s", type=float, default=45.0)
    parser.add_argument("--dyn-max-abs-jerk-mps3", type=float, default=8.0)
    parser.add_argument("--dyn-max-abs-steer-rate-deg-s", type=float, default=80.0)
    parser.add_argument("--dyn-join-window-k", type=int, default=3)

    parser.add_argument("--ratio-tolerance", type=float, default=0.05)
    parser.add_argument("--enforce-ratio-tolerance", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


# -------------------------
# Utilities
# -------------------------


def _sanitize_fragment(text: str) -> str:
    chars: List[str] = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            chars.append(ch)
        else:
            chars.append("_")
    out = "".join(chars).strip("_")
    return out or "unknown"


def _nan_score() -> Dict[str, float]:
    return {
        "no_at_fault_collisions": float("nan"),
        "drivable_area_compliance": float("nan"),
        "driving_direction_compliance": float("nan"),
        "traffic_light_compliance": float("nan"),
        "ego_progress": float("nan"),
        "pdm_score": float("nan"),
    }


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


def _cumulative_s(xy: np.ndarray) -> np.ndarray:
    if len(xy) == 0:
        return np.zeros((0,), dtype=np.float64)
    dxy = np.diff(xy, axis=0)
    ds = np.hypot(dxy[:, 0], dxy[:, 1])
    return np.concatenate([[0.0], np.cumsum(ds)])


def _local_from_global(global_xy: Tuple[float, float], origin: StateSE2) -> Tuple[float, float]:
    dx = float(global_xy[0] - origin.x)
    dy = float(global_xy[1] - origin.y)
    c = math.cos(origin.heading)
    s = math.sin(origin.heading)
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    return lx, ly


def _global_from_local(local_xy: Tuple[float, float], origin: StateSE2) -> Tuple[float, float]:
    lx, ly = float(local_xy[0]), float(local_xy[1])
    c = math.cos(origin.heading)
    s = math.sin(origin.heading)
    gx = origin.x + c * lx - s * ly
    gy = origin.y + s * lx + c * ly
    return gx, gy


def _build_default_event_mix() -> EventMixConfig:
    return EventMixConfig(
        template_weights=TemplateWeights(),
        source_weights=SourceWeights(),
        params=ParamRanges(),
        intrusion_ranges=IntrusionRanges(),
    )


def _load_event_mix_config(path: Optional[Path]) -> EventMixConfig:
    cfg = _build_default_event_mix()
    if path is None:
        return cfg
    if not path.exists():
        raise FileNotFoundError(f"event mix config not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))

    def _get(section: str, key: str, default: Any) -> Any:
        return raw.get(section, {}).get(key, default)

    cfg.template_weights = TemplateWeights(
        late_brake_stopline=float(_get("template_weights", "late_brake_stopline", cfg.template_weights.late_brake_stopline)),
        crosswalk_intrude_stop=float(_get("template_weights", "crosswalk_intrude_stop", cfg.template_weights.crosswalk_intrude_stop)),
        commit_go=float(_get("template_weights", "commit_go", cfg.template_weights.commit_go)),
    )
    cfg.source_weights = SourceWeights(
        waiting_red=float(_get("source_weights", "waiting_red", cfg.source_weights.waiting_red)),
        passing_green=float(_get("source_weights", "passing_green", cfg.source_weights.passing_green)),
    )
    cfg.params = ParamRanges(
        t_boundary_min_s=float(_get("params", "t_boundary_min_s", cfg.params.t_boundary_min_s)),
        t_boundary_max_s=float(_get("params", "t_boundary_max_s", cfg.params.t_boundary_max_s)),
        reaction_delay_min_s=float(_get("params", "reaction_delay_min_s", cfg.params.reaction_delay_min_s)),
        reaction_delay_max_s=float(_get("params", "reaction_delay_max_s", cfg.params.reaction_delay_max_s)),
        brake_delay_min_s=float(_get("params", "brake_delay_min_s", cfg.params.brake_delay_min_s)),
        brake_delay_max_s=float(_get("params", "brake_delay_max_s", cfg.params.brake_delay_max_s)),
        a_brake_peak_min_mps2=float(_get("params", "a_brake_peak_min_mps2", cfg.params.a_brake_peak_min_mps2)),
        a_brake_peak_max_mps2=float(_get("params", "a_brake_peak_max_mps2", cfg.params.a_brake_peak_max_mps2)),
        phase_shift_min_s=float(_get("params", "phase_shift_min_s", cfg.params.phase_shift_min_s)),
        phase_shift_max_s=float(_get("params", "phase_shift_max_s", cfg.params.phase_shift_max_s)),
    )
    cfg.intrusion_ranges = IntrusionRanges(
        late_brake_stopline=tuple(_get("intrusion_ranges", "late_brake_stopline", list(cfg.intrusion_ranges.late_brake_stopline))),
        crosswalk_intrude_stop=tuple(_get("intrusion_ranges", "crosswalk_intrude_stop", list(cfg.intrusion_ranges.crosswalk_intrude_stop))),
        commit_go=tuple(_get("intrusion_ranges", "commit_go", list(cfg.intrusion_ranges.commit_go))),
    )
    return cfg


def _load_seed_manifest(path: Path) -> List[SeedEntry]:
    if not path.exists():
        raise FileNotFoundError(f"seed manifest not found: {path}")

    entries: List[SeedEntry] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                row = json.loads(line)
                log_name = str(row.get("log_name") or "").strip()
                scene_token = str(row.get("scene_token") or row.get("token") or "").strip().lower()
                source_type = str(row.get("source_type") or "").strip()
                if not source_type:
                    source_type = "waiting_red" if "waiting" in path.name else "passing_green"
                if log_name and scene_token:
                    entries.append(SeedEntry(log_name=log_name, scene_token=scene_token, source_type=source_type))
    else:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                log_name = str(row.get("log_name") or "").strip()
                scene_token = str(row.get("scene_token") or row.get("token") or "").strip().lower()
                source_type = str(row.get("source_type") or "").strip()
                if not source_type:
                    if "waiting" in path.name.lower():
                        source_type = "waiting_red"
                    elif "passing" in path.name.lower() or "green" in path.name.lower():
                        source_type = "passing_green"
                    else:
                        # fallback heuristic
                        speed = float(row.get("current_speed") or row.get("raw_ego_speed") or 0.0)
                        red_count = float(row.get("raw_red_count") or 0.0)
                        source_type = "waiting_red" if (red_count > 0 and speed < 1.0) else "passing_green"
                if log_name and scene_token:
                    entries.append(SeedEntry(log_name=log_name, scene_token=scene_token, source_type=source_type))

    # deduplicate and drop known outlier from plan
    dedup: Dict[Tuple[str, str], SeedEntry] = {}
    for e in entries:
        if e.scene_token == "cd00be51b43a5281":
            continue
        if e.source_type not in ("waiting_red", "passing_green"):
            continue
        dedup[(e.log_name, e.scene_token)] = e
    return list(dedup.values())


def _reweight_seeds_for_max(entries: List[SeedEntry], max_seeds: int, cfg: EventMixConfig, rng: np.random.Generator) -> List[SeedEntry]:
    if max_seeds is None or len(entries) <= max_seeds:
        return entries

    by_source: Dict[str, List[SeedEntry]] = {"waiting_red": [], "passing_green": []}
    for e in entries:
        by_source.setdefault(e.source_type, []).append(e)

    out: List[SeedEntry] = []
    source_w = np.array([cfg.source_weights.waiting_red, cfg.source_weights.passing_green], dtype=np.float64)
    source_w = np.maximum(source_w, 1e-6)
    source_w /= source_w.sum()

    target_wait = int(round(max_seeds * float(source_w[0])))
    target_pass = max_seeds - target_wait

    for src, target in (("waiting_red", target_wait), ("passing_green", target_pass)):
        pool = by_source.get(src, [])
        if not pool or target <= 0:
            continue
        if len(pool) <= target:
            out.extend(pool)
        else:
            idx = rng.choice(np.arange(len(pool)), size=target, replace=False)
            out.extend([pool[int(i)] for i in idx])

    if len(out) < max_seeds:
        remaining = [e for e in entries if (e.log_name, e.scene_token) not in {(x.log_name, x.scene_token) for x in out}]
        if remaining:
            fill = min(len(remaining), max_seeds - len(out))
            idx = rng.choice(np.arange(len(remaining)), size=fill, replace=False)
            out.extend([remaining[int(i)] for i in idx])

    return out[:max_seeds]


def _weighted_template_choice(cfg: EventMixConfig, rng: np.random.Generator, source_type: str) -> str:
    names = ["late_brake_stopline", "crosswalk_intrude_stop", "commit_go"]
    w = np.array(
        [
            cfg.template_weights.late_brake_stopline,
            cfg.template_weights.crosswalk_intrude_stop,
            cfg.template_weights.commit_go,
        ],
        dtype=np.float64,
    )
    # Source-aware prior:
    # waiting_red seeds are more suitable for stop templates;
    # passing_green seeds are more suitable for commit_go templates.
    if source_type == "waiting_red":
        w = w * np.array([1.5, 1.3, 0.35], dtype=np.float64)
    elif source_type == "passing_green":
        w = w * np.array([0.8, 0.8, 1.35], dtype=np.float64)
    w = np.maximum(w, 1e-6)
    w /= w.sum()
    return str(rng.choice(np.array(names), p=w))


# -------------------------
# Traffic light / map context
# -------------------------


def _load_log_frames(mini_log_root: Path, log_name: str) -> List[Dict[str, Any]]:
    p = mini_log_root / f"{log_name}.pkl"
    if not p.exists():
        raise FileNotFoundError(f"mini log file not found: {p}")
    with p.open("rb") as f:
        return pickle.load(f)


def _token_to_frame_index(frames: List[Dict[str, Any]]) -> Dict[str, int]:
    return {str(frames[i]["token"]): i for i in range(len(frames))}


def _route_lane_connector_ids(
    metric_cache: Any,
    map_api: Any,
    frame_tl_ids: Sequence[str],
    ego_xy: Optional[Tuple[float, float]] = None,
) -> List[str]:
    route_set = {str(x) for x in metric_cache.route_lane_ids}
    out: List[str] = []

    # Prefer connectors that appear in current frame traffic lights.
    for lane_id in frame_tl_ids:
        lane_id_s = str(lane_id)
        obj = map_api.get_map_object(lane_id_s, SemanticMapLayer.LANE_CONNECTOR)
        if obj is not None:
            out.append(lane_id_s)
    if out:
        return sorted(set(out))

    # Next preference: connectors that appear in frame traffic lights and are on route.
    for lane_id in frame_tl_ids:
        lane_id_s = str(lane_id)
        if lane_id_s not in route_set:
            continue
        obj = map_api.get_map_object(lane_id_s, SemanticMapLayer.LANE_CONNECTOR)
        if obj is not None:
            out.append(lane_id_s)

    if out:
        return sorted(set(out))

    # Fallback: nearby route lane connectors.
    near_route: List[Tuple[float, str]] = []
    for lane_id in route_set:
        obj = map_api.get_map_object(lane_id, SemanticMapLayer.LANE_CONNECTOR)
        if obj is not None:
            poly = getattr(obj, "polygon", None)
            dist = float("inf")
            if poly is not None and ego_xy is not None:
                dist = float(poly.distance(Point(float(ego_xy[0]), float(ego_xy[1]))))
            near_route.append((dist, lane_id))

    if not near_route:
        return []
    near_route.sort(key=lambda x: x[0])
    # Keep a local neighborhood to avoid far-away route artifacts.
    k = min(20, len(near_route))
    return sorted({lane_id for _, lane_id in near_route[:k]})


def _extract_stopline_polygons(map_api: Any, lane_connector_ids: Sequence[str]) -> List[Polygon]:
    polys: List[Polygon] = []
    for lane_id in lane_connector_ids:
        lc = map_api.get_map_object(str(lane_id), SemanticMapLayer.LANE_CONNECTOR)
        if lc is None:
            continue
        for stop_line in getattr(lc, "stop_lines", []) or []:
            poly = getattr(stop_line, "polygon", None)
            if poly is not None:
                polys.append(poly)
    return polys


def _build_route_lane_connector_polygon_map(map_api: Any, lane_connector_ids: Sequence[str]) -> Dict[str, Polygon]:
    out: Dict[str, Polygon] = {}
    for lane_id in lane_connector_ids:
        lc = map_api.get_map_object(str(lane_id), SemanticMapLayer.LANE_CONNECTOR)
        if lc is None:
            continue
        poly = getattr(lc, "polygon", None)
        if poly is not None:
            out[str(lane_id)] = poly
    return out


def _estimate_stopline_s_local(
    origin: StateSE2,
    stopline_polygons: Sequence[Polygon],
    local_path_xy: Optional[np.ndarray] = None,
    default_s: float = 10.0,
) -> float:
    if not stopline_polygons:
        return float(default_s)

    # Prefer projecting stop lines to the current ego reference path.
    # This avoids choosing unrelated route connectors far ahead.
    if local_path_xy is not None and len(local_path_xy) >= 2:
        path_xy = np.asarray(local_path_xy, dtype=np.float64)
        s_path = _cumulative_s(path_xy)
        global_pts = [_global_from_local((float(x), float(y)), origin) for x, y in path_xy]
        point_objs = [Point(float(gx), float(gy)) for gx, gy in global_pts]

        projected_s: List[float] = []
        for poly in stopline_polygons:
            dists = np.array([float(poly.distance(pt)) for pt in point_objs], dtype=np.float64)
            if dists.size == 0:
                continue
            idx = int(np.argmin(dists))
            # 8m threshold keeps only stop lines plausibly tied to this path.
            if float(dists[idx]) <= 8.0:
                projected_s.append(float(s_path[idx]))

        positive_projected = [s for s in projected_s if s > 0.5]
        if positive_projected:
            return float(np.clip(min(positive_projected), 2.0, 35.0))
        # If stop lines exist but none project near the current path, use a feasible in-horizon prior.
        if s_path.size > 0:
            return float(np.clip(0.55 * float(s_path[-1]), 2.0, 20.0))

    positive: List[float] = []
    fallback: List[float] = []
    for poly in stopline_polygons:
        c = poly.centroid
        lx, _ = _local_from_global((float(c.x), float(c.y)), origin)
        fallback.append(lx)
        if lx > 0.0:
            positive.append(lx)

    if positive:
        return float(np.clip(min(positive), 2.0, 35.0))
    if fallback:
        return float(np.clip(max(fallback), 2.0, 35.0))
    return float(default_s)


def _build_light_status_from_frame_all(frame: Dict[str, Any]) -> Dict[TrafficLightStatusType, List[str]]:
    red_ids: List[str] = []
    green_ids: List[str] = []
    for lane_connector_id, is_red in frame.get("traffic_lights", []):
        lane_id = str(lane_connector_id)
        if bool(is_red):
            red_ids.append(lane_id)
        else:
            green_ids.append(lane_id)

    return {
        TrafficLightStatusType.RED: sorted(set(red_ids)),
        TrafficLightStatusType.GREEN: sorted(set(green_ids)),
    }


def _build_injected_light_schedule(
    log_frames: List[Dict[str, Any]],
    token_frame_idx: int,
    selected_lane_ids: Sequence[str],
    states_len: int,
    dt: float,
    phase_shift_s: float,
    red_start_idx: int,
) -> Dict[int, Dict[TrafficLightStatusType, List[str]]]:
    schedule: Dict[int, Dict[TrafficLightStatusType, List[str]]] = {}
    frame_dt = 0.5  # mini logs are 2Hz
    phase_shift_steps = int(round(phase_shift_s / frame_dt))
    max_idx = len(log_frames) - 1
    selected_lane_ids = [str(x) for x in selected_lane_ids]

    for step in range(states_len):
        src = token_frame_idx + int(round(step * dt / frame_dt)) + phase_shift_steps
        src = int(np.clip(src, 0, max_idx))
        status = _build_light_status_from_frame_all(log_frames[src])

        # Counterfactual injection: force route lights to red after red_start.
        if step >= red_start_idx and selected_lane_ids:
            red_set = set(str(x) for x in status.get(TrafficLightStatusType.RED, []))
            green_set = set(str(x) for x in status.get(TrafficLightStatusType.GREEN, []))
            for lane_id in selected_lane_ids:
                red_set.add(str(lane_id))
                green_set.discard(str(lane_id))
            status = {
                TrafficLightStatusType.RED: sorted(red_set),
                TrafficLightStatusType.GREEN: sorted(green_set),
            }
        schedule[step] = status

    return schedule


def _build_tl_occupancy_maps(
    schedule: Dict[int, Dict[TrafficLightStatusType, List[str]]],
    lane_polygon_map: Dict[str, Polygon],
    states_len: int,
) -> List[Tuple[List[str], np.ndarray]]:
    occupancy_maps_tl: List[Tuple[List[str], np.ndarray]] = []
    for step in range(states_len):
        status = schedule.get(step, {})
        red_lane_ids = [str(x) for x in status.get(TrafficLightStatusType.RED, [])]
        tokens: List[str] = []
        polygons: List[Polygon] = []
        for lane_id in red_lane_ids:
            poly = lane_polygon_map.get(lane_id)
            if poly is None:
                continue
            tokens.append(f"red_light_{lane_id}")
            polygons.append(poly)
        occupancy_maps_tl.append((tokens, np.array(polygons, dtype=np.object_)))
    return occupancy_maps_tl


def _slice_schedule_for_stage2(
    schedule: Dict[int, Dict[TrafficLightStatusType, List[str]]],
    start_idx: int,
    target_len: int,
) -> Dict[int, Dict[TrafficLightStatusType, List[str]]]:
    out: Dict[int, Dict[TrafficLightStatusType, List[str]]] = {}
    last_status = schedule[max(schedule.keys())] if schedule else {
        TrafficLightStatusType.RED: [],
        TrafficLightStatusType.GREEN: [],
    }
    for k in range(target_len):
        src = start_idx + k
        out[k] = schedule.get(src, last_status)
    return out


def _status_to_entries(status: Dict[TrafficLightStatusType, List[str]]) -> List[List[Any]]:
    red_ids = [str(x) for x in status.get(TrafficLightStatusType.RED, [])]
    green_ids = [str(x) for x in status.get(TrafficLightStatusType.GREEN, [])]
    red_set = set(red_ids)
    entries: List[List[Any]] = []
    for lane_id in sorted(red_set):
        entries.append([lane_id, True])
    for lane_id in sorted(set(green_ids) - red_set):
        entries.append([lane_id, False])
    return entries


def _attach_frame_traffic_lights(
    trace: Dict[str, Any],
    schedule: Dict[int, Dict[TrafficLightStatusType, List[str]]],
) -> Dict[str, Any]:
    frames = trace.get("frames", [])
    if not isinstance(frames, list) or not frames:
        return trace
    if schedule:
        max_k = max(schedule.keys())
        fallback = schedule[max_k]
    else:
        fallback = {TrafficLightStatusType.RED: [], TrafficLightStatusType.GREEN: []}
    for i, frame in enumerate(frames):
        status = schedule.get(i, fallback)
        frame["traffic_lights"] = _status_to_entries(status)
    return trace


# -------------------------
# Candidate generation
# -------------------------


def _fallback_base_poses(num_poses: int, dt: float, speed_mps: float = 6.0) -> np.ndarray:
    xs = np.array([(i + 1) * dt * speed_mps for i in range(num_poses)], dtype=np.float32)
    ys = np.zeros_like(xs)
    hs = np.zeros_like(xs)
    return np.stack([xs, ys, hs], axis=1)


def _build_base_poses(metric_cache: Any, proposal_sampling: TrajectorySampling) -> np.ndarray:
    n = proposal_sampling.num_poses
    if metric_cache.human_trajectory is None:
        speed = float(metric_cache.ego_state.dynamic_car_state.speed)
        return _fallback_base_poses(n, proposal_sampling.interval_length, max(2.0, speed))

    raw = np.asarray(metric_cache.human_trajectory.poses, dtype=np.float32)
    base_poses = _resample_local_poses(raw, n)

    disp = 0.0
    if len(base_poses) > 0:
        disp = float(np.hypot(base_poses[-1, 0], base_poses[-1, 1]))
    if disp < 2.0:
        speed = float(metric_cache.ego_state.dynamic_car_state.speed)
        return _fallback_base_poses(n, proposal_sampling.interval_length, max(2.0, speed))

    return base_poses


def _sample_pose_on_path(
    s_query: float,
    s_path: np.ndarray,
    xy_path: np.ndarray,
    heading_path_unwrap: np.ndarray,
) -> Tuple[float, float, float]:
    if len(s_path) == 0:
        return float(s_query), 0.0, 0.0

    s_max = float(s_path[-1])
    if s_query <= s_max:
        x = float(np.interp(s_query, s_path, xy_path[:, 0]))
        y = float(np.interp(s_query, s_path, xy_path[:, 1]))
        h = float(np.interp(s_query, s_path, heading_path_unwrap))
        h = float(math.atan2(math.sin(h), math.cos(h)))
        return x, y, h

    # Extrapolate linearly along last heading.
    x_last = float(xy_path[-1, 0])
    y_last = float(xy_path[-1, 1])
    h_last = float(heading_path_unwrap[-1])
    ds = s_query - s_max
    x = x_last + math.cos(h_last) * ds
    y = y_last + math.sin(h_last) * ds
    h = float(math.atan2(math.sin(h_last), math.cos(h_last)))
    return x, y, h


def _generate_candidate_from_boundary(
    base_poses: np.ndarray,
    dt: float,
    stopline_s: float,
    template: str,
    t_boundary: float,
    reaction_delay: float,
    brake_delay: float,
    a_brake_peak: float,
    intrusion_target: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    n = len(base_poses)
    if n <= 2:
        return {"ok": False, "reason": "invalid_base_poses"}

    xy_path = np.vstack([np.zeros((1, 2), dtype=np.float32), base_poses[:, :2]])
    heading_path = np.concatenate([[0.0], base_poses[:, 2].astype(np.float64)])
    heading_path_unwrap = np.unwrap(heading_path)
    s_path = _cumulative_s(xy_path)

    boundary_idx = int(np.clip(int(round(t_boundary / dt)), 1, n - 1))
    delay_steps = int(round((reaction_delay + brake_delay) / dt))
    brake_start_idx = int(np.clip(boundary_idx - delay_steps, 1, boundary_idx))

    s_nominal = np.array([float(s_path[min(i + 1, len(s_path) - 1)]) for i in range(n)], dtype=np.float64)
    boundary_s_target = float(stopline_s + intrusion_target)

    s_target = np.copy(s_nominal)

    # Before braking, keep nominal. Between brake start and boundary, force boundary state.
    s_start = float(s_target[brake_start_idx - 1]) if brake_start_idx > 0 else 0.0
    denom = max(1, boundary_idx - brake_start_idx + 1)
    for i in range(brake_start_idx, boundary_idx + 1):
        r = float(i - brake_start_idx + 1) / float(denom)
        s_target[i] = s_start + r * (boundary_s_target - s_start)

    # Recovery segment
    stop_extra = float(rng.uniform(0.5, 2.0)) if template == "late_brake_stopline" else float(rng.uniform(0.2, 1.2))
    if template == "commit_go":
        for i in range(boundary_idx + 1, n):
            delta_nominal = max(0.0, s_nominal[i] - s_nominal[boundary_idx])
            s_target[i] = boundary_s_target + 1.0 * delta_nominal
        recovery_type = "go"
    else:
        s_final = boundary_s_target + stop_extra
        k = float(np.clip(a_brake_peak / 3.0, 0.25, 1.5))
        for i in range(boundary_idx + 1, n):
            tau = (i - boundary_idx) * dt
            s_target[i] = boundary_s_target + (s_final - boundary_s_target) * (1.0 - math.exp(-k * tau))
        recovery_type = "stop"

    # Monotonic / simple accel limit shaping in s-domain.
    for i in range(1, n):
        s_target[i] = max(s_target[i], s_target[i - 1] + 0.01)

    v = np.diff(np.concatenate([[0.0], s_target])) / dt
    max_dv = a_brake_peak * dt
    for i in range(1, len(v)):
        if v[i - 1] - v[i] > max_dv:
            v[i] = max(v[i - 1] - max_dv, 0.0)
    # Keep the same length as pose horizon; dropping index 0 would shrink to n-1.
    s_target = np.cumsum(v * dt)

    candidate = np.zeros_like(base_poses, dtype=np.float32)
    for i in range(n):
        x, y, h = _sample_pose_on_path(float(s_target[i]), s_path, xy_path, heading_path_unwrap)
        candidate[i, 0] = x
        candidate[i, 1] = y
        candidate[i, 2] = h

    v_boundary = float(v[min(boundary_idx, len(v) - 1)])
    stop_distance = float(max(0.5, stop_extra))
    required_decel = float((v_boundary * v_boundary) / (2.0 * stop_distance))
    forced_commit = False
    if recovery_type == "stop" and required_decel > 4.0:
        # Stoppability rule: switch to commit_go.
        forced_commit = True
        recovery_type = "go"

    return {
        "ok": True,
        "candidate_rel": candidate,
        "boundary_idx": int(boundary_idx),
        "brake_start_idx": int(brake_start_idx),
        "boundary_s_target": float(boundary_s_target),
        "intrusion_target_m": float(intrusion_target),
        "recovery_type": recovery_type,
        "required_decel_mps2": float(required_decel),
        "forced_commit": bool(forced_commit),
    }


# -------------------------
# Metrics / gates / annotation
# -------------------------


def _compute_dynamics_metrics(states: np.ndarray, interval_s: float, split_idx: int, join_window_k: int) -> Dict[str, float]:
    n = len(states)
    if n <= 1:
        return {
            "max_abs_accel_mps2": float("nan"),
            "max_abs_steer_deg": float("nan"),
            "max_abs_yaw_rate_deg_s": float("nan"),
            "max_abs_jerk_mps3": float("nan"),
            "max_abs_steer_rate_deg_s": float("nan"),
            "window_max_abs_accel_mps2": float("nan"),
            "window_max_abs_steer_deg": float("nan"),
            "window_max_abs_yaw_rate_deg_s": float("nan"),
            "window_max_abs_jerk_mps3": float("nan"),
            "window_max_abs_steer_rate_deg_s": float("nan"),
        }

    accel = states[:, StateIndex.ACCELERATION_X]
    steer_rad = states[:, StateIndex.STEERING_ANGLE]
    heading = states[:, StateIndex.HEADING]

    steer_deg = np.degrees(steer_rad)
    yaw_rate_deg_s = np.degrees(np.array([base.normalize_angle(float(dh)) for dh in np.diff(heading)])) / interval_s
    jerk_mps3 = np.diff(accel) / interval_s
    steer_rate_deg_s = np.diff(steer_deg) / interval_s

    transition_start = max(0, split_idx - max(0, join_window_k))
    transition_end = min(n - 2, split_idx + max(0, join_window_k))

    state_window_start = max(0, split_idx - max(0, join_window_k))
    state_window_end = min(n - 1, split_idx + max(0, join_window_k))

    def _safe_max_abs(arr: np.ndarray) -> float:
        if arr.size == 0:
            return float("nan")
        return float(np.max(np.abs(arr)))

    accel_w = accel[state_window_start : state_window_end + 1]
    steer_w = steer_deg[state_window_start : state_window_end + 1]
    yaw_w = yaw_rate_deg_s[transition_start : transition_end + 1]
    jerk_w = jerk_mps3[max(0, transition_start - 1) : min(len(jerk_mps3) - 1, transition_end) + 1]
    steer_rate_w = steer_rate_deg_s[max(0, transition_start - 1) : min(len(steer_rate_deg_s) - 1, transition_end) + 1]

    return {
        "max_abs_accel_mps2": _safe_max_abs(accel),
        "max_abs_steer_deg": _safe_max_abs(steer_deg),
        "max_abs_yaw_rate_deg_s": _safe_max_abs(yaw_rate_deg_s),
        "max_abs_jerk_mps3": _safe_max_abs(jerk_mps3),
        "max_abs_steer_rate_deg_s": _safe_max_abs(steer_rate_deg_s),
        "window_max_abs_accel_mps2": _safe_max_abs(accel_w),
        "window_max_abs_steer_deg": _safe_max_abs(steer_w),
        "window_max_abs_yaw_rate_deg_s": _safe_max_abs(yaw_w),
        "window_max_abs_jerk_mps3": _safe_max_abs(jerk_w),
        "window_max_abs_steer_rate_deg_s": _safe_max_abs(steer_rate_w),
    }


def _pass_dynamics_gate(metrics: Dict[str, float], dyn: DynamicsCriteria) -> bool:
    checks = [
        ("max_abs_accel_mps2", dyn.max_abs_accel_mps2),
        ("max_abs_steer_deg", dyn.max_abs_steer_deg),
        ("max_abs_yaw_rate_deg_s", dyn.max_abs_yaw_rate_deg_s),
        ("max_abs_jerk_mps3", dyn.max_abs_jerk_mps3),
        ("max_abs_steer_rate_deg_s", dyn.max_abs_steer_rate_deg_s),
        ("window_max_abs_accel_mps2", dyn.max_abs_accel_mps2),
        ("window_max_abs_steer_deg", dyn.max_abs_steer_deg),
        ("window_max_abs_yaw_rate_deg_s", dyn.max_abs_yaw_rate_deg_s),
        ("window_max_abs_jerk_mps3", dyn.max_abs_jerk_mps3),
        ("window_max_abs_steer_rate_deg_s", dyn.max_abs_steer_rate_deg_s),
    ]
    for key, limit in checks:
        v = float(metrics.get(key, float("nan")))
        if math.isnan(v) or v > float(limit):
            return False
    return True


def _pass_gate_without_tlc(score: Dict[str, float], thresholds: base.Thresholds) -> bool:
    eps = 1e-6
    return (
        float(score["no_at_fault_collisions"]) >= 1.0 - eps
        and float(score["drivable_area_compliance"]) >= 1.0 - eps
        and float(score["driving_direction_compliance"]) >= 1.0 - eps
        and float(score["ego_progress"]) >= float(thresholds.min_progress)
    )


def _pass_stage1_gate(score: Dict[str, float], thresholds: base.Thresholds) -> bool:
    """
    Stage-A is intended to create difficult boundary states.
    We keep route/drivable/progress constraints and allow collision risk here,
    then enforce strict safety after rescue.
    """
    eps = 1e-6
    return (
        float(score["drivable_area_compliance"]) >= 1.0 - eps
        and float(score["driving_direction_compliance"]) >= 1.0 - eps
        and float(score["ego_progress"]) >= float(thresholds.min_progress)
    )


def _pass_boundary_geometry(
    base_poses: np.ndarray,
    candidate_rel: np.ndarray,
    boundary_idx: int,
    max_abs_lon_m: float = 20.0,
    max_abs_lat_m: float = 3.0,
    max_abs_heading_deg: float = 25.0,
) -> bool:
    if len(base_poses) == 0 or len(candidate_rel) == 0:
        return False
    idx = int(np.clip(boundary_idx, 0, min(len(base_poses), len(candidate_rel)) - 1))
    dlon = float(candidate_rel[idx, 0] - base_poses[idx, 0])
    dlat = float(candidate_rel[idx, 1] - base_poses[idx, 1])
    dh = abs(math.degrees(base.normalize_angle(float(candidate_rel[idx, 2] - base_poses[idx, 2]))))
    return abs(dlon) <= max_abs_lon_m and abs(dlat) <= max_abs_lat_m and dh <= max_abs_heading_deg


def _vehicle_min_dist(states: np.ndarray, tracks: List[Any]) -> float:
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


def _stage1_risk_score(min_dist_stage1: float) -> float:
    if math.isnan(min_dist_stage1):
        return 0.0
    center = 2.0
    width = 1.0
    return max(0.0, 1.0 - abs(min_dist_stage1 - center) / width)


def _strong_longtail_score(
    stage1_min_dist: float,
    stage2_min_dist: float,
    stage2_disp: float,
    stage2_mean_speed: float,
    join_translation: float,
    join_heading_deg: float,
    join_speed_delta: float,
    stage2_pdm: float,
) -> float:
    risk_score = _stage1_risk_score(stage1_min_dist)
    recovery_score = 0.6 * min(1.0, stage2_disp / 8.0) + 0.4 * min(1.0, stage2_mean_speed / 2.0)
    safety_gain = max(0.0, min(1.0, (stage2_min_dist - stage1_min_dist + 1.0) / 2.0))
    continuity = max(
        0.0,
        1.0
        - (
            0.45 * min(1.0, join_translation / 1.2)
            + 0.35 * min(1.0, join_heading_deg / 12.0)
            + 0.20 * min(1.0, join_speed_delta / 2.2)
        ),
    )
    return float((0.45 * risk_score + 0.30 * recovery_score + 0.15 * safety_gain + 0.10 * continuity) * stage2_pdm)


def _estimate_event_timestamps(
    states: np.ndarray,
    schedule: Dict[int, Dict[TrafficLightStatusType, List[str]]],
    stopline_s: float,
    boundary_idx: int,
    map_api: Any,
    recovery_type: str,
    dt: float,
) -> Dict[str, Optional[float]]:
    init = StateSE2(float(states[0, StateIndex.X]), float(states[0, StateIndex.Y]), float(states[0, StateIndex.HEADING]))

    local_x: List[float] = []
    speeds: List[float] = []
    in_crosswalk: List[bool] = []
    in_intersection: List[bool] = []
    for i in range(len(states)):
        x = float(states[i, StateIndex.X])
        y = float(states[i, StateIndex.Y])
        h = float(states[i, StateIndex.HEADING])
        lx, _ = _local_from_global((x, y), init)
        local_x.append(float(lx))
        speeds.append(float(math.hypot(states[i, StateIndex.VELOCITY_X], states[i, StateIndex.VELOCITY_Y])))
        pt = Point2D(x, y)
        in_crosswalk.append(bool(map_api.is_in_layer(pt, SemanticMapLayer.CROSSWALK)))
        in_intersection.append(bool(map_api.is_in_layer(pt, SemanticMapLayer.INTERSECTION)))

    def _first_true(mask: Sequence[bool]) -> Optional[int]:
        for idx, v in enumerate(mask):
            if bool(v):
                return idx
        return None

    idx_cross = _first_true([x >= stopline_s for x in local_x])
    idx_crosswalk = _first_true(in_crosswalk)
    idx_conflict = _first_true(in_intersection)

    red_start_idx = None
    for i in range(len(states)):
        if schedule.get(i, {}).get(TrafficLightStatusType.RED):
            red_start_idx = i
            break

    if recovery_type == "stop":
        idx_rec_end = None
        for i in range(boundary_idx, len(states)):
            if speeds[i] <= 0.3:
                idx_rec_end = i
                break
    else:
        idx_rec_end = None
        if any(in_intersection):
            entered = False
            for i in range(boundary_idx, len(states)):
                if in_intersection[i]:
                    entered = True
                if entered and (not in_intersection[i]):
                    idx_rec_end = i
                    break
        if idx_rec_end is None:
            idx_rec_end = len(states) - 1

    return {
        "t_red": None if red_start_idx is None else float(red_start_idx * dt),
        "t_cross_stopline": None if idx_cross is None else float(idx_cross * dt),
        "t_enter_crosswalk": None if idx_crosswalk is None else float(idx_crosswalk * dt),
        "t_enter_conflict_zone": None if idx_conflict is None else float(idx_conflict * dt),
        "t_recovery_start": float(boundary_idx * dt),
        "t_recovery_end": None if idx_rec_end is None else float(idx_rec_end * dt),
    }


def _compute_traffic_light_metrics(
    states: np.ndarray,
    schedule: Dict[int, Dict[TrafficLightStatusType, List[str]]],
    stopline_s: float,
    boundary_idx: int,
    template: str,
    recovery_type: str,
    boundary_state: Dict[str, float],
    map_api: Any,
    dt: float,
) -> Dict[str, Any]:
    init = StateSE2(float(states[0, StateIndex.X]), float(states[0, StateIndex.Y]), float(states[0, StateIndex.HEADING]))

    local_x: List[float] = []
    speeds: List[float] = []
    in_intersection: List[bool] = []
    red_mask: List[bool] = []
    for i in range(len(states)):
        x = float(states[i, StateIndex.X])
        y = float(states[i, StateIndex.Y])
        lx, _ = _local_from_global((x, y), init)
        local_x.append(lx)
        speeds.append(float(math.hypot(states[i, StateIndex.VELOCITY_X], states[i, StateIndex.VELOCITY_Y])))
        in_intersection.append(bool(map_api.is_in_layer(Point2D(x, y), SemanticMapLayer.INTERSECTION)))
        red_mask.append(bool(schedule.get(i, {}).get(TrafficLightStatusType.RED)))

    intrusion_series = [float(x - stopline_s) for x in local_x]
    max_intrusion = float(np.max(intrusion_series))
    boundary_intrusion = float(intrusion_series[min(boundary_idx, len(intrusion_series) - 1)])

    violation_detected = any((intrusion_series[i] > 0.1 and red_mask[i]) for i in range(len(intrusion_series)))

    # Commit-go safety: no prolonged low speed while inside intersection.
    stalled_in_intersection = False
    if recovery_type == "go":
        stall_run = 0
        for i in range(max(0, boundary_idx), len(states)):
            # Only count stalls after entering beyond stop line meaningfully.
            if in_intersection[i] and intrusion_series[i] > 0.5 and speeds[i] < 0.3:
                stall_run += 1
            else:
                stall_run = 0
            if stall_run >= 5:
                stalled_in_intersection = True
                break

    # Template consistency checks (based on realized intrusion).
    template_match = True
    if template == "late_brake_stopline":
        template_match = (0.3 <= max_intrusion <= 1.5) and (recovery_type == "stop")
    elif template == "crosswalk_intrude_stop":
        template_match = (1.5 <= max_intrusion <= 4.0) and (recovery_type == "stop")
    elif template == "commit_go":
        template_match = (max_intrusion > 4.0) and (recovery_type == "go")

    timestamps = _estimate_event_timestamps(
        states=states,
        schedule=schedule,
        stopline_s=stopline_s,
        boundary_idx=boundary_idx,
        map_api=map_api,
        recovery_type=recovery_type,
        dt=dt,
    )

    return {
        "violation_detected": bool(violation_detected),
        "template_match": bool(template_match),
        "stalled_in_intersection": bool(stalled_in_intersection),
        "max_intrusion_m": float(max_intrusion),
        "boundary_intrusion_m": float(boundary_intrusion),
        "red_steps": int(sum(red_mask)),
        "boundary_state": boundary_state,
        "event_timestamps": timestamps,
    }


def _annotate_trace(
    trace: Dict[str, Any],
    source_type: str,
    event_template: str,
    recovery_type: str,
    result_status: str,
    reject_reason: Optional[str],
    boundary_state: Dict[str, float],
    longtail_metrics: Dict[str, Any],
    traffic_light_metrics: Dict[str, Any],
    dynamics_metrics: Dict[str, Any],
    criteria_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    trace["source_type"] = source_type
    trace["event_template"] = event_template
    trace["recovery_type"] = recovery_type
    trace["result_status"] = result_status
    trace["reject_reason"] = reject_reason
    trace["search_stage"] = "redlight_v1"
    trace["boundary_state"] = boundary_state
    trace["longtail_metrics"] = longtail_metrics
    trace["traffic_light_metrics"] = traffic_light_metrics
    trace["dynamics_metrics"] = dynamics_metrics
    trace["criteria_snapshot"] = criteria_snapshot
    return trace


def _save_trace(trace: Dict[str, Any], out_dir: Path, fmt: str, rank: int, stem_suffix: str) -> Tuple[Optional[Path], Optional[Path]]:
    token = trace["token"]
    stem = f"redlight_scene_{token}_{stem_suffix}_r{rank:02d}"
    out_json: Optional[Path] = None
    out_pkl: Optional[Path] = None
    if fmt in ("json", "both"):
        out_json = out_dir / f"{stem}.json"
        out_json.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    if fmt in ("pkl", "both"):
        out_pkl = out_dir / f"{stem}.pkl"
        with out_pkl.open("wb") as f:
            pickle.dump(trace, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out_json, out_pkl


def _push_top(pool: List[Dict[str, Any]], item: Dict[str, Any], score_key: str, max_size: int) -> None:
    pool.append(item)
    pool.sort(key=lambda x: float(x.get(score_key, 0.0)), reverse=True)
    if len(pool) > max_size:
        pool.pop()


# -------------------------
# Core token runner
# -------------------------


def _run_seed_token(
    seed_entry: SeedEntry,
    metric_cache: Any,
    map_api: Any,
    log_frames: List[Dict[str, Any]],
    token_frame_idx: int,
    proposal_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    reactive_policy: Any,
    thresholds: base.Thresholds,
    dyn_cfg: DynamicsCriteria,
    event_mix: EventMixConfig,
    run_cfg: RunConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    token = seed_entry.scene_token

    # Context from current log frame
    frame = log_frames[token_frame_idx]
    frame_tl_ids = [str(x[0]) for x in frame.get("traffic_lights", [])]

    route_lc_ids = _route_lane_connector_ids(
        metric_cache,
        map_api,
        frame_tl_ids,
        ego_xy=(float(metric_cache.ego_state.rear_axle.x), float(metric_cache.ego_state.rear_axle.y)),
    )
    lane_polygon_map = _build_route_lane_connector_polygon_map(map_api, route_lc_ids)
    stopline_polygons = _extract_stopline_polygons(map_api, route_lc_ids)

    # If no route lane connectors, scene is unusable for red-light simulation.
    if not route_lc_ids:
        return {
            "token": token,
            "success_traces": [],
            "failure_traces": [],
            "stats": {
                "attempts": 0,
                "successes": 0,
                "failures_saved": 0,
                "reject_reasons": {"missing_route_lane_connectors": 1},
            },
            "rejected_reason": "missing_route_lane_connectors",
        }

    base_poses = _build_base_poses(metric_cache, proposal_sampling)
    if base_poses.shape[0] != proposal_sampling.num_poses:
        return {
            "token": token,
            "success_traces": [],
            "failure_traces": [],
            "stats": {
                "attempts": 0,
                "successes": 0,
                "failures_saved": 0,
                "reject_reasons": {"invalid_base_trajectory": 1},
            },
            "rejected_reason": "invalid_base_trajectory",
        }
    base_path_xy = np.vstack([np.zeros((1, 2), dtype=np.float32), base_poses[:, :2]])
    stopline_s = _estimate_stopline_s_local(
        metric_cache.ego_state.rear_axle,
        stopline_polygons,
        local_path_xy=base_path_xy,
    )
    base_s_end = float(_cumulative_s(base_path_xy)[-1]) if len(base_path_xy) >= 2 else 0.0
    # Keep the stop line within the horizon neighborhood; otherwise Stage-A tends to be unrealistically far.
    if base_s_end > 5.0 and stopline_s > base_s_end - 0.5:
        stopline_s = float(max(2.0, min(stopline_s, 0.75 * base_s_end)))
    # Seed must start before stop line; otherwise it's no longer a meaningful red-light boundary scene.
    if stopline_s < 3.0:
        return {
            "token": token,
            "success_traces": [],
            "failure_traces": [],
            "stats": {
                "attempts": 0,
                "successes": 0,
                "failures_saved": 0,
                "reject_reasons": {"stopline_not_ahead": 1},
            },
            "rejected_reason": "stopline_not_ahead",
        }

    scorer = PDMScorer(proposal_sampling=proposal_sampling, config=PDMScorerConfig(human_penalty_filter=False))
    planner_init = base.build_planner_initialization(metric_cache)
    expert_planner = base.build_expert_planner(proposal_sampling=proposal_sampling)

    criteria_snapshot = {
        "base_thresholds": asdict(thresholds),
        "dynamics": asdict(dyn_cfg),
        "event_mix": {
            "template_weights": asdict(event_mix.template_weights),
            "source_weights": asdict(event_mix.source_weights),
            "params": asdict(event_mix.params),
            "intrusion_ranges": {
                "late_brake_stopline": list(event_mix.intrusion_ranges.late_brake_stopline),
                "crosswalk_intrude_stop": list(event_mix.intrusion_ranges.crosswalk_intrude_stop),
                "commit_go": list(event_mix.intrusion_ranges.commit_go),
            },
        },
    }

    success_pool: List[Dict[str, Any]] = []
    failure_pool: List[Dict[str, Any]] = []
    reject_reasons: Dict[str, int] = {}

    def _inc(reason: str) -> None:
        reject_reasons[reason] = int(reject_reasons.get(reason, 0)) + 1

    def _record_failure(
        reason: str,
        ego_states: np.ndarray,
        tracks: List[Any],
        stage1_score: Dict[str, float],
        stage2_score: Dict[str, float],
        event_template: str,
        recovery_type: str,
        boundary_state: Dict[str, Any],
        light_schedule: Optional[Dict[int, Dict[TrafficLightStatusType, List[str]]]] = None,
        traffic_light_metrics: Optional[Dict[str, Any]] = None,
        dynamics_metrics: Optional[Dict[str, Any]] = None,
        longtail_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        if len(failure_pool) >= run_cfg.max_failure_traces_per_seed:
            return
        try:
            tr = base.serialize_trace(
                token=token,
                vocab_index=-1,
                ego_states=ego_states,
                tracks=tracks,
                stage1_score=stage1_score,
                stage2_score=stage2_score,
            )
        except Exception:
            return
        tr = _annotate_trace(
            trace=tr,
            source_type=seed_entry.source_type,
            event_template=event_template,
            recovery_type=recovery_type,
            result_status="rejected",
            reject_reason=reason,
            boundary_state=boundary_state,
            longtail_metrics=longtail_metrics or {},
            traffic_light_metrics=traffic_light_metrics or {},
            dynamics_metrics=dynamics_metrics or {},
            criteria_snapshot=criteria_snapshot,
        )
        if light_schedule is not None:
            tr = _attach_frame_traffic_lights(tr, light_schedule)
        tr["_rank_score"] = float(stage2_score.get("pdm_score", stage1_score.get("pdm_score", 0.0)))
        _push_top(failure_pool, tr, "_rank_score", run_cfg.max_failure_traces_per_seed)

    for _ in range(run_cfg.max_candidate_trials):
        template = _weighted_template_choice(event_mix, rng, seed_entry.source_type)

        intrusion_range = getattr(event_mix.intrusion_ranges, template)
        intrusion_target = float(rng.uniform(float(intrusion_range[0]), float(intrusion_range[1])))
        t_boundary = float(rng.uniform(event_mix.params.t_boundary_min_s, event_mix.params.t_boundary_max_s))
        reaction_delay = float(rng.uniform(event_mix.params.reaction_delay_min_s, event_mix.params.reaction_delay_max_s))
        brake_delay = float(rng.uniform(event_mix.params.brake_delay_min_s, event_mix.params.brake_delay_max_s))
        a_brake_peak = float(rng.uniform(event_mix.params.a_brake_peak_min_mps2, event_mix.params.a_brake_peak_max_mps2))
        phase_shift = float(rng.uniform(event_mix.params.phase_shift_min_s, event_mix.params.phase_shift_max_s))

        candidate_info = _generate_candidate_from_boundary(
            base_poses=base_poses,
            dt=proposal_sampling.interval_length,
            stopline_s=stopline_s,
            template=template,
            t_boundary=t_boundary,
            reaction_delay=reaction_delay,
            brake_delay=brake_delay,
            a_brake_peak=a_brake_peak,
            intrusion_target=intrusion_target,
            rng=rng,
        )
        if not candidate_info.get("ok", False):
            _inc(str(candidate_info.get("reason", "candidate_generation_fail")))
            continue

        candidate_rel = candidate_info["candidate_rel"]
        boundary_idx = int(candidate_info["boundary_idx"])
        recovery_type = str(candidate_info["recovery_type"])
        template_effective = "commit_go" if bool(candidate_info.get("forced_commit", False)) else template

        candidate_states = base.make_trajectory_states_from_local_poses(
            local_poses=candidate_rel,
            metric_cache=metric_cache,
            proposal_sampling=proposal_sampling,
        )
        phase1_states = simulator.simulate_proposals(candidate_states[None, ...], metric_cache.ego_state)[0]

        # Build synthetic TL schedule and inject into cache for both IDM and scorer.
        states_len = int(phase1_states.shape[0])
        red_start_idx = max(1, boundary_idx - int(round(reaction_delay / proposal_sampling.interval_length)))
        schedule = _build_injected_light_schedule(
            log_frames=log_frames,
            token_frame_idx=token_frame_idx,
            selected_lane_ids=route_lc_ids,
            states_len=states_len,
            dt=proposal_sampling.interval_length,
            phase_shift_s=phase_shift,
            red_start_idx=red_start_idx,
        )

        stage1_cache = copy.copy(metric_cache)
        stage1_cache.observation = copy.deepcopy(metric_cache.observation)
        stage1_cache.traffic_light_status = schedule
        stage1_cache.observation._occupancy_maps_tl = _build_tl_occupancy_maps(
            schedule=schedule,
            lane_polygon_map=lane_polygon_map,
            states_len=states_len,
        )

        phase1_tracks = reactive_policy.simulate_environment(phase1_states, stage1_cache)
        stage1_score = base.score_segment(
            scorer=scorer,
            states=phase1_states,
            metric_cache=stage1_cache,
            simulated_tracks=phase1_tracks,
        )

        if not _pass_stage1_gate(stage1_score, thresholds):
            # Stage-A may intentionally violate strict safety; keep as diagnostic only.
            _inc("stage1_fail")

        # Stage-B rescue: switch at split point and keep total horizon to 4s.
        split_idx = int(np.clip(boundary_idx + 2, 5, len(phase1_states) - 6))
        phase1_prefix_states = phase1_states[: split_idx + 1]
        phase1_prefix_tracks = phase1_tracks[: split_idx + 1]
        remaining_steps = int((len(phase1_states) - 1) - split_idx)
        if remaining_steps < 2:
            _inc("invalid_split")
            continue

        ood_time = metric_cache.ego_state.time_point + TimeDuration.from_s(split_idx * proposal_sampling.interval_length)
        ood_ego_state = state_array_to_ego_state(
            phase1_states[split_idx],
            TimePoint(int(ood_time.time_us)),
            metric_cache.ego_state.car_footprint.vehicle_parameters,
        )
        ood_obs = phase1_tracks[split_idx]

        expert_planner.initialize(planner_init)
        planner_input = PlannerInput(
            iteration=SimulationIteration(index=0, time_point=ood_ego_state.time_point),
            history=SimulationHistoryBuffer.initialize_from_list(
                buffer_size=1,
                ego_states=[ood_ego_state],
                observations=[ood_obs],
            ),
            traffic_light_data=[],
        )
        rescue_traj = expert_planner.compute_planner_trajectory(planner_input)
        rescue_states = base.get_trajectory_as_array(
            rescue_traj,
            proposal_sampling,
            start_time=ood_ego_state.time_point,
        )

        stage2_cache = base.build_stage2_metric_cache(
            metric_cache=stage1_cache,
            ego_state_ood=ood_ego_state,
            current_tracks=ood_obs,
            proposal_sampling=proposal_sampling,
        )
        schedule2 = _slice_schedule_for_stage2(
            schedule=schedule,
            start_idx=max(0, split_idx),
            target_len=len(rescue_states),
        )
        stage2_cache.traffic_light_status = schedule2
        stage2_cache.observation = copy.deepcopy(stage2_cache.observation)
        stage2_cache.observation._occupancy_maps_tl = _build_tl_occupancy_maps(
            schedule=schedule2,
            lane_polygon_map=lane_polygon_map,
            states_len=len(rescue_states),
        )

        phase2_tracks = reactive_policy.simulate_environment(rescue_states, stage2_cache)
        stage2_score = base.score_segment(
            scorer=scorer,
            states=rescue_states,
            metric_cache=stage2_cache,
            simulated_tracks=phase2_tracks,
            centerline=expert_planner._centerline,
            route_lane_ids=list(expert_planner._route_lane_dict.keys()),
            drivable_area_map=expert_planner._drivable_area_map,
        )
        if not _pass_gate_without_tlc(stage2_score, thresholds):
            _inc("stage2_fail")
            tmp_phase2_use_states = rescue_states[: remaining_steps + 1]
            tmp_phase2_use_tracks = phase2_tracks[: remaining_steps + 1]
            tmp_full_states = np.concatenate([phase1_prefix_states, tmp_phase2_use_states[1:]], axis=0)
            tmp_full_tracks = phase1_prefix_tracks + tmp_phase2_use_tracks[1:]
            boundary_state = {
                "t_boundary_s": t_boundary,
                "phase_shift_s": phase_shift,
                "reaction_delay_s": reaction_delay,
                "brake_delay_s": brake_delay,
                "a_brake_peak_mps2": a_brake_peak,
                "intrusion_target_m": float(candidate_info["intrusion_target_m"]),
                "boundary_idx": boundary_idx,
                "split_idx": split_idx,
                "required_decel_mps2": float(candidate_info["required_decel_mps2"]),
                "forced_commit": bool(candidate_info["forced_commit"]),
            }
            _record_failure(
                reason="stage2_fail",
                ego_states=tmp_full_states,
                tracks=tmp_full_tracks,
                stage1_score=stage1_score,
                stage2_score=stage2_score,
                event_template=template_effective,
                recovery_type=recovery_type,
                boundary_state=boundary_state,
                light_schedule=schedule,
            )
            continue

        # Stitch to fixed-horizon 4s sequence.
        phase2_use_states = rescue_states[: remaining_steps + 1]
        phase2_use_tracks = phase2_tracks[: remaining_steps + 1]
        full_states = np.concatenate([phase1_prefix_states, phase2_use_states[1:]], axis=0)
        full_tracks = phase1_prefix_tracks + phase2_use_tracks[1:]

        # Traffic-light metrics / template checks
        boundary_state = {
            "t_boundary_s": t_boundary,
            "phase_shift_s": phase_shift,
            "reaction_delay_s": reaction_delay,
            "brake_delay_s": brake_delay,
            "a_brake_peak_mps2": a_brake_peak,
            "intrusion_target_m": float(candidate_info["intrusion_target_m"]),
            "boundary_idx": boundary_idx,
            "split_idx": split_idx,
            "required_decel_mps2": float(candidate_info["required_decel_mps2"]),
            "forced_commit": bool(candidate_info["forced_commit"]),
        }
        tlm = _compute_traffic_light_metrics(
            states=full_states,
            schedule=schedule,
            stopline_s=stopline_s,
            boundary_idx=boundary_idx,
            template=template_effective,
            recovery_type=recovery_type,
            boundary_state=boundary_state,
            map_api=map_api,
            dt=proposal_sampling.interval_length,
        )

        if not bool(tlm["violation_detected"]):
            _inc("no_violation")
            _record_failure(
                reason="no_violation",
                ego_states=full_states,
                tracks=full_tracks,
                stage1_score=stage1_score,
                stage2_score=stage2_score,
                event_template=template_effective,
                recovery_type=recovery_type,
                boundary_state=boundary_state,
                light_schedule=schedule,
                traffic_light_metrics=tlm,
            )
            continue
        if not bool(tlm["template_match"]):
            _inc("template_mismatch")
            _record_failure(
                reason="template_mismatch",
                ego_states=full_states,
                tracks=full_tracks,
                stage1_score=stage1_score,
                stage2_score=stage2_score,
                event_template=template_effective,
                recovery_type=recovery_type,
                boundary_state=boundary_state,
                light_schedule=schedule,
                traffic_light_metrics=tlm,
            )
            continue
        if recovery_type == "go" and bool(tlm["stalled_in_intersection"]):
            _inc("commit_go_stalled_in_intersection")
            _record_failure(
                reason="commit_go_stalled_in_intersection",
                ego_states=full_states,
                tracks=full_tracks,
                stage1_score=stage1_score,
                stage2_score=stage2_score,
                event_template=template_effective,
                recovery_type=recovery_type,
                boundary_state=boundary_state,
                light_schedule=schedule,
                traffic_light_metrics=tlm,
            )
            continue

        dyn_metrics = _compute_dynamics_metrics(
            states=full_states,
            interval_s=proposal_sampling.interval_length,
            split_idx=split_idx,
            join_window_k=dyn_cfg.join_window_k,
        )
        if not _pass_dynamics_gate(dyn_metrics, dyn_cfg):
            _inc("dynamics_fail")
            _record_failure(
                reason="dynamics_fail",
                ego_states=full_states,
                tracks=full_tracks,
                stage1_score=stage1_score,
                stage2_score=stage2_score,
                event_template=template_effective,
                recovery_type=recovery_type,
                boundary_state=boundary_state,
                light_schedule=schedule,
                traffic_light_metrics=tlm,
                dynamics_metrics=dyn_metrics,
            )
            continue

        min_dist_stage1 = _vehicle_min_dist(phase1_prefix_states, phase1_prefix_tracks)
        min_dist_stage2 = _vehicle_min_dist(phase2_use_states, phase2_use_tracks)
        stage2_disp = float(math.hypot(
            phase2_use_states[-1, StateIndex.X] - phase2_use_states[0, StateIndex.X],
            phase2_use_states[-1, StateIndex.Y] - phase2_use_states[0, StateIndex.Y],
        ))
        stage2_mean_speed = float(
            np.mean(np.hypot(phase2_use_states[:, StateIndex.VELOCITY_X], phase2_use_states[:, StateIndex.VELOCITY_Y]))
        )

        prev = phase1_prefix_states[-1]
        nxt = phase2_use_states[1] if len(phase2_use_states) > 1 else phase2_use_states[0]
        join_translation = float(math.hypot(nxt[StateIndex.X] - prev[StateIndex.X], nxt[StateIndex.Y] - prev[StateIndex.Y]))
        join_heading_deg = float(abs(math.degrees(base.normalize_angle(float(nxt[StateIndex.HEADING] - prev[StateIndex.HEADING])))))
        prev_speed = float(math.hypot(prev[StateIndex.VELOCITY_X], prev[StateIndex.VELOCITY_Y]))
        next_speed = float(math.hypot(nxt[StateIndex.VELOCITY_X], nxt[StateIndex.VELOCITY_Y]))
        join_speed_delta = float(abs(next_speed - prev_speed))

        strong_score = _strong_longtail_score(
            stage1_min_dist=float(min_dist_stage1),
            stage2_min_dist=float(min_dist_stage2),
            stage2_disp=stage2_disp,
            stage2_mean_speed=stage2_mean_speed,
            join_translation=join_translation,
            join_heading_deg=join_heading_deg,
            join_speed_delta=join_speed_delta,
            stage2_pdm=float(stage2_score.get("pdm_score", 0.0)),
        )

        longtail_metrics = {
            "split_idx": float(split_idx),
            "stage1_min_dist_m": float(min_dist_stage1),
            "stage2_min_dist_m": float(min_dist_stage2),
            "stage2_displacement_m": float(stage2_disp),
            "stage2_mean_speed_mps": float(stage2_mean_speed),
            "join_translation_m": float(join_translation),
            "join_heading_deg": float(join_heading_deg),
            "join_speed_delta_mps": float(join_speed_delta),
            "risk_score": float(_stage1_risk_score(float(min_dist_stage1))),
            "strong_longtail_score": float(strong_score),
        }

        trace = base.serialize_trace(
            token=token,
            vocab_index=-1,
            ego_states=full_states,
            tracks=full_tracks,
            stage1_score=stage1_score,
            stage2_score=stage2_score,
        )
        trace = _annotate_trace(
            trace=trace,
            source_type=seed_entry.source_type,
            event_template=template_effective,
            recovery_type=recovery_type,
            result_status="success",
            reject_reason=None,
            boundary_state=boundary_state,
            longtail_metrics=longtail_metrics,
            traffic_light_metrics=tlm,
            dynamics_metrics=dyn_metrics,
            criteria_snapshot=criteria_snapshot,
        )
        trace = _attach_frame_traffic_lights(trace, schedule)
        trace["_rank_score"] = float(strong_score)
        _push_top(success_pool, trace, "_rank_score", run_cfg.top_k_per_seed)

        if run_cfg.verbose:
            print(
                f"[{token}] success template={template_effective} score={strong_score:.3f} source={seed_entry.source_type}"
            )

    # Keep top failure examples if we have none success or for diagnostics.
    stats = {
        "attempts": int(run_cfg.max_candidate_trials),
        "successes": int(len(success_pool)),
        "failures_saved": int(len(failure_pool)),
        "reject_reasons": reject_reasons,
        "source_type": seed_entry.source_type,
    }

    rejected_reason = None
    if not success_pool:
        if reject_reasons:
            rejected_reason = max(reject_reasons.items(), key=lambda kv: kv[1])[0]
        else:
            rejected_reason = "no_candidate_passed"

    for tr in success_pool:
        tr.pop("_rank_score", None)
    for tr in failure_pool:
        tr.pop("_rank_score", None)

    return {
        "token": token,
        "source_type": seed_entry.source_type,
        "success_traces": success_pool,
        "failure_traces": failure_pool,
        "stats": stats,
        "rejected_reason": rejected_reason,
    }


# -------------------------
# Main
# -------------------------


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    cache_root = base.resolve_metric_cache_root(args.metric_cache_path)
    out_dir = args.output_dir.resolve()
    success_dir = out_dir / "success"
    failure_root = out_dir / "failure"
    success_dir.mkdir(parents=True, exist_ok=True)
    failure_root.mkdir(parents=True, exist_ok=True)

    event_mix = _load_event_mix_config(args.event_mix_config)
    seeds = _load_seed_manifest(args.seed_manifest)
    if not seeds:
        raise RuntimeError(f"No valid seeds loaded from {args.seed_manifest}")

    seeds = _reweight_seeds_for_max(seeds, args.max_seeds, event_mix, rng)

    proposal_sampling = TrajectorySampling(
        num_poses=int(round(args.horizon_sec / args.interval)),
        interval_length=args.interval,
    )

    thresholds = base.Thresholds(
        max_abs_lon_m=float(args.max_abs_lon_m),
        max_abs_lat_m=float(args.max_abs_lat_m),
        max_abs_heading_deg=float(args.max_abs_heading_deg),
        max_abs_accel_mps2=float(args.max_abs_accel_mps2),
        max_abs_steer_deg=float(args.max_abs_steer_deg),
        min_progress=float(args.min_progress),
        max_join_translation_m=float(args.max_join_translation_m),
        max_join_heading_deg=float(args.max_join_heading_deg),
        max_join_speed_delta_mps=float(args.max_join_speed_delta_mps),
    )

    dyn_cfg = DynamicsCriteria(
        max_abs_accel_mps2=float(args.dyn_max_abs_accel_mps2),
        max_abs_steer_deg=float(args.dyn_max_abs_steer_deg),
        max_abs_yaw_rate_deg_s=float(args.dyn_max_abs_yaw_rate_deg_s),
        max_abs_jerk_mps3=float(args.dyn_max_abs_jerk_mps3),
        max_abs_steer_rate_deg_s=float(args.dyn_max_abs_steer_rate_deg_s),
        join_window_k=int(args.dyn_join_window_k),
    )

    run_cfg = RunConfig(
        sample_batch_size=int(args.sample_batch_size),
        max_candidate_trials=int(args.max_candidate_trials),
        top_k_per_seed=int(args.top_k_per_seed),
        max_failure_traces_per_seed=int(args.max_failure_traces_per_seed),
        verbose=bool(args.verbose),
    )

    loader = base.build_metric_cache_loader(cache_root)
    simulator = PDMSimulator(proposal_sampling=proposal_sampling)
    reactive_policy = base.build_reactive_policy(
        proposal_sampling=proposal_sampling,
        map_root_override=args.map_root_override,
    )

    print(f"Processing seeds={len(seeds)} from manifest={args.seed_manifest}")
    print(f"Metric cache root: {cache_root}")

    # Cache logs to avoid repeated IO.
    log_cache: Dict[str, List[Dict[str, Any]]] = {}
    token_index_cache: Dict[str, Dict[str, int]] = {}

    results: List[Dict[str, Any]] = []
    for idx, seed in enumerate(seeds):
        token = seed.scene_token
        if token not in loader.metric_cache_paths:
            results.append(
                {
                    "token": token,
                    "source_type": seed.source_type,
                    "success_traces": [],
                    "failure_traces": [],
                    "stats": {"attempts": 0, "successes": 0, "failures_saved": 0, "reject_reasons": {"missing_metric_cache": 1}},
                    "rejected_reason": "missing_metric_cache",
                }
            )
            continue

        metric_cache = loader.get_from_token(token)
        if args.map_root_override is not None:
            metric_cache.map_parameters.map_root = args.map_root_override

        map_api = base.get_maps_api(
            metric_cache.map_parameters.map_root,
            metric_cache.map_parameters.map_version,
            metric_cache.map_parameters.map_name,
        )

        if seed.log_name not in log_cache:
            log_cache[seed.log_name] = _load_log_frames(args.mini_log_root, seed.log_name)
            token_index_cache[seed.log_name] = _token_to_frame_index(log_cache[seed.log_name])

        frame_idx = token_index_cache[seed.log_name].get(seed.scene_token)
        if frame_idx is None:
            results.append(
                {
                    "token": token,
                    "source_type": seed.source_type,
                    "success_traces": [],
                    "failure_traces": [],
                    "stats": {"attempts": 0, "successes": 0, "failures_saved": 0, "reject_reasons": {"token_not_in_log": 1}},
                    "rejected_reason": "token_not_in_log",
                }
            )
            continue

        per_seed_rng = np.random.default_rng((args.seed + idx * 9973) % (2**32 - 1))
        result = _run_seed_token(
            seed_entry=seed,
            metric_cache=metric_cache,
            map_api=map_api,
            log_frames=log_cache[seed.log_name],
            token_frame_idx=frame_idx,
            proposal_sampling=proposal_sampling,
            simulator=simulator,
            reactive_policy=reactive_policy,
            thresholds=thresholds,
            dyn_cfg=dyn_cfg,
            event_mix=event_mix,
            run_cfg=run_cfg,
            rng=per_seed_rng,
        )
        results.append(result)

    # Save outputs + manifest
    manifest_path = out_dir / args.manifest_name
    stats_path = out_dir / "seed_stats.json"

    manifest_lines: List[str] = []
    seed_stats: Dict[str, Any] = {}
    saved_success = 0
    saved_failure = 0
    rejected_tokens = 0

    template_success_counter = {"late_brake_stopline": 0, "crosswalk_intrude_stop": 0, "commit_go": 0}
    source_success_counter = {"waiting_red": 0, "passing_green": 0}
    source_consumed_counter = {"waiting_red": 0, "passing_green": 0}

    for result in results:
        token = str(result["token"])
        source_type = str(result.get("source_type", "unknown"))
        if source_type in source_consumed_counter:
            source_consumed_counter[source_type] += 1

        success_traces = result.get("success_traces", [])
        failure_traces = result.get("failure_traces", [])
        seed_stats[token] = result.get("stats", {})

        for rank, tr in enumerate(success_traces):
            out_json, out_pkl = _save_trace(
                trace=tr,
                out_dir=success_dir,
                fmt=args.save_format,
                rank=rank,
                stem_suffix="redlight_success",
            )
            saved_success += 1
            src = str(tr.get("source_type", ""))
            tpl = str(tr.get("event_template", ""))
            if src in source_success_counter:
                source_success_counter[src] += 1
            if tpl in template_success_counter:
                template_success_counter[tpl] += 1

            manifest_lines.append(
                json.dumps(
                    {
                        "token": token,
                        "rank": rank,
                        "result_status": tr.get("result_status", "success"),
                        "source_type": tr.get("source_type"),
                        "event_template": tr.get("event_template"),
                        "recovery_type": tr.get("recovery_type"),
                        "reject_reason": tr.get("reject_reason"),
                        "boundary_state": tr.get("boundary_state"),
                        "stage1_score": tr.get("stage1_score"),
                        "stage2_score": tr.get("stage2_score"),
                        "longtail_metrics": tr.get("longtail_metrics"),
                        "traffic_light_metrics": tr.get("traffic_light_metrics"),
                        "dynamics_metrics": tr.get("dynamics_metrics"),
                        "json_path": str(out_json) if out_json is not None else None,
                        "pkl_path": str(out_pkl) if out_pkl is not None else None,
                    },
                    ensure_ascii=False,
                )
            )

        for rank, tr in enumerate(failure_traces):
            reason = _sanitize_fragment(str(tr.get("reject_reason", "rejected")))
            failure_dir = failure_root / reason
            failure_dir.mkdir(parents=True, exist_ok=True)
            out_json, out_pkl = _save_trace(
                trace=tr,
                out_dir=failure_dir,
                fmt=args.save_format,
                rank=rank,
                stem_suffix=f"redlight_failure_{reason}",
            )
            saved_failure += 1
            manifest_lines.append(
                json.dumps(
                    {
                        "token": token,
                        "rank": rank,
                        "result_status": tr.get("result_status", "rejected"),
                        "source_type": tr.get("source_type"),
                        "event_template": tr.get("event_template"),
                        "recovery_type": tr.get("recovery_type"),
                        "reject_reason": tr.get("reject_reason"),
                        "boundary_state": tr.get("boundary_state"),
                        "stage1_score": tr.get("stage1_score"),
                        "stage2_score": tr.get("stage2_score"),
                        "longtail_metrics": tr.get("longtail_metrics"),
                        "traffic_light_metrics": tr.get("traffic_light_metrics"),
                        "dynamics_metrics": tr.get("dynamics_metrics"),
                        "json_path": str(out_json) if out_json is not None else None,
                        "pkl_path": str(out_pkl) if out_pkl is not None else None,
                    },
                    ensure_ascii=False,
                )
            )

        if not success_traces and not failure_traces:
            rejected_tokens += 1
            manifest_lines.append(
                json.dumps(
                    {
                        "token": token,
                        "rank": None,
                        "result_status": "rejected",
                        "source_type": source_type,
                        "event_template": None,
                        "recovery_type": None,
                        "reject_reason": result.get("rejected_reason", "no_candidate_passed"),
                        "boundary_state": None,
                        "stage1_score": None,
                        "stage2_score": None,
                        "longtail_metrics": None,
                        "traffic_light_metrics": None,
                        "dynamics_metrics": None,
                        "json_path": None,
                        "pkl_path": None,
                    },
                    ensure_ascii=False,
                )
            )

    manifest_path.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")
    stats_path.write_text(json.dumps(seed_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # Acceptance summary + optional hard check
    template_total = sum(template_success_counter.values())
    template_dev: Dict[str, float] = {}
    if template_total > 0:
        target = {
            "late_brake_stopline": float(event_mix.template_weights.late_brake_stopline),
            "crosswalk_intrude_stop": float(event_mix.template_weights.crosswalk_intrude_stop),
            "commit_go": float(event_mix.template_weights.commit_go),
        }
        target_sum = sum(target.values())
        target = {k: (v / target_sum) for k, v in target.items()}
        for k in template_success_counter:
            actual = float(template_success_counter[k]) / float(template_total)
            template_dev[k] = abs(actual - target[k])

    source_total = sum(source_success_counter.values())
    source_dev: Dict[str, float] = {}
    if source_total > 0:
        target_s = {
            "waiting_red": float(event_mix.source_weights.waiting_red),
            "passing_green": float(event_mix.source_weights.passing_green),
        }
        target_s_sum = sum(target_s.values())
        target_s = {k: (v / target_s_sum) for k, v in target_s.items()}
        for k in source_success_counter:
            actual = float(source_success_counter[k]) / float(source_total)
            source_dev[k] = abs(actual - target_s[k])

    print(
        f"Done. success={saved_success}, failure={saved_failure}, rejected_tokens={rejected_tokens}, "
        f"manifest={manifest_path}, success_dir={success_dir}, failure_dir={failure_root}"
    )
    print(f"Template success counts: {template_success_counter}, deviation={template_dev}")
    print(f"Source consumed counts: {source_consumed_counter}")
    print(f"Source success counts: {source_success_counter}, deviation={source_dev}")

    if args.enforce_ratio_tolerance:
        tol = float(args.ratio_tolerance)
        if template_dev and max(template_dev.values()) > tol:
            raise RuntimeError(f"Template ratio deviation exceeds tolerance={tol}: {template_dev}")
        if source_dev and max(source_dev.values()) > tol:
            raise RuntimeError(f"Source ratio deviation exceeds tolerance={tol}: {source_dev}")


if __name__ == "__main__":
    main()
