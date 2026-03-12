#!/usr/bin/env python3
"""Find traffic-light-related scene tokens from NAVSIM metric caches.

Modes:
- waiting_red: ego is (near) stopped while route-relevant red light exists.
- passing_green: ego moves through a traffic-light-controlled area under green.
"""

from __future__ import annotations

import argparse
import csv
import lzma
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_metric_cache(cache_path: Path) -> Optional[Any]:
    try:
        return pickle.loads(lzma.decompress(cache_path.read_bytes()))
    except Exception as exc:
        print(f"[warn] skip unreadable cache: {cache_path} ({exc})", file=sys.stderr)
        return None


def _speed_from_ego_dynamic_state(ego_dynamic_state: List[float]) -> float:
    vx, vy = float(ego_dynamic_state[0]), float(ego_dynamic_state[1])
    return float(math.hypot(vx, vy))


def _load_log_token_index(log_path: Path) -> Dict[str, Dict[str, Any]]:
    with log_path.open("rb") as f:
        frames = pickle.load(f)

    token_index: Dict[str, Dict[str, Any]] = {}
    for frame in frames:
        red_count = sum(1 for _, is_red in frame["traffic_lights"] if is_red)
        green_count = sum(1 for _, is_red in frame["traffic_lights"] if not is_red)
        token_index[frame["token"]] = {
            "frame_idx": int(frame["frame_idx"]),
            "timestamp": int(frame["timestamp"]),
            "raw_has_red": red_count > 0,
            "raw_red_count": red_count,
            "raw_green_count": green_count,
            "raw_ego_speed": _speed_from_ego_dynamic_state(frame["ego_dynamic_state"]),
        }
    return token_index


def find_waiting_red_light_scenes(
    metric_cache_root: Path,
    logs_root: Optional[Path],
    mode: str,
    speed_threshold: float,
    pass_speed_threshold: float,
    min_red_steps: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cache_paths = sorted(metric_cache_root.glob("*/*/*/metric_cache.pkl"))

    for cache_path in cache_paths:
        metric_cache = _load_metric_cache(cache_path)
        if metric_cache is None:
            continue

        occupancy_maps_tl = getattr(metric_cache.observation, "_occupancy_maps_tl", None)
        if not occupancy_maps_tl:
            continue

        red_steps = sum(1 for tokens, _ in occupancy_maps_tl if len(tokens) > 0)
        if red_steps < min_red_steps:
            continue

        current_speed = float(metric_cache.ego_state.dynamic_car_state.speed)
        past_states = metric_cache.past_human_trajectory.get_sampled_trajectory()
        past_speeds = [float(state.dynamic_car_state.speed) for state in past_states]
        min_past_speed = min(past_speeds) if past_speeds else float("nan")

        rows.append(
            {
                "log_name": metric_cache.log_name,
                "scene_token": cache_path.parent.name,
                "red_steps": red_steps,
                "current_speed": current_speed,
                "min_past_speed": min_past_speed,
                "metric_cache_path": str(cache_path.resolve()),
            }
        )

    rows.sort(key=lambda item: (-item["red_steps"], item["current_speed"]))

    if logs_root is None:
        # If mini logs are not loaded, keep only waiting_red mode (green-pass requires raw TL colors).
        if mode == "passing_green":
            return []
        return [
            row
            for row in rows
            if row["current_speed"] <= speed_threshold
            and (row["min_past_speed"] <= speed_threshold or math.isnan(row["min_past_speed"]))
        ]

    log_index_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        log_name = row["log_name"]
        log_path = logs_root / f"{log_name}.pkl"
        if log_name not in log_index_cache:
            if log_path.exists():
                log_index_cache[log_name] = _load_log_token_index(log_path)
            else:
                log_index_cache[log_name] = {}

        frame_info = log_index_cache[log_name].get(row["scene_token"], {})
        row.update(frame_info)

    if mode == "waiting_red":
        return [
            row
            for row in rows
            if row["current_speed"] <= speed_threshold
            and (row["min_past_speed"] <= speed_threshold or math.isnan(row["min_past_speed"]))
        ]

    # passing_green
    # Heuristic:
    # 1) no route-related red occupancy in cache horizon
    # 2) current frame has green TL(s) and no red TL in mini log
    # 3) ego speed is above threshold (moving through)
    passing_rows: List[Dict[str, Any]] = []
    for cache_path in cache_paths:
        metric_cache = _load_metric_cache(cache_path)
        if metric_cache is None:
            continue

        occupancy_maps_tl = getattr(metric_cache.observation, "_occupancy_maps_tl", None)
        red_steps = 0
        if occupancy_maps_tl:
            red_steps = sum(1 for tokens, _ in occupancy_maps_tl if len(tokens) > 0)
        if red_steps > 0:
            continue

        log_name = metric_cache.log_name
        token = cache_path.parent.name
        frame_info = log_index_cache.get(log_name, {}).get(token, {})
        if not frame_info:
            continue

        raw_green_count = int(frame_info.get("raw_green_count", 0))
        raw_red_count = int(frame_info.get("raw_red_count", 0))
        raw_ego_speed = float(frame_info.get("raw_ego_speed", 0.0))
        if raw_green_count <= 0 or raw_red_count > 0 or raw_ego_speed < pass_speed_threshold:
            continue

        passing_rows.append(
            {
                "log_name": log_name,
                "scene_token": token,
                "red_steps": red_steps,
                "current_speed": float(metric_cache.ego_state.dynamic_car_state.speed),
                "min_past_speed": float(min(float(s.dynamic_car_state.speed) for s in metric_cache.past_human_trajectory.get_sampled_trajectory())),
                "frame_idx": int(frame_info["frame_idx"]),
                "timestamp": int(frame_info["timestamp"]),
                "raw_has_red": bool(frame_info["raw_has_red"]),
                "raw_red_count": raw_red_count,
                "raw_green_count": raw_green_count,
                "raw_ego_speed": raw_ego_speed,
                "metric_cache_path": str(cache_path.resolve()),
            }
        )

    passing_rows.sort(key=lambda item: (-item["raw_green_count"], -item["raw_ego_speed"]))
    return passing_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric-cache-root",
        type=Path,
        default=Path("metric_cache/navmini"),
        help="Root directory containing metric_cache.pkl files.",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("mini_navsim_logs/mini"),
        help="Root directory containing *.pkl NAVSIM logs. Use --no-logs to skip log enrichment.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="waiting_red",
        choices=["waiting_red", "passing_green"],
        help="Filtering mode.",
    )
    parser.add_argument("--no-logs", action="store_true", help="Skip loading mini logs for extra per-frame info.")
    parser.add_argument("--speed-threshold", type=float, default=0.5, help="Max ego speed (m/s) to be treated as waiting.")
    parser.add_argument(
        "--pass-speed-threshold",
        type=float,
        default=2.0,
        help="Min ego speed (m/s) for passing_green mode.",
    )
    parser.add_argument("--min-red-steps", type=int, default=5, help="Min number of red-light steps in metric cache.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/traffic_light_scenes.csv"),
        help="CSV output path.",
    )
    parser.add_argument("--print-top", type=int, default=20, help="How many rows to print to stdout.")
    args = parser.parse_args()

    logs_root = None if args.no_logs else args.logs_root
    rows = find_waiting_red_light_scenes(
        metric_cache_root=args.metric_cache_root,
        logs_root=logs_root,
        mode=args.mode,
        speed_threshold=args.speed_threshold,
        pass_speed_threshold=args.pass_speed_threshold,
        min_red_steps=args.min_red_steps,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "log_name",
        "scene_token",
        "red_steps",
        "current_speed",
        "min_past_speed",
        "frame_idx",
        "timestamp",
        "raw_has_red",
        "raw_red_count",
        "raw_green_count",
        "raw_ego_speed",
        "metric_cache_path",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Found {len(rows)} scenes for mode={args.mode}.")
    print(f"CSV saved to: {args.output.resolve()}")
    for row in rows[: args.print_top]:
        print(
            f"{row['log_name']}  {row['scene_token']}  "
            f"red_steps={row['red_steps']}  cur_speed={row['current_speed']:.3f}"
        )


if __name__ == "__main__":
    main()
