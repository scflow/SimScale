#!/usr/bin/env python3
"""
Paper-aligned pseudo-expert scene simulation generator.

This script implements a two-stage closed-loop generation flow:
1) Stage-A perturbation rollout (T -> T+H) with reactive IDM traffic.
2) Stage-B pseudo-expert rescue rollout (T+H -> T+2H) with reactive IDM traffic.

Only strong long-tail successes are saved to `success/`.
Rejected diagnostic samples are saved to `failure/<reason>/`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from nuplan.common.actor_state.state_representation import TimeDuration, TimePoint
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

import generate_ood_mini as base
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_ego_state
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex


WorkerContext = Dict[str, Any]
_CTX: Optional[WorkerContext] = None


@dataclass
class RiskCriteria:
    min_dist_min_m: float = 1.0
    min_dist_max_m: float = 3.0


@dataclass
class RecoveryCriteria:
    min_displacement_m: float = 4.0
    min_mean_speed_mps: float = 0.5
    min_distance_improve_m: float = 0.2
    min_stage2_distance_m: float = 1.2


@dataclass
class DynamicsCriteria:
    max_abs_accel_mps2: float = 6.0
    max_abs_steer_deg: float = 60.0
    max_abs_yaw_rate_deg_s: float = 45.0
    max_abs_jerk_mps3: float = 8.0
    max_abs_steer_rate_deg_s: float = 80.0
    join_window_k: int = 3


@dataclass
class RunConfig:
    sample_batch_size: int
    max_candidate_trials: int
    top_k_per_token: int
    max_failure_traces_per_token: int
    time_budget_min_per_token: float
    progress_log_interval_sec: float
    verbose: bool


@dataclass
class RetrievalCriteria:
    enabled: bool = True
    lon_tol_m: float = 15.0
    lat_min_m: float = 0.2
    lat_max_m: float = 3.5
    heading_tol_deg: float = 25.0
    disp_tol_m: float = 20.0
    mean_speed_tol_mps: float = 8.0
    end_speed_tol_mps: float = 10.0
    min_pool_size: int = 512
    fallback_k: int = 4096


@dataclass
class PerturbationCriteria:
    scale: float = 1.0
    adaptive: bool = True
    min_scale: float = 0.35
    max_scale: float = 1.1
    down_step: float = 0.1
    up_step: float = 0.05
    adapt_every_attempts: int = 40
    high_stage1_fail_ratio: float = 0.6
    high_not_risky_ratio: float = 0.25


@dataclass
class FastStage1ProxyCriteria:
    enabled: bool = True
    min_center_distance_m: float = 3.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate strong pseudo-expert OOD traces.")
    parser.add_argument("--metric-cache-path", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, default=Path("traj_final/16384.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("generated_ood_data_pseudo_expert"))
    parser.add_argument("--save-format", type=str, default="both", choices=["json", "pkl", "both"])
    parser.add_argument("--manifest-name", type=str, default="manifest.jsonl")
    parser.add_argument("--map-root-override", type=str, default=None)

    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--sample-batch-size", type=int, default=120)
    parser.add_argument("--max-candidate-trials", type=int, default=4000)
    parser.add_argument("--top-k-per-token", type=int, default=1)
    parser.add_argument("--max-failure-traces-per-token", type=int, default=20)
    parser.add_argument("--time-budget-min-per-token", type=float, default=15.0)
    parser.add_argument("--progress-log-interval-sec", type=float, default=15.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--horizon-sec", type=float, default=4.0)

    # Base filtering and safety gate thresholds.
    parser.add_argument("--max-abs-lon-m", type=float, default=20.0)
    parser.add_argument("--max-abs-lat-m", type=float, default=2.0)
    parser.add_argument("--max-abs-heading-deg", type=float, default=20.0)
    parser.add_argument("--max-abs-accel-mps2", type=float, default=6.0)
    parser.add_argument("--max-abs-steer-deg", type=float, default=60.0)
    parser.add_argument("--min-progress", type=float, default=0.5)
    parser.add_argument("--max-join-translation-m", type=float, default=1.0)
    parser.add_argument("--max-join-heading-deg", type=float, default=8.0)
    parser.add_argument("--max-join-speed-delta-mps", type=float, default=2.0)

    # Risk gate.
    parser.add_argument("--risk-min-dist-min-m", type=float, default=1.0)
    parser.add_argument("--risk-min-dist-max-m", type=float, default=3.0)

    # Recovery gate.
    parser.add_argument("--recovery-min-displacement-m", type=float, default=4.0)
    parser.add_argument("--recovery-min-mean-speed-mps", type=float, default=0.5)
    parser.add_argument("--recovery-min-distance-improve-m", type=float, default=0.2)
    parser.add_argument("--recovery-min-stage2-dist-m", type=float, default=1.2)

    # Dynamics gate.
    parser.add_argument("--dyn-max-abs-accel-mps2", type=float, default=6.0)
    parser.add_argument("--dyn-max-abs-steer-deg", type=float, default=60.0)
    parser.add_argument("--dyn-max-abs-yaw-rate-deg-s", type=float, default=45.0)
    parser.add_argument("--dyn-max-abs-jerk-mps3", type=float, default=8.0)
    parser.add_argument("--dyn-max-abs-steer-rate-deg-s", type=float, default=80.0)
    parser.add_argument("--dyn-join-window-k", type=int, default=3)

    # Token-conditioned retrieval gate for Stage-A candidate generation.
    parser.add_argument("--disable-conditioned-retrieval", action="store_true")
    parser.add_argument("--retrieval-lon-tol-m", type=float, default=15.0)
    parser.add_argument("--retrieval-lat-min-m", type=float, default=0.2)
    parser.add_argument("--retrieval-lat-max-m", type=float, default=3.5)
    parser.add_argument("--retrieval-heading-tol-deg", type=float, default=25.0)
    parser.add_argument("--retrieval-disp-tol-m", type=float, default=20.0)
    parser.add_argument("--retrieval-mean-speed-tol-mps", type=float, default=8.0)
    parser.add_argument("--retrieval-end-speed-tol-mps", type=float, default=10.0)
    parser.add_argument("--retrieval-min-pool-size", type=int, default=512)
    parser.add_argument("--retrieval-fallback-k", type=int, default=4096)

    # Candidate perturbation scaling around expert trajectory.
    parser.add_argument("--perturb-scale", type=float, default=1.0)
    parser.add_argument("--disable-adaptive-perturb-scale", action="store_true")
    parser.add_argument("--perturb-min-scale", type=float, default=0.35)
    parser.add_argument("--perturb-max-scale", type=float, default=1.1)
    parser.add_argument("--perturb-down-step", type=float, default=0.1)
    parser.add_argument("--perturb-up-step", type=float, default=0.05)
    parser.add_argument("--perturb-adapt-every-attempts", type=int, default=40)
    parser.add_argument("--perturb-high-stage1-fail-ratio", type=float, default=0.6)
    parser.add_argument("--perturb-high-not-risky-ratio", type=float, default=0.25)

    # Fast proxy filter before expensive Stage1 reactive simulation.
    parser.add_argument("--disable-fast-stage1-proxy", action="store_true")
    parser.add_argument("--fast-stage1-proxy-min-center-distance-m", type=float, default=3.2)

    parser.add_argument("--parallel-backend", type=str, default="process", choices=["process", "none"])
    parser.add_argument("--num-workers", type=int, default=0, help="0 means auto, capped to 32.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _auto_workers(num_workers: int) -> int:
    if num_workers > 0:
        return num_workers
    try:
        import os

        c = os.cpu_count() or 1
    except Exception:
        c = 1
    return max(1, min(c, 32))


def _token_seed(base_seed: int, token: str) -> int:
    h = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)
    return (base_seed + h) % (2**32 - 1)


def _nan_score() -> Dict[str, float]:
    return {
        "no_at_fault_collisions": float("nan"),
        "drivable_area_compliance": float("nan"),
        "driving_direction_compliance": float("nan"),
        "traffic_light_compliance": float("nan"),
        "ego_progress": float("nan"),
        "pdm_score": float("nan"),
    }


def _sanitize_fragment(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    token = "".join(out).strip("_")
    return token or "unknown"


def _push_top(pool: List[Dict[str, Any]], item: Dict[str, Any], score_key: str, max_size: int) -> None:
    pool.append(item)
    pool.sort(key=lambda x: float(x.get(score_key, 0.0)), reverse=True)
    if len(pool) > max_size:
        pool.pop()


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


def _ego_displacement(states: np.ndarray) -> float:
    if len(states) < 2:
        return 0.0
    dx = float(states[-1, StateIndex.X] - states[0, StateIndex.X])
    dy = float(states[-1, StateIndex.Y] - states[0, StateIndex.Y])
    return math.hypot(dx, dy)


def _ego_mean_speed(states: np.ndarray) -> float:
    v = np.hypot(states[:, StateIndex.VELOCITY_X], states[:, StateIndex.VELOCITY_Y])
    return float(np.mean(v))


def _stage1_risk_score(min_dist_stage1: float, risk: RiskCriteria) -> float:
    if math.isnan(min_dist_stage1):
        return 0.0
    center = 0.5 * (risk.min_dist_min_m + risk.min_dist_max_m)
    width = max(1e-6, 0.5 * (risk.min_dist_max_m - risk.min_dist_min_m))
    return max(0.0, 1.0 - abs(min_dist_stage1 - center) / width)


def _angle_diff_rad(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))


def _trajectory_features(local_poses: np.ndarray, interval_s: float) -> Dict[str, float]:
    poses = np.asarray(local_poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3 or len(poses) == 0:
        return {
            "end_x": 0.0,
            "end_y": 0.0,
            "end_h": 0.0,
            "disp": 0.0,
            "mean_speed": 0.0,
            "end_speed": 0.0,
        }
    end_x = float(poses[-1, 0])
    end_y = float(poses[-1, 1])
    end_h = float(poses[-1, 2])
    disp = math.hypot(end_x, end_y)
    if len(poses) <= 1:
        mean_speed = 0.0
        end_speed = 0.0
    else:
        dt = max(1e-3, float(interval_s))
        step = np.diff(poses[:, :2], axis=0)
        speed = np.hypot(step[:, 0], step[:, 1]) / dt
        mean_speed = float(np.mean(speed))
        end_speed = float(speed[-1])
    return {
        "end_x": end_x,
        "end_y": end_y,
        "end_h": end_h,
        "disp": float(disp),
        "mean_speed": mean_speed,
        "end_speed": end_speed,
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


def _build_vocab_retrieval_index(vocab: np.ndarray, interval_s: float) -> Dict[str, np.ndarray]:
    poses = np.asarray(vocab, dtype=np.float32)
    end = poses[:, -1, :]
    end_x = end[:, 0].astype(np.float32)
    end_y = end[:, 1].astype(np.float32)
    end_h = end[:, 2].astype(np.float32)
    disp = np.hypot(end_x, end_y).astype(np.float32)
    if poses.shape[1] <= 1:
        mean_speed = np.zeros(len(poses), dtype=np.float32)
        end_speed = np.zeros(len(poses), dtype=np.float32)
    else:
        dt = max(1e-3, float(interval_s))
        step = np.diff(poses[:, :, :2], axis=1)
        speed = np.hypot(step[:, :, 0], step[:, :, 1]) / dt
        mean_speed = np.mean(speed, axis=1).astype(np.float32)
        end_speed = speed[:, -1].astype(np.float32)
    return {
        "end_x": end_x,
        "end_y": end_y,
        "end_h": end_h,
        "disp": disp,
        "mean_speed": mean_speed,
        "end_speed": end_speed,
    }


def _retrieve_conditioned_vocab_indices(
    vocab_index: Dict[str, np.ndarray],
    expert_rel_poses: np.ndarray,
    retrieval: RetrievalCriteria,
    interval_s: float,
) -> np.ndarray:
    n = len(vocab_index["end_x"])
    if n == 0:
        return np.array([], dtype=np.int64)

    expert = _trajectory_features(expert_rel_poses, interval_s=interval_s)
    end_x = vocab_index["end_x"]
    end_y = vocab_index["end_y"]
    end_h = vocab_index["end_h"]
    disp = vocab_index["disp"]
    mean_speed = vocab_index["mean_speed"]
    end_speed = vocab_index["end_speed"]

    lon_tol = max(1e-3, float(retrieval.lon_tol_m))
    lat_min = max(0.0, float(retrieval.lat_min_m))
    lat_max = max(lat_min + 1e-3, float(retrieval.lat_max_m))
    heading_tol_rad = math.radians(max(1e-3, float(retrieval.heading_tol_deg)))
    disp_tol = max(1.0, float(retrieval.disp_tol_m))
    mean_speed_tol = max(0.5, float(retrieval.mean_speed_tol_mps))
    end_speed_tol = max(0.5, float(retrieval.end_speed_tol_mps))
    min_pool = max(1, int(retrieval.min_pool_size))
    fallback_k = max(1, int(retrieval.fallback_k))

    ex_x = float(expert["end_x"])
    ex_y = float(expert["end_y"])
    ex_h = float(expert["end_h"])
    ex_disp = float(expert["disp"])
    ex_mean_speed = float(expert["mean_speed"])
    ex_end_speed = float(expert["end_speed"])

    dxe = end_x - ex_x
    dye = end_y - ex_y
    c = math.cos(ex_h)
    s = math.sin(ex_h)
    lon = c * dxe + s * dye
    lat = -s * dxe + c * dye
    abs_lon = np.abs(lon)
    abs_lat = np.abs(lat)
    dh = _angle_diff_rad(end_h, ex_h)
    ddisp = np.abs(disp - ex_disp)
    dmean = np.abs(mean_speed - ex_mean_speed)
    dend = np.abs(end_speed - ex_end_speed)

    strict_mask = (
        (abs_lon <= lon_tol)
        & (abs_lat >= lat_min)
        & (abs_lat <= lat_max)
        & (dh <= heading_tol_rad)
        & (ddisp <= disp_tol)
        & (dmean <= mean_speed_tol)
        & (dend <= end_speed_tol)
    )
    strict_idx = np.where(strict_mask)[0]
    if len(strict_idx) >= min_pool:
        return strict_idx.astype(np.int64)

    relaxed_mask = (
        (abs_lon <= lon_tol * 1.8)
        & (abs_lat <= lat_max * 1.8)
        & (dh <= heading_tol_rad * 1.8)
        & (ddisp <= disp_tol * 1.8)
    )
    relaxed_idx = np.where(relaxed_mask)[0]
    base_idx = relaxed_idx if len(relaxed_idx) > 0 else np.arange(n)

    # Weighted KNN fallback: prioritize route-consistent endpoint + speed profile.
    lat_under_penalty = np.maximum(0.0, lat_min - abs_lat[base_idx]) / max(0.1, lat_min if lat_min > 0 else 1.0)
    score = (
        1.8 * (abs_lon[base_idx] / lon_tol)
        + 1.4 * (abs_lat[base_idx] / max(1.0, lat_max))
        + 0.8 * (dh[base_idx] / heading_tol_rad)
        + 0.6 * (ddisp[base_idx] / disp_tol)
        + 0.5 * (dmean[base_idx] / mean_speed_tol)
        + 0.5 * (dend[base_idx] / end_speed_tol)
        + 1.0 * lat_under_penalty
    )

    k = min(len(base_idx), fallback_k)
    if k <= 0:
        return np.arange(n, dtype=np.int64)
    if k == len(base_idx):
        order = np.argsort(score)
        return base_idx[order].astype(np.int64)
    pick = np.argpartition(score, k - 1)[:k]
    order = pick[np.argsort(score[pick])]
    return base_idx[order].astype(np.int64)


def _sample_vocab_batch(pool: np.ndarray, batch: int, rng: np.random.Generator) -> np.ndarray:
    if len(pool) == 0:
        return np.array([], dtype=np.int64)
    if len(pool) >= batch:
        return rng.choice(pool, size=batch, replace=False)
    return pool[rng.integers(0, len(pool), size=batch)]


def _adapt_perturb_scale(current: float, stats: Dict[str, Any], cfg: PerturbationCriteria) -> float:
    attempts = int(stats.get("attempts", 0))
    if attempts <= 0:
        return current
    stage1_fail = int(stats.get("stage1_fail", 0))
    stage1_not_risky = int(stats.get("stage1_not_risky", 0))
    fail_ratio = stage1_fail / max(1, attempts)
    not_risky_ratio = stage1_not_risky / max(1, attempts)
    out = float(current)
    if fail_ratio >= float(cfg.high_stage1_fail_ratio):
        out = max(float(cfg.min_scale), out - float(cfg.down_step))
    elif not_risky_ratio >= float(cfg.high_not_risky_ratio):
        out = min(float(cfg.max_scale), out + float(cfg.up_step))
    return out


def _extract_initial_vehicle_kinematics(metric_cache: Any) -> Tuple[np.ndarray, np.ndarray]:
    obs = getattr(metric_cache, "observation", None)
    tracked = getattr(obs, "tracked_objects", None)
    objects = getattr(tracked, "tracked_objects", []) if tracked is not None else []
    pos: List[Tuple[float, float]] = []
    vel: List[Tuple[float, float]] = []
    for obj in objects:
        tpe = str(getattr(obj, "tracked_object_type", "")).lower()
        if not tpe.endswith("vehicle"):
            continue
        center = getattr(obj, "center", None)
        if center is None:
            continue
        vx = 0.0
        vy = 0.0
        v = getattr(obj, "velocity", None)
        if v is not None:
            vx = float(getattr(v, "x", 0.0))
            vy = float(getattr(v, "y", 0.0))
        pos.append((float(center.x), float(center.y)))
        vel.append((vx, vy))
    if not pos:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pos, dtype=np.float32), np.asarray(vel, dtype=np.float32)


def _candidate_local_to_global_xy(candidate_rel: np.ndarray, ego_state: Any) -> np.ndarray:
    init = ego_state.rear_axle
    c = math.cos(float(init.heading))
    s = math.sin(float(init.heading))
    x = candidate_rel[:, 0].astype(np.float32)
    y = candidate_rel[:, 1].astype(np.float32)
    gx = float(init.x) + c * x - s * y
    gy = float(init.y) + s * x + c * y
    return np.stack([gx, gy], axis=1).astype(np.float32)


def _pass_fast_stage1_proxy(
    candidate_rel: np.ndarray,
    ego_state: Any,
    vehicle_pos0: np.ndarray,
    vehicle_vel: np.ndarray,
    interval_s: float,
    proxy: FastStage1ProxyCriteria,
) -> bool:
    if (not proxy.enabled) or len(vehicle_pos0) == 0 or len(candidate_rel) == 0:
        return True
    ego_xy = _candidate_local_to_global_xy(candidate_rel, ego_state)
    t = (np.arange(len(ego_xy), dtype=np.float32) * float(interval_s))[None, :, None]
    veh_xy = vehicle_pos0[:, None, :] + vehicle_vel[:, None, :] * t
    dist = np.linalg.norm(veh_xy - ego_xy[None, :, :], axis=2)
    return float(np.min(dist)) >= float(proxy.min_center_distance_m)


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
            "split_idx": float(split_idx),
            "join_window_k": float(join_window_k),
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
        "split_idx": float(split_idx),
        "join_window_k": float(join_window_k),
    }


def _pass_dynamics_gate(metrics: Dict[str, float], dyn: DynamicsCriteria) -> bool:
    def _v(name: str) -> float:
        x = float(metrics.get(name, float("nan")))
        return x

    checks = [
        (_v("max_abs_accel_mps2"), dyn.max_abs_accel_mps2),
        (_v("max_abs_steer_deg"), dyn.max_abs_steer_deg),
        (_v("max_abs_yaw_rate_deg_s"), dyn.max_abs_yaw_rate_deg_s),
        (_v("max_abs_jerk_mps3"), dyn.max_abs_jerk_mps3),
        (_v("max_abs_steer_rate_deg_s"), dyn.max_abs_steer_rate_deg_s),
        (_v("window_max_abs_accel_mps2"), dyn.max_abs_accel_mps2),
        (_v("window_max_abs_steer_deg"), dyn.max_abs_steer_deg),
        (_v("window_max_abs_yaw_rate_deg_s"), dyn.max_abs_yaw_rate_deg_s),
        (_v("window_max_abs_jerk_mps3"), dyn.max_abs_jerk_mps3),
        (_v("window_max_abs_steer_rate_deg_s"), dyn.max_abs_steer_rate_deg_s),
    ]
    for value, limit in checks:
        if math.isnan(value) or value > limit:
            return False
    return True


def _strong_longtail_score(
    stage1_min_dist: float,
    stage2_min_dist: float,
    stage2_disp: float,
    stage2_mean_speed: float,
    join_translation: float,
    join_heading_deg: float,
    join_speed_delta: float,
    stage2_pdm: float,
    risk: RiskCriteria,
    recovery: RecoveryCriteria,
) -> float:
    risk_score = _stage1_risk_score(stage1_min_dist, risk)
    recovery_score = 0.6 * min(1.0, stage2_disp / max(1e-6, recovery.min_displacement_m * 1.4)) + 0.4 * min(
        1.0, stage2_mean_speed / max(1e-6, recovery.min_mean_speed_mps * 1.8)
    )
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
    return float((0.45 * risk_score + 0.3 * recovery_score + 0.15 * safety_gain + 0.1 * continuity) * stage2_pdm)


def _default_sensor_hook(trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Placeholder for Sensor Simulation Φ.
    Keep interface stable for future image rendering integration.
    """
    _ = trace
    return None


def _make_stage1_trace(
    token: str,
    vocab_index: int,
    phase1_states: np.ndarray,
    phase1_tracks: List[Any],
    stage1_score: Dict[str, float],
) -> Dict[str, Any]:
    return base.serialize_trace(
        token=token,
        vocab_index=int(vocab_index),
        ego_states=phase1_states,
        tracks=phase1_tracks,
        stage1_score=stage1_score,
        stage2_score=_nan_score(),
    )


def _annotate_trace(
    trace: Dict[str, Any],
    status: str,
    reason: Optional[str],
    search_stage: str,
    criteria_snapshot: Dict[str, Any],
    longtail_metrics: Optional[Dict[str, float]],
    dynamics_metrics: Optional[Dict[str, float]],
) -> Dict[str, Any]:
    trace["result_status"] = status
    trace["reject_reason"] = reason
    trace["search_stage"] = search_stage
    trace["criteria_snapshot"] = criteria_snapshot
    trace["longtail_metrics"] = longtail_metrics if longtail_metrics is not None else {}
    trace["dynamics_metrics"] = dynamics_metrics if dynamics_metrics is not None else {}
    trace["sensor_payload"] = _default_sensor_hook(trace)
    return trace


def _save_trace(
    trace: Dict[str, Any],
    out_dir: Path,
    fmt: str,
    rank: int,
    stem_suffix: str,
) -> Tuple[Optional[Path], Optional[Path]]:
    token = trace["token"]
    stem = f"ood_scene_{token}_{stem_suffix}_r{rank:02d}"
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


def _base_thresholds_from_args(args: argparse.Namespace) -> base.Thresholds:
    return base.Thresholds(
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


def _run_token_pseudo_expert(
    token: str,
    metric_cache: Any,
    vocab: np.ndarray,
    vocab_retrieval_index: Dict[str, np.ndarray],
    proposal_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    reactive_policy: Any,
    thresholds: base.Thresholds,
    risk: RiskCriteria,
    recovery: RecoveryCriteria,
    dyn: DynamicsCriteria,
    retrieval: RetrievalCriteria,
    perturb: PerturbationCriteria,
    fast_proxy: FastStage1ProxyCriteria,
    run_cfg: RunConfig,
    rng: np.random.Generator,
    map_root_override: Optional[str],
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "attempts": 0,
        "geo_fail": 0,
        "physics_fail": 0,
        "stage1_fail": 0,
        "stage1_not_risky": 0,
        "join_fail": 0,
        "stage2_fail": 0,
        "recovery_fail": 0,
        "dynamics_fail": 0,
        "proxy_fail": 0,
        "successes": 0,
        "failures_saved": 0,
        "reject_reasons": {},
        "stage1_fail_breakdown": {},
    }

    def _inc_reason(reason: str) -> None:
        stats["reject_reasons"][reason] = int(stats["reject_reasons"].get(reason, 0)) + 1

    if metric_cache.human_trajectory is None:
        _inc_reason("missing_human_trajectory")
        return {
            "token": token,
            "success_traces": [],
            "failure_traces": [],
            "stats": stats,
            "rejected_reason": "missing_human_trajectory",
        }

    if map_root_override is not None:
        metric_cache.map_parameters.map_root = map_root_override

    scorer = PDMScorer(proposal_sampling=proposal_sampling, config=PDMScorerConfig(human_penalty_filter=False))
    planner_init = base.build_planner_initialization(metric_cache, map_root_override=map_root_override)
    expert_planner = base.build_expert_planner(proposal_sampling=proposal_sampling)

    criteria_snapshot = {
        "risk": asdict(risk),
        "recovery": asdict(recovery),
        "dynamics": asdict(dyn),
        "retrieval": asdict(retrieval),
        "perturb": asdict(perturb),
        "fast_stage1_proxy": asdict(fast_proxy),
        "base_thresholds": {
            "max_abs_lon_m": thresholds.max_abs_lon_m,
            "max_abs_lat_m": thresholds.max_abs_lat_m,
            "max_abs_heading_deg": thresholds.max_abs_heading_deg,
            "max_abs_accel_mps2": thresholds.max_abs_accel_mps2,
            "max_abs_steer_deg": thresholds.max_abs_steer_deg,
            "min_progress": thresholds.min_progress,
            "max_join_translation_m": thresholds.max_join_translation_m,
            "max_join_heading_deg": thresholds.max_join_heading_deg,
            "max_join_speed_delta_mps": thresholds.max_join_speed_delta_mps,
        },
    }

    success_pool: List[Dict[str, Any]] = []
    failure_pool: List[Dict[str, Any]] = []

    start_t = time.monotonic()
    budget_sec = max(1.0, float(run_cfg.time_budget_min_per_token) * 60.0)
    next_log_t = start_t + max(1.0, run_cfg.progress_log_interval_sec)
    vocab_size = len(vocab)
    expert_rel_raw = np.asarray(metric_cache.human_trajectory.poses, dtype=np.float32)
    expert_rel = _resample_local_poses(expert_rel_raw, int(vocab.shape[1]))
    if retrieval.enabled:
        candidate_pool = _retrieve_conditioned_vocab_indices(
            vocab_index=vocab_retrieval_index,
            expert_rel_poses=expert_rel,
            retrieval=retrieval,
            interval_s=proposal_sampling.interval_length,
        )
        if len(candidate_pool) == 0:
            candidate_pool = np.arange(vocab_size, dtype=np.int64)
        pool_cap = max(int(retrieval.min_pool_size), int(run_cfg.sample_batch_size) * 8)
        if len(candidate_pool) > pool_cap:
            candidate_pool = candidate_pool[:pool_cap]
    else:
        candidate_pool = np.arange(vocab_size, dtype=np.int64)
    stats["candidate_pool_size"] = int(len(candidate_pool))
    curr_perturb_scale = float(np.clip(perturb.scale, perturb.min_scale, perturb.max_scale))
    stats["perturb_scale_init"] = float(curr_perturb_scale)
    vehicle_pos0, vehicle_vel = _extract_initial_vehicle_kinematics(metric_cache)
    pool_order = rng.permutation(candidate_pool) if len(candidate_pool) > 0 else np.array([], dtype=np.int64)
    pool_cursor = 0

    def _next_batch_indices(batch_size: int) -> np.ndarray:
        nonlocal pool_order, pool_cursor
        if len(candidate_pool) == 0 or batch_size <= 0:
            return np.array([], dtype=np.int64)
        out: List[int] = []
        while len(out) < batch_size:
            if pool_cursor >= len(pool_order):
                pool_order = rng.permutation(candidate_pool)
                pool_cursor = 0
            take = min(batch_size - len(out), len(pool_order) - pool_cursor)
            if take <= 0:
                break
            out.extend(int(x) for x in pool_order[pool_cursor : pool_cursor + take])
            pool_cursor += take
        return np.asarray(out, dtype=np.int64)

    while stats["attempts"] < run_cfg.max_candidate_trials:
        now = time.monotonic()
        if now - start_t >= budget_sec:
            break
        remain = run_cfg.max_candidate_trials - stats["attempts"]
        batch = min(run_cfg.sample_batch_size, remain)
        sampled = _next_batch_indices(batch)
        if len(sampled) == 0:
            break

        for vocab_idx in sampled:
            now = time.monotonic()
            if now - start_t >= budget_sec:
                break
            if stats["attempts"] >= run_cfg.max_candidate_trials:
                break

            stats["attempts"] += 1
            if (
                perturb.adaptive
                and not success_pool
                and stats["attempts"] % max(1, int(perturb.adapt_every_attempts)) == 0
            ):
                curr_perturb_scale = _adapt_perturb_scale(curr_perturb_scale, stats, perturb)
            candidate_raw = np.asarray(vocab[vocab_idx], dtype=np.float32)
            # Blend candidate toward expert trajectory to reduce early hard collisions.
            candidate_rel = expert_rel + curr_perturb_scale * (candidate_raw - expert_rel)
            if not _pass_fast_stage1_proxy(
                candidate_rel=candidate_rel,
                ego_state=metric_cache.ego_state,
                vehicle_pos0=vehicle_pos0,
                vehicle_vel=vehicle_vel,
                interval_s=proposal_sampling.interval_length,
                proxy=fast_proxy,
            ):
                stats["proxy_fail"] += 1
                _inc_reason("proxy_fail")
                continue

            # Stage-A: perturb action simulation
            if not base.pass_geometric_filter(metric_cache, candidate_rel, thresholds):
                stats["geo_fail"] += 1
                _inc_reason("geo_fail")
                continue

            candidate_states = base.make_trajectory_states_from_local_poses(
                local_poses=candidate_rel,
                metric_cache=metric_cache,
                proposal_sampling=proposal_sampling,
            )
            if time.monotonic() - start_t >= budget_sec:
                break
            phase1_states = simulator.simulate_proposals(candidate_states[None, ...], metric_cache.ego_state)[0]
            if not base.pass_physics_filter(phase1_states, thresholds):
                stats["physics_fail"] += 1
                _inc_reason("physics_fail")
                continue

            phase1_tracks = reactive_policy.simulate_environment(phase1_states, metric_cache)
            stage1_score = base.score_segment(
                scorer=scorer,
                states=phase1_states,
                metric_cache=metric_cache,
                simulated_tracks=phase1_tracks,
            )
            if not base.pass_strict_gate(stage1_score, thresholds):
                stats["stage1_fail"] += 1
                _inc_reason("stage1_fail")
                breakdown = stats["stage1_fail_breakdown"]
                if float(stage1_score.get("no_at_fault_collisions", 1.0)) < 0.999:
                    breakdown["NC"] = int(breakdown.get("NC", 0)) + 1
                if float(stage1_score.get("drivable_area_compliance", 1.0)) < 0.999:
                    breakdown["DAC"] = int(breakdown.get("DAC", 0)) + 1
                if float(stage1_score.get("driving_direction_compliance", 1.0)) < 0.999:
                    breakdown["DDC"] = int(breakdown.get("DDC", 0)) + 1
                if float(stage1_score.get("traffic_light_compliance", 1.0)) < 0.999:
                    breakdown["TLC"] = int(breakdown.get("TLC", 0)) + 1
                if float(stage1_score.get("ego_progress", 1.0)) < float(thresholds.min_progress):
                    breakdown["EP"] = int(breakdown.get("EP", 0)) + 1
                if len(failure_pool) < run_cfg.max_failure_traces_per_token:
                    trace = _make_stage1_trace(
                        token=token,
                        vocab_index=int(vocab_idx),
                        phase1_states=phase1_states,
                        phase1_tracks=phase1_tracks,
                        stage1_score=stage1_score,
                    )
                    score = float(stage1_score.get("pdm_score", 0.0))
                    trace["_rank_score"] = score
                    _annotate_trace(
                        trace=trace,
                        status="rejected",
                        reason="stage1_fail",
                        search_stage="t=T",
                        criteria_snapshot=criteria_snapshot,
                        longtail_metrics={"diagnostic_score": score},
                        dynamics_metrics={},
                    )
                    _push_top(failure_pool, trace, "_rank_score", run_cfg.max_failure_traces_per_token)
                continue

            min_dist_stage1 = _vehicle_min_dist(phase1_states, phase1_tracks)
            elapsed_ratio = min(1.0, max(0.0, (time.monotonic() - start_t) / budget_sec))
            risk_max_effective = float(risk.min_dist_max_m)
            if not success_pool and stats["stage1_not_risky"] >= 3:
                if elapsed_ratio >= 0.80:
                    risk_max_effective += 3.0
                elif elapsed_ratio >= 0.50:
                    risk_max_effective += 1.5

            if not (risk.min_dist_min_m <= min_dist_stage1 <= risk_max_effective):
                stats["stage1_not_risky"] += 1
                _inc_reason("stage1_not_risky")
                if len(failure_pool) < run_cfg.max_failure_traces_per_token:
                    trace = _make_stage1_trace(
                        token=token,
                        vocab_index=int(vocab_idx),
                        phase1_states=phase1_states,
                        phase1_tracks=phase1_tracks,
                        stage1_score=stage1_score,
                    )
                    risk_score = _stage1_risk_score(min_dist_stage1, risk)
                    trace["_rank_score"] = risk_score
                    _annotate_trace(
                        trace=trace,
                        status="rejected",
                        reason="stage1_not_risky",
                        search_stage="t=T",
                        criteria_snapshot=criteria_snapshot,
                        longtail_metrics={
                            "stage1_min_dist_m": float(min_dist_stage1),
                            "risk_max_effective_m": float(risk_max_effective),
                            "risk_score": float(risk_score),
                            "diagnostic_score": float(risk_score),
                        },
                        dynamics_metrics={},
                    )
                    _push_top(failure_pool, trace, "_rank_score", run_cfg.max_failure_traces_per_token)
                continue

            # Stage-B: pseudo-expert rescue
            ood_time = metric_cache.ego_state.time_point + TimeDuration.from_s(proposal_sampling.time_horizon)
            ood_ego_state = state_array_to_ego_state(
                phase1_states[-1],
                TimePoint(int(ood_time.time_us)),
                metric_cache.ego_state.car_footprint.vehicle_parameters,
            )
            ood_obs = phase1_tracks[-1]
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
            if not base.pass_join_continuity_gate(phase1_states, rescue_states, thresholds):
                stats["join_fail"] += 1
                _inc_reason("join_fail")
                continue

            stage2_cache = base.build_stage2_metric_cache(
                metric_cache=metric_cache,
                ego_state_ood=ood_ego_state,
                current_tracks=ood_obs,
                proposal_sampling=proposal_sampling,
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
            if not base.pass_strict_gate(stage2_score, thresholds):
                stats["stage2_fail"] += 1
                _inc_reason("stage2_fail")
                continue

            min_dist_stage2 = _vehicle_min_dist(rescue_states, phase2_tracks)
            stage2_disp = _ego_displacement(rescue_states)
            stage2_mean_speed = _ego_mean_speed(rescue_states)

            prev = phase1_states[-1]
            nxt = rescue_states[1] if len(rescue_states) > 1 else rescue_states[0]
            join_translation = math.hypot(float(nxt[StateIndex.X] - prev[StateIndex.X]), float(nxt[StateIndex.Y] - prev[StateIndex.Y]))
            join_heading_deg = abs(
                math.degrees(base.normalize_angle(float(nxt[StateIndex.HEADING] - prev[StateIndex.HEADING])))
            )
            prev_speed = math.hypot(float(prev[StateIndex.VELOCITY_X]), float(prev[StateIndex.VELOCITY_Y]))
            next_speed = math.hypot(float(nxt[StateIndex.VELOCITY_X]), float(nxt[StateIndex.VELOCITY_Y]))
            join_speed_delta = abs(next_speed - prev_speed)

            full_ego_states = np.concatenate([phase1_states, rescue_states[1:]], axis=0)
            full_tracks = phase1_tracks + phase2_tracks[1:]
            split_idx = len(phase1_states) - 1
            dyn_metrics = _compute_dynamics_metrics(
                states=full_ego_states,
                interval_s=proposal_sampling.interval_length,
                split_idx=split_idx,
                join_window_k=dyn.join_window_k,
            )

            recovery_ok = (
                stage2_disp >= recovery.min_displacement_m
                and stage2_mean_speed >= recovery.min_mean_speed_mps
                and min_dist_stage2 >= recovery.min_stage2_distance_m
                and (min_dist_stage2 - min_dist_stage1) >= recovery.min_distance_improve_m
            )
            dynamics_ok = _pass_dynamics_gate(dyn_metrics, dyn)

            strong_score = _strong_longtail_score(
                stage1_min_dist=float(min_dist_stage1),
                stage2_min_dist=float(min_dist_stage2),
                stage2_disp=float(stage2_disp),
                stage2_mean_speed=float(stage2_mean_speed),
                join_translation=float(join_translation),
                join_heading_deg=float(join_heading_deg),
                join_speed_delta=float(join_speed_delta),
                stage2_pdm=float(stage2_score.get("pdm_score", 0.0)),
                risk=risk,
                recovery=recovery,
            )
            longtail_metrics = {
                "split_idx": float(split_idx),
                "stage1_min_dist_m": float(min_dist_stage1),
                "risk_max_effective_m": float(risk_max_effective),
                "stage2_min_dist_m": float(min_dist_stage2),
                "stage2_displacement_m": float(stage2_disp),
                "stage2_mean_speed_mps": float(stage2_mean_speed),
                "join_translation_m": float(join_translation),
                "join_heading_deg": float(join_heading_deg),
                "join_speed_delta_mps": float(join_speed_delta),
                "risk_score": float(_stage1_risk_score(float(min_dist_stage1), risk)),
                "strong_longtail_score": float(strong_score),
            }

            trace = base.serialize_trace(
                token=token,
                vocab_index=int(vocab_idx),
                ego_states=full_ego_states,
                tracks=full_tracks,
                stage1_score=stage1_score,
                stage2_score=stage2_score,
            )
            trace["_rank_score"] = float(strong_score)

            if not recovery_ok:
                stats["recovery_fail"] += 1
                _inc_reason("recovery_fail")
                if len(failure_pool) < run_cfg.max_failure_traces_per_token:
                    _annotate_trace(
                        trace=trace,
                        status="rejected",
                        reason="recovery_fail",
                        search_stage="t=T+H",
                        criteria_snapshot=criteria_snapshot,
                        longtail_metrics=longtail_metrics,
                        dynamics_metrics=dyn_metrics,
                    )
                    _push_top(failure_pool, trace, "_rank_score", run_cfg.max_failure_traces_per_token)
                continue

            if not dynamics_ok:
                stats["dynamics_fail"] += 1
                _inc_reason("dynamics_fail")
                if len(failure_pool) < run_cfg.max_failure_traces_per_token:
                    _annotate_trace(
                        trace=trace,
                        status="rejected",
                        reason="dynamics_fail",
                        search_stage="t=T+H",
                        criteria_snapshot=criteria_snapshot,
                        longtail_metrics=longtail_metrics,
                        dynamics_metrics=dyn_metrics,
                    )
                    _push_top(failure_pool, trace, "_rank_score", run_cfg.max_failure_traces_per_token)
                continue

            _annotate_trace(
                trace=trace,
                status="success_strong",
                reason=None,
                search_stage="t=T+H",
                criteria_snapshot=criteria_snapshot,
                longtail_metrics=longtail_metrics,
                dynamics_metrics=dyn_metrics,
            )
            stats["successes"] += 1
            _push_top(success_pool, trace, "_rank_score", run_cfg.top_k_per_token)

        if time.monotonic() >= next_log_t:
            elapsed = time.monotonic() - start_t
            shown = min(elapsed, budget_sec)
            pct = 100.0 * shown / budget_sec
            print(
                f"[progress][{token}] t={shown:.1f}s/{budget_sec:.1f}s ({pct:.1f}%) "
                f"attempts={stats['attempts']} success={len(success_pool)} failure={len(failure_pool)} "
                f"scale={curr_perturb_scale:.2f}"
            )
            next_log_t = time.monotonic() + max(1.0, run_cfg.progress_log_interval_sec)
            if perturb.adaptive and not success_pool:
                curr_perturb_scale = _adapt_perturb_scale(curr_perturb_scale, stats, perturb)

    stats["elapsed_sec"] = float(min(time.monotonic() - start_t, budget_sec))
    stats["failures_saved"] = len(failure_pool)
    stats["perturb_scale_final"] = float(curr_perturb_scale)

    for tr in success_pool:
        tr.pop("_rank_score", None)
    for tr in failure_pool:
        tr.pop("_rank_score", None)

    rejected_reason = None
    if not success_pool:
        reasons = stats.get("reject_reasons", {})
        rejected_reason = max(reasons.items(), key=lambda kv: kv[1])[0] if reasons else "no_candidate_passed"

    return {
        "token": token,
        "success_traces": success_pool,
        "failure_traces": failure_pool,
        "stats": stats,
        "rejected_reason": rejected_reason,
    }


def _init_worker(
    metric_cache_path: str,
    vocab_path: str,
    interval: float,
    horizon_sec: float,
    base_thresholds_dict: Dict[str, float],
    risk_dict: Dict[str, float],
    recovery_dict: Dict[str, float],
    dyn_dict: Dict[str, float],
    retrieval_dict: Dict[str, Any],
    perturb_dict: Dict[str, Any],
    fast_proxy_dict: Dict[str, Any],
    map_root_override: Optional[str],
) -> None:
    global _CTX
    cache_root = base.resolve_metric_cache_root(Path(metric_cache_path))
    loader = base.build_metric_cache_loader(cache_root)
    vocab = np.load(vocab_path, mmap_mode="r")
    proposal_sampling = TrajectorySampling(
        num_poses=int(round(horizon_sec / interval)),
        interval_length=interval,
    )
    simulator = PDMSimulator(proposal_sampling=proposal_sampling)
    reactive_policy = base.build_reactive_policy(
        proposal_sampling=proposal_sampling,
        map_root_override=map_root_override,
    )
    thresholds = base.Thresholds(**base_thresholds_dict)
    risk = RiskCriteria(**risk_dict)
    recovery = RecoveryCriteria(**recovery_dict)
    dyn = DynamicsCriteria(**dyn_dict)
    retrieval = RetrievalCriteria(**retrieval_dict)
    perturb = PerturbationCriteria(**perturb_dict)
    fast_proxy = FastStage1ProxyCriteria(**fast_proxy_dict)
    vocab_retrieval_index = _build_vocab_retrieval_index(vocab=vocab, interval_s=interval)
    _CTX = {
        "loader": loader,
        "vocab": vocab,
        "vocab_retrieval_index": vocab_retrieval_index,
        "proposal_sampling": proposal_sampling,
        "simulator": simulator,
        "reactive_policy": reactive_policy,
        "thresholds": thresholds,
        "risk": risk,
        "recovery": recovery,
        "dyn": dyn,
        "retrieval": retrieval,
        "perturb": perturb,
        "fast_proxy": fast_proxy,
        "map_root_override": map_root_override,
    }


def _run_token_worker(token: str, run_cfg: RunConfig, seed: int) -> Dict[str, Any]:
    global _CTX
    assert _CTX is not None
    try:
        metric_cache = _CTX["loader"].get_from_token(token)
        rng = np.random.default_rng(seed)
        return _run_token_pseudo_expert(
            token=token,
            metric_cache=metric_cache,
            vocab=_CTX["vocab"],
            vocab_retrieval_index=_CTX["vocab_retrieval_index"],
            proposal_sampling=_CTX["proposal_sampling"],
            simulator=_CTX["simulator"],
            reactive_policy=_CTX["reactive_policy"],
            thresholds=_CTX["thresholds"],
            risk=_CTX["risk"],
            recovery=_CTX["recovery"],
            dyn=_CTX["dyn"],
            retrieval=_CTX["retrieval"],
            perturb=_CTX["perturb"],
            fast_proxy=_CTX["fast_proxy"],
            run_cfg=run_cfg,
            rng=rng,
            map_root_override=_CTX["map_root_override"],
        )
    except Exception as e:
        return {
            "token": token,
            "success_traces": [],
            "failure_traces": [],
            "stats": {"attempts": 0, "successes": 0, "failures_saved": 0, "worker_error": str(e), "reject_reasons": {"worker_error": 1}},
            "rejected_reason": "worker_error",
        }


def _to_dict_dataclass(obj: Any) -> Dict[str, Any]:
    return asdict(obj)


def main() -> None:
    args = parse_args()
    if not args.vocab_path.exists():
        raise FileNotFoundError(f"vocab path does not exist: {args.vocab_path}")

    cache_root = base.resolve_metric_cache_root(args.metric_cache_path)
    out_dir = args.output_dir.resolve()
    success_dir = out_dir / "success"
    failure_root = out_dir / "failure"
    success_dir.mkdir(parents=True, exist_ok=True)
    failure_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / args.manifest_name
    stats_path = out_dir / "token_stats.json"

    run_cfg = RunConfig(
        sample_batch_size=int(args.sample_batch_size),
        max_candidate_trials=int(args.max_candidate_trials),
        top_k_per_token=int(args.top_k_per_token),
        max_failure_traces_per_token=int(args.max_failure_traces_per_token),
        time_budget_min_per_token=float(args.time_budget_min_per_token),
        progress_log_interval_sec=float(args.progress_log_interval_sec),
        verbose=bool(args.verbose),
    )
    base_thresholds = _base_thresholds_from_args(args)
    risk = RiskCriteria(
        min_dist_min_m=float(args.risk_min_dist_min_m),
        min_dist_max_m=float(args.risk_min_dist_max_m),
    )
    recovery = RecoveryCriteria(
        min_displacement_m=float(args.recovery_min_displacement_m),
        min_mean_speed_mps=float(args.recovery_min_mean_speed_mps),
        min_distance_improve_m=float(args.recovery_min_distance_improve_m),
        min_stage2_distance_m=float(args.recovery_min_stage2_dist_m),
    )
    dyn = DynamicsCriteria(
        max_abs_accel_mps2=float(args.dyn_max_abs_accel_mps2),
        max_abs_steer_deg=float(args.dyn_max_abs_steer_deg),
        max_abs_yaw_rate_deg_s=float(args.dyn_max_abs_yaw_rate_deg_s),
        max_abs_jerk_mps3=float(args.dyn_max_abs_jerk_mps3),
        max_abs_steer_rate_deg_s=float(args.dyn_max_abs_steer_rate_deg_s),
        join_window_k=int(args.dyn_join_window_k),
    )
    retrieval = RetrievalCriteria(
        enabled=not bool(args.disable_conditioned_retrieval),
        lon_tol_m=float(args.retrieval_lon_tol_m),
        lat_min_m=float(args.retrieval_lat_min_m),
        lat_max_m=float(args.retrieval_lat_max_m),
        heading_tol_deg=float(args.retrieval_heading_tol_deg),
        disp_tol_m=float(args.retrieval_disp_tol_m),
        mean_speed_tol_mps=float(args.retrieval_mean_speed_tol_mps),
        end_speed_tol_mps=float(args.retrieval_end_speed_tol_mps),
        min_pool_size=int(args.retrieval_min_pool_size),
        fallback_k=int(args.retrieval_fallback_k),
    )
    perturb = PerturbationCriteria(
        scale=float(args.perturb_scale),
        adaptive=not bool(args.disable_adaptive_perturb_scale),
        min_scale=float(args.perturb_min_scale),
        max_scale=float(args.perturb_max_scale),
        down_step=float(args.perturb_down_step),
        up_step=float(args.perturb_up_step),
        adapt_every_attempts=int(args.perturb_adapt_every_attempts),
        high_stage1_fail_ratio=float(args.perturb_high_stage1_fail_ratio),
        high_not_risky_ratio=float(args.perturb_high_not_risky_ratio),
    )
    fast_proxy = FastStage1ProxyCriteria(
        enabled=not bool(args.disable_fast_stage1_proxy),
        min_center_distance_m=float(args.fast_stage1_proxy_min_center_distance_m),
    )

    vocab_probe = np.load(args.vocab_path, mmap_mode="r")
    proposal_sampling = TrajectorySampling(
        num_poses=int(round(args.horizon_sec / args.interval)),
        interval_length=args.interval,
    )
    if vocab_probe.ndim != 3 or vocab_probe.shape[-1] != 3:
        raise ValueError(f"Expected vocab shape [N, H, 3], got {vocab_probe.shape}")
    if vocab_probe.shape[1] != proposal_sampling.num_poses:
        raise ValueError(
            f"Vocab horizon mismatch: vocab H={vocab_probe.shape[1]} vs proposal num_poses={proposal_sampling.num_poses}"
        )

    loader = base.build_metric_cache_loader(cache_root)
    tokens = list(loader.tokens)
    if args.max_scenes is not None:
        tokens = tokens[: args.max_scenes]

    print(f"Loaded vocab: {args.vocab_path} shape={vocab_probe.shape}")
    print(f"Processing {len(tokens)} scene tokens from {cache_root}")
    print(
        "Conditioned retrieval: "
        f"{'on' if retrieval.enabled else 'off'} "
        f"(lon_tol={retrieval.lon_tol_m:.1f}m, lat=[{retrieval.lat_min_m:.1f},{retrieval.lat_max_m:.1f}]m, "
        f"heading_tol={retrieval.heading_tol_deg:.1f}deg)"
    )
    print(
        "Perturb scale: "
        f"init={perturb.scale:.2f}, adaptive={'on' if perturb.adaptive else 'off'}, "
        f"range=[{perturb.min_scale:.2f},{perturb.max_scale:.2f}]"
    )
    print(
        "Fast Stage1 proxy: "
        f"{'on' if fast_proxy.enabled else 'off'} "
        f"(min_center_distance={fast_proxy.min_center_distance_m:.2f}m)"
    )

    base_thresholds_dict = _to_dict_dataclass(base_thresholds)
    risk_dict = _to_dict_dataclass(risk)
    recovery_dict = _to_dict_dataclass(recovery)
    dyn_dict = _to_dict_dataclass(dyn)
    retrieval_dict = _to_dict_dataclass(retrieval)
    perturb_dict = _to_dict_dataclass(perturb)
    fast_proxy_dict = _to_dict_dataclass(fast_proxy)

    results: List[Dict[str, Any]] = []
    if args.parallel_backend == "none":
        _init_worker(
            str(cache_root),
            str(args.vocab_path),
            float(args.interval),
            float(args.horizon_sec),
            base_thresholds_dict,
            risk_dict,
            recovery_dict,
            dyn_dict,
            retrieval_dict,
            perturb_dict,
            fast_proxy_dict,
            args.map_root_override,
        )
        for token in tokens:
            seed = _token_seed(args.seed, token)
            results.append(_run_token_worker(token, run_cfg, seed))
    else:
        num_workers = _auto_workers(int(args.num_workers))
        print(f"Running process pool with workers={num_workers}")
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(
                str(cache_root),
                str(args.vocab_path),
                float(args.interval),
                float(args.horizon_sec),
                base_thresholds_dict,
                risk_dict,
                recovery_dict,
                dyn_dict,
                retrieval_dict,
                perturb_dict,
                fast_proxy_dict,
                args.map_root_override,
            ),
        ) as ex:
            futs = {
                ex.submit(_run_token_worker, token, run_cfg, _token_seed(args.seed, token)): token for token in tokens
            }
            for fut in as_completed(futs):
                results.append(fut.result())

    results.sort(key=lambda r: r["token"])

    saved_success = 0
    saved_failure = 0
    rejected_tokens = 0
    manifest_lines: List[str] = []
    token_stats: Dict[str, Any] = {}

    for result in results:
        token = str(result["token"])
        success_traces = result.get("success_traces", [])
        failure_traces = result.get("failure_traces", [])
        token_stats[token] = result.get("stats", {})

        for rank, trace in enumerate(success_traces):
            out_json, out_pkl = _save_trace(
                trace=trace,
                out_dir=success_dir,
                fmt=args.save_format,
                rank=rank,
                stem_suffix="pseudo_success",
            )
            saved_success += 1
            manifest_lines.append(
                json.dumps(
                    {
                        "token": token,
                        "rank": rank,
                        "result_status": trace.get("result_status", "success_strong"),
                        "reject_reason": trace.get("reject_reason"),
                        "search_stage": trace.get("search_stage"),
                        "criteria_snapshot": trace.get("criteria_snapshot"),
                        "vocab_index": trace.get("vocab_index"),
                        "stage1_score": trace.get("stage1_score"),
                        "stage2_score": trace.get("stage2_score"),
                        "longtail_metrics": trace.get("longtail_metrics"),
                        "dynamics_metrics": trace.get("dynamics_metrics"),
                        "json_path": str(out_json) if out_json is not None else None,
                        "pkl_path": str(out_pkl) if out_pkl is not None else None,
                    },
                    ensure_ascii=False,
                )
            )

        for rank, trace in enumerate(failure_traces):
            reason = _sanitize_fragment(str(trace.get("reject_reason", "rejected")))
            failure_dir = failure_root / reason
            failure_dir.mkdir(parents=True, exist_ok=True)
            out_json, out_pkl = _save_trace(
                trace=trace,
                out_dir=failure_dir,
                fmt=args.save_format,
                rank=rank,
                stem_suffix=f"pseudo_failure_{reason}",
            )
            saved_failure += 1
            manifest_lines.append(
                json.dumps(
                    {
                        "token": token,
                        "rank": rank,
                        "result_status": trace.get("result_status", "rejected"),
                        "reject_reason": trace.get("reject_reason"),
                        "search_stage": trace.get("search_stage"),
                        "criteria_snapshot": trace.get("criteria_snapshot"),
                        "vocab_index": trace.get("vocab_index"),
                        "stage1_score": trace.get("stage1_score"),
                        "stage2_score": trace.get("stage2_score"),
                        "longtail_metrics": trace.get("longtail_metrics"),
                        "dynamics_metrics": trace.get("dynamics_metrics"),
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
                        "reject_reason": result.get("rejected_reason", "no_candidate_passed"),
                        "search_stage": None,
                        "criteria_snapshot": None,
                        "vocab_index": None,
                        "stage1_score": None,
                        "stage2_score": None,
                        "longtail_metrics": None,
                        "dynamics_metrics": None,
                        "json_path": None,
                        "pkl_path": None,
                    },
                    ensure_ascii=False,
                )
            )

    manifest_path.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")
    stats_path.write_text(json.dumps(token_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Done. success={saved_success}, failure={saved_failure}, rejected_tokens={rejected_tokens}, "
        f"success_dir={success_dir}, failure_dir={failure_root}, manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
