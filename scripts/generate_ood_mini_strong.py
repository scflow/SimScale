#!/usr/bin/env python3
"""
Strong long-tail OOD generator.

Goal:
- Build "risky but recoverable" trajectories instead of merely "valid" ones.
- Stage-1 must enter near-risk regime.
- Stage-2 must show meaningful recovery motion.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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
from navsim.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner
from navsim.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_ego_state
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex


WorkerContext = Dict[str, Any]
_CTX: Optional[WorkerContext] = None


@dataclass
class StrongCriteria:
    # Stage-1 (risk) criteria
    risk_min_dist_min_m: float = 0.8
    risk_min_dist_max_m: float = 2.5
    # Stage-2 (recovery) criteria
    recovery_min_displacement_m: float = 8.0
    recovery_min_mean_speed_mps: float = 1.0
    recovery_min_dist_m: float = 1.2
    recovery_improve_dist_m: float = 0.3


@dataclass
class RunConfig:
    sample_batch_size: int
    max_candidate_trials: int
    top_k_per_token: int
    profile: str
    time_budget_min_per_token: float
    save_failures: bool
    max_near_miss_per_token: int
    progress_log_interval_sec: float
    verbose: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate strong long-tail OOD traces.")
    parser.add_argument("--metric-cache-path", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, default=Path("traj_final/16384.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("generated_ood_data_strong"))
    parser.add_argument("--save-format", type=str, default="both", choices=["json", "pkl", "both"])

    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--sample-batch-size", type=int, default=120)
    parser.add_argument("--max-candidate-trials", type=int, default=2500)
    parser.add_argument("--top-k-per-token", type=int, default=2)
    parser.add_argument("--profile", type=str, default="adaptive", choices=["strict", "balanced", "adaptive"])
    parser.add_argument("--time-budget-min-per-token", type=float, default=15.0)
    parser.add_argument("--save-failures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-near-miss-per-token", type=int, default=20)
    parser.add_argument("--progress-log-interval-sec", type=float, default=15.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--horizon-sec", type=float, default=4.0)

    parser.add_argument("--max-abs-lon-m", type=float, default=20.0)
    parser.add_argument("--max-abs-lat-m", type=float, default=2.0)
    parser.add_argument("--max-abs-heading-deg", type=float, default=20.0)
    parser.add_argument("--max-abs-accel-mps2", type=float, default=6.0)
    parser.add_argument("--max-abs-steer-deg", type=float, default=60.0)
    parser.add_argument("--min-progress", type=float, default=0.5)
    parser.add_argument("--max-join-translation-m", type=float, default=0.6)
    parser.add_argument("--max-join-heading-deg", type=float, default=6.0)
    parser.add_argument("--max-join-speed-delta-mps", type=float, default=1.8)

    parser.add_argument("--risk-min-dist-min-m", type=float, default=0.8)
    parser.add_argument("--risk-min-dist-max-m", type=float, default=2.5)
    parser.add_argument("--recovery-min-displacement-m", type=float, default=8.0)
    parser.add_argument("--recovery-min-mean-speed-mps", type=float, default=1.0)
    parser.add_argument("--recovery-min-dist-m", type=float, default=1.2)
    parser.add_argument("--recovery-improve-dist-m", type=float, default=0.3)

    parser.add_argument("--map-root-override", type=str, default=None)
    parser.add_argument("--manifest-name", type=str, default="manifest.jsonl")

    parser.add_argument("--parallel-backend", type=str, default="process", choices=["process", "thread", "none"])
    parser.add_argument("--num-workers", type=int, default=0)
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
    return max(1, min(c, 16))


def _token_seed(base_seed: int, token: str) -> int:
    h = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)
    return (base_seed + h) % (2**32 - 1)


def _save_trace_ranked(
    trace: Dict[str, Any], out_dir: Path, fmt: str, rank: int, stem_suffix: str
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


def _vehicle_min_dist(states: np.ndarray, tracks: List[Any]) -> float:
    d_min = float("inf")
    for idx in range(min(len(states), len(tracks))):
        ex = float(states[idx, StateIndex.X])
        ey = float(states[idx, StateIndex.Y])
        objs = tracks[idx].tracked_objects.tracked_objects
        for obj in objs:
            if str(obj.tracked_object_type).lower().endswith("vehicle"):
                d = math.hypot(float(obj.center.x) - ex, float(obj.center.y) - ey)
                if d < d_min:
                    d_min = d
    return d_min if d_min < float("inf") else float("nan")


def _ego_mean_speed(states: np.ndarray) -> float:
    v = np.hypot(states[:, StateIndex.VELOCITY_X], states[:, StateIndex.VELOCITY_Y])
    return float(np.mean(v))


def _ego_displacement(states: np.ndarray) -> float:
    if len(states) < 2:
        return 0.0
    dx = float(states[-1, StateIndex.X] - states[0, StateIndex.X])
    dy = float(states[-1, StateIndex.Y] - states[0, StateIndex.Y])
    return math.hypot(dx, dy)


def _stage1_risk_score(min_dist_stage1: float, criteria: StrongCriteria) -> float:
    if math.isnan(min_dist_stage1):
        return 0.0
    # Triangle-like score with best risk around midpoint of target interval.
    center = 0.5 * (criteria.risk_min_dist_min_m + criteria.risk_min_dist_max_m)
    width = max(1e-6, 0.5 * (criteria.risk_min_dist_max_m - criteria.risk_min_dist_min_m))
    return max(0.0, 1.0 - abs(min_dist_stage1 - center) / width)


def _build_expert_variants(proposal_sampling: TrajectorySampling) -> List[Tuple[str, PDMClosedPlanner]]:
    future_poses = proposal_sampling.num_poses + int(1.0 / proposal_sampling.interval_length)
    trajectory_sampling = TrajectorySampling(
        num_poses=future_poses, interval_length=proposal_sampling.interval_length
    )

    variants: List[Tuple[str, PDMClosedPlanner]] = []
    variants.append(("default", base.build_expert_planner(proposal_sampling=proposal_sampling)))
    variants.append(
        (
            "wide",
            PDMClosedPlanner(
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
                lateral_offsets=[-2.0, -1.0, 1.0, 2.0],
                map_radius=140.0,
            ),
        )
    )
    variants.append(
        (
            "progressive",
            PDMClosedPlanner(
                trajectory_sampling=trajectory_sampling,
                proposal_sampling=proposal_sampling,
                idm_policies=BatchIDMPolicy(
                    speed_limit_fraction=[0.4, 0.7, 1.0, 1.2],
                    fallback_target_velocity=18.0,
                    min_gap_to_lead_agent=0.8,
                    headway_time=1.2,
                    accel_max=2.0,
                    decel_max=3.5,
                ),
                lateral_offsets=[-1.0, 0.5, 1.0],
                map_radius=120.0,
            ),
        )
    )
    return variants


def _satisfy_strong_criteria(
    stage1_min_dist: float,
    stage2_min_dist: float,
    stage2_disp: float,
    stage2_mean_speed: float,
    criteria: StrongCriteria,
) -> bool:
    if math.isnan(stage1_min_dist) or math.isnan(stage2_min_dist):
        return False
    return (
        criteria.risk_min_dist_min_m <= stage1_min_dist <= criteria.risk_min_dist_max_m
        and stage2_disp >= criteria.recovery_min_displacement_m
        and stage2_mean_speed >= criteria.recovery_min_mean_speed_mps
        and stage2_min_dist >= criteria.recovery_min_dist_m
        and (stage2_min_dist - stage1_min_dist) >= criteria.recovery_improve_dist_m
    )


def _strong_longtail_score(
    stage1_min_dist: float,
    stage2_min_dist: float,
    stage2_disp: float,
    stage2_mean_speed: float,
    join_translation: float,
    join_heading_deg: float,
    join_speed_delta: float,
    stage2_pdm: float,
    criteria: StrongCriteria,
) -> float:
    risk = _stage1_risk_score(stage1_min_dist, criteria)
    recovery = 0.6 * min(1.0, stage2_disp / max(1e-6, criteria.recovery_min_displacement_m * 1.2)) + 0.4 * min(
        1.0, stage2_mean_speed / max(1e-6, criteria.recovery_min_mean_speed_mps * 1.8)
    )
    safety_gain = max(0.0, min(1.0, (stage2_min_dist - stage1_min_dist + 1.0) / 2.0))
    continuity = max(
        0.0,
        1.0
        - (
            0.45 * min(1.0, join_translation / 1.0)
            + 0.35 * min(1.0, join_heading_deg / 10.0)
            + 0.20 * min(1.0, join_speed_delta / 2.0)
        ),
    )
    return float((0.4 * risk + 0.35 * recovery + 0.15 * safety_gain + 0.10 * continuity) * stage2_pdm)


def _criteria_variants(base_criteria: StrongCriteria) -> Dict[str, StrongCriteria]:
    strict = copy.deepcopy(base_criteria)
    balanced = StrongCriteria(
        risk_min_dist_min_m=max(0.5, strict.risk_min_dist_min_m - 0.2),
        risk_min_dist_max_m=strict.risk_min_dist_max_m + 0.8,
        recovery_min_displacement_m=max(3.0, strict.recovery_min_displacement_m * 0.6),
        recovery_min_mean_speed_mps=max(0.25, strict.recovery_min_mean_speed_mps * 0.6),
        recovery_min_dist_m=max(0.8, strict.recovery_min_dist_m * 0.9),
        recovery_improve_dist_m=max(0.1, strict.recovery_improve_dist_m * 0.5),
    )
    near_miss = StrongCriteria(
        risk_min_dist_min_m=max(0.5, strict.risk_min_dist_min_m - 0.3),
        risk_min_dist_max_m=strict.risk_min_dist_max_m + 1.2,
        recovery_min_displacement_m=max(1.5, strict.recovery_min_displacement_m * 0.35),
        recovery_min_mean_speed_mps=max(0.1, strict.recovery_min_mean_speed_mps * 0.35),
        recovery_min_dist_m=max(0.8, strict.recovery_min_dist_m * 0.8),
        recovery_improve_dist_m=max(0.0, strict.recovery_improve_dist_m * 0.25),
    )
    return {"strict": strict, "balanced": balanced, "near_miss": near_miss}


def _select_search_stage(profile: str, elapsed_ratio: float, success_count: int) -> str:
    if profile == "strict":
        return "strict"
    if profile == "balanced":
        return "balanced"
    if elapsed_ratio < 0.5 and success_count == 0:
        return "strict"
    if elapsed_ratio < 0.85:
        return "balanced"
    return "near_miss"


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
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out or "unknown"


def _reason_count(stats: Dict[str, Any], reason: str) -> None:
    reasons = stats.setdefault("reject_reasons", {})
    reasons[reason] = int(reasons.get(reason, 0)) + 1


def _dominant_reject_reason(stats: Dict[str, Any]) -> str:
    reasons = stats.get("reject_reasons", {})
    if not reasons:
        return "no_candidate_passed"
    return max(reasons.items(), key=lambda kv: kv[1])[0]


def _strong_failure_reason(
    stage1_min_dist: float,
    stage2_min_dist: float,
    stage2_disp: float,
    stage2_mean_speed: float,
    criteria: StrongCriteria,
) -> str:
    if math.isnan(stage1_min_dist) or math.isnan(stage2_min_dist):
        return "strong_fail_nan_distance"
    if stage2_disp < criteria.recovery_min_displacement_m:
        return "strong_fail_recovery_disp"
    if stage2_mean_speed < criteria.recovery_min_mean_speed_mps:
        return "strong_fail_recovery_speed"
    if stage2_min_dist < criteria.recovery_min_dist_m:
        return "strong_fail_recovery_min_dist"
    if (stage2_min_dist - stage1_min_dist) < criteria.recovery_improve_dist_m:
        return "strong_fail_recovery_improve_dist"
    return "strong_fail_other"


def _annotate_trace(
    trace: Dict[str, Any],
    status: str,
    reject_reason: Optional[str],
    search_stage: str,
    criteria: StrongCriteria,
) -> Dict[str, Any]:
    trace["result_status"] = status
    trace["reject_reason"] = reject_reason
    trace["search_stage"] = search_stage
    trace["criteria_snapshot"] = asdict(criteria)
    return trace


def _stage1_only_trace(
    token: str,
    vocab_index: int,
    phase1_states: np.ndarray,
    phase1_tracks: List[Any],
    stage1_score: Dict[str, float],
) -> Dict[str, Any]:
    return base.serialize_trace(
        token=token,
        vocab_index=vocab_index,
        ego_states=phase1_states,
        tracks=phase1_tracks,
        stage1_score=stage1_score,
        stage2_score=_nan_score(),
    )


def _push_top(pool: List[Dict[str, Any]], trace: Dict[str, Any], score_key: str, max_size: int) -> None:
    pool.append(trace)
    pool.sort(key=lambda x: float(x.get(score_key, 0.0)), reverse=True)
    if len(pool) > max_size:
        pool.pop()


def run_for_token_strong(
    token: str,
    metric_cache: Any,
    vocab: np.ndarray,
    proposal_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    reactive_policy: Any,
    thresholds: base.Thresholds,
    criteria: StrongCriteria,
    run_cfg: RunConfig,
    rng: np.random.Generator,
    map_root_override: Optional[str],
) -> Dict[str, Any]:
    stats = {
        "attempts": 0,
        "geo_fail": 0,
        "physics_fail": 0,
        "stage1_fail": 0,
        "stage1_not_risky": 0,
        "stage2_fail": 0,
        "join_fail": 0,
        "strong_fail": 0,
        "successes": 0,
        "near_misses": 0,
        "stage_counts": {"strict": 0, "balanced": 0, "near_miss": 0},
        "reject_reasons": {},
    }

    if metric_cache.human_trajectory is None:
        _reason_count(stats, "missing_human_trajectory")
        return {
            "token": token,
            "success_traces": [],
            "near_miss_traces": [],
            "stats": stats,
            "rejected_reason": "missing_human_trajectory",
        }

    if map_root_override is not None:
        metric_cache.map_parameters.map_root = map_root_override

    scorer = PDMScorer(proposal_sampling=proposal_sampling, config=PDMScorerConfig(human_penalty_filter=False))
    planner_init = base.build_planner_initialization(metric_cache, map_root_override=map_root_override)
    expert_variants = _build_expert_variants(proposal_sampling)
    vocab_size = len(vocab)
    criteria_map = _criteria_variants(criteria)
    success_pool: List[Dict[str, Any]] = []
    near_miss_pool: List[Dict[str, Any]] = []
    start_t = time.monotonic()
    budget_sec = max(1.0, float(run_cfg.time_budget_min_per_token) * 60.0)
    next_log_t = start_t + max(1.0, run_cfg.progress_log_interval_sec)

    while stats["attempts"] < run_cfg.max_candidate_trials:
        now = time.monotonic()
        elapsed = now - start_t
        if elapsed >= budget_sec:
            break

        elapsed_ratio = min(1.0, elapsed / budget_sec)
        search_stage = _select_search_stage(run_cfg.profile, elapsed_ratio, len(success_pool))
        active_criteria = criteria_map[search_stage]
        stats["stage_counts"][search_stage] = int(stats["stage_counts"].get(search_stage, 0)) + 1

        remain = run_cfg.max_candidate_trials - stats["attempts"]
        batch = min(run_cfg.sample_batch_size, remain)
        sampled = rng.integers(0, vocab_size, size=batch)

        for vocab_idx in sampled:
            if stats["attempts"] >= run_cfg.max_candidate_trials:
                break
            if time.monotonic() - start_t >= budget_sec:
                break

            stats["attempts"] += 1
            candidate_rel = vocab[vocab_idx]
            if not base.pass_geometric_filter(metric_cache, candidate_rel, thresholds):
                stats["geo_fail"] += 1
                _reason_count(stats, "geo_fail")
                continue

            candidate_states = base.make_trajectory_states_from_local_poses(
                local_poses=candidate_rel,
                metric_cache=metric_cache,
                proposal_sampling=proposal_sampling,
            )
            phase1_states = simulator.simulate_proposals(candidate_states[None, ...], metric_cache.ego_state)[0]
            if not base.pass_physics_filter(phase1_states, thresholds):
                stats["physics_fail"] += 1
                _reason_count(stats, "physics_fail")
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
                _reason_count(stats, "stage1_fail")
                if run_cfg.save_failures and len(near_miss_pool) < run_cfg.max_near_miss_per_token:
                    s1_trace = _stage1_only_trace(
                        token=token,
                        vocab_index=int(vocab_idx),
                        phase1_states=phase1_states,
                        phase1_tracks=phase1_tracks,
                        stage1_score=stage1_score,
                    )
                    s1_trace["strong_longtail"] = {
                        "stage1_min_dist_m": float(_vehicle_min_dist(phase1_states, phase1_tracks)),
                        "strong_longtail_score": float(stage1_score.get("pdm_score", 0.0)),
                    }
                    s1_trace["_rank_score"] = float(stage1_score.get("pdm_score", 0.0))
                    _annotate_trace(
                        s1_trace,
                        status="near_miss",
                        reject_reason="stage1_fail",
                        search_stage=search_stage,
                        criteria=active_criteria,
                    )
                    _push_top(near_miss_pool, s1_trace, "_rank_score", run_cfg.max_near_miss_per_token)
                    stats["near_misses"] += 1
                continue

            min_dist_stage1 = _vehicle_min_dist(phase1_states, phase1_tracks)
            if not (active_criteria.risk_min_dist_min_m <= min_dist_stage1 <= active_criteria.risk_min_dist_max_m):
                stats["stage1_not_risky"] += 1
                _reason_count(stats, "stage1_not_risky")
                if run_cfg.save_failures and len(near_miss_pool) < run_cfg.max_near_miss_per_token:
                    s1_trace = _stage1_only_trace(
                        token=token,
                        vocab_index=int(vocab_idx),
                        phase1_states=phase1_states,
                        phase1_tracks=phase1_tracks,
                        stage1_score=stage1_score,
                    )
                    risk_score = _stage1_risk_score(float(min_dist_stage1), active_criteria)
                    s1_trace["strong_longtail"] = {
                        "stage1_min_dist_m": float(min_dist_stage1),
                        "risk_score": float(risk_score),
                        "strong_longtail_score": float(risk_score * stage1_score.get("pdm_score", 0.0)),
                    }
                    s1_trace["_rank_score"] = float(risk_score * stage1_score.get("pdm_score", 0.0))
                    _annotate_trace(
                        s1_trace,
                        status="near_miss",
                        reject_reason="stage1_not_risky",
                        search_stage=search_stage,
                        criteria=active_criteria,
                    )
                    _push_top(near_miss_pool, s1_trace, "_rank_score", run_cfg.max_near_miss_per_token)
                    stats["near_misses"] += 1
                continue

            ood_time = metric_cache.ego_state.time_point + TimeDuration.from_s(proposal_sampling.time_horizon)
            ood_ego_state = state_array_to_ego_state(
                phase1_states[-1],
                TimePoint(int(ood_time.time_us)),
                metric_cache.ego_state.car_footprint.vehicle_parameters,
            )
            ood_obs = phase1_tracks[-1]

            best_trace: Optional[Dict[str, Any]] = None
            best_score = -1.0
            best_near_miss: Optional[Dict[str, Any]] = None
            best_near_miss_score = -1.0

            for variant_name, expert_planner in expert_variants:
                expert_planner.initialize(planner_init)
                planner_input = PlannerInput(
                    iteration=SimulationIteration(index=0, time_point=ood_ego_state.time_point),
                    history=SimulationHistoryBuffer.initialize_from_list(
                        buffer_size=1, ego_states=[ood_ego_state], observations=[ood_obs]
                    ),
                    traffic_light_data=[],
                )
                rescue_traj = expert_planner.compute_planner_trajectory(planner_input)
                rescue_states = base.get_trajectory_as_array(
                    rescue_traj, proposal_sampling, start_time=ood_ego_state.time_point
                )
                if not base.pass_join_continuity_gate(phase1_states, rescue_states, thresholds):
                    stats["join_fail"] += 1
                    _reason_count(stats, "join_fail")
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

                min_dist_stage2 = _vehicle_min_dist(rescue_states, phase2_tracks)
                stage2_disp = _ego_displacement(rescue_states)
                stage2_mean_speed = _ego_mean_speed(rescue_states)
                join_translation = math.hypot(
                    float(rescue_states[1, StateIndex.X] - phase1_states[-1, StateIndex.X]),
                    float(rescue_states[1, StateIndex.Y] - phase1_states[-1, StateIndex.Y]),
                )
                join_heading_deg = abs(
                    math.degrees(
                        base.normalize_angle(float(rescue_states[1, StateIndex.HEADING] - phase1_states[-1, StateIndex.HEADING]))
                    )
                )
                prev_speed = math.hypot(
                    float(phase1_states[-1, StateIndex.VELOCITY_X]),
                    float(phase1_states[-1, StateIndex.VELOCITY_Y]),
                )
                next_speed = math.hypot(
                    float(rescue_states[1, StateIndex.VELOCITY_X]),
                    float(rescue_states[1, StateIndex.VELOCITY_Y]),
                )
                join_speed_delta = abs(next_speed - prev_speed)
                strong_score = _strong_longtail_score(
                    stage1_min_dist=float(min_dist_stage1),
                    stage2_min_dist=float(min_dist_stage2),
                    stage2_disp=float(stage2_disp),
                    stage2_mean_speed=float(stage2_mean_speed),
                    join_translation=float(join_translation),
                    join_heading_deg=float(join_heading_deg),
                    join_speed_delta=float(join_speed_delta),
                    stage2_pdm=float(stage2_score.get("pdm_score", 0.0)),
                    criteria=active_criteria,
                )

                full_ego_states = np.concatenate([phase1_states, rescue_states[1:]], axis=0)
                full_tracks = phase1_tracks + phase2_tracks[1:]
                trace = base.serialize_trace(
                    token=token,
                    vocab_index=int(vocab_idx),
                    ego_states=full_ego_states,
                    tracks=full_tracks,
                    stage1_score=stage1_score,
                    stage2_score=stage2_score,
                )
                trace["strong_longtail"] = {
                    "expert_variant": variant_name,
                    "stage1_min_dist_m": float(min_dist_stage1),
                    "stage2_min_dist_m": float(min_dist_stage2),
                    "stage2_displacement_m": float(stage2_disp),
                    "stage2_mean_speed_mps": float(stage2_mean_speed),
                    "join_translation_m": float(join_translation),
                    "join_heading_deg": float(join_heading_deg),
                    "join_speed_delta_mps": float(join_speed_delta),
                    "strong_longtail_score": float(strong_score),
                }
                trace["_rank_score"] = float(strong_score)

                if not base.pass_strict_gate(stage2_score, thresholds):
                    stats["stage2_fail"] += 1
                    _reason_count(stats, "stage2_fail")
                    if strong_score > best_near_miss_score:
                        near = copy.deepcopy(trace)
                        _annotate_trace(near, "near_miss", "stage2_fail", search_stage, active_criteria)
                        best_near_miss = near
                        best_near_miss_score = strong_score
                    continue

                if not _satisfy_strong_criteria(
                    stage1_min_dist=float(min_dist_stage1),
                    stage2_min_dist=float(min_dist_stage2),
                    stage2_disp=float(stage2_disp),
                    stage2_mean_speed=float(stage2_mean_speed),
                    criteria=active_criteria,
                ):
                    stats["strong_fail"] += 1
                    reason = _strong_failure_reason(
                        stage1_min_dist=float(min_dist_stage1),
                        stage2_min_dist=float(min_dist_stage2),
                        stage2_disp=float(stage2_disp),
                        stage2_mean_speed=float(stage2_mean_speed),
                        criteria=active_criteria,
                    )
                    _reason_count(stats, reason)
                    if strong_score > best_near_miss_score:
                        near = copy.deepcopy(trace)
                        _annotate_trace(near, "near_miss", reason, search_stage, active_criteria)
                        best_near_miss = near
                        best_near_miss_score = strong_score
                    continue

                if search_stage == "near_miss":
                    _reason_count(stats, "near_miss_collection_window")
                    if strong_score > best_near_miss_score:
                        near = copy.deepcopy(trace)
                        _annotate_trace(near, "near_miss", "near_miss_collection_window", search_stage, active_criteria)
                        best_near_miss = near
                        best_near_miss_score = strong_score
                    continue

                if strong_score > best_score:
                    ok = copy.deepcopy(trace)
                    _annotate_trace(ok, "success", None, search_stage, active_criteria)
                    best_trace = ok
                    best_score = strong_score

            if best_trace is not None:
                stats["successes"] += 1
                _push_top(success_pool, best_trace, "_rank_score", run_cfg.top_k_per_token)
            elif run_cfg.save_failures and best_near_miss is not None:
                stats["near_misses"] += 1
                _push_top(near_miss_pool, best_near_miss, "_rank_score", run_cfg.max_near_miss_per_token)

        if time.monotonic() >= next_log_t:
            elapsed = time.monotonic() - start_t
            pct = 100.0 * min(1.0, elapsed / budget_sec)
            print(
                f"[progress][{token}] t={elapsed:.1f}s/{budget_sec:.1f}s ({pct:.1f}%) "
                f"attempts={stats['attempts']} success={len(success_pool)} near_miss={len(near_miss_pool)}"
            )
            next_log_t = time.monotonic() + max(1.0, run_cfg.progress_log_interval_sec)

    stats["elapsed_sec"] = float(time.monotonic() - start_t)
    for tr in success_pool:
        tr.pop("_rank_score", None)
    for tr in near_miss_pool:
        tr.pop("_rank_score", None)
    rejected_reason = _dominant_reject_reason(stats) if not success_pool and not near_miss_pool else None
    return {
        "token": token,
        "success_traces": success_pool,
        "near_miss_traces": near_miss_pool if run_cfg.save_failures else [],
        "stats": stats,
        "rejected_reason": rejected_reason,
    }


def _init_worker(
    metric_cache_path: str,
    vocab_path: str,
    interval: float,
    horizon_sec: float,
    thresholds_dict: Dict[str, float],
    criteria_dict: Dict[str, float],
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
    thresholds = base.Thresholds(**thresholds_dict)
    criteria = StrongCriteria(**criteria_dict)
    _CTX = {
        "loader": loader,
        "vocab": vocab,
        "proposal_sampling": proposal_sampling,
        "simulator": simulator,
        "reactive_policy": reactive_policy,
        "thresholds": thresholds,
        "criteria": criteria,
        "map_root_override": map_root_override,
    }


def _run_token_worker(token: str, run_cfg: RunConfig, seed: int) -> Dict[str, Any]:
    global _CTX
    assert _CTX is not None
    metric_cache = _CTX["loader"].get_from_token(token)
    rng = np.random.default_rng(seed)
    return run_for_token_strong(
        token=token,
        metric_cache=metric_cache,
        vocab=_CTX["vocab"],
        proposal_sampling=_CTX["proposal_sampling"],
        simulator=_CTX["simulator"],
        reactive_policy=_CTX["reactive_policy"],
        thresholds=_CTX["thresholds"],
        criteria=_CTX["criteria"],
        run_cfg=run_cfg,
        rng=rng,
        map_root_override=_CTX["map_root_override"],
    )


def _thresholds_from_args(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "max_abs_lon_m": float(args.max_abs_lon_m),
        "max_abs_lat_m": float(args.max_abs_lat_m),
        "max_abs_heading_deg": float(args.max_abs_heading_deg),
        "max_abs_accel_mps2": float(args.max_abs_accel_mps2),
        "max_abs_steer_deg": float(args.max_abs_steer_deg),
        "min_progress": float(args.min_progress),
        "max_join_translation_m": float(args.max_join_translation_m),
        "max_join_heading_deg": float(args.max_join_heading_deg),
        "max_join_speed_delta_mps": float(args.max_join_speed_delta_mps),
    }


def _criteria_from_args(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "risk_min_dist_min_m": float(args.risk_min_dist_min_m),
        "risk_min_dist_max_m": float(args.risk_min_dist_max_m),
        "recovery_min_displacement_m": float(args.recovery_min_displacement_m),
        "recovery_min_mean_speed_mps": float(args.recovery_min_mean_speed_mps),
        "recovery_min_dist_m": float(args.recovery_min_dist_m),
        "recovery_improve_dist_m": float(args.recovery_improve_dist_m),
    }


def _run_thread_pool(tokens: List[str], run_cfg: RunConfig, num_workers: int, seed: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        future_map = {
            ex.submit(_run_token_worker, token, run_cfg, _token_seed(seed, token)): token for token in tokens
        }
        for fut in as_completed(future_map):
            results.append(fut.result())
    return results


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
        sample_batch_size=args.sample_batch_size,
        max_candidate_trials=args.max_candidate_trials,
        top_k_per_token=args.top_k_per_token,
        profile=args.profile,
        time_budget_min_per_token=args.time_budget_min_per_token,
        save_failures=bool(args.save_failures),
        max_near_miss_per_token=args.max_near_miss_per_token,
        progress_log_interval_sec=args.progress_log_interval_sec,
        verbose=args.verbose,
    )

    thresholds_dict = _thresholds_from_args(args)
    criteria_dict = _criteria_from_args(args)

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

    num_workers = _auto_workers(args.num_workers)
    results: List[Dict[str, Any]] = []

    if args.parallel_backend == "none":
        _init_worker(
            str(cache_root),
            str(args.vocab_path),
            args.interval,
            args.horizon_sec,
            thresholds_dict,
            criteria_dict,
            args.map_root_override,
        )
        for token in tokens:
            results.append(_run_token_worker(token, run_cfg, _token_seed(args.seed, token)))
    elif args.parallel_backend == "thread":
        _init_worker(
            str(cache_root),
            str(args.vocab_path),
            args.interval,
            args.horizon_sec,
            thresholds_dict,
            criteria_dict,
            args.map_root_override,
        )
        print(f"Running threaded with workers={num_workers}")
        results = _run_thread_pool(tokens, run_cfg, num_workers, args.seed)
    else:
        print(f"Running process pool with workers={num_workers}")
        try:
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_init_worker,
                initargs=(
                    str(cache_root),
                    str(args.vocab_path),
                    args.interval,
                    args.horizon_sec,
                    thresholds_dict,
                    criteria_dict,
                    args.map_root_override,
                ),
            ) as ex:
                future_map = {
                    ex.submit(_run_token_worker, token, run_cfg, _token_seed(args.seed, token)): token for token in tokens
                }
                for fut in as_completed(future_map):
                    results.append(fut.result())
        except (PermissionError, OSError) as e:
            print(f"[warn] process backend unavailable ({e}); fallback to thread backend.")
            _init_worker(
                str(cache_root),
                str(args.vocab_path),
                args.interval,
                args.horizon_sec,
                thresholds_dict,
                criteria_dict,
                args.map_root_override,
            )
            results = _run_thread_pool(tokens, run_cfg, num_workers, args.seed)

    results.sort(key=lambda r: r["token"])

    saved_success = 0
    saved_near_miss = 0
    rejected_tokens = 0
    manifest_lines: List[str] = []
    token_stats: Dict[str, Any] = {}

    for result in results:
        token = result["token"]
        success_traces = result.get("success_traces", [])
        near_miss_traces = result.get("near_miss_traces", [])
        token_stats[token] = result["stats"]

        for rank, trace in enumerate(success_traces):
            out_json, out_pkl = _save_trace_ranked(
                trace=trace,
                out_dir=success_dir,
                fmt=args.save_format,
                rank=rank,
                stem_suffix="strong_success",
            )
            saved_success += 1
            manifest_lines.append(
                json.dumps(
                    {
                        "token": token,
                        "rank": rank,
                        "result_status": trace.get("result_status", "success"),
                        "reject_reason": trace.get("reject_reason"),
                        "search_stage": trace.get("search_stage"),
                        "criteria_snapshot": trace.get("criteria_snapshot"),
                        "vocab_index": trace["vocab_index"],
                        "stage1_score": trace["stage1_score"],
                        "stage2_score": trace["stage2_score"],
                        "strong_longtail": trace["strong_longtail"],
                        "strong_longtail_score": trace["strong_longtail"].get("strong_longtail_score"),
                        "json_path": str(out_json) if out_json is not None else None,
                        "pkl_path": str(out_pkl) if out_pkl is not None else None,
                    },
                    ensure_ascii=False,
                )
            )

        for rank, trace in enumerate(near_miss_traces):
            reason = _sanitize_fragment(str(trace.get("reject_reason", "near_miss")))
            failure_dir = failure_root / reason
            failure_dir.mkdir(parents=True, exist_ok=True)
            out_json, out_pkl = _save_trace_ranked(
                trace=trace,
                out_dir=failure_dir,
                fmt=args.save_format,
                rank=rank,
                stem_suffix=f"strong_near_miss_{reason}",
            )
            saved_near_miss += 1
            manifest_lines.append(
                json.dumps(
                    {
                        "token": token,
                        "rank": rank,
                        "result_status": trace.get("result_status", "near_miss"),
                        "reject_reason": trace.get("reject_reason"),
                        "search_stage": trace.get("search_stage"),
                        "criteria_snapshot": trace.get("criteria_snapshot"),
                        "vocab_index": trace["vocab_index"],
                        "stage1_score": trace["stage1_score"],
                        "stage2_score": trace["stage2_score"],
                        "strong_longtail": trace.get("strong_longtail", {}),
                        "strong_longtail_score": trace.get("strong_longtail", {}).get("strong_longtail_score"),
                        "json_path": str(out_json) if out_json is not None else None,
                        "pkl_path": str(out_pkl) if out_pkl is not None else None,
                    },
                    ensure_ascii=False,
                )
            )

        if not success_traces and not near_miss_traces:
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
                        "strong_longtail": None,
                        "strong_longtail_score": None,
                        "json_path": None,
                        "pkl_path": None,
                    },
                    ensure_ascii=False,
                )
            )

    manifest_path.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")
    stats_path.write_text(json.dumps(token_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"Done. success={saved_success}, near_miss={saved_near_miss}, rejected_tokens={rejected_tokens}, "
        f"success_dir={success_dir}, failure_dir={failure_root}, manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
