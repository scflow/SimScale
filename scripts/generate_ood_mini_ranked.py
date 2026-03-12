#!/usr/bin/env python3
"""
Ranked OOD generator with optional parallelism.

Compared with scripts/generate_ood_mini.py:
- keeps top-k successful traces per token (soft quality ranking)
- writes manifest + per-token stats for analysis
- supports process/thread/serial execution across tokens
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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


WorkerContext = Dict[str, Any]
_CTX: Optional[WorkerContext] = None


@dataclass
class RunConfig:
    sample_batch_size: int
    max_candidate_trials: int
    top_k_per_token: int
    target_success_per_token: int
    emit_near_miss: bool
    max_near_miss_per_token: int
    min_attempts_before_early_stop: int
    verbose: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ranked OOD traces on NAVSIM mini.")
    parser.add_argument("--metric-cache-path", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, default=Path("traj_final/16384.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("generated_ood_data_ranked"))
    parser.add_argument("--save-format", type=str, default="both", choices=["json", "pkl", "both"])

    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--sample-batch-size", type=int, default=100)
    parser.add_argument("--max-candidate-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--horizon-sec", type=float, default=4.0)

    parser.add_argument("--max-abs-lon-m", type=float, default=20.0)
    parser.add_argument("--max-abs-lat-m", type=float, default=2.0)
    parser.add_argument("--max-abs-heading-deg", type=float, default=20.0)
    parser.add_argument("--max-abs-accel-mps2", type=float, default=6.0)
    parser.add_argument("--max-abs-steer-deg", type=float, default=60.0)
    parser.add_argument("--min-progress", type=float, default=0.5)
    parser.add_argument("--max-join-translation-m", type=float, default=1.0)
    parser.add_argument("--max-join-heading-deg", type=float, default=8.0)
    parser.add_argument("--max-join-speed-delta-mps", type=float, default=2.0)

    parser.add_argument("--map-root-override", type=str, default=None)

    parser.add_argument("--top-k-per-token", type=int, default=1, help="Number of best successful traces to keep per token.")
    parser.add_argument(
        "--target-success-per-token",
        type=int,
        default=3,
        help="Early-stop token search after collecting this many successes (with min attempts guard).",
    )
    parser.add_argument("--emit-near-miss", action="store_true", help="Emit near-miss traces to failure folders.")
    parser.add_argument("--max-near-miss-per-token", type=int, default=10)
    parser.add_argument("--min-attempts-before-early-stop", type=int, default=200)

    parser.add_argument("--parallel-backend", type=str, default="process", choices=["process", "thread", "none"])
    parser.add_argument("--num-workers", type=int, default=0, help="0 means auto.")
    parser.add_argument("--manifest-name", type=str, default="manifest.jsonl")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _auto_workers(num_workers: int) -> int:
    if num_workers > 0:
        return num_workers
    cpu = (Path("/proc/cpuinfo").exists() and 4) or None
    # portable fallback
    try:
        import os

        cpu = os.cpu_count() or 1
    except Exception:
        cpu = 1
    return max(1, min(cpu, 16))


def _token_seed(base_seed: int, token: str) -> int:
    h = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)
    return (base_seed + h) % (2**32 - 1)


def _save_ranked_trace(
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


def _nearest_distance(frame: Dict[str, Any]) -> float:
    ex, ey = frame["ego"]["x"], frame["ego"]["y"]
    vehicles = frame.get("vehicles", [])
    if not vehicles:
        return float("nan")
    return min(math.hypot(v["x"] - ex, v["y"] - ey) for v in vehicles)


def compute_longtail_metrics(trace: Dict[str, Any]) -> Dict[str, float]:
    frames = trace["frames"]
    n = len(frames)
    split = (n - 1) // 2

    dists = [_nearest_distance(f) for f in frames]
    d1 = [d for d in dists[: split + 1] if not math.isnan(d)]
    d2 = [d for d in dists[split:] if not math.isnan(d)]
    min_s1 = min(d1) if d1 else float("nan")
    min_s2 = min(d2) if d2 else float("nan")

    e_split = frames[split]["ego"]
    e_next = frames[min(split + 1, n - 1)]["ego"]
    e_end = frames[-1]["ego"]
    join_translation = math.hypot(e_next["x"] - e_split["x"], e_next["y"] - e_split["y"])
    dh = (e_next["heading"] - e_split["heading"] + math.pi) % (2 * math.pi) - math.pi
    join_heading_deg = abs(math.degrees(dh))
    join_speed_delta = abs(e_next["velocity"] - e_split["velocity"])

    stage2_disp = math.hypot(e_end["x"] - e_split["x"], e_end["y"] - e_split["y"])
    stage2_mean_speed = float(np.mean([f["ego"]["velocity"] for f in frames[split:]]))

    risk_score = 0.0 if math.isnan(min_s1) else max(0.0, min(1.0, (6.0 - min_s1) / 6.0))
    recovery_score = 0.6 * max(0.0, min(1.0, stage2_disp / 8.0)) + 0.4 * max(
        0.0, min(1.0, stage2_mean_speed / 2.0)
    )
    continuity_penalty = (
        0.45 * min(1.0, join_translation / 2.0)
        + 0.35 * min(1.0, join_heading_deg / 20.0)
        + 0.20 * min(1.0, join_speed_delta / 3.0)
    )
    continuity_score = max(0.0, 1.0 - continuity_penalty)

    stage2_pdm = float(trace["stage2_score"]["pdm_score"])
    longtail_score = (0.5 * risk_score + 0.3 * recovery_score + 0.2 * continuity_score) * stage2_pdm

    return {
        "split_idx": float(split),
        "min_dist_stage1_m": float(min_s1),
        "min_dist_stage2_m": float(min_s2),
        "join_translation_m": float(join_translation),
        "join_heading_deg": float(join_heading_deg),
        "join_speed_delta_mps": float(join_speed_delta),
        "stage2_displacement_m": float(stage2_disp),
        "stage2_mean_speed_mps": float(stage2_mean_speed),
        "risk_score": float(risk_score),
        "recovery_score": float(recovery_score),
        "continuity_score": float(continuity_score),
        "longtail_score": float(longtail_score),
    }


def _nan_score() -> Dict[str, float]:
    return {
        "no_at_fault_collisions": float("nan"),
        "drivable_area_compliance": float("nan"),
        "driving_direction_compliance": float("nan"),
        "traffic_light_compliance": float("nan"),
        "ego_progress": float("nan"),
        "pdm_score": float("nan"),
    }


def _annotate_trace(
    trace: Dict[str, Any],
    status: str,
    reject_reason: Optional[str],
    criteria_snapshot: Dict[str, float],
) -> Dict[str, Any]:
    trace["result_status"] = status
    trace["reject_reason"] = reject_reason
    trace["search_stage"] = "ranked"
    trace["criteria_snapshot"] = criteria_snapshot
    return trace


def _sanitize_fragment(text: str) -> str:
    chars = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            chars.append(ch)
        else:
            chars.append("_")
    out = "".join(chars).strip("_")
    return out or "unknown"


def run_for_token_ranked(
    token: str,
    metric_cache: Any,
    vocab: np.ndarray,
    proposal_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    reactive_policy: Any,
    thresholds: base.Thresholds,
    run_cfg: RunConfig,
    rng: np.random.Generator,
    map_root_override: Optional[str],
) -> Dict[str, Any]:
    stats = {
        "attempts": 0,
        "geo_fail": 0,
        "physics_fail": 0,
        "stage1_fail": 0,
        "join_fail": 0,
        "stage2_fail": 0,
        "successes": 0,
        "near_misses": 0,
        "reject_reasons": {},
    }

    if metric_cache.human_trajectory is None:
        stats["reject_reasons"]["missing_human_trajectory"] = 1
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
    expert_planner = base.build_expert_planner(proposal_sampling=proposal_sampling)

    vocab_size = len(vocab)
    candidates: List[Dict[str, Any]] = []
    near_miss_candidates: List[Dict[str, Any]] = []
    criteria_snapshot = {
        "max_abs_lon_m": float(thresholds.max_abs_lon_m),
        "max_abs_lat_m": float(thresholds.max_abs_lat_m),
        "max_abs_heading_deg": float(thresholds.max_abs_heading_deg),
        "max_abs_accel_mps2": float(thresholds.max_abs_accel_mps2),
        "max_abs_steer_deg": float(thresholds.max_abs_steer_deg),
        "min_progress": float(thresholds.min_progress),
        "max_join_translation_m": float(thresholds.max_join_translation_m),
        "max_join_heading_deg": float(thresholds.max_join_heading_deg),
        "max_join_speed_delta_mps": float(thresholds.max_join_speed_delta_mps),
    }

    def _inc_reason(reason: str) -> None:
        stats["reject_reasons"][reason] = int(stats["reject_reasons"].get(reason, 0)) + 1

    while stats["attempts"] < run_cfg.max_candidate_trials:
        remain = run_cfg.max_candidate_trials - stats["attempts"]
        batch = min(run_cfg.sample_batch_size, remain)
        sampled_indices = rng.integers(0, vocab_size, size=batch)

        for vocab_idx in sampled_indices:
            stats["attempts"] += 1
            candidate_rel = vocab[vocab_idx]

            if not base.pass_geometric_filter(metric_cache, candidate_rel, thresholds):
                stats["geo_fail"] += 1
                _inc_reason("geo_fail")
                continue

            candidate_states = base.make_trajectory_states_from_local_poses(
                local_poses=candidate_rel,
                metric_cache=metric_cache,
                proposal_sampling=proposal_sampling,
            )
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
                continue

            expert_planner.initialize(planner_init)
            ood_time = metric_cache.ego_state.time_point + TimeDuration.from_s(proposal_sampling.time_horizon)
            ood_ego_state = state_array_to_ego_state(
                phase1_states[-1],
                TimePoint(int(ood_time.time_us)),
                metric_cache.ego_state.car_footprint.vehicle_parameters,
            )
            ood_obs = phase1_tracks[-1]
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
                _inc_reason("join_fail")
                if run_cfg.emit_near_miss:
                    stage1_only = base.serialize_trace(
                        token=token,
                        vocab_index=int(vocab_idx),
                        ego_states=phase1_states,
                        tracks=phase1_tracks,
                        stage1_score=stage1_score,
                        stage2_score=_nan_score(),
                    )
                    stage1_only["longtail"] = {
                        "split_idx": float(len(stage1_only["frames"]) - 1),
                        "longtail_score": float(stage1_score.get("pdm_score", 0.0)),
                    }
                    stage1_only["_rank_score"] = float(stage1_score.get("pdm_score", 0.0))
                    _annotate_trace(stage1_only, "near_miss", "join_fail", criteria_snapshot)
                    near_miss_candidates.append(stage1_only)
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
                if run_cfg.emit_near_miss:
                    full_ego_states = np.concatenate([phase1_states, rescue_states[1:]], axis=0)
                    full_tracks = phase1_tracks + phase2_tracks[1:]
                    near_trace = base.serialize_trace(
                        token=token,
                        vocab_index=int(vocab_idx),
                        ego_states=full_ego_states,
                        tracks=full_tracks,
                        stage1_score=stage1_score,
                        stage2_score=stage2_score,
                    )
                    near_trace["longtail"] = compute_longtail_metrics(near_trace)
                    near_trace["_rank_score"] = float(
                        near_trace["longtail"]["longtail_score"]
                        if not math.isnan(near_trace["longtail"]["longtail_score"])
                        else 0.0
                    )
                    _annotate_trace(near_trace, "near_miss", "stage2_fail", criteria_snapshot)
                    near_miss_candidates.append(near_trace)
                continue

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
            trace["longtail"] = compute_longtail_metrics(trace)
            trace["_rank_score"] = float(trace["longtail"]["longtail_score"])
            _annotate_trace(trace, "success", None, criteria_snapshot)
            candidates.append(trace)
            stats["successes"] += 1

            if (
                stats["successes"] >= run_cfg.target_success_per_token
                and stats["attempts"] >= run_cfg.min_attempts_before_early_stop
            ):
                break

        if (
            stats["successes"] >= run_cfg.target_success_per_token
            and stats["attempts"] >= run_cfg.min_attempts_before_early_stop
        ):
            break

    candidates.sort(key=lambda t: float(t.get("_rank_score", 0.0)), reverse=True)
    topk_success = candidates[: run_cfg.top_k_per_token]

    near_miss_candidates.sort(key=lambda t: float(t.get("_rank_score", 0.0)), reverse=True)
    topk_near = near_miss_candidates[: run_cfg.max_near_miss_per_token] if run_cfg.emit_near_miss else []
    stats["near_misses"] = len(topk_near)

    for tr in topk_success:
        tr.pop("_rank_score", None)
    for tr in topk_near:
        tr.pop("_rank_score", None)

    rejected_reason = None
    if not topk_success and not topk_near:
        reasons = stats.get("reject_reasons", {})
        rejected_reason = max(reasons.items(), key=lambda kv: kv[1])[0] if reasons else "no_candidate_passed"
    return {
        "token": token,
        "success_traces": topk_success,
        "near_miss_traces": topk_near,
        "stats": stats,
        "rejected_reason": rejected_reason,
    }


def _init_worker(
    metric_cache_path: str,
    vocab_path: str,
    interval: float,
    horizon_sec: float,
    thresholds_dict: Dict[str, float],
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
    _CTX = {
        "loader": loader,
        "vocab": vocab,
        "proposal_sampling": proposal_sampling,
        "simulator": simulator,
        "reactive_policy": reactive_policy,
        "thresholds": thresholds,
        "map_root_override": map_root_override,
    }


def _run_token_worker(token: str, run_cfg: RunConfig, seed: int) -> Dict[str, Any]:
    global _CTX
    assert _CTX is not None
    metric_cache = _CTX["loader"].get_from_token(token)
    rng = np.random.default_rng(seed)
    return run_for_token_ranked(
        token=token,
        metric_cache=metric_cache,
        vocab=_CTX["vocab"],
        proposal_sampling=_CTX["proposal_sampling"],
        simulator=_CTX["simulator"],
        reactive_policy=_CTX["reactive_policy"],
        thresholds=_CTX["thresholds"],
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
        target_success_per_token=max(args.top_k_per_token, args.target_success_per_token),
        emit_near_miss=args.emit_near_miss,
        max_near_miss_per_token=args.max_near_miss_per_token,
        min_attempts_before_early_stop=args.min_attempts_before_early_stop,
        verbose=args.verbose,
    )

    thresholds_dict = _thresholds_from_args(args)
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
            args.map_root_override,
        )
        for token in tokens:
            seed = _token_seed(args.seed, token)
            results.append(_run_token_worker(token, run_cfg, seed))
    elif args.parallel_backend == "thread":
        _init_worker(
            str(cache_root),
            str(args.vocab_path),
            args.interval,
            args.horizon_sec,
            thresholds_dict,
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
            out_json, out_pkl = _save_ranked_trace(
                trace=trace,
                out_dir=success_dir,
                fmt=args.save_format,
                rank=rank,
                stem_suffix="ranked_success",
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
                        "longtail_score": trace.get("longtail", {}).get("longtail_score"),
                        "vocab_index": trace["vocab_index"],
                        "stage1_score": trace["stage1_score"],
                        "stage2_score": trace["stage2_score"],
                        "longtail": trace.get("longtail", {}),
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
            out_json, out_pkl = _save_ranked_trace(
                trace=trace,
                out_dir=failure_dir,
                fmt=args.save_format,
                rank=rank,
                stem_suffix=f"ranked_near_miss_{reason}",
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
                        "longtail_score": trace.get("longtail", {}).get("longtail_score"),
                        "vocab_index": trace["vocab_index"],
                        "stage1_score": trace["stage1_score"],
                        "stage2_score": trace["stage2_score"],
                        "longtail": trace.get("longtail", {}),
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
                        "search_stage": "ranked",
                        "criteria_snapshot": None,
                        "longtail_score": None,
                        "vocab_index": None,
                        "stage1_score": None,
                        "stage2_score": None,
                        "longtail": None,
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
