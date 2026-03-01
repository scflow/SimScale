#!/usr/bin/env python3
"""
Generate OOD long-tail traces on NAVSIM mini split.

Pipeline:
1) Sample perturbation trajectories from vocabulary (relative frame).
2) Geometric pre-filter against human endpoint.
3) Physics feasibility filter with PDM simulator / bicycle model dynamics.
4) Reactive rollout with IDM traffic agents (T -> T+H).
5) Rescue rollout with PDM-Closed expert from OOD state (T+H -> T+2H).
6) Score segment(s) with PDMScorer and keep only strict-valid samples.
7) Save structured vector traces (JSON / Pickle).

Note:
- This script depends on generated metric cache.
- No camera/lidar sensor blobs are required.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from nuplan.common.actor_state.state_representation import StateSE2, TimeDuration, TimePoint
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.observation.navsim_idm_agents import NavsimIDMAgents
from navsim.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner
from navsim.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import state_array_to_ego_state
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from navsim.traffic_agents_policies.navsim_IDM_traffic_agents import NavsimIDMTrafficAgents


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def rotate_to_local(dx: float, dy: float, heading: float) -> Tuple[float, float]:
    """Rotate global delta into local frame aligned to heading."""
    c = math.cos(heading)
    s = math.sin(heading)
    lon = c * dx + s * dy
    lat = -s * dx + c * dy
    return lon, lat


@dataclass
class Thresholds:
    max_abs_lon_m: float = 20.0
    max_abs_lat_m: float = 2.0
    max_abs_heading_deg: float = 20.0
    max_abs_accel_mps2: float = 6.0
    max_abs_steer_deg: float = 60.0
    min_progress: float = 0.5


def build_reactive_policy(
    proposal_sampling: TrajectorySampling, map_root_override: Optional[str] = None
) -> NavsimIDMTrafficAgents:
    """Instantiate IDM reactive traffic policy using repo defaults."""
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
        map_root_override=map_root_override,
    )


def build_expert_planner(
    proposal_sampling: TrajectorySampling,
    map_radius: float = 100.0,
) -> PDMClosedPlanner:
    """
    Build PDM-Closed planner with the same style as metric cache processor.
    """
    # +1s in trajectory sampling for internal TTC robustness.
    future_poses = proposal_sampling.num_poses + int(1.0 / proposal_sampling.interval_length)
    trajectory_sampling = TrajectorySampling(
        num_poses=future_poses, interval_length=proposal_sampling.interval_length
    )
    return PDMClosedPlanner(
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
        lateral_offsets=[-1.0, 1.0],
        map_radius=map_radius,
    )


def build_planner_initialization(metric_cache: MetricCache, mission_goal: StateSE2 = StateSE2(0.0, 0.0, 0.0)) -> PlannerInitialization:
    """
    Infer route roadblocks from route lane IDs so we can initialize PDM-Closed
    without loading full Scene objects.
    """
    map_params = metric_cache.map_parameters
    map_api = get_maps_api(map_params.map_root, map_params.map_version, map_params.map_name)

    roadblock_ids: List[str] = []
    seen = set()
    for lane_id in metric_cache.route_lane_ids:
        lane = map_api.get_map_object(lane_id, SemanticMapLayer.LANE)
        if lane is None:
            lane = map_api.get_map_object(lane_id, SemanticMapLayer.LANE_CONNECTOR)
        if lane is None:
            continue
        roadblock_id = lane.get_roadblock_id()
        if roadblock_id not in seen:
            seen.add(roadblock_id)
            roadblock_ids.append(roadblock_id)

    if not roadblock_ids:
        raise RuntimeError("Failed to infer route_roadblock_ids from metric cache route_lane_ids.")

    return PlannerInitialization(route_roadblock_ids=roadblock_ids, mission_goal=mission_goal, map_api=map_api)


def make_trajectory_states_from_local_poses(
    local_poses: np.ndarray,
    metric_cache: MetricCache,
    proposal_sampling: TrajectorySampling,
) -> np.ndarray:
    """
    Convert local (relative-to-initial-ego) poses into global state array
    sampled as [num_poses + 1, StateIndex.size()].
    """
    traj = Trajectory(poses=local_poses.astype(np.float32), trajectory_sampling=proposal_sampling)
    interpolated = transform_trajectory(traj, metric_cache.ego_state)
    states = get_trajectory_as_array(interpolated, proposal_sampling, metric_cache.ego_state.time_point)
    return states


def extract_human_endpoint_global(metric_cache: MetricCache) -> Optional[StateSE2]:
    """Get human endpoint in global frame from metric cache."""
    if metric_cache.human_trajectory is None:
        return None
    human_rel_end = metric_cache.human_trajectory.poses[-1]
    init = metric_cache.ego_state.rear_axle
    c = math.cos(init.heading)
    s = math.sin(init.heading)
    gx = init.x + c * float(human_rel_end[0]) - s * float(human_rel_end[1])
    gy = init.y + s * float(human_rel_end[0]) + c * float(human_rel_end[1])
    gh = normalize_angle(init.heading + float(human_rel_end[2]))
    return StateSE2(gx, gy, gh)


def extract_candidate_endpoint_global(metric_cache: MetricCache, candidate_rel_poses: np.ndarray) -> StateSE2:
    """Convert candidate endpoint (relative) into global StateSE2."""
    end_rel = candidate_rel_poses[-1]
    init = metric_cache.ego_state.rear_axle
    c = math.cos(init.heading)
    s = math.sin(init.heading)
    gx = init.x + c * float(end_rel[0]) - s * float(end_rel[1])
    gy = init.y + s * float(end_rel[0]) + c * float(end_rel[1])
    gh = normalize_angle(init.heading + float(end_rel[2]))
    return StateSE2(gx, gy, gh)


def pass_geometric_filter(
    metric_cache: MetricCache,
    candidate_rel_poses: np.ndarray,
    thresholds: Thresholds,
) -> bool:
    """Initial geometric clipping against human endpoint."""
    human_end = extract_human_endpoint_global(metric_cache)
    if human_end is None:
        return False
    cand_end = extract_candidate_endpoint_global(metric_cache, candidate_rel_poses)
    dx, dy = cand_end.x - human_end.x, cand_end.y - human_end.y
    lon, lat = rotate_to_local(dx, dy, human_end.heading)
    dh = abs(math.degrees(normalize_angle(cand_end.heading - human_end.heading)))
    return (
        abs(lon) <= thresholds.max_abs_lon_m
        and abs(lat) <= thresholds.max_abs_lat_m
        and dh <= thresholds.max_abs_heading_deg
    )


def pass_physics_filter(simulated_states: np.ndarray, thresholds: Thresholds) -> bool:
    """Secondary physical filter on steering / acceleration limits."""
    max_abs_steer = np.max(np.abs(simulated_states[:, StateIndex.STEERING_ANGLE]))
    max_abs_accel = np.max(np.abs(simulated_states[:, StateIndex.ACCELERATION_X]))
    return (
        math.degrees(float(max_abs_steer)) <= thresholds.max_abs_steer_deg
        and float(max_abs_accel) <= thresholds.max_abs_accel_mps2
    )


def score_segment(
    scorer: PDMScorer,
    states: np.ndarray,
    metric_cache: MetricCache,
    simulated_tracks: List[DetectionsTracks],
    centerline=None,
    route_lane_ids=None,
    drivable_area_map=None,
) -> Dict[str, float]:
    """Run PDMScorer for one proposal and return compact metrics."""
    # PDMScorer mutates observation via update_detections_tracks; isolate calls.
    observation = copy.deepcopy(metric_cache.observation)
    pdm_row = scorer.score_proposals(
        states=states[None, ...],
        observation=observation,
        centerline=centerline if centerline is not None else metric_cache.centerline,
        route_lane_ids=route_lane_ids if route_lane_ids is not None else metric_cache.route_lane_ids,
        drivable_area_map=drivable_area_map if drivable_area_map is not None else metric_cache.drivable_area_map,
        map_parameters=metric_cache.map_parameters,
        simulated_agent_detections_tracks=simulated_tracks,
        human_past_trajectory=metric_cache.past_human_trajectory,
    )[0].iloc[0]
    return {
        "no_at_fault_collisions": float(pdm_row["no_at_fault_collisions"]),
        "drivable_area_compliance": float(pdm_row["drivable_area_compliance"]),
        "driving_direction_compliance": float(pdm_row["driving_direction_compliance"]),
        "traffic_light_compliance": float(pdm_row["traffic_light_compliance"]),
        "ego_progress": float(pdm_row["ego_progress"]),
        "pdm_score": float(pdm_row["pdm_score"]),
    }


def pass_strict_gate(score: Dict[str, float], thresholds: Thresholds) -> bool:
    """One-veto quality gate."""
    eps = 1e-6
    return (
        score["no_at_fault_collisions"] >= 1.0 - eps
        and score["drivable_area_compliance"] >= 1.0 - eps
        and score["driving_direction_compliance"] >= 1.0 - eps
        and score["traffic_light_compliance"] >= 1.0 - eps
        and score["ego_progress"] >= thresholds.min_progress
    )


def build_stage2_metric_cache(
    metric_cache: MetricCache,
    ego_state_ood,
    current_tracks: DetectionsTracks,
    proposal_sampling: TrajectorySampling,
) -> MetricCache:
    """Create a lightweight metric cache view rooted at OOD state."""
    stage2_cache = copy.copy(metric_cache)
    stage2_cache.ego_state = ego_state_ood
    stage2_cache.current_tracked_objects = [current_tracks]
    stage2_cache.future_tracked_objects = [current_tracks for _ in range(proposal_sampling.num_poses)]
    return stage2_cache


def serialize_trace(
    token: str,
    vocab_index: int,
    ego_states: np.ndarray,
    tracks: List[DetectionsTracks],
    stage1_score: Dict[str, float],
    stage2_score: Dict[str, float],
) -> Dict[str, Any]:
    """Convert simulation arrays/tracks to structured output payload."""
    if len(ego_states) != len(tracks):
        raise ValueError(f"Ego states length ({len(ego_states)}) != tracks length ({len(tracks)})")

    frames: List[Dict[str, Any]] = []
    for t in range(len(ego_states)):
        ego = ego_states[t]
        vx = float(ego[StateIndex.VELOCITY_X])
        vy = float(ego[StateIndex.VELOCITY_Y])
        ego_payload = {
            "x": float(ego[StateIndex.X]),
            "y": float(ego[StateIndex.Y]),
            "heading": float(ego[StateIndex.HEADING]),
            "velocity": float(math.hypot(vx, vy)),
        }

        vehicles = []
        for obj in tracks[t].tracked_objects.tracked_objects:
            # We only keep vehicle-like payload as requested.
            if str(obj.tracked_object_type).lower().endswith("vehicle"):
                vehicles.append(
                    {
                        "id": str(obj.track_token),
                        "x": float(obj.center.x),
                        "y": float(obj.center.y),
                        "heading": float(obj.center.heading),
                        "width": float(obj.box.width),
                        "length": float(obj.box.length),
                    }
                )

        frames.append({"t_idx": t, "ego": ego_payload, "vehicles": vehicles})

    return {
        "token": token,
        "vocab_index": int(vocab_index),
        "num_frames": len(frames),
        "stage1_score": stage1_score,
        "stage2_score": stage2_score,
        "frames": frames,
    }


def save_trace(trace: Dict[str, Any], output_dir: Path, fmt: str) -> None:
    """Save trace to JSON/PKL."""
    token = trace["token"]
    if fmt in ("json", "both"):
        out_json = output_dir / f"ood_scene_{token}.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)
    if fmt in ("pkl", "both"):
        out_pkl = output_dir / f"ood_scene_{token}.pkl"
        with out_pkl.open("wb") as f:
            pickle.dump(trace, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_for_token(
    token: str,
    metric_cache: MetricCache,
    vocab: np.ndarray,
    proposal_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    reactive_policy: NavsimIDMTrafficAgents,
    thresholds: Thresholds,
    sample_batch_size: int,
    max_candidate_trials: int,
    rng: np.random.Generator,
    verbose: bool,
) -> Optional[Dict[str, Any]]:
    """Try to generate one valid OOD trace for one token."""
    if metric_cache.human_trajectory is None:
        if verbose:
            print(f"[{token}] skip: human_trajectory is None.")
        return None

    scorer_cfg = PDMScorerConfig(human_penalty_filter=False)
    scorer = PDMScorer(proposal_sampling=proposal_sampling, config=scorer_cfg)

    planner_init = build_planner_initialization(metric_cache)
    expert_planner = build_expert_planner(proposal_sampling=proposal_sampling)

    vocab_size = len(vocab)
    attempts = 0
    while attempts < max_candidate_trials:
        remaining = max_candidate_trials - attempts
        batch = min(sample_batch_size, remaining)
        sampled_indices = rng.integers(0, vocab_size, size=batch)

        for vocab_idx in sampled_indices:
            attempts += 1
            candidate_rel = vocab[vocab_idx]

            # 1) Geometric clipping vs human endpoint
            if not pass_geometric_filter(metric_cache, candidate_rel, thresholds):
                continue

            # 2) Physics simulation + filter
            candidate_states = make_trajectory_states_from_local_poses(
                local_poses=candidate_rel,
                metric_cache=metric_cache,
                proposal_sampling=proposal_sampling,
            )
            phase1_states = simulator.simulate_proposals(candidate_states[None, ...], metric_cache.ego_state)[0]
            if not pass_physics_filter(phase1_states, thresholds):
                continue

            # 3) Reactive rollout (T -> T+H)
            phase1_tracks = reactive_policy.simulate_environment(phase1_states, metric_cache)
            stage1_score = score_segment(
                scorer=scorer,
                states=phase1_states,
                metric_cache=metric_cache,
                simulated_tracks=phase1_tracks,
            )
            if not pass_strict_gate(stage1_score, thresholds):
                continue

            # 4) Build OOD start state at T+H and call rescue planner
            # Re-initialize planner each candidate to avoid state leakage between rollouts.
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
            rescue_states = get_trajectory_as_array(
                rescue_traj, proposal_sampling, start_time=ood_ego_state.time_point
            )

            # 5) Reactive rollout for rescue segment (T+H -> T+2H)
            stage2_cache = build_stage2_metric_cache(
                metric_cache=metric_cache,
                ego_state_ood=ood_ego_state,
                current_tracks=ood_obs,
                proposal_sampling=proposal_sampling,
            )
            phase2_tracks = reactive_policy.simulate_environment(rescue_states, stage2_cache)

            stage2_score = score_segment(
                scorer=scorer,
                states=rescue_states,
                metric_cache=stage2_cache,
                simulated_tracks=phase2_tracks,
                centerline=expert_planner._centerline,
                route_lane_ids=list(expert_planner._route_lane_dict.keys()),
                drivable_area_map=expert_planner._drivable_area_map,
            )
            if not pass_strict_gate(stage2_score, thresholds):
                continue

            # 6) Stitch full trace and return
            full_ego_states = np.concatenate([phase1_states, rescue_states[1:]], axis=0)
            full_tracks = phase1_tracks + phase2_tracks[1:]
            return serialize_trace(
                token=token,
                vocab_index=int(vocab_idx),
                ego_states=full_ego_states,
                tracks=full_tracks,
                stage1_score=stage1_score,
                stage2_score=stage2_score,
            )

    if verbose:
        print(f"[{token}] no valid trace found after {attempts} candidates.")
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OOD mini traces using NAVSIM metric cache.")
    parser.add_argument("--metric-cache-path", type=Path, required=True, help="Path to metric cache root.")
    parser.add_argument("--vocab-path", type=Path, default=Path("traj_final/16384.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("generated_ood_data"))
    parser.add_argument("--save-format", type=str, default="both", choices=["json", "pkl", "both"])
    parser.add_argument("--max-scenes", type=int, default=None, help="Limit number of tokens to process.")
    parser.add_argument("--sample-batch-size", type=int, default=100, help="Candidates sampled per batch.")
    parser.add_argument("--max-candidate-trials", type=int, default=1000, help="Max candidates per scene.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--horizon-sec", type=float, default=4.0)
    parser.add_argument("--max-abs-lon-m", type=float, default=20.0)
    parser.add_argument("--max-abs-lat-m", type=float, default=2.0)
    parser.add_argument("--max-abs-heading-deg", type=float, default=20.0)
    parser.add_argument("--max-abs-accel-mps2", type=float, default=6.0)
    parser.add_argument("--max-abs-steer-deg", type=float, default=60.0)
    parser.add_argument("--min-progress", type=float, default=0.5)
    parser.add_argument("--map-root-override", type=str, default=None, help="Optional map root override for IDM.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if not args.metric_cache_path.exists():
        raise FileNotFoundError(f"metric cache path does not exist: {args.metric_cache_path}")
    if not args.vocab_path.exists():
        raise FileNotFoundError(f"vocab path does not exist: {args.vocab_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    vocab = np.load(args.vocab_path)
    if vocab.ndim != 3 or vocab.shape[-1] != 3:
        raise ValueError(f"Expected vocab shape [N, H, 3], got {vocab.shape}")

    proposal_sampling = TrajectorySampling(
        num_poses=int(round(args.horizon_sec / args.interval)),
        interval_length=args.interval,
    )
    if vocab.shape[1] != proposal_sampling.num_poses:
        raise ValueError(
            f"Vocab horizon mismatch: vocab H={vocab.shape[1]} vs proposal num_poses={proposal_sampling.num_poses}"
        )

    thresholds = Thresholds(
        max_abs_lon_m=args.max_abs_lon_m,
        max_abs_lat_m=args.max_abs_lat_m,
        max_abs_heading_deg=args.max_abs_heading_deg,
        max_abs_accel_mps2=args.max_abs_accel_mps2,
        max_abs_steer_deg=args.max_abs_steer_deg,
        min_progress=args.min_progress,
    )

    metric_cache_loader = MetricCacheLoader(args.metric_cache_path)
    tokens = list(metric_cache_loader.tokens)
    if args.max_scenes is not None:
        tokens = tokens[: args.max_scenes]

    simulator = PDMSimulator(proposal_sampling=proposal_sampling)
    reactive_policy = build_reactive_policy(
        proposal_sampling=proposal_sampling, map_root_override=args.map_root_override
    )

    print(f"Loaded vocab: {args.vocab_path} shape={vocab.shape}")
    print(f"Processing {len(tokens)} scene tokens from {args.metric_cache_path}")
    saved = 0
    failed = 0

    for i, token in enumerate(tokens):
        if args.verbose:
            print(f"[{i + 1}/{len(tokens)}] token={token}")
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            trace = run_for_token(
                token=token,
                metric_cache=metric_cache,
                vocab=vocab,
                proposal_sampling=proposal_sampling,
                simulator=simulator,
                reactive_policy=reactive_policy,
                thresholds=thresholds,
                sample_batch_size=args.sample_batch_size,
                max_candidate_trials=args.max_candidate_trials,
                rng=rng,
                verbose=args.verbose,
            )
            if trace is None:
                failed += 1
                continue
            save_trace(trace, args.output_dir, args.save_format)
            saved += 1
            if args.verbose:
                print(f"  saved token={token}")
        except Exception as e:
            failed += 1
            print(f"  token={token} failed: {e}")

    print(
        f"Done. saved={saved}, failed_or_skipped={failed}, "
        f"output_dir={args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
