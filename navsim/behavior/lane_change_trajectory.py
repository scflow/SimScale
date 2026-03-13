from __future__ import annotations

from typing import List, Optional

import numpy as np
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.behavior.scene_adapter import ActorState, select_best_lane
from navsim.common.dataclasses import Trajectory
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
    normalize_angle,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath


def smoothstep(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return 10.0 * clipped**3 - 15.0 * clipped**4 + 6.0 * clipped**5


def _pick_outgoing_lane(
    lane_object: LaneGraphEdgeMapObject,
    route_roadblock_ids: list[str],
) -> Optional[LaneGraphEdgeMapObject]:
    outgoing = list(lane_object.outgoing_edges)
    if not outgoing:
        return None

    if route_roadblock_ids:
        for roadblock_id in route_roadblock_ids:
            route_matches = [candidate for candidate in outgoing if candidate.get_roadblock_id() == roadblock_id]
            if route_matches:
                return route_matches[0]

    return outgoing[0]


def build_lane_path(
    lane_object: Optional[LaneGraphEdgeMapObject],
    route_roadblock_ids: list[str],
    min_length: float,
) -> Optional[PDMPath]:
    if lane_object is None:
        return None

    discrete_path: List[StateSE2] = []
    current_lane = lane_object
    visited_ids = set()
    while current_lane is not None and current_lane.id not in visited_ids:
        visited_ids.add(current_lane.id)
        discrete_path.extend(current_lane.baseline_path.discrete_path)
        if len(discrete_path) >= 2:
            path_length = PDMPath(discrete_path).length
            if path_length >= min_length:
                break
        current_lane = _pick_outgoing_lane(current_lane, route_roadblock_ids)

    if len(discrete_path) < 2:
        return None

    return PDMPath(discrete_path)


def _blend_angles(source_heading: np.ndarray, target_heading: np.ndarray, blend: np.ndarray) -> np.ndarray:
    blended_cos = (1.0 - blend) * np.cos(source_heading) + blend * np.cos(target_heading)
    blended_sin = (1.0 - blend) * np.sin(source_heading) + blend * np.sin(target_heading)
    return np.arctan2(blended_sin, blended_cos)


def rollout_absolute_states(
    actor: ActorState,
    current_lane: Optional[LaneGraphEdgeMapObject],
    target_lane: Optional[LaneGraphEdgeMapObject],
    acceleration: float,
    times: np.ndarray,
    lane_change_duration: float,
    lane_change_elapsed: float,
    route_roadblock_ids: list[str],
) -> np.ndarray:
    if current_lane is None:
        current_lane = target_lane

    if current_lane is None:
        x = actor.x + np.cos(actor.heading) * (actor.speed * times + 0.5 * acceleration * times**2)
        y = actor.y + np.sin(actor.heading) * (actor.speed * times + 0.5 * acceleration * times**2)
        headings = np.full_like(times, actor.heading, dtype=np.float64)
        return np.stack([x, y, headings], axis=-1)

    max_distance = max(actor.speed * float(times[-1]) + 0.5 * abs(acceleration) * float(times[-1]) ** 2 + 30.0, 40.0)
    current_path = build_lane_path(current_lane, route_roadblock_ids, max_distance)
    if current_path is None:
        x = actor.x + np.cos(actor.heading) * (actor.speed * times + 0.5 * acceleration * times**2)
        y = actor.y + np.sin(actor.heading) * (actor.speed * times + 0.5 * acceleration * times**2)
        headings = np.full_like(times, actor.heading, dtype=np.float64)
        return np.stack([x, y, headings], axis=-1)

    start_progress = float(current_path.project(actor.point))
    distances = start_progress + actor.speed * times + 0.5 * acceleration * times**2
    source_states = current_path.interpolate(distances, as_array=True)

    if target_lane is None or target_lane.id == current_lane.id:
        return source_states

    target_path = build_lane_path(target_lane, route_roadblock_ids, max_distance)
    if target_path is None:
        return source_states

    target_start = float(target_path.project(actor.point))
    target_states = target_path.interpolate(target_start + actor.speed * times + 0.5 * acceleration * times**2, as_array=True)

    blend = smoothstep((lane_change_elapsed + times) / max(lane_change_duration, 1e-3))
    output = source_states.copy()
    output[:, 0] = (1.0 - blend) * source_states[:, 0] + blend * target_states[:, 0]
    output[:, 1] = (1.0 - blend) * source_states[:, 1] + blend * target_states[:, 1]
    output[:, 2] = _blend_angles(source_states[:, 2], target_states[:, 2], blend)
    output[:, 2] = np.array([normalize_angle(angle) for angle in output[:, 2]], dtype=np.float64)
    return output


def rollout_ego_trajectory(
    ego: ActorState,
    current_lane: Optional[LaneGraphEdgeMapObject],
    target_lane: Optional[LaneGraphEdgeMapObject],
    acceleration: float,
    trajectory_sampling: TrajectorySampling,
    route_roadblock_ids: list[str],
    lane_change_duration: float,
) -> Trajectory:
    times = trajectory_sampling.interval_length * np.arange(1, trajectory_sampling.num_poses + 1, dtype=np.float64)
    absolute_states = rollout_absolute_states(
        actor=ego,
        current_lane=current_lane,
        target_lane=target_lane,
        acceleration=acceleration,
        times=times,
        lane_change_duration=lane_change_duration,
        lane_change_elapsed=0.0,
        route_roadblock_ids=route_roadblock_ids,
    )
    relative_states = convert_absolute_to_relative_se2_array(
        origin=StateSE2(ego.x, ego.y, ego.heading),
        state_se2_array=absolute_states,
    ).astype(np.float32)
    return Trajectory(relative_states, trajectory_sampling)


def propagate_actor(
    actor: ActorState,
    map_api,
    current_lane: Optional[LaneGraphEdgeMapObject],
    target_lane: Optional[LaneGraphEdgeMapObject],
    acceleration: float,
    dt: float,
    route_roadblock_ids: list[str],
    lane_change_duration: float,
    lane_change_elapsed: float,
) -> tuple[ActorState, float]:
    absolute_state = rollout_absolute_states(
        actor=actor,
        current_lane=current_lane,
        target_lane=target_lane,
        acceleration=acceleration,
        times=np.array([dt], dtype=np.float64),
        lane_change_duration=lane_change_duration,
        lane_change_elapsed=lane_change_elapsed,
        route_roadblock_ids=route_roadblock_ids,
    )[0]
    next_speed = max(0.0, actor.speed + acceleration * dt)
    next_actor = ActorState(
        x=float(absolute_state[0]),
        y=float(absolute_state[1]),
        heading=float(absolute_state[2]),
        speed=float(next_speed),
        velocity_x=float(np.cos(absolute_state[2]) * next_speed),
        velocity_y=float(np.sin(absolute_state[2]) * next_speed),
        length=actor.length,
        width=actor.width,
        height=actor.height,
        token=actor.token,
        metadata=actor.metadata,
        lane_object=select_best_lane(map_api, float(absolute_state[0]), float(absolute_state[1]), float(absolute_state[2])),
    )
    next_elapsed = min(lane_change_duration, lane_change_elapsed + dt) if target_lane is not None else 0.0
    return next_actor, next_elapsed

