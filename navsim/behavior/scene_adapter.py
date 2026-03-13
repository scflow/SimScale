from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from shapely.geometry import Point

from navsim.common.dataclasses import Scene
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import normalize_angle


@dataclass
class ActorState:
    x: float
    y: float
    heading: float
    speed: float
    velocity_x: float
    velocity_y: float
    length: float
    width: float
    height: float
    token: str
    lane_object: Optional[LaneGraphEdgeMapObject] = None
    metadata: Optional[SceneObjectMetadata] = None

    @property
    def point(self) -> Point:
        return Point(self.x, self.y)


@dataclass
class NeighborSet:
    curr_front: Optional[ActorState] = None
    curr_rear: Optional[ActorState] = None
    left_front: Optional[ActorState] = None
    left_rear: Optional[ActorState] = None
    right_front: Optional[ActorState] = None
    right_rear: Optional[ActorState] = None
    left_lane: Optional[LaneGraphEdgeMapObject] = None
    right_lane: Optional[LaneGraphEdgeMapObject] = None


def speed_from_xy(vx: float, vy: float) -> float:
    return float(np.hypot(vx, vy))


def rotate_local_to_global(x: float, y: float, heading: float, origin_x: float, origin_y: float, origin_heading: float):
    cos_h = np.cos(origin_heading)
    sin_h = np.sin(origin_heading)
    global_x = origin_x + x * cos_h - y * sin_h
    global_y = origin_y + x * sin_h + y * cos_h
    global_heading = normalize_angle(heading + origin_heading)
    return global_x, global_y, global_heading


def rotate_vector_to_global(vx: float, vy: float, origin_heading: float):
    cos_h = np.cos(origin_heading)
    sin_h = np.sin(origin_heading)
    global_vx = vx * cos_h - vy * sin_h
    global_vy = vx * sin_h + vy * cos_h
    return global_vx, global_vy


def lane_heading_at_point(lane_object: LaneGraphEdgeMapObject, x: float, y: float) -> float:
    discrete_path = lane_object.baseline_path.discrete_path
    lane_points = np.array([[state.x, state.y] for state in discrete_path], dtype=np.float64)
    if len(lane_points) == 0:
        return 0.0
    nearest_idx = int(np.argmin(np.linalg.norm(lane_points - np.array([x, y]), axis=1)))
    return discrete_path[nearest_idx].heading


def select_best_lane(
    map_api: Optional[AbstractMap],
    x: float,
    y: float,
    heading: Optional[float] = None,
    radius: float = 8.0,
) -> Optional[LaneGraphEdgeMapObject]:
    if map_api is None:
        return None

    query_point = Point2D(float(x), float(y))
    proximal = map_api.get_proximal_map_objects(
        query_point,
        radius,
        [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
    )
    candidates = [obj for objects in proximal.values() for obj in objects]

    if not candidates:
        for layer in (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR):
            object_id, distance = map_api.get_distance_to_nearest_map_object(query_point, layer)
            if object_id is None or distance is None or not np.isfinite(distance):
                continue
            map_object = map_api.get_map_object(object_id, layer)
            if map_object is not None:
                candidates.append(map_object)

    if not candidates:
        return None

    point = Point(x, y)
    best_lane = None
    best_score = float("inf")
    for candidate in candidates:
        distance = candidate.polygon.distance(point)
        score = float(distance)
        if heading is not None:
            candidate_heading = lane_heading_at_point(candidate, x, y)
            score += 0.5 * abs(normalize_angle(candidate_heading - heading))
        if score < best_score:
            best_score = score
            best_lane = candidate
    return best_lane


def lane_progress(actor: ActorState, lane_object: LaneGraphEdgeMapObject) -> float:
    return float(lane_object.baseline_path.linestring.project(actor.point))


def lane_relative_gap(
    follower: ActorState,
    leader: ActorState,
    lane_object: LaneGraphEdgeMapObject,
) -> float:
    follower_progress = lane_progress(follower, lane_object)
    leader_progress = lane_progress(leader, lane_object)
    return leader_progress - follower_progress - 0.5 * follower.length - 0.5 * leader.length


def _front_and_rear_in_lane(
    actor: ActorState,
    others: Iterable[ActorState],
    lane_object: Optional[LaneGraphEdgeMapObject],
) -> tuple[Optional[ActorState], Optional[ActorState]]:
    if lane_object is None:
        return None, None

    actor_progress = lane_progress(actor, lane_object)
    front_actor = None
    rear_actor = None
    best_front = float("inf")
    best_rear = float("inf")

    for other in others:
        if other.token == actor.token:
            continue
        if other.lane_object is None or other.lane_object.id != lane_object.id:
            continue

        other_progress = lane_progress(other, lane_object)
        delta = other_progress - actor_progress
        if delta >= 0.0 and delta < best_front:
            best_front = delta
            front_actor = other
        elif delta < 0.0 and -delta < best_rear:
            best_rear = -delta
            rear_actor = other

    return front_actor, rear_actor


def build_neighbor_set(actor: ActorState, others: Iterable[ActorState]) -> NeighborSet:
    current_lane = actor.lane_object
    left_lane = None
    right_lane = None
    if current_lane is not None:
        left_lane, right_lane = current_lane.adjacent_edges

    curr_front, curr_rear = _front_and_rear_in_lane(actor, others, current_lane)
    left_front, left_rear = _front_and_rear_in_lane(actor, others, left_lane)
    right_front, right_rear = _front_and_rear_in_lane(actor, others, right_lane)
    return NeighborSet(
        curr_front=curr_front,
        curr_rear=curr_rear,
        left_front=left_front,
        left_rear=left_rear,
        right_front=right_front,
        right_rear=right_rear,
        left_lane=left_lane,
        right_lane=right_lane,
    )


def build_ego_actor_from_scene(scene: Scene) -> tuple[ActorState, list[ActorState], list[str]]:
    current_frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
    ego_status = current_frame.ego_status
    ego_x, ego_y, ego_heading = ego_status.ego_pose
    ego_vx, ego_vy = ego_status.ego_velocity
    ego_actor = ActorState(
        x=float(ego_x),
        y=float(ego_y),
        heading=float(ego_heading),
        speed=speed_from_xy(float(ego_vx), float(ego_vy)),
        velocity_x=float(ego_vx),
        velocity_y=float(ego_vy),
        length=4.9,
        width=2.1,
        height=1.8,
        token="ego",
        lane_object=select_best_lane(scene.map_api, float(ego_x), float(ego_y), float(ego_heading)),
    )

    actors: list[ActorState] = []
    annotations = current_frame.annotations
    if annotations is not None:
        for box, name, velocity_3d, track_token in zip(
            annotations.boxes,
            annotations.names,
            annotations.velocity_3d,
            annotations.track_tokens,
        ):
            if name != "vehicle":
                continue
            actor_x, actor_y, actor_heading = rotate_local_to_global(
                float(box[0]),
                float(box[1]),
                float(box[6]),
                ego_actor.x,
                ego_actor.y,
                ego_actor.heading,
            )
            actor_vx, actor_vy = rotate_vector_to_global(
                float(velocity_3d[0]),
                float(velocity_3d[1]),
                ego_actor.heading,
            )
            actors.append(
                ActorState(
                    x=actor_x,
                    y=actor_y,
                    heading=actor_heading,
                    speed=speed_from_xy(actor_vx, actor_vy),
                    velocity_x=actor_vx,
                    velocity_y=actor_vy,
                    length=float(box[3]),
                    width=float(box[4]),
                    height=float(box[5]),
                    token=str(track_token),
                    lane_object=select_best_lane(scene.map_api, actor_x, actor_y, actor_heading),
                )
            )

    return ego_actor, actors, list(current_frame.roadblock_ids)


def build_actor_from_detection_track(track, map_api: Optional[AbstractMap]) -> ActorState:
    velocity_x = float(track.velocity.x)
    velocity_y = float(track.velocity.y)
    center = track.center
    return ActorState(
        x=float(center.x),
        y=float(center.y),
        heading=float(center.heading),
        speed=speed_from_xy(velocity_x, velocity_y),
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        length=float(track.box.length),
        width=float(track.box.width),
        height=float(track.box.height),
        token=str(track.track_token),
        lane_object=select_best_lane(map_api, float(center.x), float(center.y), float(center.heading)),
        metadata=track.metadata,
    )


def build_ego_actor_from_ego_state(ego_state: EgoState, map_api: Optional[AbstractMap]) -> ActorState:
    center = ego_state.rear_axle
    velocity = ego_state.dynamic_car_state.rear_axle_velocity_2d
    vehicle_parameters = ego_state.car_footprint.vehicle_parameters
    return ActorState(
        x=float(center.x),
        y=float(center.y),
        heading=float(center.heading),
        speed=speed_from_xy(float(velocity.x), float(velocity.y)),
        velocity_x=float(velocity.x),
        velocity_y=float(velocity.y),
        length=float(vehicle_parameters.length),
        width=float(vehicle_parameters.width),
        height=float(vehicle_parameters.height),
        token="ego",
        lane_object=select_best_lane(map_api, float(center.x), float(center.y), float(center.heading)),
    )

