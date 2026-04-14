from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import Point2D, StateSE2, StateVector2D
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, TrackedObjectType
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusData, TrafficLightStatusType
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely.geometry import Point

from navsim.behavior.idm import IDM, IDMParams
from navsim.behavior.lane_change_trajectory import build_lane_path, propagate_actor
from navsim.behavior.mobil import KEEP, LEFT, MobilModel, MobilParams, RIGHT, decide_with_mobil
from navsim.behavior.scene_adapter import (
    ActorState,
    build_actor_from_detection_track,
    build_ego_actor_from_ego_state,
    build_neighbor_set,
    select_best_lane,
)
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import normalize_angle
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import (
    AbstractTrafficAgentsPolicy,
    filter_tracked_objects_by_type,
)


@dataclass
class _TrafficActorRuntime:
    actor: ActorState
    target_lane: Optional[object] = None
    lane_change_elapsed: float = 0.0
    cooldown_remaining: float = 0.0
    route_lane_ids: Optional[List[str]] = None
    route_roadblock_ids: Optional[List[str]] = None


class MobilTrafficAgentsPolicy(AbstractTrafficAgentsPolicy):
    """Reactive traffic agents policy based on IDM + MOBIL."""

    def __init__(
        self,
        future_trajectory_sampling: TrajectorySampling,
        idm_params: IDMParams = IDMParams(),
        mobil_params: MobilParams = MobilParams(),
        lane_change_duration: float = 3.0,
        open_loop_detections_types: Optional[List[str]] = None,
        stop_line_min_gap: float = 3.0,
        open_loop_min_gap: float = 2.0,
        open_loop_path_buffer: float = 2.5,
        max_stop_brake: float = 6.0,
        map_root_override: Optional[str] = None,
    ):
        self.future_trajectory_sampling = future_trajectory_sampling
        self._idm = IDM(idm_params)
        self._mobil = MobilModel(mobil_params, self._idm)
        self._lane_change_duration = lane_change_duration
        self._open_loop_detections_types = [
            TrackedObjectType[name] for name in (open_loop_detections_types or [])
        ]
        self._stop_line_min_gap = stop_line_min_gap
        self._open_loop_min_gap = open_loop_min_gap
        self._open_loop_path_buffer = open_loop_path_buffer
        self._max_stop_brake = max_stop_brake
        self._map_root_override = map_root_override

    def get_list_of_simulated_object_types(self) -> List[TrackedObjectType]:
        return [TrackedObjectType.VEHICLE]

    def _build_runtime_actors(self, detections_tracks: DetectionsTracks, map_api) -> Dict[str, _TrafficActorRuntime]:
        runtime_actors: Dict[str, _TrafficActorRuntime] = {}
        for track in detections_tracks.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE):
            actor = build_actor_from_detection_track(track, map_api)
            runtime_actors[actor.token] = _TrafficActorRuntime(actor=actor)
        return runtime_actors

    @staticmethod
    def _dedupe_preserve_order(values: List[str]) -> List[str]:
        deduped: List[str] = []
        for value in values:
            if not deduped or deduped[-1] != value:
                deduped.append(value)
        return deduped

    def _match_lane_sequence(self, actor: ActorState, states: List[ActorState], map_api) -> List[object]:
        matched_lanes: List[object] = []
        previous_lane = actor.lane_object
        for state in states:
            candidate_lane = None
            if previous_lane is not None:
                candidate_pool = [previous_lane]
                candidate_pool.extend(previous_lane.outgoing_edges)
                left_lane, right_lane = previous_lane.adjacent_edges
                if left_lane is not None:
                    candidate_pool.append(left_lane)
                if right_lane is not None:
                    candidate_pool.append(right_lane)

                point = state.point
                map_point = Point2D(state.x, state.y)
                best_score = float("inf")
                for lane_object in candidate_pool:
                    score = float(lane_object.polygon.distance(point))
                    lane_heading = lane_object.baseline_path.get_nearest_pose_from_position(map_point).heading
                    score += 0.5 * abs(normalize_angle(state.heading - lane_heading))
                    if score < best_score:
                        best_score = score
                        candidate_lane = lane_object
                if candidate_lane is not None and candidate_lane.polygon.distance(point) > 4.0:
                    candidate_lane = None

            if candidate_lane is None:
                candidate_lane = select_best_lane(map_api, state.x, state.y, state.heading, radius=12.0)
            if candidate_lane is None:
                candidate_lane = previous_lane
            if candidate_lane is None:
                continue
            matched_lanes.append(candidate_lane)
            previous_lane = candidate_lane

        return matched_lanes

    def _infer_agent_routes(
        self,
        runtime_actors: Dict[str, _TrafficActorRuntime],
        metric_cache: MetricCache,
        map_api,
    ) -> None:
        future_vehicle_tracks = [
            detections_tracks.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
            for detections_tracks in metric_cache.future_tracked_objects
        ]

        for token, runtime_actor in runtime_actors.items():
            future_states: List[ActorState] = []
            for tracks in future_vehicle_tracks:
                matched_track = next((track for track in tracks if str(track.track_token) == token), None)
                if matched_track is None:
                    continue
                future_states.append(build_actor_from_detection_track(matched_track, map_api))

            lane_sequence = self._match_lane_sequence(runtime_actor.actor, future_states, map_api)
            if runtime_actor.actor.lane_object is not None:
                lane_sequence = [runtime_actor.actor.lane_object] + lane_sequence

            route_lane_ids = self._dedupe_preserve_order([lane.id for lane in lane_sequence if lane is not None])
            route_roadblock_ids = self._dedupe_preserve_order(
                [lane.get_roadblock_id() for lane in lane_sequence if lane is not None]
            )

            if not route_lane_ids and runtime_actor.actor.lane_object is not None:
                route_lane_ids = [runtime_actor.actor.lane_object.id]
            if not route_roadblock_ids and runtime_actor.actor.lane_object is not None:
                route_roadblock_ids = [runtime_actor.actor.lane_object.get_roadblock_id()]

            runtime_actor.route_lane_ids = route_lane_ids
            runtime_actor.route_roadblock_ids = route_roadblock_ids

    @staticmethod
    def _normalize_traffic_light_status(
        traffic_light_status: Optional[object],
    ) -> Dict[TrafficLightStatusType, List[str]]:
        if traffic_light_status is None:
            return {}
        if isinstance(traffic_light_status, dict):
            return {
                status: [str(lane_id) for lane_id in lane_ids]
                for status, lane_ids in traffic_light_status.items()
            }

        grouped: Dict[TrafficLightStatusType, List[str]] = {}
        for item in traffic_light_status:
            if not isinstance(item, TrafficLightStatusData):
                continue
            grouped.setdefault(item.status, []).append(str(item.lane_connector_id))
        return grouped

    def _compute_stopline_accel_cap(
        self,
        actor: ActorState,
        path,
        current_progress: float,
        route_lane_ids: List[str],
        traffic_light_status: Dict[TrafficLightStatusType, List[str]],
        map_api,
    ) -> float:
        red_lane_connector_ids = set(traffic_light_status.get(TrafficLightStatusType.RED, []))
        if not red_lane_connector_ids:
            return np.inf

        accel_cap = np.inf
        for lane_id in route_lane_ids:
            if lane_id not in red_lane_connector_ids:
                continue
            lane_connector = map_api.get_map_object(lane_id, SemanticMapLayer.LANE_CONNECTOR)
            if lane_connector is None:
                continue
            for stop_line in lane_connector.stop_lines:
                stop_line_center = stop_line.polygon.centroid
                stop_progress = float(path.project(stop_line_center))
                gap = stop_progress - current_progress - 0.5 * actor.length - self._stop_line_min_gap
                if gap <= 0.0:
                    accel_cap = min(accel_cap, -self._max_stop_brake)
                else:
                    accel_cap = min(accel_cap, -actor.speed**2 / max(2.0 * gap, 1e-3))

        return accel_cap

    def _compute_open_loop_accel_cap(
        self,
        actor: ActorState,
        path,
        current_progress: float,
        open_loop_tracks: List[object],
    ) -> float:
        accel_cap = np.inf
        for track in open_loop_tracks:
            if str(track.track_token) == actor.token:
                continue
            object_point = Point(track.center.x, track.center.y)
            if path.linestring.distance(object_point) > self._open_loop_path_buffer:
                continue

            object_progress = float(path.project(object_point))
            object_length = float(getattr(track.box, "length", 0.0))
            gap = object_progress - current_progress - 0.5 * actor.length - 0.5 * object_length - self._open_loop_min_gap
            if gap <= 0.0:
                accel_cap = min(accel_cap, -self._max_stop_brake)
                continue

            if track.tracked_object_type in AGENT_TYPES and hasattr(track, "velocity"):
                object_speed = float(np.hypot(track.velocity.x, track.velocity.y))
            else:
                object_speed = 0.0

            accel_cap = min(accel_cap, self._idm.accel(actor.speed, gap, actor.speed - object_speed))

        return accel_cap

    def _compute_action_acceleration_caps(
        self,
        actor: ActorState,
        runtime_actor: _TrafficActorRuntime,
        neighbors,
        open_loop_tracks: List[object],
        traffic_light_status: Dict[TrafficLightStatusType, List[str]],
        map_api,
    ) -> Dict[str, float]:
        action_to_lane = {
            KEEP: actor.lane_object,
            LEFT: neighbors.left_lane,
            RIGHT: neighbors.right_lane,
        }
        acceleration_caps: Dict[str, float] = {}
        max_distance = max(actor.speed * self.future_trajectory_sampling.time_horizon + 30.0, 40.0)

        for action, lane_object in action_to_lane.items():
            if lane_object is None:
                continue
            path = build_lane_path(
                lane_object=lane_object,
                route_lane_ids=runtime_actor.route_lane_ids or [],
                route_roadblock_ids=runtime_actor.route_roadblock_ids or [],
                min_length=max_distance,
            )
            if path is None:
                continue

            current_progress = float(path.project(actor.point))
            stopline_cap = self._compute_stopline_accel_cap(
                actor=actor,
                path=path,
                current_progress=current_progress,
                route_lane_ids=runtime_actor.route_lane_ids or [],
                traffic_light_status=traffic_light_status,
                map_api=map_api,
            )
            open_loop_cap = self._compute_open_loop_accel_cap(
                actor=actor,
                path=path,
                current_progress=current_progress,
                open_loop_tracks=open_loop_tracks,
            )
            acceleration_caps[action] = min(stopline_cap, open_loop_cap)

        return acceleration_caps

    def _build_detection_tracks(
        self,
        runtime_actors: Dict[str, _TrafficActorRuntime],
        timestamp_us: int,
    ) -> DetectionsTracks:
        output_tracks: List[Agent] = []
        for token in sorted(runtime_actors.keys()):
            actor = runtime_actors[token].actor
            metadata = actor.metadata or SceneObjectMetadata(
                timestamp_us=timestamp_us,
                token=token,
                track_id=None,
                track_token=token,
            )
            metadata = SceneObjectMetadata(
                timestamp_us=timestamp_us,
                token=metadata.token,
                track_id=metadata.track_id,
                track_token=metadata.track_token,
                category_name=metadata.category_name,
            )
            output_tracks.append(
                Agent(
                    tracked_object_type=TrackedObjectType.VEHICLE,
                    oriented_box=OrientedBox(
                        center=StateSE2(actor.x, actor.y, actor.heading),
                        length=actor.length,
                        width=actor.width,
                        height=actor.height,
                    ),
                    velocity=StateVector2D(actor.velocity_x, actor.velocity_y),
                    metadata=metadata,
                    angular_velocity=0.0,
                )
            )
        return DetectionsTracks(TrackedObjects(output_tracks))

    def _build_ego_actor_from_simulated_state(self, ego_state_arr: npt.NDArray[np.float64], metric_cache: MetricCache, map_api) -> ActorState:
        vehicle_parameters = metric_cache.ego_state.car_footprint.vehicle_parameters
        x = float(ego_state_arr[StateIndex.X])
        y = float(ego_state_arr[StateIndex.Y])
        heading = float(ego_state_arr[StateIndex.HEADING])
        velocity_x = float(ego_state_arr[StateIndex.VELOCITY_X])
        velocity_y = float(ego_state_arr[StateIndex.VELOCITY_Y])
        return ActorState(
            x=x,
            y=y,
            heading=heading,
            speed=float(np.hypot(velocity_x, velocity_y)),
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            length=float(vehicle_parameters.length),
            width=float(vehicle_parameters.width),
            height=float(vehicle_parameters.height),
            token="ego",
            lane_object=select_best_lane(map_api, x, y, heading),
        )

    def simulate_traffic_agents(
        self, simulated_ego_states: npt.NDArray[np.float64], metric_cache: MetricCache
    ) -> List[DetectionsTracks]:
        vehicle_current_tracks = filter_tracked_objects_by_type(
            metric_cache.current_tracked_objects, TrackedObjectType.VEHICLE
        )[0]
        map_root = self._map_root_override or metric_cache.map_parameters.map_root
        map_api = get_maps_api(
            map_root,
            metric_cache.map_parameters.map_version,
            metric_cache.map_parameters.map_name,
        )
        runtime_actors = self._build_runtime_actors(vehicle_current_tracks, map_api)
        self._infer_agent_routes(runtime_actors, metric_cache, map_api)

        dt = self.future_trajectory_sampling.interval_length
        timestamp_us = metric_cache.timepoint.time_us
        traffic_light_status = getattr(metric_cache, "traffic_light_status", None)
        future_tracks: List[DetectionsTracks] = []

        for timestep in range(1, self.future_trajectory_sampling.num_poses + 1):
            if timestep - 1 < len(simulated_ego_states):
                ego_actor = self._build_ego_actor_from_simulated_state(
                    simulated_ego_states[timestep - 1],
                    metric_cache,
                    map_api,
                )
            else:
                ego_actor = build_ego_actor_from_ego_state(metric_cache.ego_state, map_api)

            open_loop_tracks = []
            if self._open_loop_detections_types and timestep - 1 < len(metric_cache.future_tracked_objects):
                open_loop_tracks = metric_cache.future_tracked_objects[timestep - 1].tracked_objects.get_tracked_objects_of_types(
                    self._open_loop_detections_types
                )
            timestep_traffic_light_status = {}
            if traffic_light_status is not None and timestep < len(traffic_light_status):
                timestep_traffic_light_status = self._normalize_traffic_light_status(traffic_light_status[timestep])

            snapshot_actors = [runtime.actor for runtime in runtime_actors.values()] + [ego_actor]
            updated_runtime: Dict[str, _TrafficActorRuntime] = {}
            for token, runtime_actor in runtime_actors.items():
                actor = runtime_actor.actor
                neighbors = build_neighbor_set(actor, snapshot_actors)
                acceleration_caps = self._compute_action_acceleration_caps(
                    actor=actor,
                    runtime_actor=runtime_actor,
                    neighbors=neighbors,
                    open_loop_tracks=open_loop_tracks,
                    traffic_light_status=timestep_traffic_light_status,
                    map_api=map_api,
                )
                lane_change_allowed = runtime_actor.cooldown_remaining <= 0.0 and runtime_actor.lane_change_elapsed <= 0.0
                action, acceleration, target_lane = decide_with_mobil(
                    actor=actor,
                    neighbors=neighbors,
                    route_command=None,
                    idm=self._idm,
                    mobil=self._mobil,
                    lane_change_allowed=lane_change_allowed,
                    acceleration_caps=acceleration_caps,
                )

                selected_target = runtime_actor.target_lane
                elapsed = runtime_actor.lane_change_elapsed
                cooldown_remaining = max(0.0, runtime_actor.cooldown_remaining - dt)
                if runtime_actor.lane_change_elapsed > 0.0:
                    selected_target = runtime_actor.target_lane
                elif action in ("LEFT", "RIGHT") and target_lane is not None:
                    selected_target = target_lane
                    elapsed = 1e-3
                    cooldown_remaining = self._mobil.p.cooldown_s
                else:
                    selected_target = actor.lane_object
                    elapsed = 0.0

                next_actor, next_elapsed = propagate_actor(
                    actor=actor,
                    map_api=map_api,
                    current_lane=actor.lane_object,
                    target_lane=selected_target,
                    acceleration=acceleration,
                    dt=dt,
                    route_lane_ids=runtime_actor.route_lane_ids or [],
                    route_roadblock_ids=runtime_actor.route_roadblock_ids or [],
                    lane_change_duration=self._lane_change_duration,
                    lane_change_elapsed=elapsed,
                )
                if next_elapsed >= self._lane_change_duration:
                    next_elapsed = 0.0
                updated_runtime[token] = _TrafficActorRuntime(
                    actor=next_actor,
                    target_lane=selected_target if next_elapsed > 0.0 else next_actor.lane_object,
                    lane_change_elapsed=next_elapsed,
                    cooldown_remaining=cooldown_remaining,
                    route_lane_ids=runtime_actor.route_lane_ids,
                    route_roadblock_ids=runtime_actor.route_roadblock_ids,
                )

            runtime_actors = updated_runtime
            current_timestamp = timestamp_us + int(round(timestep * dt * 1e6))
            future_tracks.append(self._build_detection_tracks(runtime_actors, current_timestamp))

        return future_tracks
