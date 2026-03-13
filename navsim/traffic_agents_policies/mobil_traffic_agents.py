from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.behavior.idm import IDM, IDMParams
from navsim.behavior.lane_change_trajectory import propagate_actor
from navsim.behavior.mobil import MobilModel, MobilParams, decide_with_mobil
from navsim.behavior.scene_adapter import (
    ActorState,
    build_actor_from_detection_track,
    build_ego_actor_from_ego_state,
    build_neighbor_set,
    select_best_lane,
)
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
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


class MobilTrafficAgentsPolicy(AbstractTrafficAgentsPolicy):
    """Reactive traffic agents policy based on IDM + MOBIL."""

    def __init__(
        self,
        future_trajectory_sampling: TrajectorySampling,
        idm_params: IDMParams = IDMParams(),
        mobil_params: MobilParams = MobilParams(),
        lane_change_duration: float = 3.0,
        map_root_override: Optional[str] = None,
    ):
        self.future_trajectory_sampling = future_trajectory_sampling
        self._idm = IDM(idm_params)
        self._mobil = MobilModel(mobil_params, self._idm)
        self._lane_change_duration = lane_change_duration
        self._map_root_override = map_root_override

    def get_list_of_simulated_object_types(self) -> List[TrackedObjectType]:
        return [TrackedObjectType.VEHICLE]

    def _build_runtime_actors(self, detections_tracks: DetectionsTracks, map_api) -> Dict[str, _TrafficActorRuntime]:
        runtime_actors: Dict[str, _TrafficActorRuntime] = {}
        for track in detections_tracks.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE):
            actor = build_actor_from_detection_track(track, map_api)
            runtime_actors[actor.token] = _TrafficActorRuntime(actor=actor)
        return runtime_actors

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
        route_roadblock_ids = list(metric_cache.route_lane_ids)
        runtime_actors = self._build_runtime_actors(vehicle_current_tracks, map_api)

        dt = self.future_trajectory_sampling.interval_length
        timestamp_us = metric_cache.timepoint.time_us
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

            snapshot_actors = [runtime.actor for runtime in runtime_actors.values()] + [ego_actor]
            updated_runtime: Dict[str, _TrafficActorRuntime] = {}
            for token, runtime_actor in runtime_actors.items():
                actor = runtime_actor.actor
                neighbors = build_neighbor_set(actor, snapshot_actors)
                lane_change_allowed = runtime_actor.cooldown_remaining <= 0.0 and runtime_actor.lane_change_elapsed <= 0.0
                action, acceleration, target_lane = decide_with_mobil(
                    actor=actor,
                    neighbors=neighbors,
                    route_command=None,
                    idm=self._idm,
                    mobil=self._mobil,
                    lane_change_allowed=lane_change_allowed,
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
                    route_roadblock_ids=route_roadblock_ids,
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
                )

            runtime_actors = updated_runtime
            current_timestamp = timestamp_us + int(round(timestep * dt * 1e6))
            future_tracks.append(self._build_detection_tracks(runtime_actors, current_timestamp))

        return future_tracks
