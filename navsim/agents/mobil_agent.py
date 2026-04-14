from __future__ import annotations

from typing import Optional

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.behavior.idm import IDM, IDMParams
from navsim.behavior.lane_change_trajectory import rollout_ego_trajectory
from navsim.behavior.mobil import MobilModel, MobilParams, decide_with_mobil
from navsim.behavior.scene_adapter import build_ego_actor_and_route_from_scene, build_neighbor_set
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory


class MobilAgent(AbstractAgent):
    """A privileged research agent based on IDM + MOBIL."""

    requires_scene = True

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4.0, interval_length=0.1),
        idm_params: IDMParams = IDMParams(),
        mobil_params: MobilParams = MobilParams(),
        lane_change_duration: float = 3.0,
    ):
        super().__init__(trajectory_sampling, requires_scene=True)
        self._idm_params = idm_params
        self._mobil_params = mobil_params
        self._lane_change_duration = lane_change_duration
        self._idm = IDM(idm_params)
        self._mobil = MobilModel(mobil_params, self._idm)

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        self._idm = IDM(self._idm_params)
        self._mobil = MobilModel(self._mobil_params, self._idm)

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input: AgentInput, scene: Optional[Scene] = None) -> Trajectory:
        if scene is None:
            raise ValueError("MobilAgent requires the privileged Scene input.")

        ego_actor, actors, route_lane_ids, route_roadblock_ids = build_ego_actor_and_route_from_scene(scene)
        neighbors = build_neighbor_set(ego_actor, actors)
        route_command = agent_input.ego_statuses[-1].driving_command if agent_input.ego_statuses else None
        action, acceleration, target_lane = decide_with_mobil(
            actor=ego_actor,
            neighbors=neighbors,
            route_command=route_command,
            idm=self._idm,
            mobil=self._mobil,
        )
        if action == "KEEP":
            target_lane = ego_actor.lane_object

        return rollout_ego_trajectory(
            ego=ego_actor,
            current_lane=ego_actor.lane_object,
            target_lane=target_lane,
            acceleration=acceleration,
            trajectory_sampling=self._trajectory_sampling,
            route_lane_ids=route_lane_ids,
            route_roadblock_ids=route_roadblock_ids,
            lane_change_duration=self._lane_change_duration,
        )
