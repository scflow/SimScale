from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from navsim.behavior.idm import IDM
from navsim.behavior.scene_adapter import ActorState, NeighborSet, lane_relative_gap

KEEP = "KEEP"
LEFT = "LEFT"
RIGHT = "RIGHT"
UNKNOWN = "UNKNOWN"


@dataclass
class MobilParams:
    politeness: float = 0.3
    accel_threshold: float = 0.2
    safe_decel: float = 4.0
    cooldown_s: float = 3.0
    route_bias: float = 0.2
    command_left_idx: int = 0
    command_straight_idx: int = 1
    command_right_idx: int = 2
    command_unknown_idx: int = 3


class MobilModel:
    """A minimal MOBIL lane-change model."""

    def __init__(self, params: MobilParams, idm: IDM):
        self.p = params
        self.idm = idm

    def lane_change_gain(
        self,
        ego_now: float,
        ego_after: float,
        new_rear_now: float,
        new_rear_after: float,
        old_rear_now: float,
        old_rear_after: float,
        route_bonus: float = 0.0,
    ) -> float:
        return (
            (ego_after - ego_now)
            + self.p.politeness * ((new_rear_after - new_rear_now) + (old_rear_after - old_rear_now))
            + route_bonus
        )

    def safe(self, new_rear_after: float) -> bool:
        return new_rear_after > -self.p.safe_decel

    def decide(self, candidate_scores: dict[str, float]) -> str:
        best_action = max(candidate_scores, key=candidate_scores.get)
        if candidate_scores[best_action] <= self.p.accel_threshold:
            return KEEP
        return best_action


def decode_route_command(command: Optional[np.ndarray], params: MobilParams) -> str:
    if command is None:
        return UNKNOWN
    flattened = np.asarray(command).reshape(-1)
    if flattened.size == 0:
        return UNKNOWN
    index = int(np.argmax(flattened))
    if index == params.command_left_idx:
        return LEFT
    if index == params.command_right_idx:
        return RIGHT
    if index == params.command_straight_idx:
        return KEEP
    if index == params.command_unknown_idx:
        return UNKNOWN
    return UNKNOWN


def compute_route_bonus(route_command: Optional[np.ndarray], action: str, params: MobilParams) -> float:
    decoded = decode_route_command(route_command, params)
    if decoded == action:
        return params.route_bias
    if decoded in (LEFT, RIGHT) and action == KEEP:
        return -0.5 * params.route_bias
    return 0.0


def _idm_accel_with_lead(actor: ActorState, lead: Optional[ActorState], idm: IDM) -> float:
    if actor.lane_object is None:
        return idm.free_accel(actor.speed)
    if lead is None:
        return idm.free_accel(actor.speed)
    gap = lane_relative_gap(actor, lead, actor.lane_object)
    dv = actor.speed - lead.speed
    return idm.accel(actor.speed, gap, dv)


def _idm_accel_on_lane(
    actor: ActorState,
    lead: Optional[ActorState],
    lane_object,
    idm: IDM,
) -> float:
    if lane_object is None:
        return idm.free_accel(actor.speed)
    if lead is None:
        return idm.free_accel(actor.speed)
    gap = lane_relative_gap(actor, lead, lane_object)
    dv = actor.speed - lead.speed
    return idm.accel(actor.speed, gap, dv)


def decide_with_mobil(
    actor: ActorState,
    neighbors: NeighborSet,
    route_command: Optional[np.ndarray],
    idm: IDM,
    mobil: MobilModel,
    lane_change_allowed: bool = True,
) -> tuple[str, float, Optional[object]]:
    """Return action, longitudinal acceleration, and target lane object."""

    a_keep = _idm_accel_with_lead(actor, neighbors.curr_front, idm)
    scores = {KEEP: 0.0}
    outputs = {KEEP: (a_keep, actor.lane_object)}

    if actor.lane_object is None or not lane_change_allowed:
        return KEEP, a_keep, actor.lane_object

    candidates = (
        (LEFT, neighbors.left_front, neighbors.left_rear, neighbors.left_lane),
        (RIGHT, neighbors.right_front, neighbors.right_rear, neighbors.right_lane),
    )
    for action, front, rear, target_lane in candidates:
        if target_lane is None:
            scores[action] = -1e9
            continue

        ego_now = a_keep
        ego_after = _idm_accel_on_lane(actor, front, target_lane, idm)

        new_rear_now = _idm_accel_on_lane(rear, front, target_lane, idm) if rear is not None else 0.0
        new_rear_after = _idm_accel_on_lane(rear, actor, target_lane, idm) if rear is not None else 0.0
        old_rear_now = _idm_accel_with_lead(neighbors.curr_rear, actor, idm) if neighbors.curr_rear is not None else 0.0
        old_rear_after = (
            _idm_accel_with_lead(neighbors.curr_rear, neighbors.curr_front, idm)
            if neighbors.curr_rear is not None
            else 0.0
        )

        if not mobil.safe(new_rear_after):
            scores[action] = -1e9
            continue

        gain = mobil.lane_change_gain(
            ego_now=ego_now,
            ego_after=ego_after,
            new_rear_now=new_rear_now,
            new_rear_after=new_rear_after,
            old_rear_now=old_rear_now,
            old_rear_after=old_rear_after,
            route_bonus=compute_route_bonus(route_command, action, mobil.p),
        )
        scores[action] = gain
        outputs[action] = (ego_after, target_lane)

    best = mobil.decide(scores)
    acceleration, target_lane = outputs[best]
    return best, acceleration, target_lane

