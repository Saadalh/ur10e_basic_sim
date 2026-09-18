#!/usr/bin/env python3
"""Sim-free grasp planning helpers for the staged pick controller.

These pure functions implement the decision logic of the two-stage vertical
approach and the single stall-based grasp close, so they can be unit tested
without Isaac Sim:

- the stock tutorial controller drives the end-effector from the start pose
  to high above the cube along an unconstrained joint-space path, which can
  swipe laterally through cubes;
- relative gripper deltas ratchet the finger target every step, which fights
  the contact and reads as close/reopen/close bouncing.

The staged controller (``src.pick_place``) instead drives to a pre-grasp
waypoint, descends with x/y locked, closes incrementally until the finger
stalls on the cube, and holds that exact position.
"""

import numpy as np


def within_tolerance(current_xyz, target_xyz, tolerance) -> bool:
    """Check whether a Cartesian pose has converged to its target."""
    current = np.asarray(current_xyz, dtype=np.float64)
    target = np.asarray(target_xyz, dtype=np.float64)
    return bool(np.linalg.norm(current - target) <= float(tolerance))


def approach_target(pick_xyz, offset_xyz, pre_grasp_height) -> np.ndarray:
    """Pre-grasp waypoint: above the pick position, full x/y already acquired."""
    pick = np.asarray(pick_xyz, dtype=np.float64)
    offset = np.asarray(offset_xyz, dtype=np.float64)
    return np.array(
        [pick[0] + offset[0], pick[1] + offset[1], pick[2] + float(pre_grasp_height)]
    )


def grasp_target(pick_xyz, offset_xyz) -> np.ndarray:
    """Grasp pose the fingers must reach before closing."""
    pick = np.asarray(pick_xyz, dtype=np.float64)
    offset = np.asarray(offset_xyz, dtype=np.float64)
    return pick + offset


def ensure_base_pick_state(controller, picking_position) -> None:
    """Restore the per-episode state the stock controller sets in events 0/1.

    Staged overrides bypass those branches, but the stock transport phases
    still read ``_current_target_x/_current_target_y`` (blend origin) and
    ``_h0`` (height base). Call before delegating events 4+ to super().
    """
    controller._current_target_x = float(picking_position[0])
    controller._current_target_y = float(picking_position[1])
    controller._h0 = float(picking_position[2])


def descend_target(current_xyz, goal_xyz, step) -> np.ndarray:
    """Next descent setpoint: x/y snapped to the goal, z stepped toward it.

    Snapping x/y to the goal on every setpoint is what keeps the descent
    strictly vertical instead of drifting diagonally.
    """
    current = np.asarray(current_xyz, dtype=np.float64)
    goal = np.asarray(goal_xyz, dtype=np.float64)
    step = float(step)
    if step <= 0.0:
        raise ValueError(f"descent step must be positive, got {step}")
    z_motion = np.clip(goal[2] - current[2], -step, step)
    return np.array([goal[0], goal[1], current[2] + z_motion])


def close_step_target(current, open_position, closed_position, increments) -> float:
    """Next incremental finger setpoint from open toward closed.

    Works for either convention (closed above or below open) and clamps to
    the [open, closed] range, so repeated calls can never ratchet past the
    physical end of travel the way relative deltas do.
    """
    current = float(current)
    opened = float(open_position)
    closed = float(closed_position)
    low, high = (opened, closed) if opened <= closed else (closed, opened)
    span = high - low
    if span <= 0.0:
        raise ValueError(f"open ({opened}) and closed ({closed}) positions must differ")
    step = span / max(1, int(increments))
    direction = 1.0 if closed >= opened else -1.0
    return float(min(high, max(low, current + direction * step)))


class GraspStallDetector:
    """Detect when a closing finger has made contact and stopped moving.

    A stall is reported once the finger has travelled at least ``min_travel``
    from its first observed position and then moved less than
    ``position_eps`` for ``steady_steps`` consecutive updates. The travel
    guard prevents an already-stationary finger from reading as stalled.
    """

    def __init__(self, position_eps: float, steady_steps: int, min_travel: float = 0.0):
        if float(position_eps) <= 0.0:
            raise ValueError(f"position_eps must be positive, got {position_eps}")
        if int(steady_steps) < 1:
            raise ValueError(f"steady_steps must be at least 1, got {steady_steps}")
        if float(min_travel) < 0.0:
            raise ValueError(f"min_travel must be non-negative, got {min_travel}")
        self.position_eps = float(position_eps)
        self.steady_steps = int(steady_steps)
        self.min_travel = float(min_travel)
        self.reset()

    def reset(self) -> None:
        self._first = None
        self._previous = None
        self._steady = 0
        self.stalled = False

    def update(self, position) -> bool:
        """Feed one finger-joint reading; return True once stalled."""
        position = float(position)
        if self._first is None:
            self._first = position
            self._previous = position
            return False
        if abs(position - self._previous) <= self.position_eps:
            self._steady += 1
        else:
            self._steady = 0
        self._previous = position
        if (
            abs(position - self._first) >= self.min_travel
            and self._steady >= self.steady_steps
        ):
            self.stalled = True
        return self.stalled


def merge_gripper_hold(joint_positions, dof_index: int, value: float) -> list:
    """Return a copy of a joint-target list with one DOF pinned to a value.

    ``None`` entries (uncommanded joints) are preserved, so an arm-only
    Cartesian action can carry an explicit gripper hold without altering
    any other command.
    """
    dof_index = int(dof_index)
    if dof_index < 0:
        raise ValueError(f"dof_index must be non-negative, got {dof_index}")
    merged = list(joint_positions) if joint_positions is not None else []
    if dof_index >= len(merged):
        merged.extend([None] * (dof_index + 1 - len(merged)))
    merged[dof_index] = float(value)
    return merged


def scatter_merged_action(
    joint_positions,
    joint_velocities,
    joint_indices,
    hold_dof: int,
    hold_value: float,
    dof_count: int,
    joint_efforts=None,
) -> tuple:
    """Scatter a subset action into full-width lists with a gripper hold.

    Cartesian controllers return arm positions/velocities paired positionally
    with arm indices, while the dataset recorder requires every attribute to
    pair with the same shared indices. Scattering into full-width lists with
    ``None`` (uncommanded) elsewhere keeps all attributes consistent no
    matter the subset size or ordering. Commanded efforts are refused loudly
    rather than silently dropped.
    """
    hold_dof = int(hold_dof)
    dof_count = int(dof_count)
    if dof_count < 1:
        raise ValueError(f"dof_count must be at least 1, got {dof_count}")
    if not 0 <= hold_dof < dof_count:
        raise ValueError(f"hold_dof {hold_dof} outside [0, {dof_count})")
    if joint_positions is None:
        raise ValueError(
            "cannot merge a gripper hold into an action without joint positions"
        )
    if joint_indices is None:
        joint_indices = list(range(len(joint_positions)))
    if len(joint_positions) != len(joint_indices):
        raise ValueError(
            f"Cartesian action has {len(joint_positions)} positions "
            f"for {len(joint_indices)} indices"
        )
    if joint_velocities is not None and len(joint_velocities) != len(joint_indices):
        raise ValueError(
            f"Cartesian action has {len(joint_velocities)} velocities "
            f"for {len(joint_indices)} indices"
        )
    if joint_efforts is not None and any(value is not None for value in joint_efforts):
        raise ValueError("refusing to drop commanded joint efforts")
    full_positions: list = [None] * dof_count
    full_velocities: list = [None] * dof_count
    for position, index in zip(joint_positions, joint_indices):
        if not 0 <= int(index) < dof_count:
            raise ValueError(f"joint index {index} outside [0, {dof_count})")
        full_positions[int(index)] = position
    if joint_velocities is not None:
        for velocity, index in zip(joint_velocities, joint_indices):
            full_velocities[int(index)] = velocity
    full_positions[hold_dof] = float(hold_value)
    return full_positions, full_velocities
