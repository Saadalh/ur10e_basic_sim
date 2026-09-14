#!/usr/bin/env python3
"""Staged pick controller: vertical approach plus single stall-based grasp.

This module is imported after Isaac Sim has started and after the UR10e
controller example directory is on ``sys.path`` (see ``collect_cube_data``),
because it builds on the vendored tutorial state machine.

Stock behavior that is kept: settle (event 2), release (event 7), retreat
(events 8/9), and the Cartesian transport (events 4-6).

Behavior that is replaced:

- Events 0/1 (timed diagonal transit) become gated stages: drive to a
  pre-grasp waypoint above the cube, then descend with x/y locked. The
  stock time-based advance could move on mid-transit and swipe laterally
  through cubes.
- Event 3 (relative-delta close ratchet) becomes an incremental close that
  stops at the first detected finger stall and holds that exact position.
  Combined with absolute gripper positions (no ``action_deltas``), the
  gripper closes exactly once instead of fighting the contact.
- Events 4-6 reuse the stock Cartesian motion with the stall pose merged in,
  so a held cube can never be commanded open mid-transport.
"""

import numpy as np
from isaacsim.core.utils.types import ArticulationAction

from controller.pick_place import PickPlaceController

from src.collection_runtime import ManipulationFailure
from src.grasp_logic import (
    GraspStallDetector,
    approach_target,
    close_step_target,
    descend_target,
    grasp_target,
    merge_gripper_hold,
    within_tolerance,
)


class StagedPickPlaceController(PickPlaceController):
    """PickPlaceController with gated vertical approach and stall grasp."""

    def __init__(
        self,
        name,
        robot_articulation,
        gripper,
        robot,
        events_dt=None,
        *,
        pre_grasp_height=0.20,
        approach_tolerance=0.01,
        descent_step=0.01,
        xy_lock_tolerance=0.01,
        max_approach_steps=2000,
        grasp_increments=40,
        grasp_position_eps=0.25,
        grasp_steady_steps=10,
        grasp_min_travel_fraction=0.1,
        max_grasp_steps=400,
    ):
        super().__init__(
            name=name,
            robot_articulation=robot_articulation,
            gripper=gripper,
            events_dt=events_dt,
        )
        self._robot = robot
        self._gripper = gripper
        self._pre_grasp_height = float(pre_grasp_height)
        self._approach_tolerance = float(approach_tolerance)
        self._descent_step = float(descent_step)
        self._xy_lock_tolerance = float(xy_lock_tolerance)
        self._max_approach_steps = int(max_approach_steps)
        self._grasp_increments = int(grasp_increments)
        self._grasp_position_eps = float(grasp_position_eps)
        self._grasp_steady_steps = int(grasp_steady_steps)
        self._grasp_min_travel_fraction = float(grasp_min_travel_fraction)
        self._max_grasp_steps = int(max_grasp_steps)
        self._drive_dof = int(gripper.active_joint_indices[0])
        self._open_position = float(gripper.joint_opened_positions[0])
        self._closed_position = float(gripper.joint_closed_positions[0])
        self._reset_staging()

    def _reset_staging(self) -> None:
        self._approach_steps = 0
        self._grasp_steps = 0
        self._grasp_hold = None
        span = abs(self._closed_position - self._open_position)
        self._stall = GraspStallDetector(
            position_eps=self._grasp_position_eps,
            steady_steps=self._grasp_steady_steps,
            min_travel=span * self._grasp_min_travel_fraction,
        )

    def reset(self, end_effector_initial_height=None, events_dt=None) -> None:
        super().reset(end_effector_initial_height, events_dt)
        self._reset_staging()

    @property
    def grasp_hold(self):
        """Stalled finger position held since the grasp, or None if open."""
        return self._grasp_hold

    def _ee_position(self) -> np.ndarray:
        position, _ = self._robot.end_effector.get_world_pose()
        return np.asarray(position, dtype=np.float64)

    def _finger_position(self) -> float:
        return float(self._robot.get_joint_positions([self._drive_dof])[0])

    def _hold_action(self) -> ArticulationAction:
        return ArticulationAction(joint_positions=[None] * len(self._robot.dof_names))

    def _drive_cartesian(self, target_xyz, orientation):
        return self._cspace_controller.forward(
            target_end_effector_position=np.asarray(target_xyz, dtype=np.float64),
            target_end_effector_orientation=orientation,
        )

    def forward(
        self,
        picking_position,
        placing_position,
        current_joint_positions,
        end_effector_offset=None,
        end_effector_orientation=None,
    ):
        if self._pause or self.is_done():
            return super().forward(
                picking_position,
                placing_position,
                current_joint_positions,
                end_effector_offset,
                end_effector_orientation,
            )
        offset = (
            np.zeros(3)
            if end_effector_offset is None
            else np.asarray(end_effector_offset, dtype=np.float64)
        )
        if self._event == 0:
            return self._step_approach(
                picking_position, offset, end_effector_orientation
            )
        if self._event == 1:
            return self._step_descend(
                picking_position, offset, end_effector_orientation
            )
        if self._event == 3:
            return self._step_grasp()
        if self._event in (4, 5, 6) and self._grasp_hold is not None:
            action = super().forward(
                picking_position,
                placing_position,
                current_joint_positions,
                end_effector_offset,
                end_effector_orientation,
            )
            positions = (
                list(action.joint_positions)
                if action.joint_positions is not None
                else [None] * len(self._robot.dof_names)
            )
            action.joint_positions = merge_gripper_hold(
                positions, self._drive_dof, self._grasp_hold
            )
            return action
        return super().forward(
            picking_position,
            placing_position,
            current_joint_positions,
            end_effector_offset,
            end_effector_orientation,
        )

    def _step_approach(self, picking_position, offset, orientation):
        goal = approach_target(picking_position, offset, self._pre_grasp_height)
        if within_tolerance(self._ee_position(), goal, self._approach_tolerance):
            print(f"Pre-grasp waypoint reached: {goal}", flush=True)
            self._event = 1
            self._approach_steps = 0
            return self._hold_action()
        self._approach_steps += 1
        if self._approach_steps > self._max_approach_steps:
            raise ManipulationFailure(
                f"End effector failed to reach pre-grasp {goal} "
                f"from {self._ee_position()} within {self._max_approach_steps} steps"
            )
        return self._drive_cartesian(goal, orientation)

    def _step_descend(self, picking_position, offset, orientation):
        goal = grasp_target(picking_position, offset)
        current = self._ee_position()
        if within_tolerance(current, goal, self._approach_tolerance):
            print(f"Grasp pose reached: {goal}", flush=True)
            self._event = 2
            self._approach_steps = 0
            return self._hold_action()
        self._approach_steps += 1
        if self._approach_steps > self._max_approach_steps:
            raise ManipulationFailure(
                f"End effector failed to descend to grasp pose {goal} "
                f"from {current} within {self._max_approach_steps} steps"
            )
        if np.linalg.norm(current[:2] - goal[:2]) > self._xy_lock_tolerance:
            # Drifted sideways: recenter at the current height before descending.
            target = np.array([goal[0], goal[1], current[2]])
        else:
            target = descend_target(current, goal, self._descent_step)
        return self._drive_cartesian(target, orientation)

    def _step_grasp(self):
        finger = self._finger_position()
        if self._grasp_hold is None and self._stall.update(finger):
            self._grasp_hold = finger
            print(
                f"Gripper stalled at finger position {finger:.4f}; holding.",
                flush=True,
            )
        if self._grasp_hold is not None:
            if self._event == 3:
                self._event = 4
            positions = merge_gripper_hold(
                [None] * len(self._robot.dof_names), self._drive_dof, self._grasp_hold
            )
            return ArticulationAction(joint_positions=positions)
        self._grasp_steps += 1
        if self._grasp_steps > self._max_grasp_steps:
            raise ManipulationFailure(
                f"Gripper never stalled on the cube within {self._max_grasp_steps} "
                f"steps (last finger position {finger:.4f})"
            )
        target = close_step_target(
            finger, self._open_position, self._closed_position, self._grasp_increments
        )
        positions = merge_gripper_hold(
            [None] * len(self._robot.dof_names), self._drive_dof, target
        )
        return ArticulationAction(joint_positions=positions)
