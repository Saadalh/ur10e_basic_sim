#!/usr/bin/env python3
"""Canonical DROID-compatible 8-D action projection (sim-free, pure NumPy).

This module is the single implementation of the UR10e <-> DROID control
representation used by dataset statistics, training, and inference:

Training (encode)::

    a = (q_target - q_current) / DROID_ACTION_SCALE

Inference (decode)::

    q_target = q_current + DROID_ACTION_SCALE * a

The six arm dimensions are normalized joint-motion commands for the six
physical UR10e joints reconstructed from demonstrated absolute
joint-position targets (never raw RMPflow velocity targets, never
integrated velocities, never forward-filled estimates). Dimension 6 is the
synthetic seventh joint and remains 0; dimension 7 is absolute normalized
gripper position. The model action shape stays 8-D.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Joint displacement (radians) produced by a unit DROID arm action in one
# control step. Shared by the training projection and the inference decoder
# so the two directions can never drift apart.
DROID_ACTION_SCALE = 0.2

# Width of the model action vector: six arm commands, one synthetic joint,
# one gripper command.
DROID_ACTION_DIM = 8

# Position of the synthetic seventh arm joint inside the 8-D action.
SYNTHETIC_JOINT_INDEX = 6

# Position of the absolute normalized gripper command inside the 8-D action.
GRIPPER_INDEX = 7


def encode_arm_action(
    q_observed: np.ndarray, q_commanded: np.ndarray, commanded_mask: np.ndarray
) -> np.ndarray:
    """Project demonstrated joint motion into DROID arm actions.

    Frames without a position command (mask false) encode as exact zeros,
    which is the correct hold action — not an estimate.
    """
    q_observed = np.asarray(q_observed, dtype=np.float64)
    q_commanded = np.asarray(q_commanded, dtype=np.float64)
    mask = np.asarray(commanded_mask, dtype=bool)
    if q_observed.shape != q_commanded.shape or q_observed.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: observed {q_observed.shape}, commanded {q_commanded.shape}, "
            f"mask {mask.shape}"
        )
    return np.where(mask, (q_commanded - q_observed) / DROID_ACTION_SCALE, 0.0)


def limit_arm_action(arm_action: np.ndarray) -> np.ndarray:
    """Apply DROID's vector-level joint_velocity_to_delta limiting.

    If the largest absolute joint command exceeds 1, the entire six-joint
    vector is divided by that maximum. Joints are never clipped
    independently, so the motion direction is always preserved.
    """
    arm_action = np.asarray(arm_action, dtype=np.float64)
    peak = float(np.max(np.abs(arm_action))) if arm_action.size else 0.0
    if peak > 1.0:
        return arm_action / peak
    return arm_action


def decode_arm_action(q_current: np.ndarray, arm_action: np.ndarray) -> np.ndarray:
    """Invert the training projection: joint-position target for one step."""
    q_current = np.asarray(q_current, dtype=np.float64)
    arm_action = np.asarray(arm_action, dtype=np.float64)
    if q_current.shape != arm_action.shape:
        raise ValueError(
            f"Shape mismatch: current {q_current.shape}, action {arm_action.shape}"
        )
    return q_current + DROID_ACTION_SCALE * arm_action


def normalize_gripper_position(value: float, opened: float, closed: float) -> float:
    """Map an absolute gripper position into normalized [0, 1] space."""
    opened = float(opened)
    closed = float(closed)
    if closed <= opened:
        raise ValueError(
            "gripper-closed-position must be greater than gripper-open-position"
        )
    return float(np.clip((float(value) - opened) / (closed - opened), 0.0, 1.0))


def denormalize_gripper_position(action: float, opened: float, closed: float) -> float:
    """Map a normalized gripper command back to an absolute position."""
    opened = float(opened)
    closed = float(closed)
    if closed <= opened:
        raise ValueError(
            "gripper-closed-position must be greater than gripper-open-position"
        )
    return opened + float(np.clip(float(action), 0.0, 1.0)) * (closed - opened)


def _require_indices(values: Any, length: int, name: str) -> list[int]:
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f"Exactly {length} {name} are required")
    indices = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must contain non-negative integer indexes")
        indices.append(value)
    if len(set(indices)) != len(indices):
        raise ValueError(f"{name} must not contain duplicate indexes")
    return indices


def project_controls(
    raw_states: np.ndarray, raw_actions: np.ndarray, settings: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Project collected UR10e controls into 8-D state and DROID actions.

    State convention (unchanged): ``state[0:6]`` are the six physical joint
    positions, ``state[6]`` is 0, ``state[7]`` is the normalized current
    gripper position. Actions: ``action[:6]`` are DROID arm actions
    reconstructed from absolute commanded joint positions,
    ``action[6]`` is 0, ``action[7]`` is the normalized absolute gripper
    command where commanded and the normalized current gripper position
    (hold) otherwise.
    """
    arm_state_indices = _require_indices(
        settings["arm-state-indices"], 6, "arm-state-indices"
    )
    arm_action_indices = _require_indices(
        settings["arm-position-action-indices"], 6, "arm-position-action-indices"
    )
    if raw_states.ndim != 2 or max(arm_state_indices) >= raw_states.shape[1]:
        raise ValueError(
            f"Invalid arm-state-indices for state shape {raw_states.shape}"
        )
    if raw_actions.ndim != 2 or max(arm_action_indices) >= raw_actions.shape[1]:
        raise ValueError(
            f"Invalid arm-position-action-indices for action shape {raw_actions.shape}"
        )

    gripper_state_index = int(settings["gripper-state-index"])
    gripper_value_index = int(settings["gripper-action-value-index"])
    gripper_mask_index = int(settings["gripper-action-mask-index"])
    opened = float(settings["gripper-open-position"])
    closed = float(settings["gripper-closed-position"])
    if closed <= opened:
        raise ValueError(
            "gripper-closed-position must be greater than gripper-open-position"
        )
    if gripper_state_index >= raw_states.shape[1]:
        raise ValueError(
            f"Invalid gripper-state-index for state shape {raw_states.shape}"
        )
    if max(gripper_value_index, gripper_mask_index) >= raw_actions.shape[1]:
        raise ValueError(
            f"Invalid gripper action index for action shape {raw_actions.shape}"
        )

    gripper_state = np.clip(
        (raw_states[:, gripper_state_index] - opened) / (closed - opened), 0.0, 1.0
    )
    states = np.zeros((len(raw_states), DROID_ACTION_DIM), dtype=np.float32)
    states[:, :6] = raw_states[:, arm_state_indices]
    states[:, GRIPPER_INDEX] = gripper_state

    # Position masks live 12 columns after their values in the 72-D schema.
    arm_masks = raw_actions[:, [index + 12 for index in arm_action_indices]] > 0.5
    actions = np.zeros((len(raw_actions), DROID_ACTION_DIM), dtype=np.float32)
    actions[:, :6] = encode_arm_action(
        raw_states[:, arm_state_indices],
        raw_actions[:, arm_action_indices],
        arm_masks,
    )
    commanded = raw_actions[:, gripper_mask_index] > 0.5
    gripper_command = np.clip(
        (raw_actions[:, gripper_value_index] - opened) / (closed - opened), 0.0, 1.0
    )
    actions[:, GRIPPER_INDEX] = np.where(commanded, gripper_command, gripper_state)
    return states, actions
