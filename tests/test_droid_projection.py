#!/usr/bin/env python3
"""Sim-free unit tests for the canonical DROID action projection.

Covers the encode/decode inverse consistency required between training and
inference, the vector-level limiting rule, gripper semantics, mask gating,
and the guarantee that statistics generation and training resolve to the
same single implementation.
"""

import unittest

import numpy as np

from src import dataset_stats
from src import train_model
from src.droid_projection import (
    DROID_ACTION_SCALE,
    DROID_ACTION_DIM,
    GRIPPER_INDEX,
    SYNTHETIC_JOINT_INDEX,
    decode_arm_action,
    denormalize_gripper_position,
    encode_arm_action,
    limit_arm_action,
    normalize_gripper_position,
    project_controls,
)

SETTINGS = {
    "arm-state-indices": [0, 1, 2, 3, 4, 5],
    "arm-position-action-indices": [0, 1, 2, 3, 4, 5],
    "gripper-state-index": 6,
    "gripper-action-value-index": 6,
    "gripper-action-mask-index": 18,
    "gripper-open-position": 0.0,
    "gripper-closed-position": 0.376,
    "action-horizon": 15,
}

_rng = np.random.default_rng(7)


def _random_pose(n=32):
    return _rng.uniform(-3.0, 3.0, size=(n, 6))


class EncodeDecodeInverseTest(unittest.TestCase):
    def test_roundtrip_all_joints(self):
        q_current = _random_pose()
        deltas = _rng.uniform(-0.05, 0.05, size=q_current.shape)
        q_target = q_current + deltas
        mask = np.ones_like(q_current, dtype=bool)
        actions = encode_arm_action(q_current, q_target, mask)
        reconstructed = decode_arm_action(q_current, actions)
        np.testing.assert_allclose(reconstructed, q_target, rtol=0, atol=1e-9)

    def test_roundtrip_through_limit_for_representable_commands(self):
        q_current = _random_pose()
        deltas = _rng.uniform(-0.05, 0.05, size=q_current.shape)
        q_target = q_current + deltas
        actions = limit_arm_action(
            encode_arm_action(q_current, q_target, np.ones_like(q_current, bool))
        )
        np.testing.assert_allclose(
            decode_arm_action(q_current, actions), q_target, rtol=0, atol=1e-9
        )

    def test_hold_frames_encode_exact_zero(self):
        q_current = _random_pose()
        q_commanded = _random_pose()
        actions = encode_arm_action(
            q_current, q_commanded, np.zeros_like(q_current, bool)
        )
        np.testing.assert_array_equal(actions, np.zeros_like(q_current))

    def test_partial_masks_gate_per_joint(self):
        q_current = np.zeros((4, 6))
        q_commanded = np.full((4, 6), 0.04)
        mask = np.zeros((4, 6), dtype=bool)
        mask[:, 2] = True
        actions = encode_arm_action(q_current, q_commanded, mask)
        self.assertTrue((actions[:, [0, 1, 3, 4, 5]] == 0.0).all())
        np.testing.assert_allclose(actions[:, 2], 0.04 / DROID_ACTION_SCALE)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            encode_arm_action(np.zeros((3, 6)), np.zeros((3, 5)), np.ones((3, 6), bool))
        with self.assertRaises(ValueError):
            decode_arm_action(np.zeros(6), np.zeros(5))


class LimitActionTest(unittest.TestCase):
    def test_within_range_unchanged(self):
        action = np.array([0.2, -0.5, 0.0, 0.9, -1.0, 1.0])
        np.testing.assert_array_equal(limit_arm_action(action), action)

    def test_over_range_scales_whole_vector(self):
        action = np.array([2.0, 0.5, -0.5, 0.0, 0.0, 0.0])
        limited = limit_arm_action(action)
        np.testing.assert_allclose(limited, action / 2.0)
        self.assertAlmostEqual(float(np.max(np.abs(limited))), 1.0)

    def test_empty_action(self):
        np.testing.assert_array_equal(limit_arm_action(np.zeros(0)), np.zeros(0))


class GripperSemanticsTest(unittest.TestCase):
    def test_normalize_denormalize_roundtrip(self):
        for value in (0.0, 0.1, 0.229, 0.376):
            normalized = normalize_gripper_position(value, 0.0, 0.376)
            self.assertGreaterEqual(normalized, 0.0)
            self.assertLessEqual(normalized, 1.0)
            self.assertAlmostEqual(
                denormalize_gripper_position(normalized, 0.0, 0.376), value, places=6
            )

    def test_out_of_range_commands_clip(self):
        self.assertEqual(denormalize_gripper_position(1.5, 0.0, 0.376), 0.376)
        self.assertEqual(denormalize_gripper_position(-0.5, 0.0, 0.376), 0.0)
        self.assertEqual(normalize_gripper_position(0.5, 0.0, 0.376), 1.0)

    def test_invalid_calibration_raises(self):
        with self.assertRaises(ValueError):
            normalize_gripper_position(0.1, 0.376, 0.0)
        with self.assertRaises(ValueError):
            denormalize_gripper_position(0.5, 0.376, 0.0)


class ProjectControlsTest(unittest.TestCase):
    def _raw(self, n=16):
        states = np.zeros((n, 12))
        states[:, :6] = _rng.uniform(-3.0, 3.0, size=(n, 6))
        states[:, 6] = _rng.uniform(0.0, 0.376, size=n)
        actions = np.zeros((n, 72))
        return states, actions

    def test_state_convention_unchanged(self):
        states, actions = self._raw()
        projected_states, _ = project_controls(states, actions, SETTINGS)
        np.testing.assert_allclose(
            projected_states[:, :6], states[:, :6], rtol=0, atol=1e-6
        )
        np.testing.assert_array_equal(projected_states[:, 6], np.zeros(len(states)))
        np.testing.assert_allclose(
            projected_states[:, 7], states[:, 6] / 0.376, rtol=0, atol=1e-6
        )

    def test_action_layout_and_synthetic_joint(self):
        states, actions = self._raw()
        actions[:, :6] = states[:, :6] + 0.02
        actions[:, 12:18] = 1.0
        _, projected = project_controls(states, actions, SETTINGS)
        self.assertEqual(projected.shape, (len(states), DROID_ACTION_DIM))
        np.testing.assert_allclose(projected[:, :6], 0.02 / DROID_ACTION_SCALE)
        np.testing.assert_array_equal(
            projected[:, SYNTHETIC_JOINT_INDEX], np.zeros(len(states))
        )

    def test_uncommanded_frames_yield_zero_arm_action(self):
        states, actions = self._raw()
        actions[:, :6] = states[:, :6] + 0.5  # must be ignored without mask
        _, projected = project_controls(states, actions, SETTINGS)
        np.testing.assert_array_equal(projected[:, :6], np.zeros((len(states), 6)))

    def test_gripper_command_continuous_not_binary(self):
        states, actions = self._raw()
        actions[:, 6] = 0.229
        actions[:, 18] = 1.0
        _, projected = project_controls(states, actions, SETTINGS)
        np.testing.assert_allclose(
            projected[:, GRIPPER_INDEX], 0.229 / 0.376, rtol=0, atol=1e-6
        )

    def test_gripper_falls_back_to_state_when_uncommanded(self):
        states, actions = self._raw()
        _, projected = project_controls(states, actions, SETTINGS)
        np.testing.assert_allclose(
            projected[:, GRIPPER_INDEX], states[:, 6] / 0.376, rtol=0, atol=1e-6
        )


class SingleImplementationTest(unittest.TestCase):
    def test_training_uses_stats_projection(self):
        self.assertIs(train_model.convert_controls, dataset_stats.convert_controls)

    def test_stats_projection_is_canonical(self):
        states = _rng.uniform(-3.0, 3.0, size=(24, 12)).astype(np.float32)
        actions = np.zeros((24, 72), dtype=np.float32)
        actions[:, :6] = states[:, :6] + 0.01
        actions[:, 12:18] = 1.0
        expected_states, expected_actions = project_controls(states, actions, SETTINGS)
        got_states, got_actions = dataset_stats.convert_controls(
            states, actions, SETTINGS
        )
        np.testing.assert_array_equal(got_states, expected_states)
        np.testing.assert_array_equal(got_actions, expected_actions)

    def test_shared_scale_constant(self):
        self.assertEqual(DROID_ACTION_SCALE, 0.2)
        action = encode_arm_action(
            np.zeros((2, 6)), np.full((2, 6), 0.2), np.ones((2, 6), bool)
        )
        np.testing.assert_array_equal(action, np.ones((2, 6)))


if __name__ == "__main__":
    unittest.main()
