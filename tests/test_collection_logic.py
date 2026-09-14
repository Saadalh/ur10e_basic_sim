#!/usr/bin/env python3
"""Sim-free unit tests for collection planning, retries, and shutdown order.

These tests run in the lightweight project virtualenv (no Isaac Sim, no GPU)
and cover the logic that decides whether a randomized scene is attempted:

- task-aware placement clearance, including the obstructed seed-46 scene
  observed on 2026-09-11 (green destination 3.9 cm from the blue cube),
- deterministic scene sampling,
- deferred SIGINT/SIGTERM handling,
- finalize-before-close ordering in collection shutdown,
- bounded video-encoder shutdown verification,
- shutdown watchdog (stack dump + forced exit with a recorded code).
"""

import logging
import math
import signal
import tempfile
import threading
import unittest
from itertools import permutations
from pathlib import Path

import numpy as np

from src.collection_runtime import (
    CollectionStopped,
    ManipulationFailure,
    StopRequest,
    _watchdog_expired,
    close_with_watchdog,
    ensure_encoders_stopped,
    finalize_collection,
)
from src.cube_environment import (
    COLLECTION_CONFIG,
    CUBE_CONFIG,
    _random_cube_positions,
    build_task_schedule,
    placement_is_clear,
    select_episode_schedule,
)
from src.grasp_logic import (
    GraspStallDetector,
    approach_target,
    close_step_target,
    descend_target,
    grasp_target,
    merge_gripper_hold,
    within_tolerance,
)


def _offset_task(source="green", offset=(0.0, -0.25, 0.0)):
    return {
        "description": f"move the {source} cube to the right",
        "kind": "offset",
        "colors": (source, "blue"),
        "target_offset": np.asarray(offset, dtype=np.float64),
    }


def _stack_task(source="red", target="blue"):
    return {
        "description": f"stack the {source} cube on the {target} cube",
        "kind": "stack",
        "colors": (source, target),
        "target_offset": None,
    }


class PlacementIsClearTest(unittest.TestCase):
    def test_seed46_obstructed_scene_is_rejected(self):
        """Exact scene from the 2026-09-11 episode-5 failure."""
        positions = {
            "red": np.array([0.5590278, -0.2676479, 1.4312834]),
            "green": np.array([0.8236114, -0.07973151, 1.4312834]),
            "blue": np.array([0.8277066, -0.29052, 1.4312834]),
        }
        destination = positions["green"] + np.array([0.0, -0.25, 0.0])
        gap = float(np.linalg.norm(destination[:2] - positions["blue"][:2]))
        self.assertLess(gap, float(CUBE_CONFIG["size"]))
        self.assertFalse(
            placement_is_clear(positions, _offset_task(), np.eye(3)),
            f"destination {destination} is only {gap:.4f}m from the blue cube",
        )

    def test_clear_scene_is_accepted(self):
        positions = {
            "red": np.array([0.45, 0.30, 1.43]),
            "green": np.array([0.65, -0.10, 1.43]),
            "blue": np.array([0.85, 0.30, 1.43]),
        }
        self.assertTrue(placement_is_clear(positions, _offset_task(), np.eye(3)))

    def test_source_start_position_is_excluded(self):
        """The destination may be near where the source cube starts; it moves away."""
        positions = {
            "red": np.array([0.45, 0.30, 1.43]),
            "green": np.array([0.65, 0.30, 1.43]),
            "blue": np.array([0.85, 0.30, 1.43]),
        }
        task = _offset_task(source="green", offset=(0.0, -0.25, 0.0))
        # Destination lands 5 cm from the green start pose: allowed, only the
        # blue cube constrains placement.
        self.assertTrue(placement_is_clear(positions, task, np.eye(3)))

    def test_stack_task_ignores_target_but_not_third_cube(self):
        positions = {
            "red": np.array([0.45, 0.30, 1.43]),
            "green": np.array([0.80, 0.30, 1.43]),
            "blue": np.array([0.65, -0.10, 1.43]),
        }
        # Stacking red onto blue: blue may be at the destination.
        self.assertTrue(placement_is_clear(positions, _stack_task(), np.eye(3)))
        # ...but not when the green cube crowds the destination.
        positions["green"] = np.array([0.66, -0.09, 1.43])
        self.assertFalse(placement_is_clear(positions, _stack_task(), np.eye(3)))

    def test_destination_must_stay_on_table(self):
        positions = {
            "red": np.array([0.45, 0.30, 1.43]),
            "green": np.array([0.80, 0.30, 1.43]),
            "blue": np.array([0.60, -0.30, 1.43]),
        }
        lower = np.array([0.30, -0.45, 1.30])
        upper = np.array([1.00, 0.45, 1.60])
        self.assertTrue(
            placement_is_clear(positions, _offset_task(), np.eye(3), (lower, upper))
        )
        task = _offset_task(source="green", offset=(0.0, -0.75, 0.0))
        self.assertFalse(placement_is_clear(positions, task, np.eye(3), (lower, upper)))

    def test_no_task_disables_placement_check(self):
        self.assertTrue(placement_is_clear({}, None, np.eye(3)))

    def test_clearance_threshold_uses_diagonal_footprint(self):
        size = float(CUBE_CONFIG["size"])
        margin = float(CUBE_CONFIG["placement_clearance"])
        expected = math.sqrt(2) * size + margin
        task = _offset_task(source="green", offset=(0.0, 0.0, 0.0))
        base = {
            "red": np.array([0.45, 0.30, 1.43]),
            "green": np.array([0.0, 0.0, 1.43]),
        }
        just_clear = dict(base, blue=np.array([expected, 0.0, 1.43]))
        self.assertTrue(placement_is_clear(just_clear, task, np.eye(3)))
        just_blocked = dict(base, blue=np.array([expected - 1e-3, 0.0, 1.43]))
        self.assertFalse(placement_is_clear(just_blocked, task, np.eye(3)))


class RandomCubePositionsTest(unittest.TestCase):
    def _sample(self, seed, variation, task):
        return _random_cube_positions(
            1.40,
            0.05,
            seed,
            np.array([0.0, 0.0, 1.42]),
            np.eye(3),
            variation,
            task=task,
        )

    def test_sampling_is_deterministic(self):
        task = _offset_task()
        first = self._sample(42, 1, task)
        second = self._sample(42, 1, task)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])

    def test_accepted_scenes_satisfy_placement_clearance(self):
        tasks = [_offset_task(), _stack_task()]
        for seed in range(40, 48):
            for variation in (1, 2, 3, 4):
                for task in tasks:
                    with self.subTest(
                        seed=seed, variation=variation, task=task["kind"]
                    ):
                        positions = self._sample(seed, variation, task)
                        self.assertEqual(set(positions), {"red", "green", "blue"})
                        self.assertTrue(placement_is_clear(positions, task, np.eye(3)))

    def test_impossible_scene_raises_manipulation_failure(self):
        original = dict(CUBE_CONFIG)
        CUBE_CONFIG["minimum_separation"] = 10.0
        CUBE_CONFIG["maximum_scene_samples"] = 5
        self.addCleanup(CUBE_CONFIG.update, original)
        with self.assertRaises(ManipulationFailure):
            self._sample(42, 1, _offset_task())


class StopRequestTest(unittest.TestCase):
    def test_check_passes_without_signal(self):
        stop = StopRequest()
        stop.check()  # must not raise

    def test_signal_defers_until_check(self):
        stop = StopRequest()
        previous = signal.getsignal(signal.SIGINT)
        stop.install()
        try:
            stop._request(signal.SIGINT, None)
            self.assertEqual(stop.signum, signal.SIGINT)
            with self.assertRaises(CollectionStopped):
                stop.check()
        finally:
            stop.restore()
        self.assertEqual(signal.getsignal(signal.SIGINT), previous)

    def test_restore_returns_previous_handlers(self):
        stop = StopRequest()
        before = {
            signum: signal.getsignal(signum)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }
        stop.install()
        try:
            self.assertIsNot(signal.getsignal(signal.SIGINT), before[signal.SIGINT])
        finally:
            stop.restore()
        for signum, handler in before.items():
            self.assertIs(signal.getsignal(signum), handler)


class FakeDataset:
    def __init__(self, num_episodes=2):
        self.calls = []
        self._num_episodes = num_episodes

    @property
    def num_episodes(self):
        return self._num_episodes

    def clear_episode_buffer(self):
        self.calls.append("clear")

    def finalize(self):
        self.calls.append("finalize")


class FailingDataset(FakeDataset):
    def __init__(self, failure):
        super().__init__()
        self._failure = failure

    def finalize(self):
        self.calls.append("finalize")
        raise self._failure


class FinalizeCollectionTest(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test.collection")
        self.logger.addHandler(logging.NullHandler())

    def test_finalize_stats_then_close_order(self):
        dataset = FakeDataset()
        calls = []

        def generate_stats():
            calls.append("stats")

        def close_app():
            calls.append("close")

        finalize_collection(
            dataset,
            discard=True,
            generate_stats=generate_stats,
            close_app=close_app,
            logger=self.logger,
        )
        self.assertEqual(dataset.calls, ["clear", "finalize"])
        self.assertEqual(calls, ["stats", "close"])

    def test_no_discard_without_frames(self):
        dataset = FakeDataset()
        finalize_collection(
            dataset,
            discard=False,
            generate_stats=lambda: None,
            close_app=lambda: None,
            logger=self.logger,
        )
        self.assertEqual(dataset.calls, ["finalize"])

    def test_no_dataset_only_closes(self):
        calls = []
        finalize_collection(
            None,
            discard=False,
            generate_stats=lambda: calls.append("stats"),
            close_app=lambda: calls.append("close"),
            logger=self.logger,
        )
        self.assertEqual(calls, ["close"])

    def test_finalize_failure_skips_stats_but_still_closes(self):
        dataset = FailingDataset(ValueError("no footer"))
        calls = []
        with self.assertRaises(RuntimeError):
            finalize_collection(
                dataset,
                discard=False,
                generate_stats=lambda: calls.append("stats"),
                close_app=lambda: calls.append("close"),
                logger=self.logger,
            )
        self.assertEqual(calls, ["close"])

    def test_stats_failure_still_closes_and_reports(self):
        dataset = FakeDataset()

        def generate_stats():
            raise OSError("openpi missing")

        with self.assertRaises(RuntimeError):
            finalize_collection(
                dataset,
                discard=False,
                generate_stats=generate_stats,
                close_app=lambda: None,
                logger=self.logger,
            )


class _CameraEncoderThread(threading.Thread):
    """Name-matched stand-in for LeRobot's daemon encoder worker."""

    def __init__(self, stop_event):
        super().__init__(name="fake-camera-encoder", daemon=True)
        self._stop_event = stop_event

    def run(self):
        self._stop_event.wait()


class FakeEncoder:
    def __init__(self, episode_active=False):
        self._episode_active = episode_active
        self.cancel_calls = 0

    def cancel_episode(self):
        self.cancel_calls += 1
        self._episode_active = False


class EnsureEncodersStoppedTest(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test.collection")
        self.logger.addHandler(logging.NullHandler())

    def test_no_encoder_is_a_noop(self):
        self.assertTrue(
            ensure_encoders_stopped(object(), timeout_seconds=1, logger=self.logger)
        )

    def test_inactive_encoder_is_not_cancelled(self):
        encoder = FakeEncoder(episode_active=False)

        class Dataset:
            _streaming_encoder = encoder

        self.assertTrue(
            ensure_encoders_stopped(Dataset(), timeout_seconds=1, logger=self.logger)
        )
        self.assertEqual(encoder.cancel_calls, 0)

    def test_active_episode_is_cancelled(self):
        encoder = FakeEncoder(episode_active=True)

        class Dataset:
            _streaming_encoder = encoder

        self.assertTrue(
            ensure_encoders_stopped(Dataset(), timeout_seconds=5, logger=self.logger)
        )
        self.assertEqual(encoder.cancel_calls, 1)

    def test_stuck_worker_raises_after_timeout(self):
        stop_event = threading.Event()
        worker = _CameraEncoderThread(stop_event)
        worker.start()
        self.addCleanup(stop_event.set)
        self.addCleanup(worker.join, 5)

        class Dataset:
            _streaming_encoder = FakeEncoder(episode_active=False)

        with self.assertRaises(RuntimeError):
            ensure_encoders_stopped(Dataset(), timeout_seconds=0.3, logger=self.logger)

    def test_cancel_failure_is_logged_but_verified(self):
        class BrokenEncoder(FakeEncoder):
            def cancel_episode(self):
                raise OSError("already torn down")

        class Dataset:
            _streaming_encoder = BrokenEncoder(episode_active=True)

        # No workers remain, so verification still passes; the cancel error
        # is logged rather than aborting shutdown.
        self.assertTrue(
            ensure_encoders_stopped(Dataset(), timeout_seconds=1, logger=self.logger)
        )


class CloseWithWatchdogTest(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test.collection")
        self.logger.addHandler(logging.NullHandler())
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def _dump(self):
        return Path(self._tmpdir.name) / "watchdog.log"

    def test_fast_close_cancels_watchdog_without_dump(self):
        calls = []
        close_with_watchdog(
            lambda: calls.append("close"),
            timeout_seconds=30,
            dump_file=self._dump(),
            exit_code=0,
            logger=self.logger,
        )
        self.assertEqual(calls, ["close"])
        self.assertFalse(self._dump().exists())

    def test_disabled_watchdog_calls_close_directly(self):
        with self.assertRaises(ValueError):
            close_with_watchdog(
                lambda: (_ for _ in ()).throw(ValueError("boom")),
                timeout_seconds=None,
                dump_file=self._dump(),
                exit_code=3,
                logger=self.logger,
            )
        self.assertFalse(self._dump().exists())

    def test_expiry_dumps_stacks_and_exits_with_code(self):
        exits = []
        _watchdog_expired(self._dump(), 42, _exit=lambda code: exits.append(code))
        self.assertEqual(exits, [42])
        content = self._dump().read_bytes()
        self.assertIn(b"terminating with exit code 42", content)
        # faulthandler header for the current thread's stack.
        self.assertIn(b"most recent call first", content)

    def test_expiry_without_dump_file_still_exits(self):
        exits = []
        _watchdog_expired(None, 7, _exit=lambda code: exits.append(code))
        self.assertEqual(exits, [7])


class FinalizeWatchdogTest(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test.collection")
        self.logger.addHandler(logging.NullHandler())
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def test_stuck_encoder_is_recorded_but_close_still_runs(self):
        stop_event = threading.Event()
        worker = _CameraEncoderThread(stop_event)
        worker.start()
        self.addCleanup(stop_event.set)
        self.addCleanup(worker.join, 5)

        class Dataset(FakeDataset):
            _streaming_encoder = FakeEncoder(episode_active=False)

        calls = []
        with self.assertRaises(RuntimeError):
            finalize_collection(
                Dataset(),
                discard=False,
                generate_stats=lambda: calls.append("stats"),
                close_app=lambda: calls.append("close"),
                logger=self.logger,
                shutdown_watchdog_seconds=30,
                watchdog_dump_file=Path(self._tmpdir.name) / "watchdog.log",
                exit_code=0,
                encoder_stop_timeout_seconds=0.3,
            )
        # Stats and close still ran despite the stuck worker.
        self.assertEqual(calls, ["stats", "close"])


class ScheduleTest(unittest.TestCase):
    def test_selection_is_deterministic(self):
        first = select_episode_schedule(10, seed=42)
        second = select_episode_schedule(10, seed=42)
        self.assertEqual(
            [entry["description"] for entry in first],
            [entry["description"] for entry in second],
        )

    def test_schedule_entries_are_well_formed(self):
        schedule = build_task_schedule()
        self.assertTrue(schedule)
        for entry in schedule:
            self.assertIn(entry["kind"], ("offset", "stack"))
            self.assertTrue(entry["description"])
            self.assertTrue(entry["colors"])
            if entry["kind"] == "offset":
                self.assertIsNotNone(entry["target_offset"])

    def test_schedule_count_matches_config(self):
        colors = list(COLLECTION_CONFIG["cubes"]["prim_paths"])
        expected = 0
        for template, count in COLLECTION_CONFIG["tasks"].items():
            mentions = sum(1 for color in colors if color.lower() in template.lower())
            expected += len(list(permutations(colors, mentions))) * count
        self.assertEqual(len(build_task_schedule()), expected)


class ExceptionTypesTest(unittest.TestCase):
    def test_retryable_failures_are_runtime_errors(self):
        self.assertIsInstance(ManipulationFailure("x"), RuntimeError)
        self.assertIsInstance(CollectionStopped("x"), RuntimeError)


class WaypointTest(unittest.TestCase):
    def test_approach_is_above_pick_with_xy_acquired(self):
        goal = approach_target([0.5, -0.2, 1.43], [0.0, 0.0, 0.125], 0.20)
        np.testing.assert_allclose(goal, [0.5, -0.2, 1.63])

    def test_grasp_target_adds_offset(self):
        goal = grasp_target([0.5, -0.2, 1.43], [0.0, 0.0, 0.125])
        np.testing.assert_allclose(goal, [0.5, -0.2, 1.555])

    def test_within_tolerance_uses_euclidean_norm(self):
        self.assertTrue(within_tolerance([0, 0, 0], [0.006, 0.006, 0.006], 0.011))
        self.assertFalse(within_tolerance([0, 0, 0], [0.006, 0.006, 0.006], 0.009))

    def test_descend_snaps_xy_and_steps_z(self):
        target = descend_target([0.4, 0.1, 1.60], [0.5, -0.2, 1.555], 0.01)
        np.testing.assert_allclose(target, [0.5, -0.2, 1.59])

    def test_descend_clamps_at_goal(self):
        target = descend_target([0.5, -0.2, 1.557], [0.5, -0.2, 1.555], 0.01)
        np.testing.assert_allclose(target, [0.5, -0.2, 1.555])

    def test_descend_rejects_non_positive_step(self):
        with self.assertRaises(ValueError):
            descend_target([0, 0, 1.6], [0, 0, 1.5], 0.0)


class CloseStepTargetTest(unittest.TestCase):
    def test_steps_from_open_toward_closed(self):
        self.assertAlmostEqual(close_step_target(0.0, 0.0, 40.0, 40), 1.0)
        self.assertAlmostEqual(close_step_target(39.5, 0.0, 40.0, 40), 40.0)

    def test_clamps_at_closed(self):
        self.assertAlmostEqual(close_step_target(40.0, 0.0, 40.0, 40), 40.0)

    def test_supports_reversed_convention(self):
        self.assertAlmostEqual(close_step_target(0.0, 0.0, -1.0, 4), -0.25)
        self.assertAlmostEqual(close_step_target(-1.0, 0.0, -1.0, 4), -1.0)

    def test_rejects_degenerate_range(self):
        with self.assertRaises(ValueError):
            close_step_target(0.0, 1.0, 1.0, 40)


class GraspStallDetectorTest(unittest.TestCase):
    def test_reports_stall_after_steady_readings(self):
        detector = GraspStallDetector(position_eps=0.25, steady_steps=3, min_travel=4.0)
        trace = [0.0, 1.0, 2.0, 3.0, 4.0, 4.1, 4.05, 4.1]
        self.assertEqual([detector.update(p) for p in trace], [False] * 7 + [True])
        self.assertTrue(detector.stalled)

    def test_moving_finger_never_stalls(self):
        detector = GraspStallDetector(position_eps=0.25, steady_steps=3, min_travel=1.0)
        for step in range(20):
            self.assertFalse(detector.update(float(step)))

    def test_stationary_finger_without_travel_is_not_a_grasp(self):
        detector = GraspStallDetector(position_eps=0.25, steady_steps=3, min_travel=4.0)
        for _ in range(10):
            self.assertFalse(detector.update(0.05))

    def test_reset_clears_state(self):
        detector = GraspStallDetector(position_eps=0.25, steady_steps=1, min_travel=0.0)
        self.assertFalse(detector.update(0.0))
        self.assertTrue(detector.update(0.0))
        detector.reset()
        self.assertFalse(detector.stalled)
        self.assertFalse(detector.update(5.0))

    def test_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            GraspStallDetector(position_eps=0.0, steady_steps=3)
        with self.assertRaises(ValueError):
            GraspStallDetector(position_eps=0.25, steady_steps=0)
        with self.assertRaises(ValueError):
            GraspStallDetector(position_eps=0.25, steady_steps=3, min_travel=-1.0)


class MergeGripperHoldTest(unittest.TestCase):
    def test_pins_one_dof_and_preserves_the_rest(self):
        arm = [0.1, 0.2, None, 0.4]
        merged = merge_gripper_hold(arm, 2, 12.5)
        self.assertEqual(merged, [0.1, 0.2, 12.5, 0.4])
        # Input list is not mutated.
        self.assertEqual(arm, [0.1, 0.2, None, 0.4])

    def test_extends_short_action_vectors(self):
        self.assertEqual(merge_gripper_hold([0.1], 3, 7.0), [0.1, None, None, 7.0])

    def test_accepts_none_action(self):
        self.assertEqual(merge_gripper_hold(None, 0, 7.0), [7.0])

    def test_rejects_negative_dof(self):
        with self.assertRaises(ValueError):
            merge_gripper_hold([0.1], -1, 7.0)


if __name__ == "__main__":
    unittest.main()
