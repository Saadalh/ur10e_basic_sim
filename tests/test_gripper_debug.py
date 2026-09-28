#!/usr/bin/env python3
"""Sim-free unit tests for gripper-debug recording (no Isaac Sim, no matplotlib).

The PNG renderer needs matplotlib, which is unavailable in the lightweight
project virtualenv, so these tests cover the recorder, CSV output, filename
logic, and empty-run behavior. Plotting itself is validated by review and by
the live rollout.
"""

import csv
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.gripper_debug import (
    CSV_COLUMNS,
    GripperDebugRecorder,
    _silence_matplotlib_logging,
    debug_filename,
    restore_matplotlib_logging,
    write_gripper_csv,
)


def _recorder():
    return GripperDebugRecorder(opened=0.0, closed=0.376)


def _feed(recorder, n_inferences=2, per_inference=3):
    for _ in range(n_inferences):
        inference_id = recorder.new_inference()
        for chunk_action_index in range(per_inference):
            step = len(recorder)
            recorder.record_sample(
                inference_id=inference_id,
                chunk_action_index=chunk_action_index,
                raw_prediction=0.1 * step,
                clipped_command=min(0.1 * step, 1.0),
                physical_command_rad=0.0376 * step,
                actual_position_rad=0.03 * step,
            )


class FilenameTest(unittest.TestCase):
    def test_silencing_targets_matplotlib_only(self):
        matplotlib_logger = logging.getLogger("matplotlib")
        font_manager_logger = logging.getLogger("matplotlib.font_manager")
        previous = matplotlib_logger.level
        matplotlib_logger.setLevel(logging.DEBUG)
        try:
            previous_levels = _silence_matplotlib_logging()
            self.assertEqual(matplotlib_logger.level, logging.WARNING)
            # Child inherits: font-matching DEBUG lines are suppressed.
            self.assertEqual(font_manager_logger.getEffectiveLevel(), logging.WARNING)
            # Unrelated loggers are untouched.
            self.assertEqual(
                logging.getLogger("ur10e.collection").getEffectiveLevel(),
                logging.getLogger().getEffectiveLevel(),
            )
        finally:
            restore_matplotlib_logging(previous_levels)
            matplotlib_logger.setLevel(previous)

    def test_silencing_suppresses_explicit_child_level(self):
        # Even if the child logger was explicitly set to DEBUG (e.g. by
        # third-party code after our call), re-silencing must win.
        child = logging.getLogger("matplotlib.font_manager")
        child.setLevel(logging.DEBUG)
        self.addCleanup(child.setLevel, logging.NOTSET)
        previous_levels = _silence_matplotlib_logging()
        try:
            self.assertEqual(child.getEffectiveLevel(), logging.WARNING)
            self.assertFalse(child.isEnabledFor(logging.DEBUG))
        finally:
            restore_matplotlib_logging(previous_levels)
        # Pre-call state (explicit DEBUG) is restored untouched.
        self.assertEqual(child.level, logging.DEBUG)

    def test_deterministic_stem(self):
        self.assertEqual(debug_filename(0, 0, 42), "task_0_episode_0_seed_42")
        self.assertEqual(debug_filename(3, 7, 42), debug_filename(3, 7, 42))

    def test_none_seed(self):
        self.assertEqual(debug_filename(0, 0, None), "task_0_episode_0_seed_none")


class RecorderTest(unittest.TestCase):
    def test_rejects_inverted_limits(self):
        with self.assertRaises(ValueError):
            GripperDebugRecorder(opened=0.376, closed=0.0)

    def test_inference_ids_and_steps(self):
        recorder = _recorder()
        _feed(recorder, n_inferences=2, per_inference=3)
        self.assertEqual(len(recorder), 6)
        first = recorder._samples[0]
        last = recorder._samples[-1]
        self.assertEqual(
            (
                first["inference_id"],
                first["chunk_action_index"],
                first["global_control_step"],
            ),
            (0, 0, 0),
        )
        self.assertEqual(
            (
                last["inference_id"],
                last["chunk_action_index"],
                last["global_control_step"],
            ),
            (1, 2, 5),
        )

    def test_derived_fields(self):
        recorder = _recorder()
        recorder.new_inference()
        recorder.record_sample(
            inference_id=0,
            chunk_action_index=0,
            raw_prediction=1.4,
            clipped_command=1.0,
            physical_command_rad=0.376,
            actual_position_rad=0.188,
        )
        sample = recorder._samples[0]
        # Raw value preserved; actual normalized against (0.0, 0.376).
        self.assertEqual(sample["raw_gripper_prediction"], 1.4)
        self.assertAlmostEqual(sample["actual_gripper_position_normalized"], 0.5)
        self.assertAlmostEqual(sample["tracking_error_rad"], 0.188)

    def test_finalize_empty_returns_nones(self):
        recorder = _recorder()
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(
                recorder.finalize(
                    task_index=0, episode_index=0, seed=42, out_dir=tmpdir
                ),
                (None, None),
            )
            self.assertEqual(list(Path(tmpdir).iterdir()), [])

    def test_finalize_is_idempotent(self):
        # Episode-end finalization and the shutdown backup both invoke
        # finalize(); the second call must reuse the first call's result
        # without rewriting, so a stuck teardown can never lose the files.
        recorder = _recorder()
        _feed(recorder, n_inferences=1, per_inference=2)
        calls = []

        def fake_plot(samples, boundaries, png_path):
            calls.append(Path(png_path))
            Path(png_path).write_bytes(b"fake-png")
            return Path(png_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.gripper_debug.write_gripper_plot", side_effect=fake_plot):
                first = recorder.finalize(
                    task_index=0, episode_index=0, seed=42, out_dir=tmpdir
                )
                second = recorder.finalize(
                    task_index=0, episode_index=0, seed=42, out_dir=tmpdir
                )
            self.assertEqual(first, second)
            self.assertEqual(len(calls), 1)
            self.assertTrue(first[0].is_file())
            self.assertTrue(first[1].is_file())

    def test_finalize_writes_csv_before_plot(self):
        # matplotlib is unavailable in this venv, so the plot step raises;
        # the CSV must already exist, proving ordering (CSV first, PNG last).
        recorder = _recorder()
        _feed(recorder, n_inferences=1, per_inference=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "nested" / "gripper_debug"
            with self.assertRaises(ModuleNotFoundError):
                recorder.finalize(
                    task_index=0, episode_index=0, seed=42, out_dir=out_dir
                )
            self.assertTrue(out_dir.is_dir())
            self.assertTrue((out_dir / "task_0_episode_0_seed_42.csv").is_file())


class CsvTest(unittest.TestCase):
    def test_header_and_roundtrip_values(self):
        recorder = _recorder()
        _feed(recorder, n_inferences=2, per_inference=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = write_gripper_csv(recorder._samples, Path(tmpdir) / "out.csv")
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(list(rows[0].keys()) if rows else [], list(CSV_COLUMNS))
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            [row["global_control_step"] for row in rows], ["0", "1", "2", "3"]
        )
        self.assertEqual([row["inference_id"] for row in rows], ["0", "0", "1", "1"])
        # Full float precision for later quantitative comparison.
        self.assertEqual(rows[1]["raw_gripper_prediction"], repr(0.1))


if __name__ == "__main__":
    unittest.main()
