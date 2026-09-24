#!/usr/bin/env python3
"""Gripper-signal debugging for pi0.5 test rollouts (instrumentation only).

Records, per executed control step, the raw predicted gripper action, the
clipped command, the physical command sent to the joint, and the measured
joint position. Produces one diagnostic PNG plus a CSV for quantitative
comparison. Nothing here alters control behavior.

Only stdlib + NumPy are imported at module level so this stays unit-testable
without Isaac Sim; matplotlib is imported lazily inside the plot writer.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

CSV_COLUMNS = (
    "global_control_step",
    "inference_id",
    "chunk_action_index",
    "raw_gripper_prediction",
    "clipped_gripper_command",
    "physical_gripper_command_rad",
    "actual_gripper_position_rad",
    "actual_gripper_position_normalized",
    "tracking_error_rad",
)


def debug_filename(task_index: int, episode_index: int, seed) -> str:
    """Deterministic stem shared by the PNG and CSV outputs."""
    seed_text = "none" if seed is None else str(seed)
    return f"task_{task_index}_episode_{episode_index}_seed_{seed_text}"


class GripperDebugRecorder:
    """Accumulates per-executed-action gripper samples for one test episode."""

    def __init__(self, opened: float, closed: float) -> None:
        if not float(closed) > float(opened):
            raise ValueError("closed limit must exceed opened limit")
        self._opened = float(opened)
        self._closed = float(closed)
        self._control_step = 0
        self._inference_id = -1
        self._chunk_boundaries: list[int] = []
        self._samples: list[dict] = []
        self._finalized: tuple[Path | None, Path | None] | None = None

    def new_inference(self) -> int:
        """Mark the start of a model inference chunk; returns its id."""
        self._inference_id += 1
        self._chunk_boundaries.append(self._control_step)
        return self._inference_id

    def record_sample(
        self,
        *,
        inference_id: int,
        chunk_action_index: int,
        raw_prediction: float,
        clipped_command: float,
        physical_command_rad: float,
        actual_position_rad: float,
    ) -> None:
        """Store one executed action; all values are used exactly as given."""
        span = self._closed - self._opened
        actual_normalized = (float(actual_position_rad) - self._opened) / span
        self._samples.append(
            {
                "global_control_step": self._control_step,
                "inference_id": int(inference_id),
                "chunk_action_index": int(chunk_action_index),
                "raw_gripper_prediction": float(raw_prediction),
                "clipped_gripper_command": float(clipped_command),
                "physical_gripper_command_rad": float(physical_command_rad),
                "actual_gripper_position_rad": float(actual_position_rad),
                "actual_gripper_position_normalized": actual_normalized,
                "tracking_error_rad": float(physical_command_rad)
                - float(actual_position_rad),
            }
        )
        self._control_step += 1

    def __len__(self) -> int:
        return len(self._samples)

    def finalize(
        self, *, task_index: int, episode_index: int, seed, out_dir: Path | str
    ) -> tuple[Path | None, Path | None]:
        """Write PNG + CSV; returns their paths, or (None, None) if empty.

        Idempotent: repeat calls return the first call's result without
        rewriting, so episode-end finalization and the shutdown backup in
        the caller can both invoke it safely.
        """
        if self._finalized is not None:
            return self._finalized
        if not self._samples:
            self._finalized = (None, None)
            return self._finalized
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = debug_filename(task_index, episode_index, seed)
        png_path = out_dir / f"{stem}.png"
        csv_path = out_dir / f"{stem}.csv"
        write_gripper_csv(self._samples, csv_path)
        write_gripper_plot(
            self._samples,
            self._chunk_boundaries,
            png_path,
        )
        self._finalized = (png_path, csv_path)
        return self._finalized


def write_gripper_csv(samples: list[dict], csv_path: Path | str) -> Path:
    """Write recorded samples with full float precision for later comparison."""
    csv_path = Path(csv_path)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for sample in samples:
            writer.writerow({key: repr(sample[key]) for key in CSV_COLUMNS})
    return csv_path


def _silence_matplotlib_logging() -> None:
    """Keep matplotlib's font-matching chatter out of the rollout console.

    The sim process runs with a DEBUG-level root logger, which matplotlib's
    ``font_manager`` inherits, dumping one line per system font at plot
    time. Restricting matplotlib's own namespace to WARNING leaves every
    other logger untouched.
    """
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def write_gripper_plot(
    samples: list[dict], chunk_boundaries: list[int], png_path: Path | str
) -> Path:
    """Render the three-panel gripper diagnostic PNG (Agg backend, no window)."""
    _silence_matplotlib_logging()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.asarray([s["global_control_step"] for s in samples], dtype=float)
    raw = np.asarray([s["raw_gripper_prediction"] for s in samples], dtype=float)
    clipped = np.asarray([s["clipped_gripper_command"] for s in samples], dtype=float)
    actual_norm = np.asarray(
        [s["actual_gripper_position_normalized"] for s in samples], dtype=float
    )
    physical_cmd = np.asarray(
        [s["physical_gripper_command_rad"] for s in samples], dtype=float
    )
    physical_actual = np.asarray(
        [s["actual_gripper_position_rad"] for s in samples], dtype=float
    )
    error = np.asarray([s["tracking_error_rad"] for s in samples], dtype=float)

    _, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(steps, raw, label="raw prediction action[7]")
    axes[0].plot(steps, clipped, label="clipped command")
    axes[0].plot(steps, actual_norm, label="actual normalized")
    axes[0].axhline(0.5, linestyle="--", linewidth=1.0, label="0.5 reference")
    for boundary in chunk_boundaries:
        if steps[0] <= boundary <= steps[-1]:
            axes[0].axvline(boundary, linestyle="--", linewidth=0.6, alpha=0.5)
    axes[0].set_ylabel("Normalized gripper position")
    axes[0].set_ylim(
        min(-0.1, float(np.min([raw.min(), clipped.min(), actual_norm.min()])) - 0.05),
        max(1.1, float(np.max([raw.max(), clipped.max(), actual_norm.max()])) + 0.05),
    )
    axes[0].legend(loc="best", fontsize="small")

    axes[1].plot(steps, physical_cmd, label="commanded")
    axes[1].plot(steps, physical_actual, label="measured")
    axes[1].set_ylabel("Finger joint position [rad]")
    axes[1].legend(loc="best", fontsize="small")

    axes[2].plot(steps, error, label="tracking error")
    axes[2].axhline(0.0, linestyle="-", linewidth=1.0)
    axes[2].set_ylabel("Gripper position error [rad]")
    axes[2].set_xlabel("Executed control step")
    axes[2].legend(loc="best", fontsize="small")

    plt.tight_layout()
    png_path = Path(png_path)
    plt.savefig(png_path, dpi=150)
    plt.close()
    return png_path
