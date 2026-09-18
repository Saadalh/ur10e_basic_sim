#!/usr/bin/env python3
"""Generate OpenPI-based statistics for a collected UR10e dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml

from src.droid_projection import DROID_ACTION_SCALE, project_controls

RAW_STATE_KEY = "observation.state"
RAW_ACTION_KEY = "action"
OPENPI_STATE_KEY = "openpi.state"
OPENPI_ACTION_KEY = "openpi.actions"
PROJECTION_VERSION = 2
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def convert_controls(
    raw_states: np.ndarray, raw_actions: np.ndarray, settings: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Project collected UR10e controls into the 8D DROID-compatible space.

    Thin delegate to the canonical :mod:`src.droid_projection`
    implementation so statistics generation and training can never diverge.
    """
    return project_controls(raw_states, raw_actions, settings)


class StatsAccumulator:
    def __init__(self, running_stats_class: type):
        self._running = running_stats_class()
        self._minimum: np.ndarray | None = None
        self._maximum: np.ndarray | None = None
        self._count = 0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values)
        self._running.update(values)
        flat = values.reshape(-1, values.shape[-1])
        batch_minimum = np.min(flat, axis=0)
        batch_maximum = np.max(flat, axis=0)
        self._minimum = batch_minimum if self._minimum is None else np.minimum(self._minimum, batch_minimum)
        self._maximum = batch_maximum if self._maximum is None else np.maximum(self._maximum, batch_maximum)
        self._count += len(flat)

    def finish(self) -> dict[str, list[Any]]:
        norm_stats = self._running.get_statistics()
        if self._minimum is None or self._maximum is None:
            raise ValueError("Cannot compute statistics for an empty dataset")
        return {
            "min": self._minimum.tolist(),
            "max": self._maximum.tolist(),
            "mean": np.asarray(norm_stats.mean).tolist(),
            "std": np.asarray(norm_stats.std).tolist(),
            "count": [self._count],
            "q01": np.asarray(norm_stats.q01).tolist(),
            "q99": np.asarray(norm_stats.q99).tolist(),
        }


def _load_running_stats(openpi_root: Path) -> type:
    normalize_path = openpi_root / "src" / "openpi" / "shared" / "normalize.py"
    if not normalize_path.is_file():
        raise FileNotFoundError(f"OpenPI normalization implementation not found: {normalize_path}")
    sys.path.insert(0, str(openpi_root / "src"))
    from openpi.shared.normalize import RunningStats

    return RunningStats


def _load_control_settings(config_path: Path) -> dict[str, Any]:
    settings = yaml.safe_load(config_path.read_text())
    if not isinstance(settings, dict):
        raise ValueError("Training config must contain a YAML mapping")
    keys = {
        "arm-state-indices",
        "gripper-state-index",
        "arm-position-action-indices",
        "gripper-action-value-index",
        "gripper-action-mask-index",
        "gripper-open-position",
        "gripper-closed-position",
        "action-horizon",
    }
    missing = sorted(keys - settings.keys())
    if missing:
        raise ValueError(f"Training config is missing control projection keys: {missing}")
    return {key: settings[key] for key in keys}


def generate_dataset_stats(
    dataset_root: Path,
    openpi_root: Path,
    control_settings: dict[str, Any],
) -> Path:
    """Compute raw and projected statistics from every finalized episode."""
    info_path = dataset_root / "meta" / "info.json"
    episode_files = sorted((dataset_root / "meta" / "episodes").glob("**/*.parquet"))
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {info_path}")
    if not episode_files:
        raise FileNotFoundError(f"No episode metadata found under {dataset_root / 'meta' / 'episodes'}")
    info = json.loads(info_path.read_text())
    expected_frames = int(info["total_frames"])
    state_width = int(info["features"][RAW_STATE_KEY]["shape"][0])
    action_width = int(info["features"][RAW_ACTION_KEY]["shape"][0])
    if state_width < 1 or action_width < 1:
        raise ValueError(f"Invalid state/action dimensions: {state_width}D and {action_width}D")

    episode_rows = []
    episode_columns = ["episode_index", "length", "data/chunk_index", "data/file_index"]
    for path in episode_files:
        episode_rows.extend(pq.read_table(path, columns=episode_columns).to_pylist())
    episode_rows.sort(key=lambda row: int(row["episode_index"]))
    if len(episode_rows) != int(info["total_episodes"]):
        raise ValueError(
            f"Dataset declares {info['total_episodes']} episodes but metadata contains {len(episode_rows)}"
        )
    episode_indexes = [int(row["episode_index"]) for row in episode_rows]
    expected_indexes = list(range(int(info["total_episodes"])))
    if episode_indexes != expected_indexes:
        raise ValueError(f"Expected contiguous episode indexes {expected_indexes}, got {episode_indexes}")

    running_stats_class = _load_running_stats(openpi_root)
    accumulators = {
        RAW_STATE_KEY: StatsAccumulator(running_stats_class),
        RAW_ACTION_KEY: StatsAccumulator(running_stats_class),
        OPENPI_STATE_KEY: StatsAccumulator(running_stats_class),
        OPENPI_ACTION_KEY: StatsAccumulator(running_stats_class),
    }

    frame_count = 0
    action_horizon = int(control_settings["action-horizon"])
    if action_horizon < 1:
        raise ValueError("action-horizon must be at least 1")
    cached_path: Path | None = None
    cached_values: dict[str, list[Any]] | None = None
    for row in episode_rows:
        episode_index = int(row["episode_index"])
        data_path = dataset_root / info["data_path"].format(
            chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])
        )
        if data_path != cached_path:
            cached_values = pq.read_table(
                data_path,
                columns=[RAW_STATE_KEY, RAW_ACTION_KEY, "frame_index", "episode_index"],
            ).to_pydict()
            cached_path = data_path
        if cached_values is None:
            raise RuntimeError(f"Could not load dataset values from {data_path}")
        values = cached_values
        selected = [i for i, value in enumerate(values["episode_index"]) if int(value) == episode_index]
        selected.sort(key=lambda i: int(values["frame_index"][i]))
        if len(selected) != int(row["length"]):
            raise ValueError(
                f"Episode {episode_index} declares {row['length']} frames but {data_path} contains {len(selected)}"
            )

        raw_states = np.asarray([values[RAW_STATE_KEY][i] for i in selected], dtype=np.float32)
        raw_actions = np.asarray([values[RAW_ACTION_KEY][i] for i in selected], dtype=np.float32)
        if raw_states.shape != (len(selected), state_width) or raw_actions.shape != (len(selected), action_width):
            raise ValueError(
                f"Episode {episode_index} has state/action shapes {raw_states.shape}/{raw_actions.shape}; "
                f"expected (*, {state_width})/(*, {action_width})"
            )
        states, actions = convert_controls(raw_states, raw_actions, control_settings)
        action_indexes = np.minimum(
            np.arange(len(actions))[:, None] + np.arange(action_horizon)[None, :], len(actions) - 1
        )
        accumulators[RAW_STATE_KEY].update(raw_states)
        accumulators[RAW_ACTION_KEY].update(raw_actions)
        accumulators[OPENPI_STATE_KEY].update(states)
        accumulators[OPENPI_ACTION_KEY].update(actions[action_indexes])
        frame_count += len(raw_states)

    if frame_count != expected_frames:
        raise ValueError(f"Dataset declares {expected_frames} frames but finalized episodes contain {frame_count}")

    stats = {key: accumulator.finish() for key, accumulator in accumulators.items()}
    projection_metadata = {
        "projection_version": [PROJECTION_VERSION],
        "source_count": [frame_count],
        "source_state_dimensions": [state_width],
        "source_action_dimensions": [action_width],
        "action_horizon": [action_horizon],
        "arm_state_indices": list(control_settings["arm-state-indices"]),
        "arm_position_action_indices": list(control_settings["arm-position-action-indices"]),
        "droid_action_scale": [DROID_ACTION_SCALE],
        "gripper_state_index": [int(control_settings["gripper-state-index"])],
        "gripper_action_value_index": [int(control_settings["gripper-action-value-index"])],
        "gripper_action_mask_index": [int(control_settings["gripper-action-mask-index"])],
        "gripper_open_position": [float(control_settings["gripper-open-position"])],
        "gripper_closed_position": [float(control_settings["gripper-closed-position"])],
    }
    stats[OPENPI_STATE_KEY].update(projection_metadata)
    stats[OPENPI_ACTION_KEY].update(projection_metadata)
    output_path = dataset_root / "meta" / "stats.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(stats, indent=4) + "\n")
    temporary_path.replace(output_path)
    print(f"Wrote OpenPI statistics for {frame_count:,} frames to {output_path}", flush=True)
    return output_path


def load_projected_stats(path: Path, settings: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    """Load and validate the exact 8D training projection from dataset statistics."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset statistics not found: {path}")
    stats = json.loads(path.read_text())
    info_path = path.parent / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {info_path}")
    info = json.loads(info_path.read_text())
    expected_frames = int(info["total_frames"])
    state_width = int(info["features"][RAW_STATE_KEY]["shape"][0])
    action_width = int(info["features"][RAW_ACTION_KEY]["shape"][0])
    expected_metadata = {
        "projection_version": [PROJECTION_VERSION],
        "source_count": [expected_frames],
        "source_state_dimensions": [state_width],
        "source_action_dimensions": [action_width],
        "action_horizon": [int(settings["action-horizon"])],
        "arm_state_indices": list(settings["arm-state-indices"]),
        "arm_position_action_indices": list(settings["arm-position-action-indices"]),
        "droid_action_scale": [DROID_ACTION_SCALE],
        "gripper_state_index": [int(settings["gripper-state-index"])],
        "gripper_action_value_index": [int(settings["gripper-action-value-index"])],
        "gripper_action_mask_index": [int(settings["gripper-action-mask-index"])],
        "gripper_open_position": [float(settings["gripper-open-position"])],
        "gripper_closed_position": [float(settings["gripper-closed-position"])],
    }
    projected = {}
    for source_key, target_key in ((OPENPI_STATE_KEY, "state"), (OPENPI_ACTION_KEY, "actions")):
        if source_key not in stats:
            raise ValueError(
                f"Dataset statistics are missing {source_key!r}; regenerate them with src.dataset_stats"
            )
        for metadata_key, expected in expected_metadata.items():
            actual = stats[source_key].get(metadata_key)
            actual_array = np.asarray(actual)
            expected_array = np.asarray(expected)
            if (
                actual is None
                or actual_array.shape != expected_array.shape
                or not np.allclose(actual_array, expected_array, rtol=0.0, atol=1e-12)
            ):
                raise ValueError(
                    f"{source_key}.{metadata_key} does not match the dataset or training configuration; "
                    "regenerate meta/stats.json"
                )
        expected_count = expected_frames if source_key == OPENPI_STATE_KEY else expected_frames * int(settings["action-horizon"])
        if stats[source_key].get("count") != [expected_count]:
            raise ValueError(f"{source_key}.count must be [{expected_count}]; regenerate meta/stats.json")
        values = {}
        for statistic in ("mean", "std", "q01", "q99"):
            array = np.asarray(stats[source_key].get(statistic), dtype=np.float64)
            if array.shape != (8,) or not np.all(np.isfinite(array)):
                raise ValueError(f"{source_key}.{statistic} must contain eight finite values")
            values[statistic] = array
        if np.any(values["std"] < 0) or np.any(values["q01"] > values["q99"]):
            raise ValueError(f"Dataset statistics contain invalid values for {source_key}")
        projected[target_key] = values
    return projected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument(
        "--training-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "train_config.yaml",
    )
    args = parser.parse_args()
    try:
        generate_dataset_stats(
            args.dataset_root.expanduser().resolve(),
            args.openpi_root.expanduser().resolve(),
            _load_control_settings(args.training_config.expanduser().resolve()),
        )
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, yaml.YAMLError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
