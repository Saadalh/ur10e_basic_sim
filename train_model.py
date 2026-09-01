#!/usr/bin/env python3
"""Fine-tune pi0.5-DROID on the local UR10e LeRobot v3 dataset."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq
import yaml


@dataclass(frozen=True)
class Episode:
    index: int
    prompt: str
    states: np.ndarray
    actions: np.ndarray
    side_video: Path
    wrist_video: Path
    side_start_frame: int
    wrist_start_frame: int


class UR10eDataset:
    """Random-access adapter from the collector's LeRobot v3 files to DROID samples."""

    def __init__(self, root: Path, settings: dict[str, Any]):
        self.root = root
        self.action_horizon = int(settings["action-horizon"])
        self._cache_size = int(settings["video-capture-cache-size"])
        self._captures: OrderedDict[Path, tuple[cv2.VideoCapture, int]] = OrderedDict()

        info_path = root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Dataset metadata not found: {info_path}")
        info = json.loads(info_path.read_text())
        if info.get("codebase_version") != "v3.0":
            raise ValueError(f"Expected a LeRobot v3.0 dataset, got {info.get('codebase_version')!r}")
        self.fps = int(info["fps"])

        ignored = settings["ignore-episodes"]
        episode_rows = self._read_episode_rows(root / "meta" / "episodes")
        available = {int(row["episode_index"]) for row in episode_rows}
        unknown = sorted(set(ignored) - available)
        if unknown:
            raise ValueError(
                f"ignore-episodes contains unknown indexes: {unknown}; available indexes: {sorted(available)}"
            )

        selected_rows = sorted(
            (row for row in episode_rows if int(row["episode_index"]) not in ignored),
            key=lambda row: int(row["episode_index"]),
        )
        if not selected_rows:
            raise ValueError("ignore-episodes excludes every dataset episode")

        self._parquet_cache: dict[Path, dict[str, list[Any]]] = {}
        self.episodes = [self._load_episode(row, info, settings) for row in selected_rows]
        del self._parquet_cache
        self.episode_indexes = [episode.index for episode in self.episodes]
        self._ends = np.cumsum([len(episode.states) for episode in self.episodes]).tolist()

    @staticmethod
    def _read_episode_rows(directory: Path) -> list[dict[str, Any]]:
        files = sorted(directory.glob("**/*.parquet"))
        if not files:
            raise FileNotFoundError(f"No episode metadata Parquet files found under {directory}")
        columns = [
            "episode_index",
            "tasks",
            "length",
            "data/chunk_index",
            "data/file_index",
            "videos/observation.images.side_camera/chunk_index",
            "videos/observation.images.side_camera/file_index",
            "videos/observation.images.side_camera/from_timestamp",
            "videos/observation.images.wrist_camera/chunk_index",
            "videos/observation.images.wrist_camera/file_index",
            "videos/observation.images.wrist_camera/from_timestamp",
        ]
        rows: list[dict[str, Any]] = []
        for path in files:
            rows.extend(pq.read_table(path, columns=columns).to_pylist())
        return rows

    def _load_episode(self, row: dict[str, Any], info: dict[str, Any], settings: dict[str, Any]) -> Episode:
        episode_index = int(row["episode_index"])
        data_path = self.root / info["data_path"].format(
            chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])
        )
        values = self._parquet_cache.get(data_path)
        if values is None:
            table = pq.read_table(
                data_path,
                columns=["observation.state", "action", "frame_index", "episode_index"],
            )
            values = table.to_pydict()
            self._parquet_cache[data_path] = values
        selected = [i for i, value in enumerate(values["episode_index"]) if int(value) == episode_index]
        selected.sort(key=lambda i: int(values["frame_index"][i]))
        if len(selected) != int(row["length"]):
            raise ValueError(
                f"Episode {episode_index} declares {row['length']} frames but {data_path} contains {len(selected)}"
            )

        raw_states = np.asarray([values["observation.state"][i] for i in selected], dtype=np.float32)
        raw_actions = np.asarray([values["action"][i] for i in selected], dtype=np.float32)
        states, actions = self._convert_controls(raw_states, raw_actions, settings)

        side_path = self._video_path(row, info, "side_camera")
        wrist_path = self._video_path(row, info, "wrist_camera")
        for path in (side_path, wrist_path):
            if not path.is_file():
                raise FileNotFoundError(f"Episode {episode_index} video not found: {path}")

        tasks = row["tasks"] or []
        if not tasks:
            raise ValueError(f"Episode {episode_index} has no task prompt")
        return Episode(
            index=episode_index,
            prompt=str(tasks[0]),
            states=states,
            actions=actions,
            side_video=side_path,
            wrist_video=wrist_path,
            side_start_frame=round(float(row["videos/observation.images.side_camera/from_timestamp"]) * self.fps),
            wrist_start_frame=round(float(row["videos/observation.images.wrist_camera/from_timestamp"]) * self.fps),
        )

    def _video_path(self, row: dict[str, Any], info: dict[str, Any], camera: str) -> Path:
        prefix = f"videos/observation.images.{camera}"
        return self.root / info["video_path"].format(
            video_key=f"observation.images.{camera}",
            chunk_index=int(row[f"{prefix}/chunk_index"]),
            file_index=int(row[f"{prefix}/file_index"]),
        )

    @staticmethod
    def _convert_controls(
        raw_states: np.ndarray, raw_actions: np.ndarray, settings: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        arm_state_indices = settings["arm-state-indices"]
        arm_action_indices = settings["arm-velocity-action-indices"]
        if raw_states.ndim != 2 or max(arm_state_indices, default=-1) >= raw_states.shape[1]:
            raise ValueError(f"Invalid arm-state-indices for state shape {raw_states.shape}")
        if raw_actions.ndim != 2 or max(arm_action_indices, default=-1) >= raw_actions.shape[1]:
            raise ValueError(f"Invalid arm-velocity-action-indices for action shape {raw_actions.shape}")

        gripper_state_index = int(settings["gripper-state-index"])
        gripper_value_index = int(settings["gripper-action-value-index"])
        gripper_mask_index = int(settings["gripper-action-mask-index"])
        opened = float(settings["gripper-open-position"])
        closed = float(settings["gripper-closed-position"])
        if closed <= opened:
            raise ValueError("gripper-closed-position must be greater than gripper-open-position")
        if gripper_state_index >= raw_states.shape[1]:
            raise ValueError(f"Invalid gripper-state-index for state shape {raw_states.shape}")
        if max(gripper_value_index, gripper_mask_index) >= raw_actions.shape[1]:
            raise ValueError(f"Invalid gripper action index for action shape {raw_actions.shape}")

        gripper_state = np.clip((raw_states[:, gripper_state_index] - opened) / (closed - opened), 0.0, 1.0)
        states = np.zeros((len(raw_states), 8), dtype=np.float32)
        states[:, :6] = raw_states[:, arm_state_indices]
        states[:, 7] = gripper_state

        actions = np.zeros((len(raw_actions), 8), dtype=np.float32)
        actions[:, :6] = raw_actions[:, arm_action_indices]
        actions[:, 7] = gripper_state
        commanded = raw_actions[:, gripper_mask_index] > 0.5
        actions[commanded, 7] = (raw_actions[commanded, gripper_value_index] > 0.0).astype(np.float32)
        return states, actions

    def __len__(self) -> int:
        return int(self._ends[-1])

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_position = bisect_right(self._ends, index)
        episode_start = 0 if episode_position == 0 else self._ends[episode_position - 1]
        frame_index = index - episode_start
        episode = self.episodes[episode_position]

        action_indexes = np.minimum(np.arange(frame_index, frame_index + self.action_horizon), len(episode.actions) - 1)
        return {
            "observation/exterior_image_1_left": self._read_frame(
                episode.side_video, episode.side_start_frame + frame_index
            ),
            "observation/wrist_image_left": self._read_frame(
                episode.wrist_video, episode.wrist_start_frame + frame_index
            ),
            "observation/joint_position": episode.states[frame_index, :7],
            "observation/gripper_position": episode.states[frame_index, 7:],
            "actions": episode.actions[action_indexes],
            "prompt": episode.prompt,
        }

    def _read_frame(self, path: Path, frame_index: int) -> np.ndarray:
        cached = self._captures.pop(path, None)
        if cached is None:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise RuntimeError(f"Could not open video: {path}")
            next_frame = -1
        else:
            capture, next_frame = cached

        if next_frame != frame_index:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Could not decode frame {frame_index} from {path}")
        self._captures[path] = (capture, frame_index + 1)
        while len(self._captures) > self._cache_size:
            _, (old_capture, _) = self._captures.popitem(last=False)
            old_capture.release()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_captures"] = OrderedDict()
        return state


def _resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def load_settings(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    settings = yaml.safe_load(config_path.read_text())
    if not isinstance(settings, dict):
        raise ValueError("Training config must contain a YAML mapping")

    required = {
        "openpi-root",
        "dataset-path",
        "base-checkpoint",
        "pretrained-assets-id",
        "config-name",
        "experiment-name",
        "project-name",
        "checkpoint-base-dir",
        "assets-base-dir",
        "fine-tuning-method",
        "minimum-lora-vram-gb",
        "minimum-full-vram-gb",
        "ignore-episodes",
        "action-horizon",
        "model-action-dim",
        "max-token-length",
        "batch-size",
        "num-train-steps",
        "num-workers",
        "seed",
        "warmup-steps",
        "peak-learning-rate",
        "learning-rate-decay-steps",
        "final-learning-rate",
        "adam-beta1",
        "adam-beta2",
        "adam-epsilon",
        "weight-decay",
        "gradient-clip-norm",
        "log-interval",
        "save-interval",
        "keep-period",
        "fsdp-devices",
        "wandb-enabled",
        "overwrite",
        "resume",
        "arm-state-indices",
        "gripper-state-index",
        "arm-velocity-action-indices",
        "gripper-action-value-index",
        "gripper-action-mask-index",
        "gripper-open-position",
        "gripper-closed-position",
        "video-capture-cache-size",
    }
    missing = sorted(required - settings.keys())
    if missing:
        raise ValueError(f"Training config is missing keys: {missing}")

    settings["openpi-root"] = _resolve_path(settings["openpi-root"], config_path)
    settings["dataset-path"] = _resolve_path(settings["dataset-path"], config_path)
    settings["checkpoint-base-dir"] = _resolve_path(settings["checkpoint-base-dir"], config_path)
    settings["assets-base-dir"] = _resolve_path(settings["assets-base-dir"], config_path)
    if settings["fine-tuning-method"] not in {"lora", "full"}:
        raise ValueError('fine-tuning-method must be either "lora" or "full"')
    ignored = settings["ignore-episodes"]
    if not isinstance(ignored, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in ignored):
        raise ValueError("ignore-episodes must be a list of integer episode indexes")
    if len(set(ignored)) != len(ignored) or any(value < 0 for value in ignored):
        raise ValueError("ignore-episodes must contain unique, non-negative indexes")
    if settings["resume"] and settings["overwrite"]:
        raise ValueError("resume and overwrite cannot both be true")
    for key in (
        "action-horizon",
        "model-action-dim",
        "max-token-length",
        "batch-size",
        "num-train-steps",
        "warmup-steps",
        "learning-rate-decay-steps",
        "log-interval",
        "save-interval",
        "fsdp-devices",
        "video-capture-cache-size",
    ):
        if int(settings[key]) < 1:
            raise ValueError(f"{key} must be at least 1")
    if len(settings["arm-state-indices"]) != 6 or len(settings["arm-velocity-action-indices"]) != 6:
        raise ValueError("Exactly six arm state and six arm velocity action indexes are required")
    index_keys = (
        "arm-state-indices",
        "arm-velocity-action-indices",
        "gripper-state-index",
        "gripper-action-value-index",
        "gripper-action-mask-index",
    )
    for key in index_keys:
        values = settings[key] if isinstance(settings[key], list) else [settings[key]]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ValueError(f"{key} must contain non-negative integer indexes")
        if len(values) != len(set(values)):
            raise ValueError(f"{key} must not contain duplicate indexes")
    if int(settings["model-action-dim"]) < 8:
        raise ValueError("model-action-dim must be at least 8 for DROID-compatible controls")
    if int(settings["num-workers"]) < 0:
        raise ValueError("num-workers cannot be negative")
    if settings["keep-period"] is not None and int(settings["keep-period"]) < 1:
        raise ValueError("keep-period must be null or at least 1")
    return settings


def query_gpus() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise RuntimeError(f"Could not query NVIDIA GPUs with nvidia-smi: {detail.strip()}") from error

    gpus = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index, name, total_mib, free_mib = (part.strip() for part in line.split(",", maxsplit=3))
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "total_gib": float(total_mib) / 1024.0,
                "free_gib": float(free_mib) / 1024.0,
            }
        )
    if not gpus:
        raise RuntimeError("nvidia-smi did not report an NVIDIA GPU")
    return gpus


def check_vram(settings: dict[str, Any]) -> None:
    method = settings["fine-tuning-method"]
    minimum = float(settings[f"minimum-{method}-vram-gb"])
    needed_devices = int(settings["fsdp-devices"])
    gpus = sorted(query_gpus(), key=lambda gpu: gpu["free_gib"], reverse=True)
    selected = gpus[:needed_devices]
    descriptions = "; ".join(
        f"GPU {gpu['index']} {gpu['name']}: {gpu['free_gib']:.2f} GiB free / {gpu['total_gib']:.2f} GiB total"
        for gpu in selected
    )
    if len(selected) < needed_devices or any(min(gpu["free_gib"], gpu["total_gib"]) < minimum for gpu in selected):
        raise RuntimeError(
            f"Training cannot start. Available GPU memory: {descriptions or 'none'}. "
            f"pi0.5 {method} fine-tuning requires at least {minimum:.1f} GiB per GPU "
            f"across {needed_devices} GPU(s)."
        )
    print(f"GPU preflight passed: {descriptions}; required: {minimum:.1f} GiB per GPU.")


def run_training(settings: dict[str, Any], dataset: UR10eDataset) -> None:
    openpi_root = settings["openpi-root"]
    if not (openpi_root / "src" / "openpi").is_dir():
        raise FileNotFoundError(f"OpenPI source tree not found: {openpi_root}")
    sys.path.insert(0, str(openpi_root))
    sys.path.insert(0, str(openpi_root / "src"))

    try:
        import flax.nnx as nnx

        from openpi import transforms
        from openpi.models import pi0_config
        from openpi.policies import droid_policy
        from openpi.training import config as openpi_config
        from openpi.training import data_loader
        from openpi.training import optimizer
        from openpi.training import weight_loaders
        from scripts import train
    except ImportError as error:
        expected_python = openpi_root / ".venv" / "bin" / "python"
        raise RuntimeError(
            f"OpenPI dependencies are unavailable in {sys.executable}. Run with: "
            f"{expected_python} {Path(__file__).resolve()}"
        ) from error

    method = settings["fine-tuning-method"]
    model = pi0_config.Pi0Config(
        pi05=True,
        action_dim=int(settings["model-action-dim"]),
        action_horizon=int(settings["action-horizon"]),
        max_token_len=int(settings["max-token-length"]),
        paligemma_variant="gemma_2b_lora" if method == "lora" else "gemma_2b",
        action_expert_variant="gemma_300m_lora" if method == "lora" else "gemma_300m",
    )

    checkpoint = str(settings["base-checkpoint"]).rstrip("/")
    params_path = checkpoint if checkpoint.endswith("/params") else f"{checkpoint}/params"
    checkpoint_root = checkpoint.removesuffix("/params")
    data_factory = openpi_config.SimpleDataConfig(
        repo_id="local/ur10e",
        assets=openpi_config.AssetsConfig(
            assets_dir=f"{checkpoint_root}/assets",
            asset_id=str(settings["pretrained-assets-id"]),
        ),
        base_config=openpi_config.DataConfig(prompt_from_task=False),
        data_transforms=lambda model_config: transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        ),
    )
    freeze_filter = model.get_freeze_filter() if method == "lora" else nnx.Nothing
    config = openpi_config.TrainConfig(
        name=str(settings["config-name"]),
        project_name=str(settings["project-name"]),
        exp_name=str(settings["experiment-name"]),
        model=model,
        weight_loader=weight_loaders.CheckpointWeightLoader(params_path),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=int(settings["warmup-steps"]),
            peak_lr=float(settings["peak-learning-rate"]),
            decay_steps=int(settings["learning-rate-decay-steps"]),
            decay_lr=float(settings["final-learning-rate"]),
        ),
        optimizer=optimizer.AdamW(
            b1=float(settings["adam-beta1"]),
            b2=float(settings["adam-beta2"]),
            eps=float(settings["adam-epsilon"]),
            weight_decay=float(settings["weight-decay"]),
            clip_gradient_norm=float(settings["gradient-clip-norm"]),
        ),
        ema_decay=None if method == "lora" else 0.99,
        freeze_filter=freeze_filter,
        data=data_factory,
        assets_base_dir=str(settings["assets-base-dir"]),
        checkpoint_base_dir=str(settings["checkpoint-base-dir"]),
        seed=int(settings["seed"]),
        batch_size=int(settings["batch-size"]),
        num_workers=int(settings["num-workers"]),
        num_train_steps=int(settings["num-train-steps"]),
        log_interval=int(settings["log-interval"]),
        save_interval=int(settings["save-interval"]),
        keep_period=None if settings["keep-period"] is None else int(settings["keep-period"]),
        overwrite=bool(settings["overwrite"]),
        resume=bool(settings["resume"]),
        wandb_enabled=bool(settings["wandb-enabled"]),
        fsdp_devices=int(settings["fsdp-devices"]),
    )

    original_factory = data_loader.create_torch_dataset
    data_loader.create_torch_dataset = lambda *_args, **_kwargs: dataset
    try:
        print(
            f"Starting pi0.5 {method} fine-tuning with {len(dataset):,} frames from episodes {dataset.episode_indexes}."
        )
        train.main(config)
    finally:
        data_loader.create_torch_dataset = original_factory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("train_config.yaml"),
        help="Path to the YAML training configuration.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration, episode filtering, one action chunk, and both camera videos without training.",
    )
    parser.add_argument(
        "--skip-vram-check",
        action="store_true",
        help="Bypass the documented GPU-memory guard for expert/debug use.",
    )
    args = parser.parse_args()

    try:
        config_path = args.config.expanduser().resolve()
        settings = load_settings(config_path)
        dataset = UR10eDataset(settings["dataset-path"], settings)
        print(
            f"Dataset ready: {len(dataset):,} frames across {len(dataset.episodes)} episodes; "
            f"ignored episodes: {settings['ignore-episodes'] or 'none'}."
        )
        if args.validate_only:
            sample = dataset[0]
            print(
                "Validation passed: "
                f"side={sample['observation/exterior_image_1_left'].shape}, "
                f"wrist={sample['observation/wrist_image_left'].shape}, "
                f"state={(len(sample['observation/joint_position']) + len(sample['observation/gripper_position']),)}, "
                f"actions={sample['actions'].shape}, prompt={sample['prompt']!r}."
            )
            return 0
        if not args.skip_vram_check:
            check_vram(settings)
        run_training(settings, dataset)
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, yaml.YAMLError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
