import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml
from isaaclab.app import AppLauncher

from src.cube_environment import (
    ARM_JOINT_NAMES,
    CUBE_PRIM_PATHS,
    CUBE_SETUP_POSITIVE_Y_COUNTS,
    GRIPPER_CONFIG,
    PATH_CONFIG,
    CubeEnvironment,
    build_task_schedule,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PI05_CONFIG_PATH = PROJECT_ROOT / "config" / "pi05_config.yaml"
TRAIN_CONFIG_PATH = PROJECT_ROOT / "config" / "train_config.yaml"
with PI05_CONFIG_PATH.open(encoding="utf-8") as config_file:
    PI05_CONFIG = yaml.safe_load(config_file)
with TRAIN_CONFIG_PATH.open(encoding="utf-8") as config_file:
    TRAIN_CONFIG = yaml.safe_load(config_file)

_environment: CubeEnvironment | None = None
_arm_joint_indices: np.ndarray | None = None
_gripper_joint_index: int | None = None
_arm_max_velocities: np.ndarray | None = None
_task_prompt: str | None = None


def get_images() -> tuple[np.ndarray, ...]:
    """Return synchronized side and wrist RGB frames from the collection cameras."""
    if _environment is None:
        return ()
    return tuple(_environment.get_camera_rgb(name) for name in _environment.cameras)


def get_proprioception() -> np.ndarray:
    """Return direct articulation state in pi0.5-DROID order."""
    if (
        _environment is None
        or _arm_joint_indices is None
        or _gripper_joint_index is None
    ):
        return np.empty(0, dtype=np.float32)

    arm_positions = np.asarray(
        _environment.robot.get_joint_positions(_arm_joint_indices),
        dtype=np.float32,
    )
    # DROID has seven arm joints; the UR10e's nonexistent seventh joint is represented as zero.
    arm_positions = np.pad(arm_positions, (0, 7 - len(arm_positions)))
    gripper_position = float(
        _environment.robot.get_joint_positions([_gripper_joint_index])[0]
    )
    opened = float(TRAIN_CONFIG["gripper-open-position"])
    closed = float(TRAIN_CONFIG["gripper-closed-position"])
    normalized_gripper = np.clip(
        (gripper_position - opened) / (closed - opened), 0.0, 1.0
    )
    return np.append(arm_positions, np.float32(normalized_gripper))


def get_task_prompt() -> str:
    if _task_prompt is None:
        raise RuntimeError(
            "The test task is unavailable before launch_simulation initializes the environment"
        )
    return _task_prompt


def _initialize_robot_controller() -> None:
    global _arm_joint_indices, _gripper_joint_index, _arm_max_velocities

    if _environment is None:
        raise RuntimeError(
            "The collection environment must be initialized before the policy controller"
        )

    robot = _environment.robot
    _arm_joint_indices = np.asarray(
        [robot.get_dof_index(name) for name in ARM_JOINT_NAMES],
        dtype=np.int32,
    )
    _gripper_joint_index = int(
        robot.get_dof_index(GRIPPER_CONFIG["joint_prim_names"][0])
    )

    controller = robot.get_articulation_controller()
    for joint_index in _arm_joint_indices:
        controller.switch_dof_control_mode(int(joint_index), "velocity")
    controller.switch_dof_control_mode(_gripper_joint_index, "position")
    _arm_max_velocities = np.asarray(
        robot.dof_properties["maxVelocity"][_arm_joint_indices],
        dtype=np.float32,
    )


def step_simulation(action_chunk: np.ndarray, chunk_samples: int = 1) -> int:
    """Apply policy actions with the collection world's 15 Hz render and 150 Hz physics timing."""
    from isaacsim.core.utils.types import ArticulationAction

    if action_chunk.ndim != 2 or action_chunk.shape[1] != 8:
        raise ValueError(
            f"Expected a VLA action chunk shaped (steps, 8), got {action_chunk.shape}"
        )
    if chunk_samples < 1:
        raise ValueError(f"chunk_samples must be at least 1, got {chunk_samples}")
    if (
        _environment is None
        or _arm_joint_indices is None
        or _gripper_joint_index is None
        or _arm_max_velocities is None
    ):
        raise RuntimeError(
            "Robot controller is not initialized; call step_simulation from launch_simulation"
        )

    sample_count = min(chunk_samples, len(action_chunk))
    opened = float(TRAIN_CONFIG["gripper-open-position"])
    closed = float(TRAIN_CONFIG["gripper-closed-position"])
    controller = _environment.robot.get_articulation_controller()
    for action in action_chunk[:sample_count]:
        # DROID has seven arm commands; the UR10e uses the first six and drops the synthetic seventh DOF.
        arm_velocities = np.clip(action[:6], -_arm_max_velocities, _arm_max_velocities)
        gripper_position = opened + np.clip(action[7], 0.0, 1.0) * (closed - opened)
        controller.apply_action(
            ArticulationAction(
                joint_velocities=arm_velocities, joint_indices=_arm_joint_indices
            )
        )
        controller.apply_action(
            ArticulationAction(
                joint_positions=np.asarray([gripper_position], dtype=np.float32),
                joint_indices=np.asarray([_gripper_joint_index], dtype=np.int32),
            )
        )
        _environment.world.step(render=True)
    return sample_count


def launch_simulation(on_step: Callable[[], bool] | None = None) -> None:
    global _environment, _task_prompt
    global _arm_joint_indices, _gripper_joint_index, _arm_max_velocities

    parser = argparse.ArgumentParser(
        description="Launch the randomized cube environment used for data collection."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Environment randomization seed."
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=0,
        help="Index in the configured collection task schedule.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Dataset episode index used for cube setup variation and seed offset.",
    )
    parser.add_argument(
        "--tcp-y-sign",
        type=int,
        choices=(-1, 1),
        default=None,
        help="Override the initial end-effector base-frame y side.",
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(experience=PATH_CONFIG["isaac_experience"])
    args = parser.parse_args()
    if args.episode_index < 0:
        parser.error("--episode-index must be non-negative")

    task_schedule = build_task_schedule()
    if not 0 <= args.task_index < len(task_schedule):
        parser.error(f"--task-index must be between 0 and {len(task_schedule) - 1}")
    task = task_schedule[args.task_index]
    _task_prompt = str(task["description"])

    setup_variation = args.episode_index % len(CUBE_SETUP_POSITIVE_Y_COUNTS) + 1
    positive_y_count = CUBE_SETUP_POSITIVE_Y_COUNTS[setup_variation - 1]
    episode_seed = None if args.seed is None else args.seed + args.episode_index
    first_tcp_y_sign = int(np.random.default_rng(args.seed).choice((-1, 1)))
    tcp_y_sign = args.tcp_y_sign
    if tcp_y_sign is None:
        tcp_y_sign = (
            first_tcp_y_sign if args.episode_index % 2 == 0 else -first_tcp_y_sign
        )

    print(f"Task: {_task_prompt}", flush=True)
    print(
        f"Test episode index {args.episode_index}, seed {episode_seed}, "
        f"{'positive' if tcp_y_sign > 0 else 'negative'} initial TCP y",
        flush=True,
    )
    print(
        f"Cube setup variation {setup_variation}: all cubes require base-frame x > 0; "
        f"{positive_y_count} require y > 0 and {len(CUBE_PRIM_PATHS) - positive_y_count} require y < 0.",
        flush=True,
    )

    simulation_app = AppLauncher(args).app
    try:
        _environment = CubeEnvironment.create(simulation_app)
        _environment.prepare_episode(
            seed=episode_seed,
            tcp_y_sign=tcp_y_sign,
            setup_variation=setup_variation,
            episode_label="Test episode",
            reset_world=False,
            task=task,
        )
        # Collection advances once before recording its first observation.
        _environment.world.step(render=True)
        _initialize_robot_controller()

        stop_simulation = False
        while simulation_app.is_running() and not stop_simulation:
            if on_step is None:
                _environment.world.step(render=True)
            else:
                stop_simulation = on_step()
    finally:
        simulation_app.close()
        _environment = None
        _task_prompt = None
        _arm_joint_indices = None
        _gripper_joint_index = None
        _arm_max_velocities = None
