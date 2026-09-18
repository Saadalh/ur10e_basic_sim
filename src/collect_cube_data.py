import argparse
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from isaaclab.app import AppLauncher

from src.collection_runtime import (
    CollectionStopped,
    ManipulationFailure,
    StopRequest,
    finalize_collection,
    is_incomplete_dataset_stub,
)

from src.cube_environment import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    COLLECTION_CONFIG,
    CUBE_PRIM_PATHS,
    CUBE_SETUP_POSITIVE_Y_COUNTS,
    DATASET_CONFIG,
    GRIPPER_CONFIG,
    PATH_CONFIG,
    PROJECT_ROOT,
    ROBOT_CONFIG,
    SIMULATION_CONFIG,
    CubeEnvironment,
    config_path,
    select_episode_schedule,
)

CONTROLLER_CONFIG = COLLECTION_CONFIG["controller"]

DATASET_ROOT = config_path(DATASET_CONFIG["root"])
LEROBOT_V3_PATH = config_path(PATH_CONFIG["lerobot_v3"])
OPENPI_ROOT = config_path(PATH_CONFIG["openpi"])
UR10E_EXAMPLE_ROOT = config_path(PATH_CONFIG["ur10e_controller"])
TRAINING_CONFIG_PATH = PROJECT_ROOT / "config" / "train_config.yaml"
DATASET_REPO_ID = DATASET_CONFIG["repo_id"]
FILES_PER_CHUNK = int(DATASET_CONFIG["files_per_chunk"])
DATA_FILE_SIZE_LIMIT_MB = int(DATASET_CONFIG["data_file_size_limit_mb"])
VIDEO_FILE_SIZE_LIMIT_MB = int(DATASET_CONFIG["video_file_size_limit_mb"])
EPISODE_METADATA_BUFFER_SIZE = int(DATASET_CONFIG["episode_metadata_buffer_size"])
WRIST_3_JOINT_NAME = ROBOT_CONFIG["wrist_3_joint_name"]
WRIST_3_PREFERRED_POSITION = np.deg2rad(
    ROBOT_CONFIG["wrist_3_preferred_position_degrees"]
)
END_EFFECTOR_POSITION_TOLERANCE = float(CONTROLLER_CONFIG["position_tolerance"])
MAXIMUM_ALIGNMENT_STEPS = int(CONTROLLER_CONFIG["maximum_alignment_steps"])
END_EFFECTOR_OFFSET = np.asarray(
    CONTROLLER_CONFIG["end_effector_offset"], dtype=np.float64
)
MINIMUM_LIFT_HEIGHT = float(CONTROLLER_CONFIG["minimum_lift_height"])
MAXIMUM_PLACEMENT_ERROR = float(CONTROLLER_CONFIG["maximum_placement_error"])


def _generate_dataset_stats() -> None:
    openpi_python = OPENPI_ROOT / ".venv" / "bin" / "python"
    if not openpi_python.is_file():
        raise FileNotFoundError(f"OpenPI Python environment not found: {openpi_python}")
    command = [
        str(openpi_python),
        "-m",
        "src.dataset_stats",
        "--dataset-root",
        str(DATASET_ROOT),
        "--openpi-root",
        str(OPENPI_ROOT),
        "--training-config",
        str(TRAINING_CONFIG_PATH),
    ]
    try:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"OpenPI dataset statistics generation failed with exit code {error.returncode}"
        ) from error


def _action_feature_names(dof_names: list[str]) -> list[str]:
    names = []
    for command_name in ("position", "velocity", "effort"):
        names.extend(f"{command_name}.{dof_name}" for dof_name in dof_names)
        names.extend(
            f"{command_name}.{dof_name}.is_commanded" for dof_name in dof_names
        )
    return names


def _articulation_action_vector(action: object, dof_count: int) -> np.ndarray:
    indices = (
        np.arange(dof_count, dtype=np.int64)
        if action.joint_indices is None
        else np.asarray(action.joint_indices, dtype=np.int64)
    )
    if np.any(indices < 0) or np.any(indices >= dof_count):
        raise ValueError(
            f"ArticulationAction contains invalid joint indices: {indices}"
        )

    vector_parts = []
    for attribute in ("joint_positions", "joint_velocities", "joint_efforts"):
        values = np.zeros(dof_count, dtype=np.float32)
        commanded = np.zeros(dof_count, dtype=np.float32)
        commands = getattr(action, attribute)
        if commands is not None:
            commands = np.asarray(commands, dtype=object).reshape(-1)
            if len(commands) != len(indices):
                raise ValueError(
                    f"{attribute} has {len(commands)} values for {len(indices)} ArticulationAction indices"
                )
            for command, dof_index in zip(commands, indices, strict=True):
                if command is None or np.isnan(float(command)):
                    continue
                values[dof_index] = np.float32(command)
                commanded[dof_index] = 1.0
        vector_parts.extend((values, commanded))
    return np.concatenate(vector_parts, dtype=np.float32)


def _dataset_features(dof_names: list[str]) -> dict[str, dict]:
    image_feature = {
        "dtype": "video",
        "shape": (3, CAMERA_HEIGHT, CAMERA_WIDTH),
        "names": ["channels", "height", "width"],
    }
    action_names = _action_feature_names(dof_names)
    return {
        "observation.images.side_camera": image_feature.copy(),
        "observation.images.wrist_camera": image_feature.copy(),
        "observation.state": {
            "dtype": "float32",
            "shape": (len(dof_names),),
            "names": dof_names,
        },
        "observation.end_effector_position": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["x", "y", "z"],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": action_names,
        },
    }


def _validate_dataset_compatibility(
    dataset: object, fps: int, features: dict[str, dict]
) -> None:
    if dataset.fps != fps:
        raise ValueError(f"Existing dataset FPS is {dataset.fps}, expected {fps}")
    for name, expected in features.items():
        if name not in dataset.features:
            raise ValueError(f"Existing dataset is missing required feature: {name}")
        actual = dataset.features[name]
        if (
            actual["dtype"] != expected["dtype"]
            or tuple(actual["shape"]) != tuple(expected["shape"])
            or actual.get("names") != expected.get("names")
        ):
            raise ValueError(
                f"Existing dataset feature {name} is incompatible: {actual} != {expected}"
            )


def _open_dataset(dataset_class: type, fps: int, dof_names: list[str]) -> object:
    features = _dataset_features(dof_names)
    if DATASET_ROOT.exists() and is_incomplete_dataset_stub(DATASET_ROOT):
        print(
            f"Removing incomplete dataset stub with no saved episodes: {DATASET_ROOT}",
            flush=True,
        )
        shutil.rmtree(DATASET_ROOT)
    if DATASET_ROOT.exists():
        if not (DATASET_ROOT / "meta" / "info.json").is_file():
            raise FileExistsError(
                f"Refusing to overwrite non-LeRobot dataset directory without meta/info.json: {DATASET_ROOT}"
            )
        if not (DATASET_ROOT / "meta" / "tasks.parquet").is_file():
            raise FileExistsError(
                f"Dataset at {DATASET_ROOT} is incomplete: meta/tasks.parquet is missing "
                f"but episodes were recorded. Remove or repair the directory manually; "
                f"refusing to overwrite recorded data."
            )
        dataset = dataset_class(
            repo_id=DATASET_REPO_ID,
            root=DATASET_ROOT,
            video_backend=DATASET_CONFIG["video_backend"],
            vcodec=DATASET_CONFIG["video_codec"],
            streaming_encoding=DATASET_CONFIG["streaming_encoding"],
        )
        try:
            _validate_dataset_compatibility(dataset, fps, features)
        except Exception:
            dataset.finalize()
            raise
        dataset.meta.metadata_buffer_size = EPISODE_METADATA_BUFFER_SIZE
        dataset.meta.update_chunk_settings(
            chunks_size=FILES_PER_CHUNK,
            data_files_size_in_mb=DATA_FILE_SIZE_LIMIT_MB,
            video_files_size_in_mb=VIDEO_FILE_SIZE_LIMIT_MB,
        )
        dataset.one_file_per_episode = DATASET_CONFIG["one_file_per_episode"]
        dataset.meta.one_file_per_episode = DATASET_CONFIG["one_file_per_episode"]
        print(f"Appending to LeRobot dataset at {DATASET_ROOT}", flush=True)
        return dataset

    dataset = dataset_class.create(
        repo_id=DATASET_REPO_ID,
        root=DATASET_ROOT,
        fps=fps,
        robot_type=DATASET_CONFIG["robot_type"],
        features=features,
        use_videos=True,
        video_backend=DATASET_CONFIG["video_backend"],
        vcodec=DATASET_CONFIG["video_codec"],
        metadata_buffer_size=EPISODE_METADATA_BUFFER_SIZE,
        streaming_encoding=DATASET_CONFIG["streaming_encoding"],
    )
    dataset.meta.update_chunk_settings(
        chunks_size=FILES_PER_CHUNK,
        data_files_size_in_mb=DATA_FILE_SIZE_LIMIT_MB,
        video_files_size_in_mb=VIDEO_FILE_SIZE_LIMIT_MB,
    )
    dataset.one_file_per_episode = DATASET_CONFIG["one_file_per_episode"]
    dataset.meta.one_file_per_episode = DATASET_CONFIG["one_file_per_episode"]
    print(f"Created LeRobot dataset at {DATASET_ROOT}", flush=True)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect configured randomized cube-manipulation tasks."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Optional randomization seed."
    )
    parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        default=None,
        metavar="COUNT",
        help="Record COUNT randomly selected configured episodes instead of the complete task schedule.",
    )
    parser.add_argument(
        "--log-file", type=Path, default=PROJECT_ROOT / "collection.log"
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(experience=PATH_CONFIG["isaac_experience"], enable_cameras=True)
    args = parser.parse_args()
    if args.episodes is not None and args.episodes < 1:
        parser.error("--episodes must be at least 1")
    episode_schedule = select_episode_schedule(args.episodes, args.seed)
    episode_count = len(episode_schedule)
    # All dataset access in this collector is local (repo_id "local/...").
    # Offline mode converts any Hub fallback for local metadata problems
    # into an immediate local error instead of a confusing 401.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    max_attempts = int(CONTROLLER_CONFIG.get("maximum_episode_attempts", 3))
    if max_attempts < 1:
        parser.error("maximum_episode_attempts must be at least 1")
    logger = logging.getLogger("ur10e.collection")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(args.log_file.expanduser(), encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stderr))
    # Use Isaac Sim's default fast shutdown: extension-by-extension teardown
    # has thrown ordering errors (USD context destroyed while GUI callbacks
    # still run). All durable work (finalize, statistics, logs) finishes
    # before close(), and a watchdog bounds the shutdown itself.
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    stop_request = StopRequest()
    stop_request.install()
    failure = None
    dataset = None
    episode_has_frames = False
    episode_saved = False

    try:
        from isaacsim.core.utils.rotations import (
            euler_angles_to_quat,
            quat_to_rot_matrix,
        )

        if not LEROBOT_V3_PATH.is_dir():
            raise FileNotFoundError(f"LeRobot v3 package not found: {LEROBOT_V3_PATH}")
        sys.path.insert(0, str(LEROBOT_V3_PATH))
        from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

        if CODEBASE_VERSION != "v3.0":
            raise RuntimeError(
                f"Expected LeRobot dataset codebase v3.0, found {CODEBASE_VERSION}"
            )

        if not UR10E_EXAMPLE_ROOT.is_dir():
            raise FileNotFoundError(
                f"UR10e controller example not found: {UR10E_EXAMPLE_ROOT}"
            )
        sys.path.insert(0, str(UR10E_EXAMPLE_ROOT))
        from src.pick_place import StagedPickPlaceController

        environment = CubeEnvironment.create(simulation_app)
        world = environment.world
        cubes = environment.cubes
        gripper = environment.gripper
        robot = environment.robot
        cube_height = environment.cube_height
        end_effector_orientation = euler_angles_to_quat(
            np.deg2rad(
                np.asarray(
                    CONTROLLER_CONFIG["end_effector_orientation_degrees"],
                    dtype=np.float64,
                )
            )
        )
        dof_names = list(robot.dof_names)
        dataset_fps = round(1.0 / world.get_rendering_dt())
        if dataset_fps != int(DATASET_CONFIG["fps"]):
            raise ValueError(
                f"Configured dataset FPS {DATASET_CONFIG['fps']} does not match rendering FPS {dataset_fps}"
            )
        dataset = _open_dataset(LeRobotDataset, dataset_fps, dof_names)
        print(f"Recording controller timesteps at {dataset_fps} FPS", flush=True)

        if dataset.num_episodes > 0:
            latest_episode = dataset.meta.episodes[-1]
            latest_initial_frame = dataset.hf_dataset[
                int(latest_episode["dataset_from_index"])
            ]
            latest_tcp_world_position = np.asarray(
                latest_initial_frame["observation.end_effector_position"],
                dtype=np.float64,
            )
            base_position, base_orientation = robot.get_world_pose()
            latest_tcp_base_position = quat_to_rot_matrix(base_orientation).T @ (
                latest_tcp_world_position - base_position
            )
            first_tcp_y_sign = -1 if latest_tcp_base_position[1] > 0 else 1
            print(
                f"Latest episode initial TCP base-frame y was {latest_tcp_base_position[1]:.4f}m; "
                f"the next episode will target {'positive' if first_tcp_y_sign > 0 else 'negative'} y.",
                flush=True,
            )
        else:
            first_tcp_y_sign = int(np.random.default_rng(args.seed).choice((-1, 1)))
            print(
                f"New dataset will start with {'positive' if first_tcp_y_sign > 0 else 'negative'} "
                "base-frame TCP y.",
                flush=True,
            )

        def collect_episode(episode_index, episode_task, attempt):
            nonlocal dataset, episode_has_frames, episode_saved
            episode_number = episode_index + 1
            dataset_episode_index = dataset.num_episodes
            cube_setup_variation = (
                dataset_episode_index % len(CUBE_SETUP_POSITIVE_Y_COUNTS) + 1
            )
            positive_y_cube_count = CUBE_SETUP_POSITIVE_Y_COUNTS[
                cube_setup_variation - 1
            ]
            negative_y_cube_count = len(CUBE_PRIM_PATHS) - positive_y_cube_count
            task_description = str(episode_task["description"])
            task_colors = tuple(episode_task["colors"])
            source_color = task_colors[0]
            episode_seed = (
                None
                if args.seed is None
                else args.seed + episode_index + attempt * episode_count
            )
            tcp_y_sign = (
                first_tcp_y_sign if episode_index % 2 == 0 else -first_tcp_y_sign
            )
            episode_has_frames = False
            episode_saved = False
            logger.info(
                "Episode %s/%s attempt %s/%s seed=%s task=%s",
                episode_number,
                episode_count,
                attempt + 1,
                max_attempts,
                episode_seed,
                task_description,
            )
            stop_request.check()

            print(
                f"Starting episode {episode_number}/{episode_count} with seed {episode_seed} and "
                f"{'positive' if tcp_y_sign > 0 else 'negative'} initial TCP y",
                flush=True,
            )
            print(f"Task: {task_description}", flush=True)
            print(
                f"Cube setup variation {cube_setup_variation}: all cubes require base-frame x > 0; "
                f"{positive_y_cube_count} require y > 0 and {negative_y_cube_count} require y < 0.",
                flush=True,
            )
            base_to_world_rotation = environment.prepare_episode(
                seed=episode_seed,
                tcp_y_sign=tcp_y_sign,
                setup_variation=cube_setup_variation,
                episode_label=f"Episode {episode_number}",
                reset_world=episode_index > 0 or attempt > 0,
                task=episode_task,
                check_stop=stop_request.check,
            )

            source_cube = cubes[source_color]
            source_position, _ = source_cube.get_world_pose()
            target_cube = None
            target_color = None
            if episode_task["kind"] == "stack":
                target_color = task_colors[1]
                target_cube = cubes[target_color]
                target_position, _ = target_cube.get_world_pose()
                placing_position = target_position.copy()
                placing_position[2] += cube_height
                print(
                    f"Acquired {target_color} target pose: {target_position}",
                    flush=True,
                )
            else:
                target_offset = np.asarray(
                    episode_task["target_offset"], dtype=np.float64
                )
                placing_position = (
                    source_position + base_to_world_rotation @ target_offset
                )
                print(
                    f"Configured base-frame target offset: {target_offset}", flush=True
                )
            print(f"Acquired {source_color} pick pose: {source_position}", flush=True)
            print(
                f"Computed {source_color} placement pose: {placing_position}",
                flush=True,
            )

            events_dt = [float(value) for value in CONTROLLER_CONFIG["base_events_dt"]]
            for event_index in CONTROLLER_CONFIG["movement_event_indices"]:
                events_dt[int(event_index)] *= float(
                    CONTROLLER_CONFIG["movement_speed_multiplier"]
                )

            controller = StagedPickPlaceController(
                name=f"{source_color}_task_controller_episode_{episode_number}",
                robot_articulation=robot,
                gripper=gripper,
                robot=robot,
                events_dt=events_dt,
                pre_grasp_height=float(CONTROLLER_CONFIG["pre_grasp_height"]),
                approach_tolerance=float(CONTROLLER_CONFIG["approach_tolerance"]),
                descent_step=float(CONTROLLER_CONFIG["descent_step"]),
                xy_lock_tolerance=float(CONTROLLER_CONFIG["xy_lock_tolerance"]),
                max_approach_steps=int(CONTROLLER_CONFIG["max_approach_steps"]),
                grasp_increments=int(GRIPPER_CONFIG["grasp_increments"]),
                grasp_position_eps=float(GRIPPER_CONFIG["grasp_position_eps"]),
                grasp_steady_steps=int(GRIPPER_CONFIG["grasp_steady_steps"]),
                grasp_min_travel_fraction=float(
                    GRIPPER_CONFIG["grasp_min_travel_fraction"]
                ),
                max_grasp_steps=int(GRIPPER_CONFIG["max_grasp_steps"]),
            )
            controller.reset(
                end_effector_initial_height=max(source_position[2], placing_position[2])
                + float(CONTROLLER_CONFIG["initial_height_clearance"])
            )

            rmpflow = controller._cspace_controller.rmpflow
            active_joint_names = rmpflow.get_active_joints()
            current_joint_positions = robot.get_joint_positions()
            cspace_target = np.array(
                [
                    current_joint_positions[robot.get_dof_index(name)]
                    for name in active_joint_names
                ],
                dtype=np.float64,
            )
            wrist_3_active_index = active_joint_names.index(WRIST_3_JOINT_NAME)

            # Camera-aware soft posture preference: bias wrist_3 toward +90 degrees.
            # This is not a joint lock; RMPflow may move it whenever the Cartesian task requires it.
            cspace_target[wrist_3_active_index] = WRIST_3_PREFERRED_POSITION
            rmpflow.set_cspace_target(cspace_target)

            articulation_controller = robot.get_articulation_controller()
            grasp_verified = False
            alignment_steps = {2: 0, 7: 0}
            alignment_targets = {
                2: source_position + END_EFFECTOR_OFFSET,
                7: placing_position + END_EFFECTOR_OFFSET,
            }
            aligned_events = set()

            while simulation_app.is_running() and not controller.is_done():
                stop_request.check()
                world.step(render=True)
                event = controller.get_current_event()
                if not grasp_verified and event >= 5:
                    if controller.grasp_hold is None:
                        raise ManipulationFailure(
                            "Gripper closed without stalling on the cube"
                        )
                    lifted_source_position, _ = source_cube.get_world_pose()
                    lift_height = lifted_source_position[2] - source_position[2]
                    if lift_height < MINIMUM_LIFT_HEIGHT:
                        raise ManipulationFailure(
                            f"Physical grasp failed: {source_color} cube lifted only {lift_height:.4f}m"
                        )
                    print(
                        f"Physical grasp verified: {source_color} cube lifted {lift_height:.4f}m",
                        flush=True,
                    )
                    grasp_verified = True

                if event in alignment_targets and event not in aligned_events:
                    end_effector_position, _ = robot.end_effector.get_world_pose()
                    alignment_error = np.linalg.norm(
                        end_effector_position - alignment_targets[event]
                    )
                    if alignment_error > END_EFFECTOR_POSITION_TOLERANCE:
                        alignment_steps[event] += 1
                        if alignment_steps[event] > MAXIMUM_ALIGNMENT_STEPS:
                            raise ManipulationFailure(
                                f"End effector failed to converge before event {event}: {alignment_error:.4f}m error; "
                                f"tcp={end_effector_position}, target={alignment_targets[event]}"
                            )
                        actions = controller._cspace_controller.forward(
                            target_end_effector_position=alignment_targets[event],
                            target_end_effector_orientation=end_effector_orientation,
                        )
                    else:
                        print(
                            f"Event {event} alignment verified: {alignment_error:.4f}m error",
                            flush=True,
                        )
                        aligned_events.add(event)
                        actions = controller.forward(
                            picking_position=source_position,
                            placing_position=placing_position,
                            current_joint_positions=robot.get_joint_positions(),
                            end_effector_offset=END_EFFECTOR_OFFSET,
                            end_effector_orientation=end_effector_orientation,
                        )
                else:
                    actions = controller.forward(
                        picking_position=source_position,
                        placing_position=placing_position,
                        current_joint_positions=robot.get_joint_positions(),
                        end_effector_offset=END_EFFECTOR_OFFSET,
                        end_effector_orientation=end_effector_orientation,
                    )
                end_effector_position, _ = robot.end_effector.get_world_pose()
                dataset.add_frame(
                    {
                        "observation.images.side_camera": environment.get_camera_rgb(
                            "side_camera"
                        ),
                        "observation.images.wrist_camera": environment.get_camera_rgb(
                            "wrist_camera"
                        ),
                        "observation.state": np.asarray(
                            robot.get_joint_positions(), dtype=np.float32
                        ),
                        "observation.end_effector_position": np.asarray(
                            end_effector_position, dtype=np.float32
                        ),
                        "action": _articulation_action_vector(actions, len(dof_names)),
                        "task": task_description,
                    }
                )
                episode_has_frames = True
                articulation_controller.apply_action(actions)

            if not controller.is_done():
                raise RuntimeError(
                    f"Simulation stopped before episode {episode_number} completed"
                )

            for _ in range(int(SIMULATION_CONFIG["settling_steps_after_controller"])):
                stop_request.check()
                world.step(render=True)
            final_task_objects = {f"{source_color}_cube": source_cube}
            if target_cube is not None:
                final_task_objects[f"{target_color}_cube"] = target_cube
            environment.print_poses(
                f"Episode {episode_number} final task poses:", final_task_objects
            )
            final_source_position, _ = source_cube.get_world_pose()
            if target_cube is not None:
                final_target_position, _ = target_cube.get_world_pose()
                expected_source_position = final_target_position + np.array(
                    [0.0, 0.0, cube_height]
                )
            else:
                expected_source_position = placing_position
            final_wrist_3_position = robot.get_joint_positions()[
                robot.get_dof_index(WRIST_3_JOINT_NAME)
            ]
            placement_error = np.linalg.norm(
                final_source_position - expected_source_position
            )
            print(
                f"Final {source_color} placement error: {placement_error:.4f}m",
                flush=True,
            )
            print(
                f"Final {WRIST_3_JOINT_NAME} angle: {np.degrees(final_wrist_3_position):.2f} degrees",
                flush=True,
            )
            if placement_error > MAXIMUM_PLACEMENT_ERROR:
                raise ManipulationFailure(
                    f"Physical task failed: position error is {placement_error:.4f}m"
                )
            stop_request.check()
            dataset.save_episode()
            episode_saved = True
            # Close Parquet footers and flush episode metadata before declaring success.
            # finalize() is idempotent, so calling it again during shutdown is safe.
            dataset.finalize()
            logger.info("Saved dataset episode %s", dataset.num_episodes - 1)
            print(
                f"Saved LeRobot episode {dataset.num_episodes - 1} to {DATASET_ROOT}",
                flush=True,
            )
            print(
                f"Completed episode {episode_number}/{episode_count}: {task_description}",
                flush=True,
            )
            stop_request.check()
            if episode_index + 1 < episode_count:
                # More episodes remain: reopen for the next one rather than
                # reusing finalized writers. The final episode skips this so
                # shutdown does not churn another dataset instance.
                dataset = _open_dataset(LeRobotDataset, dataset_fps, dof_names)

        exhausted = 0
        for episode_index, episode_task in enumerate(episode_schedule):
            for attempt in range(max_attempts):
                stop_request.check()
                try:
                    collect_episode(episode_index, episode_task, attempt)
                    break
                except ManipulationFailure:
                    logger.exception(
                        "Rejected episode %s attempt %s", episode_index + 1, attempt + 1
                    )
                    if episode_has_frames and not episode_saved:
                        dataset.clear_episode_buffer()
                    episode_has_frames = False
                    if attempt + 1 == max_attempts:
                        exhausted += 1
                        logger.error(
                            "Skipping task after %s attempts: %s",
                            max_attempts,
                            episode_task["description"],
                        )
        logger.info(
            "Schedule finished: %s succeeded, %s exhausted retries",
            episode_count - exhausted,
            exhausted,
        )
        if exhausted:
            raise RuntimeError(
                f"{exhausted} scheduled tasks exhausted retries; see {args.log_file}"
            )
    except BaseException as error:
        # Emit now: logging after close() is too late on some Kit shutdown paths.
        logger.exception("Collection stopped: %s", error)
        failure = error
    finally:
        # Decide the process outcome now: nothing after close() is guaranteed
        # to run on every Kit shutdown path, so the result is logged here and
        # the watchdog below reuses the same code if it must terminate.
        if failure is None:
            exit_code = 0
        elif isinstance(failure, CollectionStopped):
            exit_code = 128 + int(stop_request.signum or 0)
        else:
            exit_code = 1
        shutdown_config = COLLECTION_CONFIG.get("shutdown", {})
        watchdog_dump = config_path(
            str(shutdown_config.get("watchdog_dump_file", "shutdown_watchdog.log"))
        )
        try:
            finalize_collection(
                dataset,
                discard=episode_has_frames and not episode_saved,
                generate_stats=_generate_dataset_stats,
                close_app=simulation_app.close,
                logger=logger,
                shutdown_watchdog_seconds=shutdown_config.get("watchdog_seconds", 300),
                watchdog_dump_file=watchdog_dump,
                exit_code=exit_code,
                encoder_stop_timeout_seconds=shutdown_config.get(
                    "encoder_stop_timeout_seconds", 60
                ),
            )
        except Exception as error:
            if failure is None:
                failure = error
        finally:
            stop_request.restore()
    if isinstance(failure, CollectionStopped):
        raise SystemExit(128 + int(stop_request.signum or 0))
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
