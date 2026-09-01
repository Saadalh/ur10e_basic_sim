import argparse
import json
import re
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

from src.usd_utils import expand_gripper_visual_instances


PROJECT_ROOT = Path(__file__).resolve().parent
COLLECTION_CONFIG_PATH = PROJECT_ROOT / "ds_collect_config.json"
with COLLECTION_CONFIG_PATH.open(encoding="utf-8") as config_file:
    COLLECTION_CONFIG = json.load(config_file)


def _config_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


PATH_CONFIG = COLLECTION_CONFIG["paths"]
DATASET_CONFIG = COLLECTION_CONFIG["dataset"]
SIMULATION_CONFIG = COLLECTION_CONFIG["simulation"]
CAMERA_CONFIG = COLLECTION_CONFIG["cameras"]
ROBOT_CONFIG = COLLECTION_CONFIG["robot"]
GRIPPER_CONFIG = COLLECTION_CONFIG["gripper"]
CUBE_CONFIG = COLLECTION_CONFIG["cubes"]
CONTROLLER_CONFIG = COLLECTION_CONFIG["controller"]

USD_PATH = _config_path(PATH_CONFIG["usd"])
CAMERA_DEBUG_DIR = _config_path(PATH_CONFIG["camera_debug_dir"])
DATASET_ROOT = _config_path(DATASET_CONFIG["root"])
LEROBOT_V3_PATH = _config_path(PATH_CONFIG["lerobot_v3"])
UR10E_EXAMPLE_ROOT = _config_path(PATH_CONFIG["ur10e_controller"])
DATASET_REPO_ID = DATASET_CONFIG["repo_id"]
FILES_PER_CHUNK = int(DATASET_CONFIG["files_per_chunk"])
DATA_FILE_SIZE_LIMIT_MB = int(DATASET_CONFIG["data_file_size_limit_mb"])
VIDEO_FILE_SIZE_LIMIT_MB = int(DATASET_CONFIG["video_file_size_limit_mb"])
EPISODE_METADATA_BUFFER_SIZE = int(DATASET_CONFIG["episode_metadata_buffer_size"])
CAMERA_WIDTH = int(CAMERA_CONFIG["width"])
CAMERA_HEIGHT = int(CAMERA_CONFIG["height"])
ARM_JOINT_NAMES = tuple(ROBOT_CONFIG["arm_joint_names"])
TABLE_PRIM_PATH = SIMULATION_CONFIG["table_prim_path"]
ROBOT_PRIM_PATH = ROBOT_CONFIG["prim_path"]
END_EFFECTOR_PRIM_PATH = ROBOT_CONFIG["end_effector_prim_path"]
DEBUG_CAMERA_PRIM_PATHS = CAMERA_CONFIG["prim_paths"]
WRIST_3_JOINT_NAME = ROBOT_CONFIG["wrist_3_joint_name"]
WRIST_3_PREFERRED_POSITION = np.deg2rad(ROBOT_CONFIG["wrist_3_preferred_position_degrees"])
WRIST_3_INITIAL_ROTATION_RANGE = tuple(
    np.deg2rad(ROBOT_CONFIG["wrist_3_initial_rotation_range_degrees"])
)
ARM_INITIAL_POSITION_HALF_RANGES = np.deg2rad(
    np.asarray(ROBOT_CONFIG["arm_initial_position_half_ranges_degrees"], dtype=np.float64)
)
MAXIMUM_INITIAL_POSE_SAMPLES = int(ROBOT_CONFIG["maximum_initial_pose_samples"])
MINIMUM_INITIAL_TCP_Y_MAGNITUDE = float(ROBOT_CONFIG["minimum_initial_tcp_y_magnitude"])
MINIMUM_BASE_FRAME_TCP_X = float(ROBOT_CONFIG["minimum_base_frame_tcp_x"])
MINIMUM_BASE_FRAME_TCP_Z = float(ROBOT_CONFIG["minimum_base_frame_tcp_z"])
END_EFFECTOR_POSITION_TOLERANCE = float(CONTROLLER_CONFIG["position_tolerance"])
MAXIMUM_ALIGNMENT_STEPS = int(CONTROLLER_CONFIG["maximum_alignment_steps"])
CUBE_SIZE = float(CUBE_CONFIG["size"])
CUBE_BASE_X_RANGE = tuple(CUBE_CONFIG["base_x_range"])
CUBE_BASE_Y_MAGNITUDE_RANGE = tuple(CUBE_CONFIG["base_y_magnitude_range"])
MINIMUM_CUBE_SEPARATION = float(CUBE_CONFIG["minimum_separation"])
END_EFFECTOR_OFFSET = np.asarray(CONTROLLER_CONFIG["end_effector_offset"], dtype=np.float64)
MINIMUM_LIFT_HEIGHT = float(CONTROLLER_CONFIG["minimum_lift_height"])
MAXIMUM_PLACEMENT_ERROR = float(CONTROLLER_CONFIG["maximum_placement_error"])
CUBE_PRIM_PATHS = CUBE_CONFIG["prim_paths"]
CUBE_SETUP_POSITIVE_Y_COUNTS = tuple(CUBE_CONFIG["setup_positive_y_counts"])
AVAILABLE_COLORS = tuple(CUBE_PRIM_PATHS)
DIRECTION_OFFSETS = {
    direction.lower(): np.asarray(offset, dtype=np.float64)
    for direction, offset in COLLECTION_CONFIG["direction_offsets_base_frame"].items()
}


def _build_task_schedule() -> list[dict[str, object]]:
    if not AVAILABLE_COLORS:
        raise ValueError("cubes.prim_paths must define at least one available color")
    if any(not 0 <= count <= len(AVAILABLE_COLORS) for count in CUBE_SETUP_POSITIVE_Y_COUNTS):
        raise ValueError("Each cubes.setup_positive_y_counts value must fit the available cube count")

    color_pattern = re.compile(
        rf"\b(?:{'|'.join(re.escape(color) for color in AVAILABLE_COLORS)})\b",
        flags=re.IGNORECASE,
    )
    schedule = []
    for task_template, sample_count in COLLECTION_CONFIG["tasks"].items():
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
            raise ValueError(f"Task sample count must be a positive integer: {task_template!r}")

        color_mentions = color_pattern.findall(task_template)
        if len(color_mentions) not in (1, 2):
            raise ValueError(
                f"Task must mention one or two available colors, found {len(color_mentions)}: {task_template!r}"
            )

        target_offset = None
        task_kind = "stack"
        if len(color_mentions) == 1:
            matched_directions = [
                direction
                for direction in DIRECTION_OFFSETS
                if re.search(rf"\b{re.escape(direction)}\b", task_template, flags=re.IGNORECASE)
            ]
            if len(matched_directions) != 1:
                raise ValueError(
                    "A one-color task must contain exactly one configured direction "
                    f"({', '.join(DIRECTION_OFFSETS)}): {task_template!r}"
                )
            task_kind = "offset"
            target_offset = DIRECTION_OFFSETS[matched_directions[0]]

        variations = []
        for color_assignment in permutations(AVAILABLE_COLORS, len(color_mentions)):
            replacements = iter(color_assignment)
            description = color_pattern.sub(lambda _: next(replacements), task_template)
            variations.append(
                {
                    "description": description,
                    "kind": task_kind,
                    "colors": color_assignment,
                    "target_offset": target_offset,
                }
            )
        for _ in range(sample_count):
            schedule.extend(variations)

    print()
    print(variations)
    print()

    if not schedule:
        raise ValueError("tasks must define at least one task")
    return schedule


def _select_episode_schedule(episode_limit: int | None, seed: int | None) -> list[dict[str, object]]:
    complete_schedule = _build_task_schedule()
    if episode_limit is None:
        return complete_schedule

    rng = np.random.default_rng(seed)
    selected_indices = rng.integers(0, len(complete_schedule), size=episode_limit)
    return [complete_schedule[int(index)] for index in selected_indices]


def _random_cube_positions(
    table_height: float,
    cube_height: float,
    seed: int | None,
    robot_base_position: np.ndarray,
    base_to_world_rotation: np.ndarray,
    setup_variation: int,
) -> dict[str, np.ndarray]:
    if not 1 <= setup_variation <= len(CUBE_SETUP_POSITIVE_Y_COUNTS):
        raise ValueError(f"Unknown cube setup variation: {setup_variation}")

    rng = np.random.default_rng(seed)
    positive_y_count = CUBE_SETUP_POSITIVE_Y_COUNTS[setup_variation - 1]
    cube_y_signs = np.array(
        [1] * positive_y_count + [-1] * (len(CUBE_PRIM_PATHS) - positive_y_count),
        dtype=np.int8,
    )
    rng.shuffle(cube_y_signs)
    target_world_z = table_height + cube_height / 2
    if abs(base_to_world_rotation[2, 2]) < 1e-6:
        raise ValueError("Robot base z-axis is parallel to the table plane")

    positions: dict[str, np.ndarray] = {}
    base_positions: list[np.ndarray] = []
    while len(positions) < len(CUBE_PRIM_PATHS):
        cube_y_sign = cube_y_signs[len(positions)]
        candidate_base = np.array(
            [
                rng.uniform(*CUBE_BASE_X_RANGE),
                cube_y_sign * rng.uniform(*CUBE_BASE_Y_MAGNITUDE_RANGE),
                0.0,
            ],
            dtype=np.float64,
        )
        candidate_base[2] = (
            target_world_z
            - robot_base_position[2]
            - base_to_world_rotation[2, 0] * candidate_base[0]
            - base_to_world_rotation[2, 1] * candidate_base[1]
        ) / base_to_world_rotation[2, 2]
        if all(
            np.linalg.norm(candidate_base[:2] - position[:2]) >= MINIMUM_CUBE_SEPARATION
            for position in base_positions
        ):
            name = tuple(CUBE_PRIM_PATHS)[len(positions)]
            candidate_world = robot_base_position + base_to_world_rotation @ candidate_base
            positions[name] = candidate_world.astype(np.float32)
            base_positions.append(candidate_base)
    return positions


def _randomize_initial_robot_pose(
    robot: object,
    world: object,
    seed: int | None,
    tcp_y_sign: int,
    articulation_action_type: type,
    quat_to_rot_matrix: object,
) -> np.ndarray:
    if tcp_y_sign not in (-1, 1):
        raise ValueError(f"tcp_y_sign must be -1 or 1, got {tcp_y_sign}")

    rng = np.random.default_rng(seed)
    arm_joint_indices = np.asarray([robot.get_dof_index(name) for name in ARM_JOINT_NAMES], dtype=np.int32)
    initial_positions = np.asarray(robot.get_joint_positions(arm_joint_indices), dtype=np.float64)
    properties = robot.dof_properties
    lower = np.maximum(properties["lower"][arm_joint_indices], initial_positions - ARM_INITIAL_POSITION_HALF_RANGES)
    upper = np.minimum(properties["upper"][arm_joint_indices], initial_positions + ARM_INITIAL_POSITION_HALF_RANGES)
    wrist_3_index = ARM_JOINT_NAMES.index(WRIST_3_JOINT_NAME)
    lower[wrist_3_index] = max(lower[wrist_3_index], WRIST_3_INITIAL_ROTATION_RANGE[0])
    upper[wrist_3_index] = min(upper[wrist_3_index], WRIST_3_INITIAL_ROTATION_RANGE[1])
    if np.any(lower >= upper):
        raise RuntimeError(f"Invalid initial arm sampling ranges: lower={lower}, upper={upper}")

    base_position, base_orientation = robot.get_world_pose()
    world_to_base_rotation = quat_to_rot_matrix(base_orientation).T
    for sample_index in range(1, MAXIMUM_INITIAL_POSE_SAMPLES + 1):
        candidate = rng.uniform(lower, upper).astype(np.float32)
        robot.set_joint_positions(candidate, joint_indices=arm_joint_indices)
        robot.set_joint_velocities(np.zeros_like(candidate), joint_indices=arm_joint_indices)
        world.render()

        tcp_world_position, _ = robot.end_effector.get_world_pose()
        tcp_base_position = world_to_base_rotation @ (tcp_world_position - base_position)
        in_positive_xz_workspace = (
            tcp_base_position[0] > MINIMUM_BASE_FRAME_TCP_X
            and tcp_base_position[2] > MINIMUM_BASE_FRAME_TCP_Z
        )
        has_target_y_sign = tcp_y_sign * tcp_base_position[1] >= MINIMUM_INITIAL_TCP_Y_MAGNITUDE
        if in_positive_xz_workspace and has_target_y_sign:
            robot.get_articulation_controller().apply_action(
                articulation_action_type(joint_positions=candidate, joint_indices=arm_joint_indices)
            )
            print(
                f"Randomized initial arm pose after {sample_index} sample(s): "
                f"joints(deg)={np.degrees(candidate)}, tcp_in_base={tcp_base_position}, "
                f"target_y_sign={'positive' if tcp_y_sign > 0 else 'negative'}, "
                f"wrist_3_z_rotation={np.degrees(candidate[wrist_3_index]):.2f} degrees",
                flush=True,
            )
            return candidate

    raise RuntimeError(
        "Failed to sample an initial robot pose with positive base-frame TCP x and z and a "
        f"{'positive' if tcp_y_sign > 0 else 'negative'} base-frame TCP y "
        f"after {MAXIMUM_INITIAL_POSE_SAMPLES} attempts"
    )


def _print_poses(title: str, objects: dict[str, object]) -> None:
    print(title, flush=True)
    for name, prim in objects.items():
        position, orientation = prim.get_world_pose()
        print(f"  {name}: position={position}, orientation(wxyz)={orientation}", flush=True)


def _save_debug_camera_frames(cameras: dict[str, object]) -> None:
    from PIL import Image

    CAMERA_DEBUG_DIR.mkdir(exist_ok=True)
    for name, camera in cameras.items():
        output_path = CAMERA_DEBUG_DIR / f"{name}.png"
        Image.fromarray(_get_camera_rgb(name, camera)).save(output_path)
        print(f"Saved debug camera frame: {output_path}", flush=True)


def _get_camera_rgb(name: str, camera: object) -> np.ndarray:
    rgba = camera.get_rgba()
    if rgba is None or rgba.size == 0:
        raise RuntimeError(f"No RGB frame available from {name}")

    rgb = np.asarray(rgba)[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        scale = 255.0 if np.max(rgb) <= 1.0 else 1.0
        rgb = np.clip(rgb * scale, 0, 255)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    expected_shape = (CAMERA_HEIGHT, CAMERA_WIDTH, 3)
    if rgb.shape != expected_shape:
        raise RuntimeError(f"Unexpected frame shape from {name}: {rgb.shape}, expected {expected_shape}")
    return rgb


def _action_feature_names(dof_names: list[str]) -> list[str]:
    names = []
    for command_name in ("position", "velocity", "effort"):
        names.extend(f"{command_name}.{dof_name}" for dof_name in dof_names)
        names.extend(f"{command_name}.{dof_name}.is_commanded" for dof_name in dof_names)
    return names


def _articulation_action_vector(action: object, dof_count: int) -> np.ndarray:
    indices = (
        np.arange(dof_count, dtype=np.int64)
        if action.joint_indices is None
        else np.asarray(action.joint_indices, dtype=np.int64)
    )
    if np.any(indices < 0) or np.any(indices >= dof_count):
        raise ValueError(f"ArticulationAction contains invalid joint indices: {indices}")

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


def _validate_dataset_compatibility(dataset: object, fps: int, features: dict[str, dict]) -> None:
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
            raise ValueError(f"Existing dataset feature {name} is incompatible: {actual} != {expected}")


def _open_dataset(dataset_class: type, fps: int, dof_names: list[str]) -> object:
    features = _dataset_features(dof_names)
    if DATASET_ROOT.exists():
        if not (DATASET_ROOT / "meta" / "info.json").is_file():
            raise FileExistsError(
                f"Refusing to overwrite non-LeRobot dataset directory without meta/info.json: {DATASET_ROOT}"
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
    parser = argparse.ArgumentParser(description="Collect configured randomized cube-manipulation tasks.")
    parser.add_argument("--seed", type=int, default=None, help="Optional randomization seed.")
    parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        default=None,
        metavar="COUNT",
        help="Record COUNT randomly selected configured episodes instead of the complete task schedule.",
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(experience=PATH_CONFIG["isaac_experience"])
    args = parser.parse_args()
    if args.episodes is not None and args.episodes < 1:
        parser.error("--episodes must be at least 1")
    episode_schedule = _select_episode_schedule(args.episodes, args.seed)
    episode_count = len(episode_schedule)
    simulation_app = AppLauncher(args).app
    dataset = None
    episode_has_frames = False
    episode_saved = False

    try:
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_rot_matrix
        from isaacsim.core.utils.stage import is_stage_loading
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.robot.manipulators import SingleManipulator
        from isaacsim.robot.manipulators.grippers import ParallelGripper
        from isaacsim.sensors.camera import Camera

        if not LEROBOT_V3_PATH.is_dir():
            raise FileNotFoundError(f"LeRobot v3 package not found: {LEROBOT_V3_PATH}")
        sys.path.insert(0, str(LEROBOT_V3_PATH))
        from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

        if CODEBASE_VERSION != "v3.0":
            raise RuntimeError(f"Expected LeRobot dataset codebase v3.0, found {CODEBASE_VERSION}")

        if not UR10E_EXAMPLE_ROOT.is_dir():
            raise FileNotFoundError(f"UR10e controller example not found: {UR10E_EXAMPLE_ROOT}")
        sys.path.insert(0, str(UR10E_EXAMPLE_ROOT))
        from controller.pick_place import PickPlaceController

        if not omni.usd.get_context().open_stage(str(USD_PATH)):
            raise RuntimeError(f"Failed to open USD simulation: {USD_PATH}")
        while is_stage_loading():
            simulation_app.update()

        expanded_visuals = expand_gripper_visual_instances(omni.usd.get_context().get_stage())
        print(f"Expanded {expanded_visuals} gripper visual instances for reliable rendering.", flush=True)
        simulation_app.update()

        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / float(SIMULATION_CONFIG["physics_hz"]),
            rendering_dt=1.0 / float(SIMULATION_CONFIG["rendering_hz"]),
        )
        table = world.scene.add(FixedCuboid(prim_path=TABLE_PRIM_PATH, name="tabletop"))
        cube_physics_material = PhysicsMaterial(
            prim_path=SIMULATION_CONFIG["cube_material_prim_path"],
            static_friction=float(SIMULATION_CONFIG["cube_static_friction"]),
            dynamic_friction=float(SIMULATION_CONFIG["cube_dynamic_friction"]),
            restitution=float(SIMULATION_CONFIG["cube_restitution"]),
        )
        cubes = {
            color: world.scene.add(
                DynamicCuboid(
                    prim_path=prim_path,
                    name=f"{color}_cube",
                    scale=np.ones(3),
                    size=CUBE_SIZE,
                    physics_material=cube_physics_material,
                    mass=float(CUBE_CONFIG["mass"]),
                )
            )
            for color, prim_path in CUBE_PRIM_PATHS.items()
        }

        gripper = ParallelGripper(
            end_effector_prim_path=END_EFFECTOR_PRIM_PATH,
            joint_prim_names=GRIPPER_CONFIG["joint_prim_names"],
            joint_opened_positions=np.asarray(GRIPPER_CONFIG["joint_opened_positions"], dtype=np.float64),
            joint_closed_positions=np.asarray(GRIPPER_CONFIG["joint_closed_positions"], dtype=np.float64),
            action_deltas=np.asarray(GRIPPER_CONFIG["action_deltas"], dtype=np.float64),
            use_mimic_joints=GRIPPER_CONFIG["use_mimic_joints"],
        )
        robot = world.scene.add(
            SingleManipulator(
                prim_path=ROBOT_PRIM_PATH,
                name="ur10e_robot",
                end_effector_prim_path=END_EFFECTOR_PRIM_PATH,
                gripper=gripper,
            )
        )
        gripper.set_default_state(
            joint_positions=np.asarray(GRIPPER_CONFIG["joint_opened_positions"], dtype=np.float64)
        )
        world.reset()
        debug_cameras = {
            name: Camera(
                prim_path=prim_path,
                name=f"{name}_debug",
                resolution=(CAMERA_WIDTH, CAMERA_HEIGHT),
            )
            for name, prim_path in DEBUG_CAMERA_PRIM_PATHS.items()
        }
        for camera in debug_cameras.values():
            camera.initialize()

        table_position, _ = table.get_world_pose()
        table_height = table_position[2] + table.get_size() * table.get_world_scale()[2] / 2
        first_cube = next(iter(cubes.values()))
        cube_height = first_cube.get_size() * first_cube.get_world_scale()[2]
        identity_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        end_effector_orientation = euler_angles_to_quat(
            np.deg2rad(np.asarray(CONTROLLER_CONFIG["end_effector_orientation_degrees"], dtype=np.float64))
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
            latest_initial_frame = dataset.hf_dataset[int(latest_episode["dataset_from_index"])]
            latest_tcp_world_position = np.asarray(
                latest_initial_frame["observation.end_effector_position"], dtype=np.float64
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

        relevant_objects = {
            "tabletop": table,
            **{f"{color}_cube": cube for color, cube in cubes.items()},
            "robot": robot,
            "end_effector": robot.end_effector,
        }
        starting_episode_index = dataset.num_episodes
        for episode_index, episode_task in enumerate(episode_schedule):
            episode_number = episode_index + 1
            dataset_episode_index = starting_episode_index + episode_index
            cube_setup_variation = dataset_episode_index % len(CUBE_SETUP_POSITIVE_Y_COUNTS) + 1
            positive_y_cube_count = CUBE_SETUP_POSITIVE_Y_COUNTS[cube_setup_variation - 1]
            negative_y_cube_count = len(CUBE_PRIM_PATHS) - positive_y_cube_count
            task_description = str(episode_task["description"])
            task_colors = tuple(episode_task["colors"])
            source_color = task_colors[0]
            episode_seed = None if args.seed is None else args.seed + episode_index
            tcp_y_sign = first_tcp_y_sign if episode_index % 2 == 0 else -first_tcp_y_sign
            episode_has_frames = False
            episode_saved = False
            if episode_index > 0:
                world.reset()

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
            _print_poses(f"Episode {episode_number} initial scene poses:", relevant_objects)
            _randomize_initial_robot_pose(
                robot=robot,
                world=world,
                seed=episode_seed,
                tcp_y_sign=tcp_y_sign,
                articulation_action_type=ArticulationAction,
                quat_to_rot_matrix=quat_to_rot_matrix,
            )

            robot_base_position, robot_base_orientation = robot.get_world_pose()
            base_to_world_rotation = quat_to_rot_matrix(robot_base_orientation)
            randomized_positions = _random_cube_positions(
                table_height,
                cube_height,
                episode_seed,
                robot_base_position,
                base_to_world_rotation,
                cube_setup_variation,
            )
            for name, position in randomized_positions.items():
                cube = cubes[name]
                cube.set_world_pose(position=position, orientation=identity_orientation)
                cube.set_linear_velocity(np.zeros(3))
                cube.set_angular_velocity(np.zeros(3))
                cube_in_base = base_to_world_rotation.T @ (position - robot_base_position)
                print(f"{name}_cube initial base-frame position: {cube_in_base}", flush=True)

            for _ in range(int(SIMULATION_CONFIG["settling_steps_before_recording"])):
                world.step(render=True)
            _save_debug_camera_frames(debug_cameras)
            _print_poses(f"Episode {episode_number} randomized scene poses:", relevant_objects)

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
                print(f"Acquired {target_color} target pose: {target_position}", flush=True)
            else:
                target_offset = np.asarray(episode_task["target_offset"], dtype=np.float64)
                placing_position = source_position + base_to_world_rotation @ target_offset
                print(f"Configured base-frame target offset: {target_offset}", flush=True)
            print(f"Acquired {source_color} pick pose: {source_position}", flush=True)
            print(f"Computed {source_color} placement pose: {placing_position}", flush=True)

            events_dt = [float(value) for value in CONTROLLER_CONFIG["base_events_dt"]]
            for event_index in CONTROLLER_CONFIG["movement_event_indices"]:
                events_dt[int(event_index)] *= float(CONTROLLER_CONFIG["movement_speed_multiplier"])

            controller = PickPlaceController(
                name=f"{source_color}_task_controller_episode_{episode_number}",
                robot_articulation=robot,
                gripper=gripper,
                events_dt=events_dt,
            )
            controller.reset(
                end_effector_initial_height=max(source_position[2], placing_position[2])
                + float(CONTROLLER_CONFIG["initial_height_clearance"])
            )

            rmpflow = controller._cspace_controller.rmpflow
            active_joint_names = rmpflow.get_active_joints()
            current_joint_positions = robot.get_joint_positions()
            cspace_target = np.array(
                [current_joint_positions[robot.get_dof_index(name)] for name in active_joint_names],
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
                world.step(render=True)
                event = controller.get_current_event()
                if not grasp_verified and event >= 5:
                    lifted_source_position, _ = source_cube.get_world_pose()
                    lift_height = lifted_source_position[2] - source_position[2]
                    if lift_height < MINIMUM_LIFT_HEIGHT:
                        raise RuntimeError(
                            f"Physical grasp failed: {source_color} cube lifted only {lift_height:.4f}m"
                        )
                    print(
                        f"Physical grasp verified: {source_color} cube lifted {lift_height:.4f}m",
                        flush=True,
                    )
                    grasp_verified = True

                if event in alignment_targets and event not in aligned_events:
                    end_effector_position, _ = robot.end_effector.get_world_pose()
                    alignment_error = np.linalg.norm(end_effector_position - alignment_targets[event])
                    if alignment_error > END_EFFECTOR_POSITION_TOLERANCE:
                        alignment_steps[event] += 1
                        if alignment_steps[event] > MAXIMUM_ALIGNMENT_STEPS:
                            raise RuntimeError(
                                f"End effector failed to converge before event {event}: {alignment_error:.4f}m error"
                            )
                        actions = controller._cspace_controller.forward(
                            target_end_effector_position=alignment_targets[event],
                            target_end_effector_orientation=end_effector_orientation,
                        )
                    else:
                        print(f"Event {event} alignment verified: {alignment_error:.4f}m error", flush=True)
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
                        "observation.images.side_camera": _get_camera_rgb(
                            "side_camera", debug_cameras["side_camera"]
                        ),
                        "observation.images.wrist_camera": _get_camera_rgb(
                            "wrist_camera", debug_cameras["wrist_camera"]
                        ),
                        "observation.state": np.asarray(robot.get_joint_positions(), dtype=np.float32),
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
                raise RuntimeError(f"Simulation stopped before episode {episode_number} completed")

            for _ in range(int(SIMULATION_CONFIG["settling_steps_after_controller"])):
                world.step(render=True)
            final_task_objects = {f"{source_color}_cube": source_cube}
            if target_cube is not None:
                final_task_objects[f"{target_color}_cube"] = target_cube
            _print_poses(f"Episode {episode_number} final task poses:", final_task_objects)
            final_source_position, _ = source_cube.get_world_pose()
            if target_cube is not None:
                final_target_position, _ = target_cube.get_world_pose()
                expected_source_position = final_target_position + np.array([0.0, 0.0, cube_height])
            else:
                expected_source_position = placing_position
            final_wrist_3_position = robot.get_joint_positions()[robot.get_dof_index(WRIST_3_JOINT_NAME)]
            placement_error = np.linalg.norm(final_source_position - expected_source_position)
            print(f"Final {source_color} placement error: {placement_error:.4f}m", flush=True)
            print(f"Final {WRIST_3_JOINT_NAME} angle: {np.degrees(final_wrist_3_position):.2f} degrees", flush=True)
            if placement_error > MAXIMUM_PLACEMENT_ERROR:
                raise RuntimeError(f"Physical task failed: position error is {placement_error:.4f}m")
            dataset.save_episode()
            episode_saved = True
            print(f"Saved LeRobot episode {dataset.num_episodes - 1} to {DATASET_ROOT}", flush=True)
            print(f"Completed episode {episode_number}/{episode_count}: {task_description}", flush=True)
    finally:
        try:
            if dataset is not None:
                try:
                    if episode_has_frames and not episode_saved:
                        dataset.clear_episode_buffer()
                        print("Discarded incomplete LeRobot episode.", flush=True)
                finally:
                    dataset.finalize()
        finally:
            simulation_app.close()


if __name__ == "__main__":
    main()
