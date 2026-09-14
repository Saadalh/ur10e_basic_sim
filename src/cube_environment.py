import json
import re
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np

from src.usd_utils import expand_gripper_visual_instances
from src.collection_runtime import ManipulationFailure

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "ds_collect_config.json"
with CONFIG_PATH.open(encoding="utf-8") as config_file:
    COLLECTION_CONFIG = json.load(config_file)

PATH_CONFIG = COLLECTION_CONFIG["paths"]
DATASET_CONFIG = COLLECTION_CONFIG["dataset"]
SIMULATION_CONFIG = COLLECTION_CONFIG["simulation"]
CAMERA_CONFIG = COLLECTION_CONFIG["cameras"]
ROBOT_CONFIG = COLLECTION_CONFIG["robot"]
GRIPPER_CONFIG = COLLECTION_CONFIG["gripper"]
CUBE_CONFIG = COLLECTION_CONFIG["cubes"]

CAMERA_WIDTH = int(CAMERA_CONFIG["width"])
CAMERA_HEIGHT = int(CAMERA_CONFIG["height"])
ARM_JOINT_NAMES = tuple(ROBOT_CONFIG["arm_joint_names"])
CUBE_PRIM_PATHS = CUBE_CONFIG["prim_paths"]
CUBE_SETUP_POSITIVE_Y_COUNTS = tuple(CUBE_CONFIG["setup_positive_y_counts"])
AVAILABLE_COLORS = tuple(CUBE_PRIM_PATHS)
DIRECTION_OFFSETS = {
    direction.lower(): np.asarray(offset, dtype=np.float64)
    for direction, offset in COLLECTION_CONFIG["direction_offsets_base_frame"].items()
}


def config_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def build_task_schedule() -> list[dict[str, object]]:
    if not AVAILABLE_COLORS:
        raise ValueError("cubes.prim_paths must define at least one available color")
    if any(
        not 0 <= count <= len(AVAILABLE_COLORS)
        for count in CUBE_SETUP_POSITIVE_Y_COUNTS
    ):
        raise ValueError(
            "Each cubes.setup_positive_y_counts value must fit the available cube count"
        )

    color_pattern = re.compile(
        rf"\b(?:{'|'.join(re.escape(color) for color in AVAILABLE_COLORS)})\b",
        flags=re.IGNORECASE,
    )
    schedule = []
    for task_template, sample_count in COLLECTION_CONFIG["tasks"].items():
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count < 1
        ):
            raise ValueError(
                f"Task sample count must be a positive integer: {task_template!r}"
            )

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
                if re.search(
                    rf"\b{re.escape(direction)}\b", task_template, flags=re.IGNORECASE
                )
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
            description = color_pattern.sub(
                lambda _, replacements=replacements: next(replacements), task_template
            )
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

    if not schedule:
        raise ValueError("tasks must define at least one task")
    return schedule


def select_episode_schedule(
    episode_limit: int | None, seed: int | None
) -> list[dict[str, object]]:
    complete_schedule = build_task_schedule()
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
    task: dict[str, object] | None = None,
    table_bounds: tuple[np.ndarray, np.ndarray] | None = None,
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
    for _ in range(int(CUBE_CONFIG.get("maximum_scene_samples", 10000))):
        cube_y_sign = cube_y_signs[len(positions)]
        candidate_base = np.array(
            [
                rng.uniform(*CUBE_CONFIG["base_x_range"]),
                cube_y_sign * rng.uniform(*CUBE_CONFIG["base_y_magnitude_range"]),
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
            np.linalg.norm(candidate_base[:2] - position[:2])
            >= float(CUBE_CONFIG["minimum_separation"])
            for position in base_positions
        ):
            name = tuple(CUBE_PRIM_PATHS)[len(positions)]
            positions[name] = (
                robot_base_position + base_to_world_rotation @ candidate_base
            ).astype(np.float32)
            base_positions.append(candidate_base)
        if len(positions) == len(CUBE_PRIM_PATHS):
            if placement_is_clear(
                positions, task, base_to_world_rotation, table_bounds
            ):
                return positions
            positions.clear()
            base_positions.clear()
    raise ManipulationFailure(
        "Could not sample a scene with a clear, on-table placement target"
    )


def placement_is_clear(
    positions, task, base_to_world_rotation, table_bounds=None
) -> bool:
    """Reserve placement clearance around the destination, not just initial cubes."""
    size = float(CUBE_CONFIG["size"])
    margin = float(CUBE_CONFIG.get("placement_clearance", 0.05))
    if task is None:
        return True
    source = f"{task['colors'][0]}"
    excluded = {source}
    if task["kind"] == "stack":
        target = str(task["colors"][1])
        destination = positions[target]
        excluded.add(target)
    else:
        destination = positions[source] + base_to_world_rotation @ np.asarray(
            task["target_offset"]
        )
    if table_bounds is not None:
        lower, upper = table_bounds
        for position in [*positions.values(), destination]:
            if np.any(position[:2] < lower[:2] + size / 2 + margin) or np.any(
                position[:2] > upper[:2] - size / 2 - margin
            ):
                return False
    # The diagonal bound remains conservative for rotated cube footprints.
    return all(
        np.linalg.norm(destination[:2] - position[:2]) >= np.sqrt(2) * size + margin
        for name, position in positions.items()
        if name not in excluded
    )


def _randomize_initial_robot_pose(
    robot: Any,
    world: Any,
    seed: int | None,
    tcp_y_sign: int,
) -> np.ndarray:
    from isaacsim.core.utils.rotations import quat_to_rot_matrix
    from isaacsim.core.utils.types import ArticulationAction

    if tcp_y_sign not in (-1, 1):
        raise ValueError(f"tcp_y_sign must be -1 or 1, got {tcp_y_sign}")

    rng = np.random.default_rng(seed)
    arm_joint_indices = np.asarray(
        [robot.get_dof_index(name) for name in ARM_JOINT_NAMES], dtype=np.int32
    )
    initial_positions = np.asarray(
        robot.get_joint_positions(arm_joint_indices), dtype=np.float64
    )
    half_ranges = np.deg2rad(
        np.asarray(
            ROBOT_CONFIG["arm_initial_position_half_ranges_degrees"], dtype=np.float64
        )
    )
    lower = np.maximum(
        robot.dof_properties["lower"][arm_joint_indices],
        initial_positions - half_ranges,
    )
    upper = np.minimum(
        robot.dof_properties["upper"][arm_joint_indices],
        initial_positions + half_ranges,
    )
    wrist_index = ARM_JOINT_NAMES.index(ROBOT_CONFIG["wrist_3_joint_name"])
    wrist_range = np.deg2rad(ROBOT_CONFIG["wrist_3_initial_rotation_range_degrees"])
    lower[wrist_index] = max(lower[wrist_index], wrist_range[0])
    upper[wrist_index] = min(upper[wrist_index], wrist_range[1])
    if np.any(lower >= upper):
        raise RuntimeError(
            f"Invalid initial arm sampling ranges: lower={lower}, upper={upper}"
        )

    base_position, base_orientation = robot.get_world_pose()
    world_to_base_rotation = quat_to_rot_matrix(base_orientation).T
    maximum_samples = int(ROBOT_CONFIG["maximum_initial_pose_samples"])
    for sample_index in range(1, maximum_samples + 1):
        candidate = rng.uniform(lower, upper).astype(np.float32)
        robot.set_joint_positions(candidate, joint_indices=arm_joint_indices)
        robot.set_joint_velocities(
            np.zeros_like(candidate), joint_indices=arm_joint_indices
        )
        world.render()

        tcp_world_position, _ = robot.end_effector.get_world_pose()
        tcp_base_position = world_to_base_rotation @ (
            tcp_world_position - base_position
        )
        in_positive_xz_workspace = tcp_base_position[0] > float(
            ROBOT_CONFIG["minimum_base_frame_tcp_x"]
        ) and tcp_base_position[2] > float(ROBOT_CONFIG["minimum_base_frame_tcp_z"])
        has_target_y_sign = tcp_y_sign * tcp_base_position[1] >= float(
            ROBOT_CONFIG["minimum_initial_tcp_y_magnitude"]
        )
        if in_positive_xz_workspace and has_target_y_sign:
            robot.get_articulation_controller().apply_action(
                ArticulationAction(
                    joint_positions=candidate, joint_indices=arm_joint_indices
                )
            )
            print(
                f"Randomized initial arm pose after {sample_index} sample(s): "
                f"joints(deg)={np.degrees(candidate)}, tcp_in_base={tcp_base_position}, "
                f"target_y_sign={'positive' if tcp_y_sign > 0 else 'negative'}, "
                f"wrist_3_z_rotation={np.degrees(candidate[wrist_index]):.2f} degrees",
                flush=True,
            )
            return candidate

    raise ManipulationFailure(
        "Failed to sample an initial robot pose with positive base-frame TCP x and z and a "
        f"{'positive' if tcp_y_sign > 0 else 'negative'} base-frame TCP y after {maximum_samples} attempts"
    )


@dataclass
class CubeEnvironment:
    simulation_app: Any
    world: Any
    table: Any
    cubes: dict[str, Any]
    gripper: Any
    robot: Any
    cameras: dict[str, Any]
    table_height: float
    cube_height: float

    @classmethod
    def create(cls, simulation_app: Any) -> "CubeEnvironment":
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        from isaacsim.core.utils.stage import is_stage_loading
        from isaacsim.robot.manipulators import SingleManipulator
        from isaacsim.robot.manipulators.grippers import ParallelGripper
        from isaacsim.sensors.camera import Camera

        usd_path = config_path(PATH_CONFIG["usd"])
        if not usd_path.is_file():
            raise FileNotFoundError(f"USD simulation file not found: {usd_path}")
        if not omni.usd.get_context().open_stage(str(usd_path)):
            raise RuntimeError(f"Failed to open USD simulation: {usd_path}")
        while is_stage_loading():
            simulation_app.update()

        expanded_visuals = expand_gripper_visual_instances(
            omni.usd.get_context().get_stage()
        )
        print(
            f"Expanded {expanded_visuals} gripper visual instances for reliable rendering.",
            flush=True,
        )
        simulation_app.update()

        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / float(SIMULATION_CONFIG["physics_hz"]),
            rendering_dt=1.0 / float(SIMULATION_CONFIG["rendering_hz"]),
        )
        table = world.scene.add(
            FixedCuboid(prim_path=SIMULATION_CONFIG["table_prim_path"], name="tabletop")
        )
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
                    size=float(CUBE_CONFIG["size"]),
                    physics_material=cube_physics_material,
                    mass=float(CUBE_CONFIG["mass"]),
                )
            )
            for color, prim_path in CUBE_PRIM_PATHS.items()
        }

        gripper = ParallelGripper(
            end_effector_prim_path=ROBOT_CONFIG["end_effector_prim_path"],
            joint_prim_names=GRIPPER_CONFIG["joint_prim_names"],
            joint_opened_positions=np.asarray(
                GRIPPER_CONFIG["joint_opened_positions"], dtype=np.float64
            ),
            joint_closed_positions=np.asarray(
                GRIPPER_CONFIG["joint_closed_positions"], dtype=np.float64
            ),
            # No action deltas: the staged controller commands absolute open
            # positions and a single stall-based close, so the gripper can
            # never ratchet open/closed across steps.
            use_mimic_joints=GRIPPER_CONFIG["use_mimic_joints"],
        )
        robot = world.scene.add(
            SingleManipulator(
                prim_path=ROBOT_CONFIG["prim_path"],
                name="ur10e_robot",
                end_effector_prim_path=ROBOT_CONFIG["end_effector_prim_path"],
                gripper=gripper,
            )
        )
        gripper.set_default_state(
            joint_positions=np.asarray(
                GRIPPER_CONFIG["joint_opened_positions"], dtype=np.float64
            )
        )
        world.reset()

        cameras = {
            name: Camera(
                prim_path=prim_path,
                name=f"{name}_debug",
                resolution=(CAMERA_WIDTH, CAMERA_HEIGHT),
            )
            for name, prim_path in CAMERA_CONFIG["prim_paths"].items()
        }
        for camera in cameras.values():
            camera.initialize()

        table_position, _ = table.get_world_pose()
        table_height = (
            table_position[2] + table.get_size() * table.get_world_scale()[2] / 2
        )
        first_cube = next(iter(cubes.values()))
        cube_height = first_cube.get_size() * first_cube.get_world_scale()[2]
        return cls(
            simulation_app=simulation_app,
            world=world,
            table=table,
            cubes=cubes,
            gripper=gripper,
            robot=robot,
            cameras=cameras,
            table_height=table_height,
            cube_height=cube_height,
        )

    def relevant_objects(self) -> dict[str, Any]:
        return {
            "tabletop": self.table,
            **{f"{color}_cube": cube for color, cube in self.cubes.items()},
            "robot": self.robot,
            "end_effector": self.robot.end_effector,
        }

    def print_poses(self, title: str, objects: dict[str, Any] | None = None) -> None:
        print(title, flush=True)
        for name, prim in (objects or self.relevant_objects()).items():
            position, orientation = prim.get_world_pose()
            print(
                f"  {name}: position={position}, orientation(wxyz)={orientation}",
                flush=True,
            )

    def get_camera_rgb(self, name: str) -> np.ndarray:
        rgba = self.cameras[name].get_rgba()
        if rgba is None or rgba.size == 0:
            raise RuntimeError(f"No RGB frame available from {name}")

        rgb = np.asarray(rgba)[..., :3]
        if np.issubdtype(rgb.dtype, np.floating):
            scale = 255.0 if np.max(rgb) <= 1.0 else 1.0
            rgb = np.clip(rgb * scale, 0, 255)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        expected_shape = (CAMERA_HEIGHT, CAMERA_WIDTH, 3)
        if rgb.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected frame shape from {name}: {rgb.shape}, expected {expected_shape}"
            )
        return rgb

    def save_debug_camera_frames(self) -> None:
        from PIL import Image

        output_dir = config_path(PATH_CONFIG["camera_debug_dir"])
        output_dir.mkdir(exist_ok=True)
        for name in self.cameras:
            output_path = output_dir / f"{name}.png"
            Image.fromarray(self.get_camera_rgb(name)).save(output_path)
            print(f"Saved debug camera frame: {output_path}", flush=True)

    def prepare_episode(
        self,
        *,
        seed: int | None,
        tcp_y_sign: int,
        setup_variation: int,
        episode_label: str,
        reset_world: bool,
        task: dict[str, object] | None = None,
        check_stop=None,
    ) -> np.ndarray:
        from isaacsim.core.utils.rotations import quat_to_rot_matrix

        if reset_world:
            self.world.reset()
        if check_stop is not None:
            check_stop()
        self.print_poses(f"{episode_label} initial scene poses:")
        _randomize_initial_robot_pose(self.robot, self.world, seed, tcp_y_sign)

        robot_base_position, robot_base_orientation = self.robot.get_world_pose()
        base_to_world_rotation = quat_to_rot_matrix(robot_base_orientation)
        table_position, table_orientation = self.table.get_world_pose()
        half_extents = np.abs(quat_to_rot_matrix(table_orientation)) @ (
            self.table.get_size() * self.table.get_world_scale() / 2
        )
        randomized_positions = _random_cube_positions(
            self.table_height,
            self.cube_height,
            seed,
            robot_base_position,
            base_to_world_rotation,
            setup_variation,
            task=task,
            table_bounds=(table_position - half_extents, table_position + half_extents),
        )
        identity_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        for name, position in randomized_positions.items():
            cube = self.cubes[name]
            cube.set_world_pose(position=position, orientation=identity_orientation)
            cube.set_linear_velocity(np.zeros(3))
            cube.set_angular_velocity(np.zeros(3))
            cube_in_base = base_to_world_rotation.T @ (position - robot_base_position)
            print(
                f"{name}_cube initial base-frame position: {cube_in_base}", flush=True
            )

        for _ in range(int(SIMULATION_CONFIG["settling_steps_before_recording"])):
            if check_stop is not None:
                check_stop()
            self.world.step(render=True)
        settled_positions = {
            name: cube.get_world_pose()[0] for name, cube in self.cubes.items()
        }
        if not placement_is_clear(
            settled_positions,
            task,
            base_to_world_rotation,
            (table_position - half_extents, table_position + half_extents),
        ):
            raise ManipulationFailure(
                "Settled scene no longer has a clear placement target"
            )
        self.save_debug_camera_frames()
        self.print_poses(f"{episode_label} randomized scene poses:")
        return base_to_world_rotation
