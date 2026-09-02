import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml
from isaaclab.app import AppLauncher

from src.usd_utils import expand_gripper_visual_instances

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "pi05_config.yaml"
with CONFIG_PATH.open(encoding="utf-8") as config_file:
    PI05_CONFIG = yaml.safe_load(config_file)

_ros_node = None
_ros_subscriptions = []
_latest_images: dict[str, np.ndarray] = {}
_latest_joint_positions: dict[str, float] = {}
_simulation_app = None
_robot = None
_arm_joint_indices: np.ndarray | None = None
_gripper_joint_indices: np.ndarray | None = None
_arm_max_velocities: np.ndarray | None = None
_gripper_limits: tuple[float, float] | None = None


def _image_to_numpy(message) -> np.ndarray:
    channels_by_encoding = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}
    if message.encoding not in channels_by_encoding:
        raise ValueError(f"Unsupported ROS image encoding: {message.encoding}")

    channels = channels_by_encoding[message.encoding]
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    image = rows[:, : message.width * channels].reshape(message.height, message.width, channels)
    if message.encoding in {"bgr8", "bgra8"}:
        image = image[..., 2::-1]
    elif message.encoding == "rgba8":
        image = image[..., :3]
    elif message.encoding == "mono8":
        image = np.repeat(image, 3, axis=2)
    return image.copy()


def _ensure_ros2_subscribers() -> None:
    global _ros_node, _ros_subscriptions
    if _ros_node is not None:
        return

    import rclpy
    from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import Image, JointState

    if not rclpy.ok():
        rclpy.init(args=None)
    _ros_node = rclpy.create_node("ur10e_pi05_observation_reader")
    qos = QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
    )

    def image_callback(name: str):
        def callback(message: Image) -> None:
            _latest_images[name] = _image_to_numpy(message)

        return callback

    def joint_state_callback(message: JointState) -> None:
        _latest_joint_positions.update(zip(message.name, message.position, strict=False))

    _ros_subscriptions = [
        _ros_node.create_subscription(Image, camera["topic"], image_callback(name), qos)
        for name, camera in PI05_CONFIG["cameras"].items()
    ]
    _ros_subscriptions.append(
        _ros_node.create_subscription(JointState, PI05_CONFIG["joint_state_topic"], joint_state_callback, qos)
    )


def _spin_ros2() -> None:
    import rclpy

    _ensure_ros2_subscribers()
    for _ in _ros_subscriptions:
        rclpy.spin_once(_ros_node, timeout_sec=0.0)


def get_images() -> tuple[np.ndarray, ...]:
    """Return the latest side and wrist RGB frames in configured order."""
    _spin_ros2()
    camera_names = PI05_CONFIG["cameras"]
    if any(name not in _latest_images for name in camera_names):
        return ()
    return tuple(_latest_images[name].copy() for name in camera_names)


def get_proprioception() -> np.ndarray:
    """Return the latest arm and gripper positions in pi0.5-DROID state order."""
    _spin_ros2()
    if _gripper_limits is None:
        return np.empty(0, dtype=np.float32)
    arm_joint_names = PI05_CONFIG["arm_joint_names"]
    gripper_joint_name = PI05_CONFIG["gripper_joint_name"]
    required_names = [*arm_joint_names, gripper_joint_name]
    if any(name not in _latest_joint_positions for name in required_names):
        return np.empty(0, dtype=np.float32)

    arm_positions = np.array([_latest_joint_positions[name] for name in arm_joint_names], dtype=np.float32)
    # DROID has seven arm joints; the UR10e's nonexistent seventh joint is represented as zero.
    arm_positions = np.pad(arm_positions, (0, 7 - len(arm_positions)))
    gripper_lower, gripper_upper = _gripper_limits
    gripper_position = np.clip(
        (_latest_joint_positions[gripper_joint_name] - gripper_lower) / (gripper_upper - gripper_lower), 0.0, 1.0
    )
    return np.append(arm_positions, np.float32(gripper_position))


def _initialize_robot_controller() -> None:
    global _robot, _arm_joint_indices, _gripper_joint_indices, _arm_max_velocities, _gripper_limits

    from isaacsim.core.prims import SingleArticulation

    _robot = SingleArticulation(PI05_CONFIG["robot_prim_path"])
    _robot.initialize()
    _arm_joint_indices = np.asarray(
        [_robot.get_dof_index(name) for name in PI05_CONFIG["arm_joint_names"]], dtype=np.int32
    )
    _gripper_joint_indices = np.asarray(
        [_robot.get_dof_index(PI05_CONFIG["gripper_joint_name"])], dtype=np.int32
    )

    controller = _robot.get_articulation_controller()
    for joint_index in _arm_joint_indices:
        controller.switch_dof_control_mode(int(joint_index), "velocity")
    controller.switch_dof_control_mode(int(_gripper_joint_indices[0]), "position")

    properties = _robot.dof_properties
    _arm_max_velocities = np.asarray(properties["maxVelocity"][_arm_joint_indices], dtype=np.float32)
    gripper_index = _gripper_joint_indices[0]
    _gripper_limits = (float(properties["lower"][gripper_index]), float(properties["upper"][gripper_index]))


def step_simulation(action_chunk: np.ndarray, chunk_samples: int = 1) -> int:
    """Execute the requested prefix of a pi0.5-DROID action chunk."""
    from isaacsim.core.utils.types import ArticulationAction

    if action_chunk.ndim != 2 or action_chunk.shape[1] != 8:
        raise ValueError(f"Expected a VLA action chunk shaped (steps, 8), got {action_chunk.shape}")
    if chunk_samples < 1:
        raise ValueError(f"chunk_samples must be at least 1, got {chunk_samples}")
    if any(
        value is None
        for value in (
            _simulation_app,
            _robot,
            _arm_joint_indices,
            _gripper_joint_indices,
            _arm_max_velocities,
            _gripper_limits,
        )
    ):
        raise RuntimeError("Robot controller is not initialized; call step_simulation from launch_simulation")

    sample_count = min(chunk_samples, len(action_chunk))
    gripper_lower, gripper_upper = _gripper_limits
    for action in action_chunk[:sample_count]:
        # DROID has seven arm commands; the UR10e uses the first six and drops the synthetic seventh DOF.
        arm_velocities = np.clip(action[:6], -_arm_max_velocities, _arm_max_velocities)
        gripper_position = gripper_lower + np.clip(action[7], 0.0, 1.0) * (gripper_upper - gripper_lower)
        _robot.apply_action(
            ArticulationAction(joint_velocities=arm_velocities, joint_indices=_arm_joint_indices)
        )
        _robot.apply_action(
            ArticulationAction(
                joint_positions=np.asarray([gripper_position], dtype=np.float32),
                joint_indices=_gripper_joint_indices,
            )
        )
        for _ in range(PI05_CONFIG["simulation_steps_per_action"]):
            _simulation_app.update()

    _robot.apply_action(
        ArticulationAction(
            joint_velocities=np.zeros(len(_arm_joint_indices), dtype=np.float32),
            joint_indices=_arm_joint_indices,
        )
    )
    _simulation_app.update()
    return sample_count


def _create_joint_state_publisher() -> None:
    import omni.graph.core as og
    import omni.usd
    import usdrt.Sdf

    graph_path = PI05_CONFIG["joint_state_graph_path"]
    if omni.usd.get_context().get_stage().GetPrimAtPath(graph_path).IsValid():
        return

    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
            ],
            keys.SET_VALUES: [
                ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path(PI05_CONFIG["robot_prim_path"])]),
                ("PublishJointState.inputs:topicName", PI05_CONFIG["joint_state_topic"]),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                ("Context.outputs:context", "PublishJointState.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
            ],
        },
    )

def _shutdown_ros2() -> None:
    global _ros_node, _ros_subscriptions
    if _ros_node is None:
        return

    import rclpy

    _ros_node.destroy_node()
    _ros_node = None
    _ros_subscriptions = []
    if rclpy.ok():
        rclpy.shutdown()


def _resolve_usd_path(value: str) -> Path:
    usd_path = Path(value).expanduser()
    if usd_path.parent == Path("."):
        usd_path = Path(__file__).resolve().parent.parent / usd_path

    usd_path = usd_path.resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(f"USD simulation file not found: {usd_path}")
    return usd_path


def launch_simulation(on_step: Callable[[], bool] | None = None) -> None:
    global _simulation_app

    parser = argparse.ArgumentParser(description="Launch an Isaac Sim USD simulation.")
    parser.add_argument(
        "-s",
        "--simulation",
        default="ur10e_with_table.usd",
        help="USD file name or path.",
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(experience="/home/rahmlab/isaacsim/apps/isaacsim.exp.base.python.kit")
    args = parser.parse_args()
    usd_path = _resolve_usd_path(args.simulation)

    simulation_app = AppLauncher(args).app
    _simulation_app = simulation_app
    try:
        import omni.timeline
        import omni.usd
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.core.utils.stage import is_stage_loading

        enable_extension("isaacsim.ros2.bridge")
        simulation_app.update()

        if not omni.usd.get_context().open_stage(str(usd_path)):
            raise RuntimeError(f"Failed to open USD simulation: {usd_path}")
        while is_stage_loading():
            simulation_app.update()

        stage = omni.usd.get_context().get_stage()
        expanded_visuals = expand_gripper_visual_instances(stage)
        print(f"Expanded {expanded_visuals} gripper visual instances for reliable rendering.", flush=True)
        simulation_app.update()
        for camera in PI05_CONFIG["cameras"].values():
            if not stage.GetPrimAtPath(camera["prim_path"]).IsValid():
                raise RuntimeError(f"Camera prim not found: {camera['prim_path']}")
        _create_joint_state_publisher()

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        simulation_app.update()
        _initialize_robot_controller()
        stop_sim = False
        while simulation_app.is_running() and not stop_sim:
            simulation_app.update()
            if on_step is not None:
                stop_sim = on_step()
    finally:
        _shutdown_ros2()
        simulation_app.close()
        _simulation_app = None
