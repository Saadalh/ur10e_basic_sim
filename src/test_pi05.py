import os
from pathlib import Path
from time import perf_counter

from PIL import Image

from src.cube_environment import CAMERA_CONFIG
from src.gripper_debug import GripperDebugRecorder
from src.sim import (
    PI05_CONFIG,
    TRAIN_CONFIG,
    get_images,
    get_launch_args,
    get_proprioception,
    get_task_prompt,
    launch_simulation,
    step_simulation,
)


def run() -> None:
    # Load JAX before Isaac Sim starts native worker threads; late loading can break glibc TLS setup.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from src.init_pi05 import init_pi05

    model = init_pi05()
    debug_dir = Path(__file__).resolve().parent.parent / "camera_debug"
    debug_dir.mkdir(exist_ok=True)
    steps = PI05_CONFIG["steps"]
    chunk_samples = PI05_CONFIG["chunk_samples"]
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    if chunk_samples < 1:
        raise ValueError(f"chunk_samples must be at least 1, got {chunk_samples}")
    executed_steps = 0
    gripper_recorder = None

    def finalize_gripper_debug() -> None:
        # Writes files while the app is still alive when called at episode
        # end; the outer finally reuses it as a backup. finalize() itself
        # is idempotent, so calling both paths is safe.
        if gripper_recorder is None:
            return
        if len(gripper_recorder) == 0:
            print("Gripper debug recording is empty; no plot written.", flush=True)
            return
        args = get_launch_args()
        out_dir = Path(__file__).resolve().parent.parent / "gripper_debug"
        png_path, csv_path = gripper_recorder.finalize(
            task_index=args.task_index,
            episode_index=args.episode_index,
            seed=args.seed,
            out_dir=out_dir,
        )
        print(f"Saved gripper debug plot: {png_path}", flush=True)
        print(f"Saved gripper debug CSV: {csv_path}", flush=True)

    def capture_observations() -> bool:
        nonlocal executed_steps, gripper_recorder
        images = get_images()
        proprioception = get_proprioception()
        if not images or proprioception.size == 0:
            return False

        for camera_name, image in zip(CAMERA_CONFIG["prim_paths"], images, strict=True):
            Image.fromarray(image).save(debug_dir / f"{camera_name}.png")

        # pi05_droid expects one unbatched observation with this exact raw format:
        # {
        #   "observation/exterior_image_1_left": uint8 RGB array shaped (H, W, 3),
        #   "observation/wrist_image_left": uint8 RGB array shaped (H, W, 3),
        #   "observation/joint_position": float32 array shaped (7,), in radians,
        #   "observation/gripper_position": float32 array shaped (1,), in [0, 1],
        #   "prompt": str,
        # }
        # The policy preprocessing resizes both images to (224, 224, 3).
        observation = {
            "observation/exterior_image_1_left": images[0],
            "observation/wrist_image_left": images[1],
            "observation/joint_position": proprioception[:7],
            "observation/gripper_position": proprioception[7:],
            "prompt": get_task_prompt(),
        }
        # pi05_droid returns a float array shaped (15, 8): 15 future control steps,
        # with six DROID joint-motion commands in columns 0:6, a synthetic
        # seventh joint in column 6, and an absolute normalized gripper
        # command in column 7. step_simulation decodes the arm commands to
        # absolute joint-position targets (q + 0.2 * a).
        inference_start = perf_counter()
        actions = model.infer(observation)
        inference_ms = (perf_counter() - inference_start) * 1000
        executed_steps += 1
        print(f"Model exec {executed_steps}: {inference_ms:.0f}ms", flush=True)
        print(f"VLA actions ({actions.shape}):\n{actions}", flush=True)
        # Lazily created on the first observation: launch_simulation() has
        # parsed the CLI by the time this callback runs.
        if gripper_recorder is None:
            launch_args = get_launch_args()
            if launch_args is not None and launch_args.plot_gripper:
                gripper_recorder = GripperDebugRecorder(
                    opened=float(TRAIN_CONFIG["gripper-open-position"]),
                    closed=float(TRAIN_CONFIG["gripper-closed-position"]),
                )
        if gripper_recorder is not None:
            inference_id = gripper_recorder.new_inference()
        else:
            inference_id = None
        step_simulation(
            actions,
            chunk_samples,
            gripper_recorder=gripper_recorder,
            inference_id=inference_id,
        )
        print(f"Executed {executed_steps}/{steps} action steps", flush=True)
        finished = executed_steps >= steps
        if finished:
            # Finalize before returning: launch_simulation() shuts the app
            # down next, and a stuck teardown must not eat the recording.
            finalize_gripper_debug()
        return finished

    try:
        launch_simulation(capture_observations)
    finally:
        finalize_gripper_debug()
