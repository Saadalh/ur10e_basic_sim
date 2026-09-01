from pathlib import Path
from time import perf_counter

from PIL import Image

from src.init_pi05 import init_pi05
from src.sim import PI05_CONFIG, get_images, get_proprioception, launch_simulation, step_simulation


def run() -> None:
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

    def capture_observations() -> bool:
        nonlocal executed_steps
        images = get_images()
        proprioception = get_proprioception()
        if not images or proprioception.size == 0:
            return False

        for camera_name, image in zip(PI05_CONFIG["cameras"], images, strict=True):
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
            "prompt": PI05_CONFIG["default_prompt"],
        }
        # pi05_droid returns a float array shaped (15, 8): 15 future control steps,
        # with seven joint-velocity commands in columns 0:7 and a gripper-position
        # command in column 7. These are policy actions, not absolute joint positions;
        # the gripper command should be clipped/thresholded as required by the controller.
        inference_start = perf_counter()
        actions = model.infer(observation)
        inference_ms = (perf_counter() - inference_start) * 1000
        executed_steps += 1
        print(f"Model exec {executed_steps}: {inference_ms:.0f}ms", flush=True)
        print(f"VLA actions ({actions.shape}):\n{actions}", flush=True)
        step_simulation(actions, chunk_samples)
        print(f"Executed {executed_steps}/{steps} action steps", flush=True)
        return executed_steps >= steps

    launch_simulation(capture_observations)
