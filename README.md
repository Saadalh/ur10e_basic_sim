# UR10e Cube Manipulation Simulation

Isaac Sim/Isaac Lab simulation for collecting language-conditioned UR10e and Robotiq 2F-85 cube-manipulation demonstrations, running pi0.5-DROID inference, and launching pi0.5 fine-tuning through OpenPI.

## Contents

- `ur10e_with_table.usd`: simulation scene.
- `main.py` and `src/`: simulation and pi0.5 inference entry points.
- `collect_cube_data.py`: configurable LeRobot v3 dataset collector.
- `ds_collect_config.json`: robot, camera, task, and collection settings.
- `train_model.py`: LeRobot v3 to DROID adapter and OpenPI training launcher.
- `train_config.yaml`: commented pi0.5 fine-tuning configuration.

## Prerequisites

- NVIDIA Isaac Sim and Isaac Lab.
- ROS 2 support configured for Isaac Sim.
- A local [OpenPI](https://github.com/Physical-Intelligence/openpi) checkout for inference and training.
- LeRobot v3 and the Python packages imported by the collection and training scripts.

Update the machine-specific OpenPI and Isaac Sim paths in `configs/pi05_config.yaml`, `ds_collect_config.json`, and `train_config.yaml` before running the project on another system.

## Simulation And Inference

Launch the simulation:

```bash
python main.py
```

Run pi0.5-DROID inference in the simulation:

```bash
python main.py --test
```

## Dataset Collection

The collector expands configured color templates into concrete task instructions. Without `--episodes`, it records the complete weighted task schedule. With `--episodes`, it randomly selects the requested number of configured tasks.

```bash
python collect_cube_data.py --episodes 10 --seed 42
```

Generated datasets are written to `data/`, one file per episode by default, and are intentionally excluded from Git.

## Training

Validate the dataset adapter without allocating the pi0.5 model:

```bash
python train_model.py --validate-only
```

Start fine-tuning from the OpenPI environment on a capable GPU:

```bash
/path/to/openpi/.venv/bin/python train_model.py
```

The default configuration uses LoRA and requires at least 22.5 GiB of VRAM per GPU. Full fine-tuning requires approximately 70 GiB. Set `ignore-episodes` in `train_config.yaml` to a list of integer episode indexes that should not be used for training.

## Local Artifacts

The repository excludes virtual environments, generated datasets, camera captures, model assets, checkpoints, logs, and W&B outputs. Keep large datasets and trained weights in external storage rather than committing them to Git.
