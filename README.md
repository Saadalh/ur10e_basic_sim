# UR10e Cube Manipulation Simulation

Isaac Sim/Isaac Lab simulation for collecting language-conditioned UR10e and Robotiq 2F-85 cube-manipulation demonstrations, running pi0.5-DROID inference, and launching pi0.5 fine-tuning through OpenPI.

## Contents

- `ur10e_with_table.usd`: simulation scene.
- `main.py` and `src/`: simulation, collection, inference, and training entry points.
- `src/collect_cube_data.py`: configurable LeRobot v3 dataset collector.
- `src/train_model.py`: LeRobot v3 to DROID adapter and OpenPI training launcher.
- `config/ds_collect_config.json`: robot, camera, task, and collection settings.
- `config/pi05_config.yaml`: pi0.5 inference settings.
- `config/train_config.yaml`: commented pi0.5 fine-tuning settings.

## Requirements

- Ubuntu 22.04 with a recent NVIDIA production driver.
- Python 3.11 for Isaac Sim 5.x and the current OpenPI environment.
- At least 16 GB of GPU VRAM for the simulation workflow.
- More than 8 GB of GPU VRAM for pi0.5 inference.
- At least 22.5 GB of GPU VRAM for LoRA fine-tuning or 70 GB for full fine-tuning.

Check the current [Isaac Sim system requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html) and [OpenPI requirements](https://github.com/Physical-Intelligence/openpi#requirements) before installation.

## Install Isaac Sim And Isaac Lab

This project imports `AppLauncher` and other APIs from Isaac Lab, so install both Isaac Sim and Isaac Lab. The commands below follow NVIDIA's recommended Isaac Sim 5.1 pip workflow on Linux. Keep the Isaac Sim and Isaac Lab versions compatible and consult the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) if a newer release is available.

Create and activate a Python 3.11 environment:

```bash
python3.11 -m venv env_isaaclab
source env_isaaclab/bin/activate
python -m pip install --upgrade pip
```

Install Isaac Sim and a compatible CUDA-enabled PyTorch build:

```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

Verify Isaac Sim. The first launch downloads extensions and asks you to accept NVIDIA's license:

```bash
isaacsim
```

Install Isaac Lab from source into the same environment:

```bash
sudo apt install cmake build-essential
git clone https://github.com/isaac-sim/IsaacLab.git --branch main
cd IsaacLab
./isaaclab.sh --install none
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

The final command should open an empty Isaac Sim viewport. Configure the [Isaac Sim ROS 2 bridge](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_installation.html) before running pi0.5 inference, because this project receives camera and joint-state observations through ROS 2.

## Install OpenPI And pi0.5

OpenPI contains the pi0.5 model implementation and downloads released model weights. Install `uv` by following its [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/), then clone OpenPI with submodules:

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

This project uses the released pi0.5-DROID checkpoint at `gs://openpi-assets/checkpoints/pi05_droid`. OpenPI downloads it automatically on first use and caches it under `~/.cache/openpi`. To download it in advance, run this from the OpenPI directory:

```bash
uv run python -c 'from openpi.shared import download; print(download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid"))'
```

Set `OPENPI_DATA_HOME` if the model cache should live somewhere other than `~/.cache/openpi`.

## Configure This Project

Clone this repository and run commands from its root directory:

```bash
git clone https://github.com/Saadalh/ur10e_basic_sim.git
cd ur10e_basic_sim
```

Update machine-specific paths before running:

- `config/pi05_config.yaml`: set `openpi_root` to the OpenPI checkout.
- `config/train_config.yaml`: set `openpi-root` to the OpenPI checkout.
- `config/ds_collect_config.json`: set `paths.ur10e_controller` and `paths.isaac_experience` for the installed Isaac Sim version.

The collector also expects a LeRobot v3 source tree at the configured `paths.lerobot_v3` location. Dataset, camera capture, checkpoint, and asset paths are relative to the project root by default.

## Simulation And Inference

Run these commands from the Isaac Lab environment.

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
python -m src.collect_cube_data --episodes 10 --seed 42
```

Generated datasets are written to `data/`, one file per episode by default, and are intentionally excluded from Git.

## Training

Validate the dataset adapter without allocating the pi0.5 model:

```bash
python -m src.train_model --validate-only
```

Start fine-tuning from the OpenPI environment on a capable GPU:

```bash
/path/to/openpi/.venv/bin/python -m src.train_model
```

Set `ignore-episodes` in `config/train_config.yaml` to a list of integer episode indexes that should not be used for training. An empty list keeps every episode.

## Local Artifacts

The repository excludes virtual environments, generated datasets, camera captures, model assets, checkpoints, logs, and W&B outputs. Keep large datasets and trained weights in external storage rather than committing them to Git.
