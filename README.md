# UR10e Cube Manipulation Simulation

Tools for generating language-conditioned UR10e and Robotiq 2F-85 demonstrations in Isaac Sim, running pi0.5-DROID in the simulation, and fine-tuning pi0.5 from an existing LeRobot dataset.

## Contents

- `ur10e_with_table.usd`: Isaac Sim scene.
- `main.py`: simulation and simulated-policy inference entry point.
- `src/collect_cube_data.py`: configurable LeRobot v3 dataset collector.
- `src/train_model.py`: LeRobot v3 to DROID adapter and OpenPI training launcher.
- `config/ds_collect_config.json`: robot, camera, task, and collection settings.
- `config/pi05_config.yaml`: pi0.5 inference settings.
- `config/train_config.yaml`: commented pi0.5 fine-tuning settings.

## Choose A Use Case

| Use case | Required | Not required |
| --- | --- | --- |
| Fine-tune pi0.5 from an existing dataset | OpenPI, pi0.5 checkpoint, LeRobot dataset, NVIDIA GPU | Isaac Sim, Isaac Lab, ROS 2 |
| Validate the training dataset adapter | Python, OpenCV, PyArrow, PyYAML, LeRobot dataset | Isaac Sim, Isaac Lab, ROS 2, pi0.5 weights |
| Generate a new simulated dataset | Isaac Sim, Isaac Lab, LeRobot v3, NVIDIA GPU | OpenPI and pi0.5 |
| Launch or inspect the simulation | Isaac Sim, Isaac Lab, NVIDIA GPU | OpenPI and pi0.5 |
| Run pi0.5 inference inside the simulation | Isaac Sim, Isaac Lab, Isaac Sim ROS 2 bridge, OpenPI, pi0.5-DROID checkpoint, NVIDIA GPU | Training dataset |

Use the installation documentation linked below rather than assuming that commands or version numbers from another environment are compatible.

## Existing Dataset

An existing UR10e dataset generated with this Isaac Sim project is available from KIT bwSync&Share:

**[Download the UR10e_Basic Isaac Sim dataset](https://bwsyncandshare.kit.edu/s/Bj8tZGxsNTBaqNY)**

Download and extract it so the repository has the following layout:

```text
ur10e_basic_sim/
  data/
    meta/
      info.json
    data/
    videos/
```

The downloaded dataset can be used directly by `src/train_model.py`. You do not need to install Isaac Sim or Isaac Lab when fine-tuning exclusively from this existing dataset.

## Official Installation References

### OpenPI And pi0.5

Required for fine-tuning and pi0.5 inference.

- [OpenPI repository, requirements, and installation](https://github.com/Physical-Intelligence/openpi)
- [OpenPI model checkpoints](https://github.com/Physical-Intelligence/openpi#model-checkpoints)
- [pi0.5 model overview](https://www.physicalintelligence.company/blog/pi05)
- [`uv` installation](https://docs.astral.sh/uv/getting-started/installation/), used by OpenPI
- [NVIDIA driver downloads](https://www.nvidia.com/en-us/drivers/)

Clone OpenPI with its submodules and follow its current installation guide. This project uses the `pi05_droid` configuration and the checkpoint at `gs://openpi-assets/checkpoints/pi05_droid`. OpenPI downloads released checkpoints on first use and caches them under `~/.cache/openpi`; set `OPENPI_DATA_HOME` to choose another cache directory.

OpenPI currently documents these approximate GPU memory requirements:

| Mode | GPU memory |
| --- | --- |
| Inference | More than 8 GB |
| LoRA fine-tuning | More than 22.5 GB |
| Full fine-tuning | More than 70 GB |

### Isaac Sim And Isaac Lab

Required only for launching the simulation, generating datasets, or evaluating a policy in the simulation.

- [Isaac Sim installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/index.html)
- [Isaac Sim system requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
- [Isaac Lab installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
- [Isaac Lab installation using Isaac Sim pip packages](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)

Install mutually compatible Isaac Sim and Isaac Lab releases by following the Isaac Lab documentation. Run simulation and collection commands from the resulting Isaac Lab Python environment.

### ROS 2

Required only for pi0.5 inference inside Isaac Sim because `src/sim.py` receives camera images and joint states through ROS 2.

- [Isaac Sim ROS 2 installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_ros.html)
- [Isaac Sim ROS 2 tutorials](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/index.html)
- [ROS 2 installation documentation](https://docs.ros.org/en/jazzy/Installation.html)

Use the ROS 2 version supported by the installed Isaac Sim release.

### LeRobot And Dataset Utilities

LeRobot v3 is required to create new datasets. The training adapter also imports OpenCV, PyArrow, and PyYAML; these should be installed in the Python environment used to run it.

- [LeRobot installation](https://huggingface.co/docs/lerobot/installation)
- [OpenCV Python packages](https://pypi.org/project/opencv-python/)
- [PyArrow installation](https://arrow.apache.org/docs/python/install.html)
- [PyYAML package](https://pypi.org/project/PyYAML/)

OpenPI manages most training dependencies in its own environment. Follow OpenPI's dependency versions when packages overlap rather than mixing the Isaac Sim and OpenPI environments.

## Configure This Project

Clone the repository and run commands from its root directory:

```bash
git clone https://github.com/Saadalh/ur10e_basic_sim.git
cd ur10e_basic_sim
```

Update only the configuration relevant to the selected workflow:

- Fine-tuning: set `openpi-root` in `config/train_config.yaml` and review its dataset, output, memory, and optimization settings.
- Simulated inference: set `openpi_root` in `config/pi05_config.yaml` and verify the checkpoint and ROS 2 topics.
- Dataset generation: set `paths.lerobot_v3`, `paths.ur10e_controller`, and `paths.isaac_experience` in `config/ds_collect_config.json`.

Dataset, camera capture, checkpoint, and asset paths are relative to the repository root by default.

## Fine-Tune From An Existing Dataset

Isaac Sim, Isaac Lab, and ROS 2 are not used by the training launcher.

Validate the downloaded dataset without allocating the pi0.5 model:

```bash
/path/to/openpi/.venv/bin/python -m src.train_model --validate-only
```

Start fine-tuning from the OpenPI environment on a capable GPU:

```bash
/path/to/openpi/.venv/bin/python -m src.train_model
```

Set `ignore-episodes` in `config/train_config.yaml` to integer episode indexes that should not be used. An empty list keeps every episode.

## Generate A Dataset In Isaac Sim

Run this command from the Isaac Lab environment:

```bash
python -m src.collect_cube_data --episodes 10 --seed 42
```

Without `--episodes`, the collector records the complete weighted task schedule. Generated datasets are written to `data/`, one file per episode by default, and are intentionally excluded from Git.

## Run The Simulation

Run these commands from the Isaac Lab environment.

Launch the scene without pi0.5:

```bash
python main.py
```

Run pi0.5-DROID inference in the simulation:

```bash
python main.py --test
```

## Local Artifacts

The repository excludes virtual environments, generated datasets, camera captures, model assets, checkpoints, logs, and W&B outputs. Keep large datasets and trained weights in external storage rather than committing them to Git.
