# UR10e Cube Manipulation Simulation

Tools for generating language-conditioned UR10e and Robotiq 2F-85 demonstrations in Isaac Sim, running pi0.5-DROID in the simulation, and fine-tuning pi0.5 from an existing LeRobot dataset.

## Contents

- `ur10e_with_table.usd`: Isaac Sim scene.
- `main.py`: simulation and simulated-policy inference entry point.
- `src/collect_cube_data.py`: configurable LeRobot v3 dataset collector.
- `src/dataset_stats.py`: OpenPI-based raw and projected dataset statistics generator.
- `src/train_model.py`: LeRobot v3 to DROID adapter and OpenPI training launcher.
- `config/ds_collect_config.json`: robot, camera, task, and collection settings.
- `config/pi05_config.yaml`: pi0.5 inference settings.
- `config/train_config.yaml`: commented pi0.5 fine-tuning settings.

## Choose A Use Case

| Use case | Required | Not required |
| --- | --- | --- |
| Fine-tune pi0.5 from an existing dataset | OpenPI, pi0.5 checkpoint, LeRobot dataset, NVIDIA GPU | Isaac Sim, Isaac Lab, ROS 2 |
| Validate the training dataset adapter | Python, OpenCV, PyArrow, PyYAML, LeRobot dataset | Isaac Sim, Isaac Lab, ROS 2, pi0.5 weights |
| Generate a new simulated dataset | Isaac Sim, Isaac Lab, LeRobot v3, OpenPI, NVIDIA GPU | pi0.5 weights |
| Launch or inspect the simulation | Isaac Sim, Isaac Lab, NVIDIA GPU | OpenPI and pi0.5 |
| Run pi0.5 inference inside the simulation | Isaac Sim, Isaac Lab, OpenPI, pi0.5-DROID checkpoint, NVIDIA GPU | Training dataset, ROS 2 |

Use the installation documentation linked below rather than assuming that commands or version numbers from another environment are compatible.

## Existing Dataset

An existing UR10e dataset generated with this Isaac Sim project is available from KIT bwSync&Share:

**[Download the UR10e_Basic Isaac Sim dataset](https://bwsyncandshare.kit.edu/s/Bj8tZGxsNTBaqNY)**

Download and extract it so the repository has the following layout:

```text
ur10e_basic_sim/
  dataset/
    meta/
      info.json
      stats.json
    data/
    videos/
```

The downloaded dataset can be used by `src/train_model.py`. Regenerate `meta/stats.json` with `src.dataset_stats` if it does not contain the `openpi.state` and `openpi.actions` entries described below, or train with `--use-base-norm-stats`. You do not need to install Isaac Sim or Isaac Lab when fine-tuning exclusively from this existing dataset.

## Official Installation References

### OpenPI And pi0.5

Required for dataset statistics generation, fine-tuning, and pi0.5 inference.

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
- Simulated inference: set `openpi_root` and `checkpoint` in `config/pi05_config.yaml`.
- Dataset generation: set `paths.lerobot_v3`, `paths.openpi`, `paths.ur10e_controller`, and `paths.isaac_experience` in `config/ds_collect_config.json`.

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

By default, the launcher reads the exact 8D `openpi.state` and `openpi.actions` projections from the dataset's `meta/stats.json`. It writes those values to `assets/pi05_ur10e_lora/droid/norm_stats.json` and embeds them in each saved checkpoint as `assets/droid/norm_stats.json`.

To retain the released pi0.5-DROID checkpoint statistics instead, use:

```bash
/path/to/openpi/.venv/bin/python -m src.train_model --use-base-norm-stats
```

Set `ignore-episodes` in `config/train_config.yaml` to integer episode indexes that should not be used. An empty list keeps every episode. Dataset statistics describe the complete collected dataset; ignored episodes are excluded from training but not from the collection-time statistics. When `resume` is enabled, the launcher preserves the normalization statistics embedded in the newest checkpoint. `--validate-only` does not load normalization statistics.

## Generate A Dataset In Isaac Sim

Run this command from the Isaac Lab environment:

```bash
python -m src.collect_cube_data --episodes 10 --seed 42
```

Without `--episodes`, the collector records the complete weighted task schedule. Generated datasets are written to `dataset/`, one file per episode by default, and are intentionally excluded from Git. After finalizing the dataset, the collector uses OpenPI's `RunningStats` implementation to replace LeRobot's provisional `meta/stats.json`. The final file contains all 12 raw state dimensions, all 72 raw action dimensions, and exact 8D `openpi.state` and `openpi.actions` projections.

Collection is a long unattended run: keep the workstation awake for its whole
duration. Background execution does not prevent suspend, and a suspended run
loses GPU state, so wrap the command in a sleep inhibitor:

```bash
set -o pipefail
systemd-inhibit --what=sleep:idle --mode=block \
  --why="UR10e dataset collection" \
  /home/rahmlab/projects/IsaacLab/isaaclab.sh -p \
  -m src.collect_cube_data --seed 42 --headless --enable_cameras \
  2>&1 | tee /tmp/collect_full.log
```

Scene randomization is task-aware: a sampled layout is rejected unless the
placement destination keeps `placement_clearance` beyond the diagonal cube
footprint from every non-participating cube and stays on the table. Rejected
scenes and failed demonstrations raise a retryable error instead of aborting
the run; each scheduled task is retried up to
`controller.maximum_episode_attempts` times with a fresh seed, then skipped
with the reason recorded. Programming, storage, and shutdown errors still
abort immediately.

Grasping is staged: the controller drives to a pre-grasp waypoint
(`controller.pre_grasp_height`, default 0.20 m above the cube), descends with
x/y locked in `controller.descent_step` increments, then closes incrementally
until the finger stalls on the cube and holds that exact position — one
close, no relative-delta ratcheting (`action_deltas` was removed for this
reason). Grasp tuning lives under `gripper` (`grasp_increments`,
`grasp_position_eps`, `grasp_steady_steps`, `grasp_min_travel_fraction`,
`max_grasp_steps`); values are in finger-joint radians matching the training
calibration (`gripper-open-position` 0.0, `gripper-closed-position` 0.376), and
a unit test fails if collection calibration ever drifts from it. Transport
reuses the Cartesian motion with the stall pose merged in, so a held cube is
never commanded open mid-transport; release still opens once at the place pose.

Simulation, recording, and policy execution share a 15 Hz control rate
(150 Hz physics, one action per render step), matching the pi0.5/DROID step
convention so a 15-action chunk always spans one second. Datasets carry their
rate in `meta/info.json`, and the collector refuses to append to a dataset
recorded at a different rate — collect 15 Hz data into a fresh dataset root
rather than extending the older 20 Hz one. If a run is interrupted before the
first episode saves, only `meta/info.json` exists; the next run detects this
empty stub and recreates the dataset automatically. A directory that recorded
episodes but lost `meta/tasks.parquet` is treated as corruption and aborts
with instructions instead of falling back to the Hub. All Hub access is
disabled during collection, so local metadata problems fail fast instead of
surfacing as `401 Repository Not Found` for the local repo id.

Every run writes `collection.log` (per-episode attempts, failures, and the
full traceback, flushed before simulator shutdown) alongside the terminal
output. SIGINT/SIGTERM are deferred to safe points so the current episode
buffer is discarded, Parquet footers are finalized, statistics are
regenerated, and only then the simulator shuts down; the exit code is
`128 + signal number`.

Shutdown uses Isaac Sim's default fast mode: all dataset work (episode
discard, Parquet finalize, statistics generation, log flush) finishes before
`close()`, and a watchdog bounds the shutdown itself. If closing takes longer
than `shutdown.watchdog_seconds` (default 300), every thread's stack is dumped
to `shutdown.watchdog_dump_file` and the process exits with the already
recorded outcome code, so a stuck teardown can never hang the terminal
overnight. Video-encoder workers are likewise cancelled with a bounded,
verified wait (`shutdown.encoder_stop_timeout_seconds`) before finalization.

Sim-free unit tests cover the placement check (including the obstructed
seed-46 scene), deterministic sampling, signal handling, and shutdown order:

```bash
.venv/bin/python -m unittest discover -s tests
```

Regenerate these statistics for an existing dataset with:

```bash
/path/to/openpi/.venv/bin/python -m src.dataset_stats \
  --dataset-root dataset \
  --openpi-root /path/to/openpi
```

## Run The Simulation

Run these commands from the Isaac Lab environment.

Launch the scene without pi0.5:

```bash
python main.py
```

Run pi0.5-DROID inference in the simulation:

```bash
python main.py --test --seed 42 --task-index 0 --episode-index 0
```

Test mode uses the same scene construction, dynamic cubes, physics material, robot and gripper wrappers, direct cameras, randomization, settling, and simulation timing as data collection. `--task-index` selects an entry from the configured collection task schedule, while `--episode-index` selects the cube setup variation and offsets the random seed. Use `--tcp-y-sign -1` or `--tcp-y-sign 1` to force the initial end-effector side.

## Local Artifacts

The repository excludes virtual environments, generated datasets, camera captures, model assets, checkpoints, logs, and W&B outputs. Keep large datasets and trained weights in external storage rather than committing them to Git.
