import sys
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "pi05_config.yaml"
with CONFIG_PATH.open(encoding="utf-8") as config_file:
    PI05_CONFIG = yaml.safe_load(config_file)

OPENPI_ROOT = Path(PI05_CONFIG["openpi_root"]).expanduser()
sys.path.insert(0, str(OPENPI_ROOT / "src"))

from openpi.policies import policy_config
from openpi.training import config as config_module


class Pi05Model:
    def __init__(self, policy: Any) -> None:
        self._policy = policy

    def infer(self, observations: dict[str, Any]):
        return self._policy.infer(observations)["actions"]


def init_pi05() -> Pi05Model:
    config = config_module.get_config(PI05_CONFIG["config_name"])
    policy = policy_config.create_trained_policy(
        config,
        PI05_CONFIG["checkpoint"],
        default_prompt=PI05_CONFIG["default_prompt"],
    )
    return Pi05Model(policy)
