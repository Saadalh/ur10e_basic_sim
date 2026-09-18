import dataclasses
import sys
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "pi05_config.yaml"
with CONFIG_PATH.open(encoding="utf-8") as config_file:
    PI05_CONFIG = yaml.safe_load(config_file)

OPENPI_ROOT = Path(PI05_CONFIG["openpi_root"]).expanduser()
sys.path.insert(0, str(OPENPI_ROOT / "src"))

from openpi.models import pi0_config
from openpi.policies import policy_config
from openpi.training import config as config_module


class Pi05Model:
    def __init__(self, policy: Any) -> None:
        self._policy = policy

    def infer(self, observations: dict[str, Any]):
        return self._policy.infer(observations)["actions"]


def init_pi05() -> Pi05Model:
    base_config = config_module.get_config(PI05_CONFIG["config_name"])
    # Reconstruct the exact LoRA architecture this checkpoint was trained
    # with. The stock pi05_droid config builds a non-LoRA model, whose
    # parameter tree has no LoRA leaves, so restore discards the trained
    # adapter parameters. Only the model field is replaced; the DROID data
    # pipeline, transforms, and checkpoint normalization behavior carry over.
    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=15,
        max_token_len=200,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    print(
        "Inference model: "
        f"{model_config.paligemma_variant}/{model_config.action_expert_variant}, "
        f"action_dim={model_config.action_dim}, "
        f"action_horizon={model_config.action_horizon}, "
        f"max_token_len={model_config.max_token_len}, "
        f"pi05={model_config.pi05}",
        flush=True,
    )
    config = dataclasses.replace(base_config, model=model_config)
    policy = policy_config.create_trained_policy(
        config,
        PI05_CONFIG["checkpoint"],
        default_prompt=PI05_CONFIG["default_prompt"],
    )
    return Pi05Model(policy)
