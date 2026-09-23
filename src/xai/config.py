"""Configuration loading, reproducible seeding and device selection."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class TaskConfig:
    """Configuration for a single task, with attribute access over the raw dict."""

    key: str
    raw: dict[str, Any]

    def __getattr__(self, item: str) -> Any:
        try:
            return self.raw[item]
        except KeyError as exc:
            raise AttributeError(f"task '{self.key}' has no setting '{item}'") from exc

    def get(self, item: str, default: Any = None) -> Any:
        return self.raw.get(item, default)

    def path(self, item: str) -> Path:
        """Resolve a path-valued setting against the project root."""
        return PROJECT_ROOT / self.raw[item]

    @property
    def n_labels(self) -> int:
        return len(self.raw["labels"])


class Config:
    """Top-level configuration object."""

    def __init__(self, path: Path | str = DEFAULT_CONFIG):
        self.path = Path(path)
        with open(self.path) as fh:
            self.raw: dict[str, Any] = yaml.safe_load(fh)
        self.seed: int = self.raw["seed"]

    def task(self, key: str) -> TaskConfig:
        if key not in self.raw["tasks"]:
            raise KeyError(f"unknown task '{key}'; have {list(self.raw['tasks'])}")
        return TaskConfig(key, self.raw["tasks"][key])

    @property
    def task_keys(self) -> list[str]:
        return list(self.raw["tasks"])

    @property
    def artifacts(self) -> Path:
        p = PROJECT_ROOT / self.raw["paths"]["artifacts"]
        p.mkdir(parents=True, exist_ok=True)
        return p

    def artifact_dir(self, *parts: str) -> Path:
        p = self.artifacts.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def explain(self) -> dict[str, Any]:
        return self.raw["explain"]


def set_seed(seed: int) -> None:
    """Seed every source of randomness the pipeline touches.

    LIME and SHAP both sample perturbations; without this, re-running the
    pipeline produces different attributions for the same input and no
    explanation in the report can be checked against the code.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(spec: str = "auto") -> torch.device:
    """Pick a compute device, preferring Apple Metal then CUDA then CPU."""
    if spec != "auto":
        return torch.device(spec)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load(path: Path | str = DEFAULT_CONFIG) -> Config:
    """Load the config and apply its seed immediately."""
    cfg = Config(path)
    set_seed(cfg.seed)
    return cfg
