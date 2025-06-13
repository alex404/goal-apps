"""Shared utilities for GOAL examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import wandb as wandb
from omegaconf import MISSING

from ..interface import (
    ClusteringDatasetConfig,
    ClusteringModelConfig,
)
from ..runtime import LogLevel

### Run Configs ###


@dataclass
class RunConfig:
    """Base configuration for a single run. Subclasses should extend this and be set to the base config by `cs.store(name="config_schema", node=MyRunConfig)`."""

    run_name: str
    device: str
    jit: bool
    use_local: bool
    repeat: int
    use_wandb: bool
    log_level: LogLevel
    project: str
    group: str | None
    job_type: str | None
    from_epoch: int | None
    run_id: str | None
    sweep_id: str | None
    from_scratch: bool


defaults: list[Any] = [
    {"model": MISSING},
    {"dataset": MISSING},
]


### Clustering Run Configs ###


@dataclass
class ClusteringRunConfig(RunConfig):
    """Base configuration for clustering simulations."""

    dataset: ClusteringDatasetConfig = MISSING
    model: ClusteringModelConfig = MISSING
    defaults: list[Any] = field(default_factory=lambda: defaults)
