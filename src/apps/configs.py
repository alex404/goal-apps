import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from omegaconf import MISSING

### Runtime Configs ###


# Basic Configs

# Define a custom level
STATS_NUM = 15  # Between INFO (20) and DEBUG (10)
logging.addLevelName(STATS_NUM, "STATS")


class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    STATS = STATS_NUM
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


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


@dataclass
class DatasetConfig:
    """Base configuration for models."""

    _target_: str


@dataclass
class ExperimentConfig:
    """Base configuration for models."""

    _target_: str


### Clustering Configs ###


@dataclass
class ClusteringDatasetConfig(DatasetConfig):
    """Base configuration for clustering datasets."""

    _target_: str


@dataclass
class ClusteringExperimentConfig(ExperimentConfig):
    """Base configuration for clustering models."""

    _target_: str
    data_dim: int = MISSING
    n_clusters: int = MISSING


defaults: list[Any] = [
    {"model": MISSING},
    {"dataset": MISSING},
]


@dataclass
class ClusteringRunConfig(RunConfig):
    """Base configuration for clustering simulations."""

    dataset: ClusteringDatasetConfig
    model: ClusteringExperimentConfig
    defaults: list[Any] = field(default_factory=lambda: defaults)
