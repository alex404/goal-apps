from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

### Runtime Configs ###

cs = ConfigStore.instance()

# Base Component Configs


@dataclass
class LoggerConfig:
    """Base configuration for logging."""

    _target_: str = MISSING  # Will point to logger implementation


@dataclass
class RunConfig:
    """Base configuration for a single run. Subclasses should extend this and be set to the base config by `cs.store(name="config_schema", node=MyRunConfig)`."""

    run_name: str
    device: str
    jit: bool
    logger: LoggerConfig


@dataclass
class DatasetConfig:
    """Base configuration for models."""

    _target_: str


@dataclass
class ModelConfig:
    """Base configuration for models."""

    _target_: str


# Logger Configs


@dataclass
class WandbLoggerConfig(LoggerConfig):
    """Configuration for Weights & Biases logging."""

    _target_: str = "apps.runtime.logger.WandbLogger"
    project: str = "goal"
    group: str | None = None
    job_type: str | None = None


@dataclass
class LocalLoggerConfig(LoggerConfig):
    """Configuration for local file logging."""

    _target_: str = "apps.runtime.logger.LocalLogger"


@dataclass
class NullLoggerConfig(LoggerConfig):
    """Configuration for disabled logging."""

    _target_: str = "apps.runtime.logger.NullLogger"


cs.store(group="logger", name="wandb", node=WandbLoggerConfig)
cs.store(group="logger", name="local", node=LocalLoggerConfig)
cs.store(group="logger", name="null", node=NullLoggerConfig)


### Clustering Configs ###


@dataclass
class ClusteringDatasetConfig(DatasetConfig):
    """Base configuration for clustering datasets."""

    _target_: str


@dataclass
class ClusteringModelConfig(ModelConfig):
    """Base configuration for clustering models."""

    _target_: str
    data_dim: int = MISSING
    n_clusters: int = MISSING


defaults: list[Any] = [
    {"model": MISSING},
    {"dataset": MISSING},
    {"logger": "wandb"},
]


@dataclass
class ClusteringRunConfig(RunConfig):
    """Base configuration for clustering simulations."""

    dataset: ClusteringDatasetConfig
    model: ClusteringModelConfig
    defaults: list[Any] = field(default_factory=lambda: defaults)
