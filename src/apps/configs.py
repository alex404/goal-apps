from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

### Runtime Configs ###


# Basic Configs


@dataclass
class RunConfig:
    """Base configuration for a single run. Subclasses should extend this and be set to the base config by `cs.store(name="config_schema", node=MyRunConfig)`."""

    run_name: str
    device: str
    jit: bool
    use_local: bool
    use_wandb: bool
    project: str
    group: str | None
    job_type: str | None


@dataclass
class DatasetConfig:
    """Base configuration for models."""

    _target_: str


@dataclass
class ModelConfig:
    """Base configuration for models."""

    _target_: str


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
]


@dataclass
class ClusteringRunConfig(RunConfig):
    """Base configuration for clustering simulations."""

    dataset: ClusteringDatasetConfig
    model: ClusteringModelConfig
    defaults: list[Any] = field(default_factory=lambda: defaults)
