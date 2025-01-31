from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING


@dataclass
class DatasetConfig:
    """Base configuration for datasets."""

    _target_: str


@dataclass
class ModelConfig:
    """Base configuration for models."""

    _target_: str
    latent_dim: int
    n_clusters: int


defaults = [{"dataset": MISSING}, {"model": MISSING}]


@dataclass
class ClusteringConfig:
    """Base configuration for clustering experiments."""

    experiment: str
    dataset: DatasetConfig
    model: ModelConfig
    defaults: list[Any] = field(default_factory=lambda: defaults)
    device: str = "gpu"
    jit: bool = True
