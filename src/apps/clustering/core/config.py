from dataclasses import dataclass

from omegaconf import MISSING

from ...experiments import ExperimentConfig

defaults = [{"dataset": MISSING}, {"model": MISSING}]


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


@dataclass
class ClusteringConfig(ExperimentConfig):
    """Base configuration for clustering experiments."""

    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING
