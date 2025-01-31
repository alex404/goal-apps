from dataclasses import dataclass


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
class ClusteringConfig:
    """Base configuration for clustering experiments."""

    dataset: DatasetConfig
    model: ModelConfig
    experiment: str
    device: str = "gpu"
    jit: bool = True
