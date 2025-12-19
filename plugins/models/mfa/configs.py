"""Configuration dataclasses for MFA model."""

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from apps.interface import ClusteringModelConfig


@dataclass
class GradientTrainerConfig:
    """Configuration for gradient descent trainer with AdamW optimizer."""

    _target_: str = "plugins.models.mfa.trainers.GradientTrainer"
    lr: float = 1e-2
    """Learning rate for AdamW optimizer."""

    n_epochs: int = 200
    """Number of training epochs."""

    batch_size: int | None = None
    """Batch size (None = full batch gradient descent)."""


@dataclass
class MFAConfig(ClusteringModelConfig):
    """Configuration for Mixture of Factor Analyzers model."""

    _target_: str = "plugins.models.mfa.model.MFAModel"

    # Model architecture
    data_dim: int = MISSING
    """Dimension of observable data (set by dataset)."""

    latent_dim: int = 10
    """Dimension of latent factors."""

    n_clusters: int = 10
    """Number of mixture components."""

    # Trainer
    trainer: GradientTrainerConfig = field(default=MISSING)
    """Trainer configuration."""


# Register configurations with Hydra
cs = ConfigStore.instance()
cs.store(group="model/trainer", name="default", node=GradientTrainerConfig)
cs.store(group="model", name="mfa", node=MFAConfig)
