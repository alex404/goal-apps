"""Configuration for HMoG implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from goal.geometry import Diagonal, PositiveDefinite, Scale
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from apps.configs import ClusteringModelConfig

### Covariance Reps ###


class RepresentationType(Enum):
    scale = Scale
    diagonal = Diagonal
    positive_definite = PositiveDefinite


### LGM Trainer Configs ###


@dataclass
class LGMTrainerConfig:
    """Base configuration for LGM trainers."""

    n_epochs: int = 100
    min_var: float = 1e-6
    jitter: float = 0


@dataclass
class EMLGMTrainerConfig(LGMTrainerConfig):
    """Configuration for EM-based LGM trainer."""

    _target_: str = "plugins.models.hmog.trainers.EMLGMTrainer"


### Mixture Trainer Configs ###


@dataclass
class MixtureTrainerConfig:
    """Base configuration for mixture trainers."""

    n_epochs: int = 100
    min_prob: float = 1e-3
    min_var: float = 1e-6
    jitter: float = 0


@dataclass
class GradientMixtureTrainerConfig(MixtureTrainerConfig):
    """Configuration for gradient-based mixture trainer."""

    _target_: str = "plugins.models.hmog.trainers.GradientMixtureTrainer"
    lr_init: float = 1e-3
    lr_final_ratio: float = 1
    batch_size: int = 256


### Full Model Trainer Configs ###


@dataclass
class FullModelTrainerConfig:
    """Base configuration for full model trainers."""

    min_prob: float = 1e-3
    obs_min_var: float = 1e-6
    lat_min_var: float = 1e-6
    obs_jitter: float = 0
    lat_jitter: float = 0
    n_epochs: int = 100


@dataclass
class GradientFullModelTrainerConfig(FullModelTrainerConfig):
    """Configuration for gradient-based full model trainer."""

    _target_: str = "plugins.models.hmog.trainers.GradientFullModelTrainer"
    lr_init: float = 3e-4
    lr_final_ratio: float = 0.2
    grad_clip: float = 8
    batch_size: int = 256


### Main Configuration ###


hmog_defaults: list[Any] = [
    {"stage1": "em"},
    {"stage2": "gradient"},
    {"stage3": "gradient"},
]


@dataclass
class HMoGConfig(ClusteringModelConfig):
    """Configuration for Hierarchical Mixture of Gaussians model.

    Model Architecture:
        latent_dim: Dimension of latent space
        n_clusters: Number of mixture components
        data_dim: Dimension of input data (set by dataset)
        obs_rep: Representation type for observations
        lat_rep: Representation type for latents

    Training Parameters:
        stage1: Configuration for LGM training stage
        stage2: Configuration for mixture component training stage
        stage3: Configuration for full model training stage
        stage1_epochs: Number of epochs for stage 1
        stage2_epochs: Number of epochs for stage 2
        stage3_epochs: Number of epochs for stage 3
    """

    _target_: str = "plugins.models.hmog.experiment.HMoGExperiment"

    # Model architecture
    data_dim: int = MISSING
    latent_dim: int = 10
    n_clusters: int = 10
    obs_rep: RepresentationType = RepresentationType.diagonal
    lat_rep: RepresentationType = RepresentationType.diagonal

    # Training configuration
    stage1: LGMTrainerConfig = MISSING
    stage2: MixtureTrainerConfig = MISSING
    stage3: FullModelTrainerConfig = MISSING

    # Analysis configuration
    from_scratch: bool = False
    analysis_epoch: int | None = None

    # Defaults
    defaults: list[Any] = field(default_factory=lambda: hmog_defaults)


### Config Registration ###

cs = ConfigStore.instance()

# Register base configs
cs.store(group="model/stage1", name="em", node=EMLGMTrainerConfig)

cs.store(group="model/stage2", name="gradient", node=GradientMixtureTrainerConfig)

cs.store(group="model/stage3", name="gradient", node=GradientFullModelTrainerConfig)

cs.store(group="model", name="hmog", node=HMoGConfig)
