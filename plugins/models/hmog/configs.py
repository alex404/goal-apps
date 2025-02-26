"""Configuration for HMoG implementations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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


@dataclass
class EMLGMTrainerConfig(LGMTrainerConfig):
    """Configuration for EM-based LGM trainer."""

    _target_: str = "plugins.models.hmog.trainers.EMLGMTrainer"
    min_var: float = 1e-4
    jitter: float = 1e-5


### Mixture Trainer Configs ###


@dataclass
class MixtureTrainerConfig:
    """Base configuration for mixture trainers."""

    min_prob: float = 1e-3


@dataclass
class GradientMixtureTrainerConfig(MixtureTrainerConfig):
    """Configuration for gradient-based mixture trainer."""

    _target_: str = "plugins.models.hmog.trainers.GradientMixtureTrainer"
    learning_rate: float = 1e-3
    min_var: float = 1e-4
    jitter: float = 1e-5


### Full Model Trainer Configs ###


@dataclass
class FullModelTrainerConfig:
    """Base configuration for full model trainers."""

    min_prob: float = 1e-3


@dataclass
class GradientFullModelTrainerConfig(FullModelTrainerConfig):
    """Configuration for gradient-based full model trainer."""

    _target_: str = "plugins.models.hmog.trainers.GradientFullModelTrainer"
    learning_rate: float = 3e-4
    obs_min_var: float = 1e-4
    lat_min_var: float = 1e-6
    obs_jitter: float = 1e-5
    lat_jitter: float = 1e-7


### Main Configuration ###


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
        stage1_trainer: Configuration for LGM training stage
        stage2_trainer: Configuration for mixture component training stage
        stage3_trainer: Configuration for full model training stage
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
    stage1_trainer: LGMTrainerConfig = MISSING
    stage2_trainer: MixtureTrainerConfig = MISSING
    stage3_trainer: FullModelTrainerConfig = MISSING

    # Analysis configuration
    from_scratch: bool = False
    analysis_epoch: int | None = None


### Config Registration ###

cs = ConfigStore.instance()

# Register base configs
cs.store(group="lgm_trainer", name="em", node=EMLGMTrainerConfig)

cs.store(group="mixture_trainer", name="gradient", node=GradientMixtureTrainerConfig)

cs.store(
    group="full_model_trainer", name="gradient", node=GradientFullModelTrainerConfig
)

# Register default configurations with their default trainers
default_config = HMoGConfig(
    stage1_trainer=EMLGMTrainerConfig(),
    stage2_trainer=GradientMixtureTrainerConfig(),
    stage3_trainer=GradientFullModelTrainerConfig(),
)

cs.store(group="model", name="hmog", node=default_config)
