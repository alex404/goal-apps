"""Configuration for HMoG implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from apps.configs import ClusteringModelConfig

### Gradient Trainer Configs ###


@dataclass
class GradientTrainerConfig:
    """Base configuration for gradient-based trainers."""

    _target_: str = "plugins.models.hmog.trainers.GradientTrainer"
    n_epochs: int = 200
    lr_init: float = 3e-4
    lr_final_ratio: float = 1.0
    batch_size: int | None = None
    batch_steps: int = 1000
    l1_reg: float = 0
    l2_reg: float = 0
    grad_clip: float = 8.0
    min_prob: float = 1e-4
    obs_min_var: float = 1e-5
    lat_min_var: float = 1e-6
    obs_jitter_var: float = 0.0
    lat_jitter_var: float = 0.0
    eigen_floor_prs: float = 1e-6


@dataclass
class GradientLGMTrainerConfig(GradientTrainerConfig):
    """Configuration for gradient-based LGM trainer."""

    _target_: str = "plugins.models.hmog.trainers.GradientTrainer"
    mask_type: str = "LGM"


@dataclass
class GradientMixtureTrainerConfig(GradientTrainerConfig):
    """Configuration for gradient-based mixture trainer."""

    _target_: str = "plugins.models.hmog.trainers.GradientTrainer"
    mask_type: str = "MIXTURE"


@dataclass
class GradientFullModelTrainerConfig(GradientTrainerConfig):
    """Configuration for gradient-based full model trainer."""

    _target_: str = "plugins.models.hmog.trainers.GradientTrainer"
    mask_type: str = "FULL"


### Analysis Config ###


@dataclass
class AnalysisConfig:
    """Configuration for analysis of trained model."""

    _target_: str = "plugins.models.hmog.artifacts.AnalysisArgs"
    from_scratch: bool = False
    epoch: int | None = None


### Main Configuration ###

cycle_defaults: list[Any] = [
    {"lgm": "gradient_lgm"},
    {"mix": "gradient_mixture"},
    {"full": "gradient_full"},
]


@dataclass
class HMoGConfig(ClusteringModelConfig):
    # Model architecture
    data_dim: int = MISSING
    latent_dim: int = 10
    n_clusters: int = 10

    # Training configuration
    lgm: GradientTrainerConfig = field(default=MISSING)
    mix: GradientTrainerConfig = field(default=MISSING)
    full: GradientTrainerConfig = field(default=MISSING)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)


@dataclass
class DifferentiableHMoGConfig(HMoGConfig):
    """HMoG configuration with cyclic training."""

    _target_: str = "plugins.models.hmog.experiment.HMoGExperiment"
    num_cycles: int = 10
    defaults: list[Any] = field(default_factory=lambda: cycle_defaults)


### Config Registration ###

cs = ConfigStore.instance()

# Register base configs
cs.store(group="model/lgm", name="gradient_lgm", node=GradientLGMTrainerConfig)
cs.store(group="model/mix", name="gradient_mixture", node=GradientMixtureTrainerConfig)
cs.store(group="model/full", name="gradient_full", node=GradientFullModelTrainerConfig)

# Register model configs
cs.store(group="model", name="hmog_diff", node=DifferentiableHMoGConfig)
