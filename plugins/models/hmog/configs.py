"""Configuration for HMoG implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from apps.configs import ClusteringModelConfig

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


@dataclass
class GradientLGMPretrainerConfig(LGMTrainerConfig):
    """Configuration for gradient-based LGM trainer."""

    _target_: str = "plugins.models.hmog.trainers.GradientLGMPretrainer"
    n_epochs: int = 1000
    lr_init: float = 1e-3
    lr_final_ratio: float = 0.1
    batch_size: int = 256
    l1_reg: float = 0
    l2_reg: float = 0.0001
    re_reg: float = 0
    grad_clip: float = 8


@dataclass
class GradientLGMTrainerConfig(LGMTrainerConfig):
    """Configuration for gradient-based LGM trainer."""

    _target_: str = "plugins.models.hmog.trainers.GradientLGMTrainer"
    n_epochs: int = 200
    lr_init: float = 1e-4
    lr_final_ratio: float = 1
    batch_size: int = 256
    l1_reg: float = 0
    l2_reg: float = 0.0001
    re_reg: float = 0
    grad_clip: float = 8


### Mixture Trainer Configs ###


@dataclass
class MixtureTrainerConfig:
    """Base configuration for mixture trainers."""

    n_epochs: int = 200
    min_prob: float = 1e-4
    min_var: float = 0
    jitter: float = 0


@dataclass
class GradientMixtureTrainerConfig(MixtureTrainerConfig):
    """Configuration for gradient-based mixture trainer."""

    _target_: str = "plugins.models.hmog.trainers.GradientMixtureTrainer"
    lr_init: float = 1e-4
    lr_final_ratio: float = 1
    batch_size: int = 256
    l2_reg: float = 0.0001
    grad_clip: float = 8


### Full Model Trainer Configs ###


@dataclass
class FullModelTrainerConfig:
    """Base configuration for full model trainers."""

    n_epochs: int = 0
    min_prob: float = 1e-4
    obs_min_var: float = 1e-6
    obs_jitter: float = 0
    lat_min_var: float = 0
    lat_jitter: float = 0


@dataclass
class GradientFullModelTrainerConfig(FullModelTrainerConfig):
    """Configuration for gradient-based full model trainer."""

    _target_: str = "plugins.models.hmog.trainers.GradientFullModelTrainer"
    lr_init: float = 3e-5
    lr_final_ratio: float = 1
    batch_size: int = 256
    l1_reg: float = 0
    l2_reg: float = 0.0001
    re_reg: float = 0
    grad_clip: float = 8


### Analysis Config ###


@dataclass
class AnalysisConfig:
    """Configuration for analysis of trained model."""

    _target_: str = "plugins.models.hmog.artifacts.AnalysisArgs"
    from_scratch: bool = False
    epoch: int | None = None


### Main Configuration ###

three_stage_defaults: list[Any] = [
    {"lgm": "em"},
    {"mix": "gradient"},
    {"full": "gradient"},
]

cycle_defaults: list[Any] = [
    {"pre": "pretrain"},
    {"lgm": "gradient"},
    {"mix": "gradient"},
    {"full": "gradient"},
]


@dataclass
class HMoGConfig(ClusteringModelConfig):
    # Model architecture
    data_dim: int = MISSING
    latent_dim: int = 10
    n_clusters: int = 10

    # Training configuration
    lgm: LGMTrainerConfig = field(default=MISSING)
    mix: MixtureTrainerConfig = field(default=MISSING)
    full: FullModelTrainerConfig = field(default=MISSING)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)


@dataclass
class SymmetricHMoGConfig(HMoGConfig):
    """HMoG configuration with cyclic training."""

    _target_: str = "plugins.models.hmog.experiment.SymmetricHMoGExperiment"
    num_cycles: int = 10
    defaults: list[Any] = field(default_factory=lambda: cycle_defaults)


### DifferentiableHMoG Trainer Config ###


@dataclass
class DifferentiableModelTrainerConfig:
    """Configuration for single-stage DifferentiableHMoG trainer."""

    _target_: str = "plugins.models.hmog.trainers.DifferentiableModelTrainer"
    n_epochs: int = 5000
    pretrain_epochs: int = 1000
    lr_init: float = 1e-4
    lr_final_ratio: float = 1
    batch_size: int = 256
    l1_reg: float = 0
    l2_reg: float = 0.001
    re_reg: float = 0
    grad_clip: float = 8
    min_prob: float = 1e-4
    obs_min_var: float = 1e-6
    obs_jitter: float = 0
    lat_min_var: float = 1e-6
    lat_jitter: float = 0


### Main Configuration ###

differentiable_defaults: list[Any] = [
    {"trainer": "differentiable"},
]


@dataclass
class DifferentiableHMoGConfig(ClusteringModelConfig):
    """Configuration for DifferentiableHMoG model."""

    _target_: str = "plugins.models.hmog.experiment.DifferentiableHMoGExperiment"
    data_dim: int = MISSING
    latent_dim: int = 10
    n_clusters: int = 10

    # Single trainer for the entire model
    trainer: DifferentiableModelTrainerConfig = field(default=MISSING)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    defaults: list[Any] = field(default_factory=lambda: differentiable_defaults)


### Config Registration ###

cs = ConfigStore.instance()

# Register base configs
cs.store(group="model/lgm", name="em", node=EMLGMTrainerConfig)
cs.store(group="model/pre", name="pretrain", node=GradientLGMPretrainerConfig)
cs.store(group="model/lgm", name="gradient", node=GradientLGMTrainerConfig)

cs.store(group="model/mix", name="gradient", node=GradientMixtureTrainerConfig)

cs.store(group="model/full", name="gradient", node=GradientFullModelTrainerConfig)

cs.store(group="model", name="hmog_sym", node=SymmetricHMoGConfig)

### Config Registration ###

# Add these lines to the existing registration section:
cs.store(
    group="model/trainer", name="differentiable", node=DifferentiableModelTrainerConfig
)
cs.store(group="model", name="hmog_diff", node=DifferentiableHMoGConfig)
