"""Configuration for HMoG implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from apps.configs import ClusteringModelConfig

### Gradient Trainer Configs ###


@dataclass
class PreTrainerConfig:
    """Base configuration for pre-trainers."""

    _target_: str = "plugins.models.hmog.trainers.PreTrainer"
    lr: float = 1e-3
    n_epochs: int = 1000
    batch_size: int | None = None
    batch_steps: int = 1000
    l1_reg: float = 0
    l2_reg: float = 0
    grad_clip: float = 8.0
    min_var: float = 1e-5
    jitter_var: float = 0.0


@dataclass
class GradientTrainerConfig:
    """Base configuration for gradient-based trainers."""

    _target_: str = "plugins.models.hmog.trainers.GradientTrainer"
    lr: float = 1e-4
    n_epochs: int = 200
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
    upr_prs_reg: float = 1e-3
    lwr_prs_reg: float = 1e-3


@dataclass
class GradientLGMTrainerConfig(GradientTrainerConfig):
    """Configuration for gradient-based LGM trainer."""

    _target_: str = "plugins.models.hmog.trainers.GradientTrainer"
    mask_type: str = "LGM"


# @dataclass
# class GradientMixtureTrainerConfig(GradientTrainerConfig):
#     """Configuration for gradient-based mixture trainer."""
#
#     _target_: str = "plugins.models.hmog.trainers.GradientTrainer"
#     mask_type: str = "MIXTURE"
#


@dataclass
class GradientFullModelTrainerConfig(GradientTrainerConfig):
    """Configuration for gradient-based full model trainer."""

    _target_: str = "plugins.models.hmog.trainers.GradientTrainer"
    mask_type: str = "FULL"


@dataclass
class FixedObservableTrainerConfig:
    """Configuration for fixed observable trainer.

    This trainer holds observable parameters fixed and only updates mixture parameters,
    which is much more efficient for high-dimensional data.
    """

    _target_: str = "plugins.models.hmog.trainers.FixedObservableTrainer"

    # Training hyperparameters
    lr: float = 1e-4
    n_epochs: int = 200
    batch_size: int | None = None
    batch_steps: int = 1000
    grad_clip: float = 8.0

    # Regularization parameters
    l1_reg: float = 0
    l2_reg: float = 0

    # Parameter bounds for mixture components
    min_prob: float = 1e-4
    lat_min_var: float = 1e-6
    lat_jitter_var: float = 0.0

    # Precision matrix regularization
    upr_prs_reg: float = 1e-3
    lwr_prs_reg: float = 1e-3


### Analysis Config ###


@dataclass
class AnalysisConfig:
    """Configuration for analysis of trained model."""

    _target_: str = "plugins.models.hmog.artifacts.AnalysisArgs"
    from_scratch: bool = False
    epoch: int | None = None


### Main Configuration ###

cycle_defaults: list[Any] = [
    {"pre": "gradient_pre"},
    {"lgm": "gradient_lgm"},
    {"mix": "fixed_observable"},
    {"full": "gradient_full"},
]


@dataclass
class HMoGConfig(ClusteringModelConfig):
    # Model architecture
    data_dim: int = MISSING
    latent_dim: int = 10
    n_clusters: int = 10
    lgm_noise_scale: float = 0.01
    mix_noise_scale: float = 0.01
    pretrain: bool = False

    # Training configuration
    pre: PreTrainerConfig = field(default=MISSING)
    lgm: GradientTrainerConfig = field(default=MISSING)
    mix: FixedObservableTrainerConfig = field(default=MISSING)
    full: GradientTrainerConfig = field(default=MISSING)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)


@dataclass
class DifferentiableHMoGConfig(HMoGConfig):
    """HMoG configuration with cyclic training."""

    _target_: str = "plugins.models.hmog.experiment.HMoGExperiment"
    num_cycles: int = 10
    lr_scale_init: float = 0.1
    lr_scale_final: float = 0.1
    defaults: list[Any] = field(default_factory=lambda: cycle_defaults)


@dataclass
class ProjectionTrainerConfig:
    """Configuration for projection-based trainer."""

    _target_: str = "plugins.models.hmog.projection.ProjectionTrainer"
    lr: float = 1e-3
    n_epochs: int = 1000
    batch_size: int | None = None
    batch_steps: int = 1000
    l1_reg: float = 0
    l2_reg: float = 0
    grad_clip: float = 1.0
    min_prob: float = 1e-4
    lat_min_var: float = 1e-6
    lat_jitter_var: float = 0.0


@dataclass
class ProjectionHMoGConfig(ClusteringModelConfig):
    """Configuration for projection-based HMoG training."""

    _target_: str = "plugins.models.hmog.projection.ProjectionHMoGExperiment"

    # Model architecture
    pretrain: bool = False
    data_dim: int = MISSING
    latent_dim: int = 10
    n_clusters: int = 10
    lgm_noise_scale: float = 0.01
    mix_noise_scale: float = 0.01

    # Training configuration
    pre: PreTrainerConfig = field(default=MISSING)
    pro: ProjectionTrainerConfig = field(default_factory=ProjectionTrainerConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)


### Config Registration ###

cs = ConfigStore.instance()

# Register base configs
cs.store(group="model/pre", name="gradient_pre", node=PreTrainerConfig)
cs.store(group="model/lgm", name="gradient_lgm", node=GradientLGMTrainerConfig)
cs.store(group="model/mix", name="fixed_observable", node=FixedObservableTrainerConfig)
cs.store(group="model/full", name="gradient_full", node=GradientFullModelTrainerConfig)

# Register model configs
cs.store(group="model", name="hmog_diff", node=DifferentiableHMoGConfig)

# two stage
cs = ConfigStore.instance()
cs.store(group="model", name="hmog_proj", node=ProjectionHMoGConfig)

cs = ConfigStore.instance()
