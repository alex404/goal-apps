"""Configuration dataclasses for MFA model."""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from apps.interface import ClusteringModelConfig
from apps.interface.clustering.config import ClusteringAnalysesConfig


@dataclass
class GradientTrainerConfig:
    """Configuration for gradient descent trainer using HMOG-style gradient computation."""

    _target_: str = "plugins.models.mfa.trainers.GradientTrainer"

    # Training hyperparameters
    lr: float = 1e-3
    """Learning rate for optimizer."""

    n_epochs: int = 200
    """Number of training epochs."""

    batch_size: int | None = None
    """Batch size (None = full batch)."""

    batch_steps: int = 1
    """Number of gradient steps per batch."""

    grad_clip: float = 1.0
    """Maximum gradient norm for clipping."""

    # Regularization parameters
    l1_reg: float = 0.0
    """L1 regularization on interaction parameters (promotes sparse loadings)."""

    l2_reg: float = 0.0
    """L2 regularization on all parameters."""

    upr_prs_reg: float = 1e-3
    """Upper bound regularization on precision eigenvalues (trace penalty)."""

    lwr_prs_reg: float = 1e-3
    """Lower bound regularization on precision eigenvalues (log-det penalty)."""

    log_freq: int = 10
    """Log metrics every log_freq epochs."""

    # Parameter bounds (applied in mean coordinate space)
    min_prob: float = 1e-4
    """Minimum cluster probability to prevent collapse."""

    obs_min_var: float = 1e-5
    """Minimum observable variance."""

    lat_min_var: float = 1e-6
    """Minimum latent variance."""

    obs_jitter_var: float = 0.0
    """Jitter to add to observable variance."""

    lat_jitter_var: float = 0.0
    """Jitter to add to latent variance."""


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

    init_scale: float = 0.01
    """Scale for parameter initialization (smaller for high-dim data)."""

    min_var: float = 0.01
    """Minimum variance for regularization (prevents NaN for zero-variance pixels)."""

    # Trainer
    trainer: GradientTrainerConfig = field(default=MISSING)
    """Trainer configuration."""

    # Analyses
    analyses: ClusteringAnalysesConfig = field(default_factory=ClusteringAnalysesConfig)
    """Configuration for analyses to run."""


# Register configurations with Hydra
cs = ConfigStore.instance()
cs.store(group="model/trainer", name="default", node=GradientTrainerConfig)
cs.store(group="model", name="mfa", node=MFAConfig)
