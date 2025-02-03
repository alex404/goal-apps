from dataclasses import dataclass, field

from omegaconf import MISSING

defaults = [{"dataset": MISSING}, {"model": MISSING}]


### Base Runtime Configs ###


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases integration.

    Attributes:
        enabled: Whether to use wandb logging
        project: Name of the wandb project
        entity: Username or team name in wandb
        group: Optional group name for organizing related runs (e.g. sweeps)
        tags: List of tags to apply to the run
        notes: Optional notes about the run
        mode: Wandb run mode - 'online', 'offline', or 'disabled'
        save_code: Whether to save code to wandb
        job_type: Type of job (e.g. 'training', 'evaluation')
    """

    enabled: bool = False
    project: str = "goal"
    entity: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str | None = None
    mode: str = "online"
    save_code: bool = False
    job_type: str | None = None


@dataclass
class RunConfig:
    """Base configuration for a single run.

    Attributes:
        run_name: Name of the run
        device: Device to run on ('cpu' or 'gpu')
        jit: Whether to enable JAX JIT compilation
        wandb: Weights & Biases configuration
    """

    run_name: str
    device: str
    jit: bool
    wandb: WandbConfig = field(default_factory=WandbConfig)


### Base Clustering Configs ###


@dataclass
class DatasetConfig:
    """Base configuration for datasets."""

    _target_: str


@dataclass
class ModelConfig:
    """Base configuration for models."""

    _target_: str
    data_dim: int = MISSING
    latent_dim: int = MISSING
    n_clusters: int = MISSING


@dataclass
class ClusteringConfig(RunConfig):
    """Base configuration for clustering simulations."""

    dataset: DatasetConfig = MISSING
    model: ModelConfig = MISSING
