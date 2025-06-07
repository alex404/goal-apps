from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from ..interface import ClusteringDatasetConfig, ClusteringExperimentConfig
from ..runtime import LogLevel

### Runtime Config ###


@dataclass
class RunConfig:
    """Base configuration for a single run. Subclasses should extend this and be set to the base config by `cs.store(name="config_schema", node=MyRunConfig)`."""

    run_name: str
    device: str
    jit: bool
    use_local: bool
    repeat: int
    use_wandb: bool
    log_level: LogLevel
    project: str
    group: str | None
    job_type: str | None
    sweep_id: str | None = None
    from_epoch: int = 0
    from_scratch: bool = False


defaults: list[Any] = [
    {"model": MISSING},
    {"dataset": MISSING},
]


@dataclass
class ClusteringRunConfig(RunConfig):
    """Base configuration for clustering simulations."""

    dataset: ClusteringDatasetConfig = MISSING
    model: ClusteringExperimentConfig = MISSING
    defaults: list[Any] = field(default_factory=lambda: defaults)
