"""Base configuration classes for goal-apps."""

from __future__ import annotations

from dataclasses import dataclass

from ..runtime import LogLevel


@dataclass
class RunConfig:
    """Base configuration for a single run."""

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
    resume_epoch: int | None
    run_id: str | None
    sweep_id: str | None
    recompute_artifacts: bool
