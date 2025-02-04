"""Shared utilities for GOAL examples."""

import json
import os
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from jax import Array
from matplotlib.figure import Figure
from omegaconf import DictConfig, OmegaConf

from .config import RunConfig

### Loggers ###


@dataclass(frozen=True)
class Logger(ABC):
    """Protocol defining interface for metric logging.

    Attributes:
        metrics: List of metric names to track
        steps: Number of logging steps expected in simulation
    """

    metrics: list[str]
    steps: int

    def log_metrics(self, values: dict[str, Array], step: int) -> None:
        """Log metrics for current step.

        Args:
            values: Dictionary mapping metric names to values
            step: Current step number (0-based)

        Raises:
            ValueError: If metrics don't match those specified at initialization
        """
        if set(values.keys()) != set(self.metrics):
            raise ValueError(
                f"Logged metrics {set(values.keys())} do not match "
                f"specified metrics {set(self.metrics)}"
            )
        if not 0 <= step < self.steps:
            raise ValueError(f"Step {step} out of range [0, {self.steps})")


@dataclass(frozen=True)
class WandbLogger(Logger):
    """Logger implementation using Weights & Biases.

    Logs metrics in real-time to wandb platform.
    """

    metrics: list[str]
    steps: int

    def log_metrics(self, values: dict[str, Array], step: int) -> None:
        super().log_metrics(values, step)
        wandb.log({k: float(v) for k, v in values.items()}, step=step)

    def log_array(self, key: str, array: Array) -> None:
        """Log array data to wandb."""
        wandb.log({key: wandb.Image(array)})

    def finalize(self) -> None:
        """Clean up wandb run."""
        wandb.finish()


@dataclass(frozen=True)
class LocalLogger(Logger):
    """Logger implementation for local storage.

    Stores metrics in memory and provides debug output.
    """

    metrics: list[str]
    steps: int

    # Initialize buffer for each metric
    buffers: dict[str, Array] = field(init=False)

    def __post_init__(self) -> None:
        # Create zero-initialized buffer for each metric
        buffers = {metric: jnp.zeros(self.steps) for metric in self.metrics}
        # Set buffers through __setattr__ to work with frozen dataclass
        object.__setattr__(self, "buffers", buffers)

    def log_metrics(self, values: dict[str, Array], step: int) -> None:
        """Log metrics to internal buffers and print debug output."""
        super().log_metrics(values, step)

        # Update buffers
        for metric, value in values.items():
            self.buffers[metric] = self.buffers[metric].at[step].set(float(value))

        # Print debug output
        debug_str = f"Step {step}: " + ", ".join(
            f"{k}={float(v):.6f}" for k, v in values.items()
        )
        print(debug_str)

    def log_array(self, key: str, array: Array) -> None:
        """Store array data locally."""
        # Note: Implementation depends on specific needs
        pass

    def finalize(self) -> dict[str, Array]:
        """Return complete history of all metrics."""
        return dict(self.buffers)


### Path and IO Handler ###


@dataclass(frozen=True)
class RunHandler:
    """Handles file management and organization for a single run."""

    name: str

    @property
    def project_root(self) -> Path:
        return Path(__file__).parents[2]

    @property
    def run_dir(self) -> Path:
        base = self.project_root / "runs"
        sweep_id = os.environ.get("WANDB_SWEEP_ID")
        if sweep_id is not None:
            return base / "sweep" / sweep_id / self.name
        return base / "single" / self.name

    @property
    def cache_dir(self) -> Path:
        return self.project_root / ".cache"

    def save_analysis(self, results: Any, name: str = "analysis") -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        analysis_path = self.run_dir / f"{name}.json"
        with open(analysis_path, "w") as f:
            json.dump(results, f, indent=2)

    def load_analysis(self, name: str = "analysis") -> Any:
        analysis_path = self.run_dir / f"{name}.json"
        with open(analysis_path) as f:
            return json.load(f)

    def save_plot(self, fig: Figure, name: str = "plot") -> None:
        plot_path = self.run_dir / f"{name}.png"
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Log arbitrary metrics dictionary to wandb if enabled."""
        if not wandb.run:
            return
        wandb.log(metrics)

    def log_image(
        self, key: str, image: Array | Figure, caption: str | None = None
    ) -> None:
        """Log an image to wandb if enabled."""
        if not wandb.run:
            return
        wandb.log({key: wandb.Image(image, caption=caption)})

    def finish(self) -> None:
        """Clean up wandb run if active."""
        if wandb.run:
            wandb.finish()


### Initialization ###


def _initialize_jax(device: str = "cpu", disable_jit: bool = False) -> None:
    jax.config.update("jax_platform_name", device)
    if disable_jit:
        jax.config.update("jax_disable_jit", True)


def initialize_run(
    config_type: type[RunConfig],
    overrides: list[str],
) -> tuple[RunHandler, DictConfig]:
    """Initialize a new run with hydra config and wandb logging."""
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=config_type)

    with initialize_config_dir(
        version_base="1.3", config_dir=str(Path(__file__).parents[2] / "config")
    ):
        cfg = compose(config_name="config", overrides=overrides)

    _initialize_jax(device=cfg.device, disable_jit=not cfg.jit)

    handler = RunHandler(name=cfg.run_name)

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            name=handler.name,
            group=cfg.wandb.group,
            job_type=cfg.wandb.job_type,
            tags=cfg.wandb.tags,
            notes=cfg.wandb.notes,
            config=OmegaConf.to_container(cfg, resolve=True),  # pyright: ignore[reportArgumentType]
        )

    return handler, cfg
