"""Shared utilities for GOAL examples."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import matplotlib.pyplot as plt
import wandb
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from jax import Array
from matplotlib.figure import Figure
from omegaconf import DictConfig, OmegaConf

from .config import RunConfig

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
