"""Shared utilities for GOAL examples."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


@dataclass
class ExperimentConfig:
    """Base configuration for clustering experiments."""

    experiment: str
    device: str
    jit: bool
    wandb: bool


### Initialization and Path Management ###


@dataclass(frozen=True)
class ExperimentHandler:
    name: str

    @property
    def project_root(self) -> Path:
        return Path(__file__).parents[2]

    @property
    def experiments_dir(self) -> Path:
        return self.project_root / "experiments" / self.name

    @property
    def cache_dir(self) -> Path:
        return self.project_root / ".cache"

    def save_analysis(self, results: Any, name: str = "analysis") -> None:
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        analysis_path = self.experiments_dir / f"{name}.json"
        with open(analysis_path, "w") as f:
            json.dump(results, f, indent=2)

    def load_analysis(self, name: str = "analysis") -> Any:
        analysis_path = self.experiments_dir / f"{name}.json"
        with open(analysis_path) as f:
            return json.load(f)

    def save_plot(self, fig: Figure, name: str = "plot") -> None:
        plot_path = self.experiments_dir / f"{name}.png"
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)


def initialize_jax(device: str = "cpu", disable_jit: bool = False) -> None:
    jax.config.update("jax_platform_name", device)
    if disable_jit:
        jax.config.update("jax_disable_jit", True)
