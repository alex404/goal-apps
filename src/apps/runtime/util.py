"""Shared utilities for GOAL examples."""

from __future__ import annotations

import logging
import math
import re
from abc import ABC
from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt
import wandb as wandb
from jax import Array
from matplotlib.figure import Figure

## Logging ###

log = logging.getLogger(__name__)

# Define a custom level
STATS_NUM = 15  # Between INFO (20) and DEBUG (10)
logging.addLevelName(STATS_NUM, "STATS")


class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    STATS = STATS_NUM
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


### Metrics ###

type MetricDict = dict[str, tuple[Array, Array]]  # Single snapshot
type MetricHistory = dict[str, list[tuple[int, float]]]  # Time series


def update_stats(
    group: str, name: str, stats: Array, metrics: MetricDict, level: int = STATS_NUM
) -> MetricDict:
    """Add min/median/max statistics for an array to a metrics dict.

    Args:
        group: Metric group name (e.g., "Params", "Grad Norms")
        name: Metric name within group (e.g., "Obs Location")
        stats: Array of values to compute statistics over
        metrics: Existing metrics dict to update
        level: Log level for these metrics (default: STATS_NUM)

    Returns:
        Updated metrics dict with three new entries:
        - "{group}/{name} Min"
        - "{group}/{name} Median"
        - "{group}/{name} Max"
    """
    import jax.numpy as jnp

    level_arr = jnp.array(level)
    metrics.update(
        {
            f"{group}/{name} Min": (level_arr, jnp.min(stats)),
            f"{group}/{name} Median": (level_arr, jnp.median(stats)),
            f"{group}/{name} Max": (level_arr, jnp.max(stats)),
        }
    )
    return metrics


### Artifacts ###


@dataclass(frozen=True)
class Artifact(ABC):
    """Base class for data that can be logged and visualized."""


### Helpers ###


def to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case."""
    name = re.sub("([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub("([a-z])([A-Z])", r"\1_\2", name)
    return name.lower()


def plot_metrics(metrics: MetricHistory) -> Figure:
    """Create a summary plot of all metrics over training.

    Args:
        metrics: Dictionary mapping metric names to lists of (epoch, value) pairs

    Returns:
        Figure containing subplots for each metric
    """
    # Create figure
    n_metrics = len(metrics)
    side_length = math.ceil(math.sqrt(n_metrics))
    fig, axes = plt.subplots(
        side_length, side_length, figsize=(6 * side_length, 4 * side_length)
    )

    axes = axes.ravel()

    # Plot each metric
    for ax, (name, values) in zip(axes, metrics.items()):
        epochs, metric_values = zip(*values)
        ax.plot(epochs, metric_values)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.grid(True)

    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    return fig
