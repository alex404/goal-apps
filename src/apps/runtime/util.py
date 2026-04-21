"""Shared utilities for GOAL examples."""

from __future__ import annotations

import logging
import math
import re
from abc import ABC
from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from matplotlib.figure import Figure

## Logging ###

log = logging.getLogger(__name__)

# Define a custom level
STATS_NUM = 15  # Between INFO (20) and DEBUG (10)
logging.addLevelName(STATS_NUM, "STATS")
STATS_LEVEL: Array = jnp.array(STATS_NUM)
INFO_LEVEL: Array = jnp.array(logging.INFO)


class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    STATS = STATS_NUM
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


### Exceptions ###


class DivergentTrainingError(Exception):
    """Raised when NaN parameters are detected during training."""


### Metrics ###

type MetricDict = dict[str, tuple[Array, Array]]  # Single snapshot
type MetricHistory = dict[str, list[tuple[int, float]]]  # Time series


def update_stats(
    group: str,
    name: str,
    stats: Array,
    metrics: MetricDict,
    level: Array = STATS_LEVEL,
) -> MetricDict:
    """Add min/median/max statistics for an array to a metrics dict.

    Args:
        group: Metric group name (e.g., "Params", "Grad Norms")
        name: Metric name within group (e.g., "Obs Location")
        stats: Array of values to compute statistics over
        metrics: Existing metrics dict to update
        level: Log level for these metrics. Pass the module-level
            ``STATS_LEVEL`` / ``INFO_LEVEL`` constants — they're pre-built
            JAX arrays so no per-call allocation.

    Returns:
        Updated metrics dict with three new entries:
        - "{group}/{name} Min"
        - "{group}/{name} Median"
        - "{group}/{name} Max"
    """
    metrics.update(
        {
            f"{group}/{name} Min": (level, jnp.min(stats)),
            f"{group}/{name} Median": (level, jnp.median(stats)),
            f"{group}/{name} Max": (level, jnp.max(stats)),
        }
    )
    return metrics


def stats_keys(group: str, *names: str) -> frozenset[str]:
    """Declare the metric keys that update_stats() will produce."""
    return frozenset(
        f"{group}/{name} {stat}" for name in names for stat in ("Min", "Median", "Max")
    )


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
    if n_metrics == 0:
        return plt.figure()
    side_length = math.ceil(math.sqrt(n_metrics))
    fig, axes = plt.subplots(
        side_length, side_length, figsize=(6 * side_length, 4 * side_length)
    )

    axes = np.atleast_1d(axes).ravel()

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
