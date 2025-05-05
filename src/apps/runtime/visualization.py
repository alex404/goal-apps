"""Core visualization utilities."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .handler import MetricHistory


def setup_matplotlib_style() -> None:
    """Load and set the default matplotlib style."""
    style_path = Path(__file__).parents[3] / "config" / "default.mplstyle"
    if style_path.exists():
        plt.style.use(str(style_path))


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
