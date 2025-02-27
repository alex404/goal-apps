"""Core visualization utilities."""

import math
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score

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
    side_length = round(math.sqrt(n_metrics))
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


def evaluate_clustering(cluster_assignments: Array, true_labels: Array) -> float:
    """Evaluate clustering by finding optimal label assignment."""
    n_clusters = int(jnp.max(cluster_assignments)) + 1
    n_classes = int(jnp.max(true_labels)) + 1

    # Compute cluster-class frequency matrix
    freq_matrix = np.zeros((n_clusters, n_classes))
    for i in range(len(true_labels)):
        freq_matrix[int(cluster_assignments[i]), int(true_labels[i])] += 1

    # Assign each cluster to its most frequent class
    cluster_to_class = np.argmax(freq_matrix, axis=1)
    predicted_labels = jnp.array([cluster_to_class[i] for i in cluster_assignments])

    return float(accuracy_score(true_labels, predicted_labels))
