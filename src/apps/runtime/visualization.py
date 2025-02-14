"""Core visualization utilities."""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score


def setup_matplotlib_style() -> None:
    """Load and set the default matplotlib style."""
    style_path = (
        Path(__file__).parent.parent.parent.parent / "misc" / "default.mplstyle"
    )
    if style_path.exists():
        plt.style.use(str(style_path))


def plot_metrics(buffer: list[tuple[int, dict[str, float]]]) -> Figure:
    """Create a summary plot of all metrics over training.

    Args:
        buffer: List of (epoch, metrics) pairs

    Returns:
        Figure containing subplots for each metric
    """
    if not buffer:
        raise ValueError("No metrics to plot")

    # Extract epochs and reorganize metrics
    epochs, metric_dicts = zip(*sorted(buffer))
    metrics = {name: [d[name] for d in metric_dicts] for name in metric_dicts[0]}

    # Create figure
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics))

    # Handle single metric case
    if n_metrics == 1:
        axes = [axes]

    # Plot each metric
    for ax, (name, values) in zip(axes, metrics.items()):
        ax.plot(epochs, values)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.grid(True)

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
