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


def create_heatmap(
    matrix: Array,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[int, int] = (10, 8),
    cmap: str = "viridis",
) -> Figure:
    """Create a heatmap figure.

    Pure plotting function that returns a figure without saving.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap=cmap)
    plt.colorbar(im)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

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
