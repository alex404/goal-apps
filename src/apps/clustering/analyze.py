"""Analysis of trained clustering models."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from apps.experiments import Experiment

from .core.common import ProbabilisticResults


def analyze(paths: Experiment) -> None:
    """Analyze results from a trained model.

    Args:
        paths: Experiment paths containing training results
    """
    # Load training results
    results: ProbabilisticResults = paths.load_analysis("training_results")

    # Create visualization of prototypes
    visualize_prototypes(results, paths)


def visualize_prototypes(results: ProbabilisticResults, paths: Experiment) -> None:
    """Visualize model prototypes.

    Args:
        results: Training results containing model info and parameters
        paths: Experiment paths for saving outputs
    """
    n_components = results["n_clusters"]
    grid_size = int(jnp.ceil(jnp.sqrt(n_components)))

    # Create figure
    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(2 * grid_size, 2 * grid_size), squeeze=False
    )
    fig.suptitle(f"{results['model_name']} Component Prototypes")

    # Plot each prototype
    prototypes = jnp.array(results["prototypes"])
    for k in range(n_components):
        i, j = k // grid_size, k % grid_size
        ax = axes[i, j]

        # Reshape to square image and plot
        img = prototypes[k].reshape(28, 28)
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.axis("off")
        ax.set_title(f"Component {k}")

    # Remove empty subplots
    for k in range(n_components, grid_size * grid_size):
        i, j = k // grid_size, k % grid_size
        fig.delaxes(axes[i, j])

    # Save figure
    plt.tight_layout()
    paths.save_plot(fig, "prototypes")
