"""Analysis of trained MNIST HMoG models."""

from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array

from ..shared import ExperimentPaths
from .core.types import BaseModel, MNISTData
from .train import create_model, load_mnist


def load_results(paths: ExperimentPaths) -> tuple[Any, MNISTData]:
    """Load training results and data."""
    training_results = paths.load_analysis("training")
    data = load_mnist(paths)
    return training_results, data


def visualize_prototypes(
    prototypes: Array,
    n_components: int,
    paths: ExperimentPaths,
    title: str = "Component Prototypes",
    cmap: str = "gray",
) -> None:
    """Create a grid visualization of component prototypes.

    Args:
        prototypes: Array of shape (n_components, data_dim) containing prototype vectors
        n_components: Number of components to visualize
        paths: Paths object for saving results
        title: Title for the overall figure
        cmap: Colormap to use for visualization
    """
    # Compute grid dimensions
    grid_size = int(jnp.ceil(jnp.sqrt(n_components)))

    # Create figure
    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(2 * grid_size, 2 * grid_size), squeeze=False
    )
    fig.suptitle(title)

    # Plot each prototype
    for k in range(n_components):
        i, j = k // grid_size, k % grid_size
        ax = axes[i, j]

        # Reshape to square image and plot
        img = prototypes[k].reshape(28, 28)  # Assuming MNIST dimensions
        ax.imshow(img, cmap=cmap, interpolation="nearest")
        ax.axis("off")
        ax.set_title(f"Component {k}")

    # Remove empty subplots
    for k in range(n_components, grid_size * grid_size):
        i, j = k // grid_size, k % grid_size
        fig.delaxes(axes[i, j])

    # Save figure
    plt.tight_layout()
    paths.save_plot(fig, "prototypes")


def analyze_model[P](
    model: BaseModel[P, Any], params: P, paths: ExperimentPaths
) -> None:
    """Analyze a trained model by visualizing its components.

    Args:
        model: Trained model instance
        params: Trained model parameters
        paths: Paths object for saving results
    """
    # Extract and visualize prototypes
    prototypes = model.get_component_prototypes(params)
    visualize_prototypes(
        prototypes=prototypes,
        n_components=model.n_clusters,
        paths=paths,
        title=f"{model.__class__.__name__} Component Prototypes",
    )


def main() -> None:
    """Run analysis on trained model."""
    paths = example_paths(__file__)

    # Load results and reconstruct model
    results, data = load_results(paths)
    model = create_model(
        model_name=results["model_name"].lower(),
        latent_dim=results["latent_dim"],
        n_clusters=results["n_clusters"],
        data_dim=data.train_images.shape[1],
        stage1_epochs=100,  # Not used for analysis
        stage2_epochs=100,  # Not used for analysis
        n_epochs=300,  # Not used for analysis
        batch_size=0,  # Not used for analysis
    )

    # Initialize model parameters (needed to get correct shapes)
    key = jax.random.PRNGKey(0)
    params = model.model.natural_point(results["parameters"])

    # Extract and visualize prototypes
    prototypes = extract_component_prototypes(model.model, params)
    visualize_prototypes(prototypes, results["n_clusters"], paths)


if __name__ == "__main__":
    main()
