"""Analysis of trained MNIST HMoG models."""

import json
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array

from goal.geometry import Natural, Point, PositiveDefinite
from goal.models import DifferentiableHMoG

from ...shared import ExamplePaths, initialize_paths
from .run import create_model, load_mnist
from .types import MNISTData


def load_results(paths: ExamplePaths) -> tuple[Any, MNISTData]:
    """Load training results and data."""
    with open(paths.analysis_path) as f:
        results = json.load(f)
    data = load_mnist(paths)
    return results, data


def extract_component_prototypes[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    model: DifferentiableHMoG[ObsRep, LatRep],
    params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
) -> Array:
    r"""Extract the mean image for each mixture component.

    $$
    \mu_k = A_k \mu_k^z + b_k
    $$
    where $A_k, b_k$ are the linear transformation parameters for component k,
    and $\mu_k^z$ is the mean of the latent distribution for component k.
    """
    # Split into likelihood and mixture parameters
    lkl_params, mix_params = model.split_conjugated(params)

    # Extract components from mixture
    comp_lats, _ = model.upr_hrm.split_natural_mixture(mix_params)

    # For each component, compute the observable distribution and get its mean
    prototypes = []
    for comp_lat_params in comp_lats:
        # Get latent mean for this component
        with model.lwr_hrm as lh:
            lwr_hrm_params = lh.join_conjugated(lkl_params, comp_lat_params)
            lwr_hrm_means = lh.to_mean(lwr_hrm_params)
            lwr_hrm_obs = lh.split_params(lwr_hrm_means)[0]
            obs_means = lh.obs_man.split_mean_second_moment(lwr_hrm_obs)[0].array

        prototypes.append(obs_means)

    return jnp.stack(prototypes)


def visualize_prototypes(
    prototypes: Array, n_components: int, paths: ExamplePaths
) -> None:
    """Create a grid visualization of the prototype digits."""
    # Compute grid dimensions
    grid_size = int(jnp.ceil(jnp.sqrt(n_components)))

    # Create figure
    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(2 * grid_size, 2 * grid_size), squeeze=False
    )

    # Plot each prototype
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
        axes[i, j].remove()

    # Save figure
    plt.tight_layout()
    plt.savefig(paths.results_dir / "prototypes.png")
    plt.close()


def main() -> None:
    """Run analysis on trained model."""
    paths = initialize_paths(__file__)

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
