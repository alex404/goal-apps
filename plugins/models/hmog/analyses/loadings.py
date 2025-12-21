"""Base class for DifferentiableHMoG implementations."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, override

import jax.numpy as jnp
import matplotlib.pyplot as plt
from goal.models import DifferentiableHMoG
from jax import Array
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from apps.interface import (
    Analysis,
    ClusteringDataset,
)
from apps.runtime import Artifact, RunHandler

### Loading Matrix Artifacts ###


@dataclass(frozen=True)
class LoadingMatrixArtifact(Artifact):
    """Loading matrix visualizations in both natural and mean parameterizations."""

    natural_loadings: Array  # Shape: (data_dim, latent_dim)
    mean_loadings: Array  # Shape: (data_dim, latent_dim)


def generate_loading_matrices(
    model: DifferentiableHMoG,
    params: Array,
) -> LoadingMatrixArtifact:
    """Extract loading matrices in both natural and mean parameterizations."""
    # Extract the interaction parameters
    obs_params, int_params, _ = model.split_coords(params)

    # Get natural loadings (directly from the parameters)
    natural_loadings = model.int_man.to_matrix(int_params)

    # Convert to mean parameterization for interpretability
    _, obs_prs = model.obs_man.split_coords(obs_params)
    obs_prs_dense = model.obs_man.cov_man.to_matrix(obs_prs)

    # In mean coordinates, the loading matrix is Σ_x * Λ
    obs_cov_dense = jnp.linalg.inv(obs_prs_dense)
    mean_loadings = jnp.matmul(obs_cov_dense, natural_loadings)

    return LoadingMatrixArtifact(
        natural_loadings=natural_loadings, mean_loadings=mean_loadings
    )


def loading_matrix_plotter(
    dataset: ClusteringDataset,
) -> Callable[[LoadingMatrixArtifact], Figure]:
    """Visualize loading matrices using existing dataset visualization routines."""

    def plot_loading_matrices(artifact: LoadingMatrixArtifact) -> Figure:
        # Get dimensions
        _, latent_dim = artifact.mean_loadings.shape

        # Create figure for all latent dimensions
        # We'll show natural and mean parameterizations side by side
        fig_width = 5 * min(6, latent_dim)  # Limit width for many latent dims
        fig_height = 2 * math.ceil(
            latent_dim / 3
        )  # Adjust height based on num dimensions

        fig = plt.figure(figsize=(fig_width, fig_height))

        # Create grid layout
        grid = GridSpec(math.ceil(latent_dim / 3), 6, figure=fig)

        # Add title
        fig.suptitle("Loading Matrix Visualization", fontsize=16)

        # Plot each latent dimension
        for i in range(latent_dim):
            row = i // 3
            col = (i % 3) * 2

            # Create axes for this latent dimension
            ax_natural = fig.add_subplot(grid[row, col])
            ax_mean = fig.add_subplot(grid[row, col + 1])

            # Extract patterns for this latent dimension
            natural_pattern = artifact.natural_loadings[:, i]
            mean_pattern = artifact.mean_loadings[:, i]

            # Use dataset visualization to plot
            dataset.paint_observable(natural_pattern, ax_natural)
            dataset.paint_observable(mean_pattern, ax_mean)

            # Add titles
            ax_natural.set_title(f"Natural Z{i + 1}")
            ax_mean.set_title(f"Mean Z{i + 1}")

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # pyright: ignore[reportArgumentType]
        return fig

    return plot_loading_matrices


### Analysis ###


@dataclass(frozen=True)
class LoadingMatrixAnalysis(Analysis[ClusteringDataset, Any, LoadingMatrixArtifact]):
    """Analysis of loading matrices from the linear Gaussian model."""

    @property
    @override
    def artifact_type(self) -> type[LoadingMatrixArtifact]:
        return LoadingMatrixArtifact

    @override
    def generate(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: Any,
        epoch: int,
        params: Array,
    ) -> LoadingMatrixArtifact:
        """Generate loading matrix visualization."""
        return generate_loading_matrices(model.manifold, params)

    @override
    def plot(
        self, artifact: LoadingMatrixArtifact, dataset: ClusteringDataset
    ) -> Figure:
        """Create grid of cluster prototype visualizations."""
        return loading_matrix_plotter(dataset)(artifact)
