"""Base class for DifferentiableHMoG implementations."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, override

import jax.numpy as jnp
import matplotlib.pyplot as plt
from goal.geometry import Diagonal
from jax import Array
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from apps.interface import (
    Analysis,
    ClusteringDataset,
)
from apps.runtime import Artifact, RunHandler

from ..types import AnyHMoG

### Loading Matrix Artifacts ###


@dataclass(frozen=True)
class LoadingMatrixArtifact(Artifact):
    """Loading matrix visualizations in both natural and mean parameterizations."""

    natural_loadings: Array  # Shape: (data_dim, latent_dim)
    mean_loadings: Array  # Shape: (data_dim, latent_dim)


def generate_loading_matrices(
    model: AnyHMoG,
    params: Array,
) -> LoadingMatrixArtifact:
    """Extract loading matrices in both natural and mean parameterizations.

    The natural loading matrix is the model's interaction matrix Λ (the raw
    weights that link latent codes to observable natural parameters). For
    interpretability we also return the mean-parameter loading
    ``Σ_x · Λ``, which expresses the loading as the covariance-weighted
    influence on the observable mean. Both are (data_dim, latent_dim).
    """
    # Extract the interaction parameters
    obs_params, int_params, _ = model.split_coords(params)

    # Get natural loadings (directly from the parameters)
    natural_loadings = model.int_man.to_matrix(int_params)

    # Convert to mean parameterization for interpretability.
    # In mean coordinates the loading matrix is Sigma_x * Lambda.
    _, obs_prs = model.obs_man.split_coords(obs_params)
    cov_man = model.obs_man.cov_man
    if isinstance(cov_man.rep, Diagonal):
        # Diagonal precision: covariance is element-wise reciprocal of the
        # precision diagonal. Avoid densifying to DxD (6 GB at D=39k) --
        # extract the diagonal and broadcast-multiply instead.
        obs_cov_diag = 1.0 / cov_man.get_diagonal(obs_prs)
        mean_loadings = obs_cov_diag[:, None] * natural_loadings
    else:
        obs_prs_dense = cov_man.to_matrix(obs_prs)
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

        # Cap displayed dimensions to keep figure size reasonable
        max_display = min(latent_dim, 24)
        n_cols = min(3, max_display)
        n_rows = math.ceil(max_display / n_cols)

        fig_width = 5 * n_cols * 2  # 2 panels (natural + mean) per dimension
        fig_height = 2 * n_rows

        fig = plt.figure(figsize=(fig_width, fig_height))
        grid = GridSpec(n_rows, n_cols * 2, figure=fig)

        title = "Loading Matrix Visualization"
        if max_display < latent_dim:
            title += f" (showing {max_display}/{latent_dim} dims)"
        fig.suptitle(title, fontsize=16)

        # Plot each latent dimension
        for i in range(max_display):
            row = i // n_cols
            col = (i % n_cols) * 2

            ax_natural = fig.add_subplot(grid[row, col])
            ax_mean = fig.add_subplot(grid[row, col + 1])

            natural_pattern = artifact.natural_loadings[:, i]
            mean_pattern = artifact.mean_loadings[:, i]

            dataset.paint_observable(natural_pattern, ax_natural)
            dataset.paint_observable(mean_pattern, ax_mean)

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
