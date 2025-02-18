"""Base class for HMoG implementations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, TypedDict, override

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy
from goal.geometry import (
    Natural,
    Point,
    PositiveDefinite,
)
from goal.models import (
    DifferentiableHMoG,
)
from jax import Array
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray

from apps.plugins import (
    ClusteringDataset,
)
from apps.runtime.handler import Artifact, JSONDict, RunHandler
from apps.runtime.logger import JaxLogger

### Metrics ###


class HMoGMetrics(TypedDict):
    train_ll: Array
    test_ll: Array
    train_average_bic: Array


### Prototypes ###


@dataclass(frozen=True)
class Prototypes(Artifact):
    prototypes: list[Array]

    @override
    def to_json(self) -> JSONDict:
        return {
            "prototypes": [p.tolist() for p in self.prototypes],
        }

    @classmethod
    @override
    def from_json(cls, json_dict: JSONDict) -> Prototypes:  # pyright: ignore[reportIncompatibleMethodOverride]
        return cls([jnp.array(p) for p in json_dict["prototypes"]])


def get_component_prototypes[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    model: DifferentiableHMoG[ObsRep, LatRep],
    params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
) -> Prototypes:
    # Split into likelihood and mixture parameters
    lkl_params, mix_params = model.split_conjugated(params)

    # Extract components from mixture
    comp_lats, _ = model.upr_hrm.split_natural_mixture(mix_params)

    # For each component, compute the observable distribution and get its mean
    prototypes = []
    for i in range(comp_lats.shape[0]):
        # Get latent mean for this component
        with model.lwr_hrm as lh:
            comp_lat_params = model.upr_hrm.cmp_man.get_replicate(
                comp_lats, jnp.asarray(i)
            )
            lwr_hrm_params = lh.join_conjugated(lkl_params, comp_lat_params)
            lwr_hrm_means = lh.to_mean(lwr_hrm_params)
            lwr_hrm_obs = lh.split_params(lwr_hrm_means)[0]
            obs_means = lh.obs_man.split_mean_second_moment(lwr_hrm_obs)[0].array

        prototypes.append(obs_means)

    return Prototypes(prototypes)


def prototypes_plotter(
    dataset: ClusteringDataset,
) -> Callable[[Prototypes], Figure]:
    def plot_prototypes(prototypes: Prototypes) -> Figure:
        n_prots = len(prototypes.prototypes)

        obs_arts = [dataset.observable_artifact(p) for p in prototypes.prototypes]
        shape = obs_arts[0].shape

        n_cols = math.ceil(math.sqrt(n_prots))
        n_rows = math.ceil(n_prots / n_cols)

        height, width = shape
        wh = (width + height) / 2
        height = 2 * height / wh
        width = 2 * width / wh
        figsize = (width * n_cols, height * n_rows)

        fig, axes = plt.subplots(
            nrows=n_rows, ncols=n_cols, figsize=figsize, constrained_layout=True
        )

        # Handle single subplot case
        if n_prots == 1:
            dataset.paint_observable(obs_arts[0], axes)
            return fig

        # Paint the observables using axes.flat iterator
        for ax, obs_art in zip(axes.flat, obs_arts):
            dataset.paint_observable(obs_art, ax)

        # Remove empty subplots
        for ax in axes.flat[n_prots:]:
            fig.delaxes(ax)

        return fig

    return plot_prototypes


### Divergences ###


@dataclass(frozen=True)
class Divergences(Artifact):
    """Artifact containing divergence analysis results."""

    kl_matrix: Array  # Raw KL divergences
    symmetric_kl: Array  # Symmetrized KL divergences
    linkage_matrix: NDArray[np.float64]  # Linkage matrix for hierarchical clustering

    @override
    def to_json(self) -> JSONDict:
        return {
            "kl_matrix": self.kl_matrix.tolist(),
            "symmetric_kl": self.symmetric_kl.tolist(),
            "linkage_matrix": self.linkage_matrix.tolist(),
        }

    @classmethod
    @override
    def from_json(cls, json_dict: JSONDict) -> Divergences:  # pyright: ignore[reportIncompatibleMethodOverride]
        return cls(
            kl_matrix=jnp.array(json_dict["kl_matrix"]),
            symmetric_kl=jnp.array(json_dict["symmetric_kl"]),
            linkage_matrix=np.array(
                json_dict["linkage_matrix"], dtype=np.float64
            ),  # Keep as numpy array
        )


def get_component_divergences[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    model: DifferentiableHMoG[ObsRep, LatRep],
    params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
) -> Divergences:
    # Get raw KL divergences (keeping existing code)
    mix_params = model.prior(params)
    comp_lats, _ = model.upr_hrm.split_natural_mixture(mix_params)

    def kl_div_between_components(i: Array, j: Array) -> Array:
        comp_i = model.upr_hrm.cmp_man.get_replicate(comp_lats, i)
        comp_i_mean = model.upr_hrm.obs_man.to_mean(comp_i)
        comp_j = model.upr_hrm.cmp_man.get_replicate(comp_lats, j)
        return model.upr_hrm.obs_man.relative_entropy(comp_i_mean, comp_j)

    idxs = jnp.arange(model.upr_hrm.n_categories)

    def kl_div_from_one_to_all(i: Array) -> Array:
        return jax.vmap(kl_div_between_components, in_axes=(None, 0))(i, idxs)

    kl_matrix = jax.vmap(kl_div_from_one_to_all)(idxs)
    symmetric_kl = (kl_matrix + kl_matrix.T) / 2

    # Convert to numpy and handle distances
    dist_matrix = np.array(symmetric_kl, dtype=np.float64)

    # Ensure non-negative distances and exact zero diagonal
    min_off_diag = np.min(dist_matrix[~np.eye(dist_matrix.shape[0], dtype=bool)])
    if min_off_diag < 0:
        dist_matrix = (
            dist_matrix - min_off_diag
        )  # Shift everything up to make minimum 0

    # Ensure perfect symmetry
    dist_matrix = (dist_matrix + dist_matrix.T) / 2

    # Force diagonal to exactly zero
    np.fill_diagonal(dist_matrix, 0.0)

    # Double check validity before converting
    assert np.allclose(dist_matrix, dist_matrix.T)  # Symmetry
    assert np.allclose(np.diag(dist_matrix), 0)  # Zero diagonal
    assert np.all(dist_matrix >= 0)  # Non-negative distances

    # Convert to condensed form
    dist_vector = scipy.spatial.distance.squareform(dist_matrix)

    # Compute linkage matrix using average linkage
    linkage_matrix = scipy.cluster.hierarchy.linkage(
        dist_vector,
        method="average",  # Using UPGMA clustering
    )

    return Divergences(
        kl_matrix=kl_matrix, symmetric_kl=symmetric_kl, linkage_matrix=linkage_matrix
    )


def plot_divergence_matrix(
    divergences: Divergences,
) -> Figure:
    fig = plt.figure(figsize=(15, 5))

    # Create GridSpec for layout
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1.2])

    # Plot raw KL divergences
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(divergences.kl_matrix, cmap="viridis")
    plt.colorbar(im1, ax=ax1, label="KL Divergence")
    ax1.set_title("Raw KL Divergences")
    ax1.set_xlabel("Component j")
    ax1.set_ylabel("Component i")

    # Plot symmetric KL divergences
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(divergences.symmetric_kl, cmap="viridis")
    plt.colorbar(im2, ax=ax2, label="Symmetric KL Divergence")
    ax2.set_title("Symmetric KL Divergences")
    ax2.set_xlabel("Component j")
    ax2.set_ylabel("Component i")

    # Plot dendrogram
    ax3 = fig.add_subplot(gs[2])
    scipy.cluster.hierarchy.dendrogram(
        divergences.linkage_matrix, ax=ax3, leaf_rotation=90, leaf_font_size=8
    )
    ax3.set_title("Hierarchical Clustering Dendrogram")

    # Add grid lines to matrix plots
    for ax in [ax1, ax2]:
        n_rws = divergences.kl_matrix.shape[0]
        ax.set_xticks(np.arange(-0.5, n_rws, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rws, 1), minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    return fig


### Cluster Hierarchy ###


@dataclass(frozen=True)
class ClusterHierarchy(Artifact):
    """Artifact containing clustering hierarchy analysis results."""

    prototypes: list[Array]
    linkage_matrix: NDArray[np.float64]

    @override
    def to_json(self) -> JSONDict:
        return {
            "prototypes": [p.tolist() for p in self.prototypes],
            "linkage_matrix": self.linkage_matrix.tolist(),
        }

    @classmethod
    @override
    def from_json(cls, json_dict: JSONDict) -> ClusterHierarchy:
        return cls(
            prototypes=[jnp.array(p) for p in json_dict["prototypes"]],
            linkage_matrix=np.array(json_dict["linkage_matrix"], dtype=np.float64),
        )


def get_cluster_hierarchy[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    prototypes: Prototypes, divergences: Divergences
) -> ClusterHierarchy:
    """Generate combined clustering hierarchy analysis."""
    return ClusterHierarchy(
        prototypes=prototypes.prototypes, linkage_matrix=divergences.linkage_matrix
    )


def hierarchy_plotter(
    dataset: ClusteringDataset,
) -> Callable[[ClusterHierarchy], Figure]:
    """Plot dendrogram with corresponding prototype visualizations."""

    def plot_cluster_hierarchy(hierarchy: ClusterHierarchy) -> Figure:
        n_clusters = len(hierarchy.prototypes)

        # Create prototype artifacts
        prototype_artifacts = [
            dataset.observable_artifact(p) for p in hierarchy.prototypes
        ]
        prototype_shape = prototype_artifacts[0].shape

        # Compute figure dimensions
        dendrogram_width = 6  # Fixed width for dendrogram
        prototype_width = 4  # Fixed width for each prototype
        spacing = 0.5  # Spacing between dendrogram and prototypes

        # Total figure width
        fig_width = dendrogram_width + spacing + prototype_width

        # Height per prototype/cluster
        cluster_height = 1.0  # Base height per cluster
        fig_height = n_clusters * cluster_height

        # Create figure with grid
        fig = plt.figure(figsize=(fig_width, fig_height))

        # Create gridspec with two columns
        gs = GridSpec(
            n_clusters,
            2,
            width_ratios=[dendrogram_width, prototype_width],
            wspace=spacing / fig_width,  # Normalize spacing
            figure=fig,
        )

        # Plot dendrogram in left column
        dendrogram_ax = fig.add_subplot(gs[:, 0])
        # Using scipy's dendrogram with modified orientation
        scipy.cluster.hierarchy.dendrogram(
            hierarchy.linkage_matrix,
            orientation="left",
            ax=dendrogram_ax,
            leaf_font_size=8,
        )
        dendrogram_ax.set_title("Cluster Hierarchy")

        # Get the order of leaves from the dendrogram
        # This ensures prototypes are shown in the same order as the dendrogram
        leaf_order = scipy.cluster.hierarchy.dendrogram(
            hierarchy.linkage_matrix, no_plot=True
        )["leaves"]

        # Plot prototypes in right column
        for i, leaf_idx in enumerate(leaf_order):
            prototype_ax = fig.add_subplot(gs[i, 1])
            dataset.paint_observable(prototype_artifacts[leaf_idx], prototype_ax)
            prototype_ax.set_title(f"Cluster {leaf_idx}", fontsize=8)

        return fig

    return plot_cluster_hierarchy


def log_artifacts[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    handler: RunHandler,
    dataset: ClusteringDataset,
    logger: JaxLogger,
    model: DifferentiableHMoG[ObsRep, LatRep],
    epoch: int,
    params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]] | None = None,
) -> None:
    """Generate and save plots from artifacts.

    Args:
        handler: Run handler containing saved artifacts
        dataset: Dataset used for visualization
        logger: Logger for saving artifacts and figures
        model: Model used for analysis and artifact generation
        params: If provided, generate new artifacts from these parameters
        epoch: Specific epoch to analyze, defaults to latest
    """
    # from_scratch if params is provided
    if params is not None:
        prototypes = get_component_prototypes(model, params)
        divergences = get_component_divergences(model, params)
        clusters = get_cluster_hierarchy(prototypes, divergences)
        handler.save_params(epoch, params.array)
    else:
        prototypes = handler.load_artifact(epoch, Prototypes)
        divergences = handler.load_artifact(epoch, Divergences)
        clusters = handler.load_artifact(epoch, ClusterHierarchy)

    # Plot and save
    plot_prototypes = prototypes_plotter(dataset)
    plot_hierarchy = hierarchy_plotter(dataset)
    logger.log_artifact(handler, epoch, prototypes, plot_prototypes)
    logger.log_artifact(handler, epoch, divergences, plot_divergence_matrix)
    logger.log_artifact(handler, epoch, clusters, plot_hierarchy)
