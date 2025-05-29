"""Base class for HMoG implementations."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Callable, override

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy
from goal.geometry import (
    Natural,
    Point,
)
from h5py import File, Group
from jax import Array
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray

from apps.plugins import (
    Analysis,
    ClusteringDataset,
)
from apps.runtime.handler import Artifact, RunHandler

from ..base import HMoG
from .base import cluster_probabilities, get_component_prototypes, symmetric_kl_matrix

### Helpers ###


def posterior_co_assignment_matrix[M: HMoG](
    model: M,
    params: Point[Natural, M],
    data: Array,
) -> Array:
    """Compute posterior co-assignment matrix between components.

    This computes how often two clusters are assigned to the same data points,
    based on their posterior probabilities.

    Args:
        model: HMoG model
        params: Model parameters
        data: Data points for calculating empirical similarities

    Returns:
        Co-assignment similarity matrix (higher values = more similar)
    """
    # Get cluster probabilities for each data point
    probs = cluster_probabilities(model, params.array, data)

    # Compute co-assignment matrix efficiently through matrix multiplication
    # co_assignment[i,j] = sum_x p(x,i) * p(x,j)
    co_assignment = probs.T @ probs

    # Normalize to get correlation-like measure
    diag_sqrt = jnp.sqrt(jnp.diag(co_assignment))
    # Build outer product of sqrt-diag entries, add small epsilon for stability
    denom = diag_sqrt[:, None] * diag_sqrt[None, :] + 1e-12
    normalized_co_assignment = co_assignment / denom

    # Ensure perfect symmetry by averaging with transpose
    return 0.5 * (normalized_co_assignment + normalized_co_assignment.T)


### Artifacts ###


@dataclass(frozen=True)
class ClusterHierarchy(Artifact):
    """Base artifact for clustering hierarchy analysis."""

    prototypes: list[Array]
    linkage_matrix: NDArray[np.float64]
    similarity_matrix: NDArray[np.float64]

    @override
    def save_to_hdf5(self, file: File) -> None:
        """Save hierarchy data to HDF5 file."""
        # Save prototypes
        proto_group = file.create_group("prototypes")
        for i, proto in enumerate(self.prototypes):
            proto_group.create_dataset(f"{i}", data=np.array(proto))

        # Save linkage matrix and similarity matrix
        file.create_dataset("linkage_matrix", data=self.linkage_matrix)
        file.create_dataset("similarity_matrix", data=np.array(self.similarity_matrix))

    @classmethod
    @override
    def load_from_hdf5(cls, file: File) -> ClusterHierarchy:
        """Load hierarchy data from HDF5 file."""
        # Load prototypes
        proto_group = file["prototypes"]
        assert isinstance(proto_group, Group)
        n_protos = len(proto_group)
        prototypes = [jnp.array(proto_group[f"{i}"]) for i in range(n_protos)]

        # Load matrices
        linkage_matrix = np.array(file["linkage_matrix"][()])  # pyright: ignore[reportIndexIssue,reportArgumentType]
        similarity_matrix = np.array(file["similarity_matrix"][()])  # pyright: ignore[reportIndexIssue,reportArgumentType]

        return cls(
            prototypes=prototypes,
            linkage_matrix=linkage_matrix,
            similarity_matrix=similarity_matrix,
        )


@dataclass(frozen=True)
class KLClusterHierarchy(ClusterHierarchy):
    """KL divergence-based clustering hierarchy."""


@dataclass(frozen=True)
class CoAssignmentClusterHierarchy(ClusterHierarchy):
    """Co-assignment probability-based clustering hierarchy."""


def generate_cluster_hierarchy[M: HMoG, C: ClusterHierarchy](
    model: M,
    params: Point[Natural, M],
    hierarchy_type: type[C],
    data: Array,
) -> C:
    """Generate clustering hierarchy analysis with specified similarity metric.

    Args:
        model: HMoG model
        params: Model parameters
        hierarchy_type: Type of cluster hierarchy to create
        data: Data points for calculating empirical similarities (only needed for co-assignment)

    Returns:
        Specified type of cluster hierarchy
    """
    prototypes = get_component_prototypes(model, params)

    # Calculate the similarity matrix based on the hierarchy type
    if hierarchy_type == KLClusterHierarchy:
        similarity_matrix = symmetric_kl_matrix(model, params)
        # convert to numpy
    elif hierarchy_type == CoAssignmentClusterHierarchy:
        similarity_matrix = posterior_co_assignment_matrix(model, params, data)
    else:
        raise TypeError(f"Unsupported hierarchy type: {hierarchy_type}")
    similarity_matrix = np.array(similarity_matrix, dtype=np.float64)

    # Convert to numpy and prepare for hierarchical clustering
    if hierarchy_type == CoAssignmentClusterHierarchy:
        # Co-assignment is a similarity matrix (higher = more similar)
        # Convert to distance (lower = more similar)
        dist_matrix = np.array(1.0 - similarity_matrix, dtype=np.float64)
    else:
        # KL divergence is already a distance (lower = more similar)
        dist_matrix = np.array(similarity_matrix, dtype=np.float64)

    # Ensure non-negative distances
    min_off_diag = np.min(dist_matrix[~np.eye(dist_matrix.shape[0], dtype=bool)])
    if min_off_diag < 0:
        dist_matrix = dist_matrix - min_off_diag

    # Ensure perfect symmetry
    dist_matrix = (dist_matrix + dist_matrix.T) / 2

    # Force diagonal to exactly zero
    np.fill_diagonal(dist_matrix, 0.0)

    # Import scipy here for clarity
    import scipy.cluster.hierarchy
    import scipy.spatial.distance

    # Convert to condensed form
    dist_vector = scipy.spatial.distance.squareform(dist_matrix)

    # Compute linkage matrix using average linkage
    linkage_matrix = scipy.cluster.hierarchy.linkage(
        dist_vector,
        method="average",  # Using UPGMA clustering
    )

    # Create the hierarchy object of the requested type
    return hierarchy_type(
        prototypes=prototypes,
        linkage_matrix=linkage_matrix,  # pyright: ignore[reportArgumentType]
        similarity_matrix=similarity_matrix,
    )


def hierarchy_plotter(
    dataset: ClusteringDataset,
) -> Callable[[ClusterHierarchy], Figure]:
    """Plot dendrogram with corresponding prototype visualizations."""

    def plot_cluster_hierarchy(hierarchy: ClusterHierarchy) -> Figure:
        n_clusters = len(hierarchy.prototypes)
        prototype_shape = dataset.observable_shape

        # Compute figure dimensions
        dendrogram_width = 6  # Fixed width for dendrogram
        height, width = prototype_shape
        prototype_width = (
            width / height * dendrogram_width
        )  # Scale width based on shape
        spacing = 4

        # Total figure width
        fig_width = dendrogram_width + spacing + prototype_width

        # Height per prototype/cluster
        cluster_height = 1.0  # Base height per cluster
        fig_height = n_clusters * cluster_height

        # Create figure with grid
        fig = plt.figure(figsize=(fig_width, fig_height))

        # Set title and metric label based on hierarchy type
        if isinstance(hierarchy, KLClusterHierarchy):
            fig.suptitle("Hierarchical Clustering using KL Divergence", fontsize=12)
            metric_label = "KL Divergence"
            linkage_matrix = hierarchy.linkage_matrix
        elif isinstance(hierarchy, CoAssignmentClusterHierarchy):
            fig.suptitle("Hierarchical Clustering using Co-assignment", fontsize=12)

            # For co-assignment, use log-scale for better visualization
            # Check if most distances are near 0
            linkage_matrix = hierarchy.linkage_matrix.copy()
            distances = linkage_matrix[:, 2]
            if np.percentile(distances, 75) < 0.1:
                # Transform distances to better visualize when most are near 0
                linkage_matrix[:, 2] = -np.log(
                    1 - np.minimum(0.999, linkage_matrix[:, 2])
                )
                metric_label = "-log(Co-assignment)"
            else:
                metric_label = "1 - Co-assignment"
        else:
            fig.suptitle("Hierarchical Clustering", fontsize=12)
            metric_label = "Distance"
            linkage_matrix = hierarchy.linkage_matrix

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
        dendrogram_results = scipy.cluster.hierarchy.dendrogram(
            linkage_matrix,
            orientation="left",
            ax=dendrogram_ax,
            leaf_font_size=10,
            leaf_label_func=lambda x: f"Cluster {x}",
        )
        dendrogram_ax.set_xlabel(metric_label)

        leaf_order = dendrogram_results["leaves"]
        if leaf_order is None:
            raise ValueError("Failed to get leaf order from dendrogram.")

        leaf_order = leaf_order[::-1]

        for i, leaf_idx in enumerate(leaf_order):
            prototype_ax = fig.add_subplot(gs[i, 1])
            dataset.paint_observable(hierarchy.prototypes[leaf_idx], prototype_ax)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # pyright: ignore[reportArgumentType]
        return fig

    return plot_cluster_hierarchy


### Analysis ###


@dataclass(frozen=True)
class HierarchyAnalysis[T: ClusterHierarchy](Analysis[ClusteringDataset, HMoG, T], ABC):
    @override
    def generate(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: HMoG,
        epoch: int,
        params: Array,
    ) -> T:
        typed_params = model.natural_point(params)
        return generate_cluster_hierarchy(
            model, typed_params, self.artifact_type, dataset.train_data
        )

    @override
    def plot(self, artifact: T, dataset: ClusteringDataset) -> Figure:
        plotter = hierarchy_plotter(dataset)
        return plotter(artifact)


@dataclass(frozen=True)
class KLHierarchyAnalysis(HierarchyAnalysis[KLClusterHierarchy]):
    """Co-assignment probability-based hierarchical clustering analysis."""

    @property
    @override
    def artifact_type(self) -> type[KLClusterHierarchy]:
        return KLClusterHierarchy


@dataclass(frozen=True)
class CoAssignmentHierarchyAnalysis(HierarchyAnalysis[CoAssignmentClusterHierarchy]):
    """Co-assignment probability-based hierarchical clustering analysis."""

    @property
    @override
    def artifact_type(self) -> type[CoAssignmentClusterHierarchy]:
        return CoAssignmentClusterHierarchy
