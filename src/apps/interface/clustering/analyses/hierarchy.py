"""Hierarchical clustering analyses for clustering models.

Provides shared utilities for building and visualizing cluster hierarchies,
plus the co-assignment based analysis that works with any HasSoftAssignments model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy
import scipy.spatial.distance
from jax import Array
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray

from ....runtime import Artifact, RunHandler
from ...analysis import Analysis
from ..dataset import ClusteringDataset

# Base artifact


@dataclass(frozen=True)
class ClusterHierarchy(Artifact):
    """Base artifact for cluster hierarchy analysis.

    Contains the hierarchical clustering results that can be used
    for visualization and downstream analyses (like merge).
    """

    prototypes: list[Array]
    linkage_matrix: NDArray[np.float64]
    distance_matrix: NDArray[np.float64]


@dataclass(frozen=True)
class CoAssignmentHierarchy(ClusterHierarchy):
    """Co-assignment based clustering hierarchy."""

    pass


# Shared utilities


def build_hierarchy_from_distance(
    distance_matrix: Array,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build hierarchical clustering from a distance matrix.

    Args:
        distance_matrix: Symmetric matrix where lower values = more similar

    Returns:
        Tuple of (linkage_matrix, cleaned_distance_matrix)
    """
    dist_np = np.array(distance_matrix, dtype=np.float64)

    # Ensure non-negative distances
    min_off_diag = np.min(dist_np[~np.eye(dist_np.shape[0], dtype=bool)])
    if min_off_diag < 0:
        dist_np = dist_np - min_off_diag

    # Ensure perfect symmetry
    dist_np = (dist_np + dist_np.T) / 2

    # Force diagonal to exactly zero
    np.fill_diagonal(dist_np, 0.0)

    # Convert to condensed form for scipy
    dist_vector = scipy.spatial.distance.squareform(dist_np)

    # Compute linkage using average linkage (UPGMA)
    linkage_matrix = scipy.cluster.hierarchy.linkage(dist_vector, method="average")

    return linkage_matrix, dist_np


def plot_hierarchy_dendrogram(
    hierarchy: ClusterHierarchy,
    dataset: ClusteringDataset,
    metric_label: str,
    title: str,
) -> Figure:
    """Plot dendrogram with corresponding prototype visualizations.

    Args:
        hierarchy: Cluster hierarchy artifact
        dataset: Dataset for prototype visualization
        metric_label: Label for the x-axis (e.g., "KL Divergence", "1 - Co-assignment")
        title: Figure title
    """
    n_clusters = len(hierarchy.prototypes)
    prototype_shape = dataset.observable_shape

    # Compute figure dimensions
    dendrogram_width = 6
    height, width = prototype_shape
    prototype_width = width / height * dendrogram_width
    spacing = 4

    fig_width = dendrogram_width + spacing + prototype_width
    cluster_height = 1.0
    fig_height = n_clusters * cluster_height

    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.suptitle(title, fontsize=12)

    # Create grid with two columns
    gs = GridSpec(
        n_clusters,
        2,
        width_ratios=[dendrogram_width, prototype_width],
        wspace=spacing / fig_width,
        figure=fig,
    )

    # Plot dendrogram
    dendrogram_ax = fig.add_subplot(gs[:, 0])
    dendrogram_results = scipy.cluster.hierarchy.dendrogram(
        hierarchy.linkage_matrix,
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

    # Plot prototypes aligned with dendrogram leaves
    for i, leaf_idx in enumerate(leaf_order):
        prototype_ax = fig.add_subplot(gs[i, 1])
        dataset.paint_observable(hierarchy.prototypes[leaf_idx], prototype_ax)

    return fig


# Co-assignment specific


def compute_co_assignment_matrix(responsibilities: Array) -> Array:
    """Compute co-assignment matrix from posterior responsibilities.

    Args:
        responsibilities: Array of shape (n_samples, n_clusters) where
            responsibilities[i, k] = p(z=k | x=i)

    Returns:
        Normalized co-assignment matrix of shape (n_clusters, n_clusters)
        where entry (i, j) measures how often clusters i and j are
        assigned to the same data points.
    """
    # co_assignment[i,j] = sum_x p(z=i|x) * p(z=j|x)
    co_assignment = responsibilities.T @ responsibilities

    # Normalize to get correlation-like measure
    diag_sqrt = jnp.sqrt(jnp.diag(co_assignment))
    denom = diag_sqrt[:, None] * diag_sqrt[None, :] + 1e-12
    normalized = co_assignment / denom

    # Ensure perfect symmetry
    return 0.5 * (normalized + normalized.T)


@dataclass(frozen=True)
class CoAssignmentHierarchyAnalysis(
    Analysis[ClusteringDataset, Any, CoAssignmentHierarchy]
):
    """Hierarchical clustering analysis based on posterior co-assignments.

    Works with any model implementing:
    - HasSoftAssignments (posterior_soft_assignments)
    - CanComputePrototypes (compute_cluster_prototypes)
    """

    @property
    @override
    def artifact_type(self) -> type[CoAssignmentHierarchy]:
        return CoAssignmentHierarchy

    @override
    def generate(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: Any,
        epoch: int,
        params: Array,
    ) -> CoAssignmentHierarchy:
        """Generate hierarchy from posterior co-assignments."""
        # Get responsibilities from model
        responsibilities = model.posterior_soft_assignments(params, dataset.train_data)

        # Compute co-assignment similarity matrix
        similarity_matrix = compute_co_assignment_matrix(responsibilities)

        # Convert similarity to distance
        distance_matrix = 1.0 - similarity_matrix

        # Build hierarchy
        linkage_matrix, cleaned_distance = build_hierarchy_from_distance(
            distance_matrix
        )

        # Get prototypes for visualization
        prototypes = model.compute_cluster_prototypes(params)

        return CoAssignmentHierarchy(
            prototypes=prototypes,
            linkage_matrix=linkage_matrix,
            distance_matrix=cleaned_distance,
        )

    @override
    def plot(
        self, artifact: CoAssignmentHierarchy, dataset: ClusteringDataset
    ) -> Figure:
        # Transform for visualization when most distances are near 0
        linkage_copy = artifact.linkage_matrix.copy()
        distances = linkage_copy[:, 2]
        if np.percentile(distances, 75) < 0.1:
            linkage_copy[:, 2] = -np.log(1 - np.minimum(0.999, distances))
            metric_label = "-log(Co-assignment)"
        else:
            metric_label = "1 - Co-assignment"

        # Create a temporary hierarchy with transformed linkage for plotting
        plot_hierarchy = CoAssignmentHierarchy(
            prototypes=artifact.prototypes,
            linkage_matrix=linkage_copy,
            distance_matrix=artifact.distance_matrix,
        )

        return plot_hierarchy_dendrogram(
            plot_hierarchy,
            dataset,
            metric_label=metric_label,
            title="Hierarchical Clustering (Co-assignment)",
        )
