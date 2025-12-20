"""Co-assignment based hierarchical clustering analysis.

Requires models to implement:
- HasSoftAssignments (posterior_soft_assignments)
- CanComputePrototypes (compute_cluster_prototypes)
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


@dataclass(frozen=True)
class CoAssignmentHierarchy(Artifact):
    """Co-assignment based clustering hierarchy.

    The co-assignment matrix measures how often pairs of clusters
    are assigned to the same data points based on posterior probabilities.
    """

    prototypes: list[Array]
    linkage_matrix: NDArray[np.float64]
    distance_matrix: NDArray[np.float64]


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


def build_hierarchy_from_similarity(
    similarity_matrix: Array,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build hierarchical clustering from a similarity matrix.

    Args:
        similarity_matrix: Symmetric matrix where higher values = more similar

    Returns:
        Tuple of (linkage_matrix, distance_matrix)
    """
    # Convert to numpy
    sim_np = np.array(similarity_matrix, dtype=np.float64)

    # Convert similarity to distance (lower = more similar)
    dist_matrix = 1.0 - sim_np

    # Ensure non-negative distances
    min_off_diag = np.min(dist_matrix[~np.eye(dist_matrix.shape[0], dtype=bool)])
    if min_off_diag < 0:
        dist_matrix = dist_matrix - min_off_diag

    # Ensure perfect symmetry
    dist_matrix = (dist_matrix + dist_matrix.T) / 2

    # Force diagonal to exactly zero
    np.fill_diagonal(dist_matrix, 0.0)

    # Convert to condensed form for scipy
    dist_vector = scipy.spatial.distance.squareform(dist_matrix)

    # Compute linkage using average linkage (UPGMA)
    linkage_matrix = scipy.cluster.hierarchy.linkage(dist_vector, method="average")

    return linkage_matrix, dist_matrix


def plot_hierarchy_dendrogram(
    hierarchy: CoAssignmentHierarchy,
    dataset: ClusteringDataset,
) -> Figure:
    """Plot dendrogram with corresponding prototype visualizations."""
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
    fig.suptitle("Hierarchical Clustering (Co-assignment)", fontsize=12)

    # Transform distances for better visualization when most are near 0
    linkage_matrix = hierarchy.linkage_matrix.copy()
    distances = linkage_matrix[:, 2]
    if np.percentile(distances, 75) < 0.1:
        linkage_matrix[:, 2] = -np.log(1 - np.minimum(0.999, distances))
        metric_label = "-log(Co-assignment)"
    else:
        metric_label = "1 - Co-assignment"

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

    # Plot prototypes aligned with dendrogram leaves
    for i, leaf_idx in enumerate(leaf_order):
        prototype_ax = fig.add_subplot(gs[i, 1])
        dataset.paint_observable(hierarchy.prototypes[leaf_idx], prototype_ax)

    return fig


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

        # Build hierarchy
        linkage_matrix, distance_matrix = build_hierarchy_from_similarity(
            similarity_matrix
        )

        # Get prototypes for visualization
        prototypes = model.compute_cluster_prototypes(params)

        return CoAssignmentHierarchy(
            prototypes=prototypes,
            linkage_matrix=linkage_matrix,
            distance_matrix=distance_matrix,
        )

    @override
    def plot(
        self, artifact: CoAssignmentHierarchy, dataset: ClusteringDataset
    ) -> Figure:
        return plot_hierarchy_dendrogram(artifact, dataset)
