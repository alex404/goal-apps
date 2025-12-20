"""KL divergence based hierarchical clustering analysis for HMoG.

This is HMoG-specific because it requires computing KL divergence between
mixture components, which depends on the model's internal structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override

import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy
import scipy.spatial.distance
from goal.models import DifferentiableHMoG
from jax import Array
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray

from apps.interface import Analysis, ClusteringDataset
from apps.runtime import Artifact, RunHandler

from .base import get_component_prototypes, symmetric_kl_matrix


@dataclass(frozen=True)
class KLClusterHierarchy(Artifact):
    """KL divergence-based clustering hierarchy.

    Uses symmetric KL divergence between mixture components to build
    a hierarchical clustering of the learned clusters.
    """

    prototypes: list[Array]
    linkage_matrix: NDArray[np.float64]
    distance_matrix: NDArray[np.float64]


def build_hierarchy_from_distance(
    distance_matrix: Array,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build hierarchical clustering from a distance matrix.

    Args:
        distance_matrix: Symmetric matrix where lower values = more similar

    Returns:
        Tuple of (linkage_matrix, distance_matrix)
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


def plot_kl_hierarchy(
    hierarchy: KLClusterHierarchy,
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
    fig.suptitle("Hierarchical Clustering (KL Divergence)", fontsize=12)

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
    dendrogram_ax.set_xlabel("KL Divergence")

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
class KLHierarchyAnalysis(Analysis[ClusteringDataset, Any, KLClusterHierarchy]):
    """KL divergence based hierarchical clustering analysis.

    HMoG-specific: requires access to model.manifold for computing
    KL divergence between mixture components.
    """

    @property
    @override
    def artifact_type(self) -> type[KLClusterHierarchy]:
        return KLClusterHierarchy

    @override
    def generate(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: Any,
        epoch: int,
        params: Array,
    ) -> KLClusterHierarchy:
        """Generate hierarchy from KL divergence between components."""
        manifold: DifferentiableHMoG = model.manifold

        # Get prototypes for visualization
        prototypes = get_component_prototypes(manifold, params)

        # Compute symmetric KL divergence matrix
        kl_matrix = symmetric_kl_matrix(manifold, params)

        # Build hierarchy (KL is already a distance - lower = more similar)
        linkage_matrix, distance_matrix = build_hierarchy_from_distance(kl_matrix)

        return KLClusterHierarchy(
            prototypes=prototypes,
            linkage_matrix=linkage_matrix,
            distance_matrix=distance_matrix,
        )

    @override
    def plot(
        self, artifact: KLClusterHierarchy, dataset: ClusteringDataset
    ) -> Figure:
        return plot_kl_hierarchy(artifact, dataset)
