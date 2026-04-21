"""Hierarchical clustering analyses for clustering models.

Provides shared utilities for building and visualizing cluster hierarchies,
plus the co-assignment based analysis that works with any HasSoftAssignments model.
"""

from __future__ import annotations

import logging
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

from ....runtime import Artifact, DivergentTrainingError, RunHandler
from ...analysis import Analysis
from ..dataset import ClusteringDataset

log = logging.getLogger(__name__)


def get_valid_clusters(
    train_assignments: Array,
    n_clusters: int,
    n_classes: int,
    filter_empty_clusters: bool,
    min_cluster_size: float,
    n_train_samples: int,
) -> Array:
    """Get valid cluster indices, filtering empty/small clusters if requested.

    If the threshold-pass leaves fewer than ``min(n_clusters, n_classes)``
    clusters, falls back to the top-N by assignment count and logs a
    warning. The fallback keeps downstream merge/linkage runnable but is a
    strong signal that the model has collapsed — inspect before trusting.
    """
    valid_clusters = jnp.arange(n_clusters)

    if filter_empty_clusters:
        cluster_counts = jnp.zeros(n_clusters).at[train_assignments].add(1)

        min_count = max(1, int(min_cluster_size * n_train_samples))
        valid_clusters = jnp.where(cluster_counts >= min_count)[0]

        # Ensure we have at least min(n_clusters, n_classes) valid clusters
        min_required = min(n_clusters, n_classes)
        if len(valid_clusters) < min_required:
            log.warning(
                "Only %d/%d clusters passed min_cluster_size=%.6g (>= %d samples); falling back to top-%d by assignment count. Model may be collapsing — review cluster occupancy.",
                len(valid_clusters),
                n_clusters,
                min_cluster_size,
                min_count,
                min_required,
            )
            valid_clusters = jnp.argsort(-cluster_counts)[:min_required]

    return valid_clusters


# Base artifact


@dataclass(frozen=True)
class ClusterHierarchy(Artifact):
    """Base artifact for cluster hierarchy analysis.

    ``linkage_matrix`` and ``distance_matrix`` are computed on the filtered
    subset of clusters identified by ``valid_clusters``, so they have shape
    ``(len(valid_clusters) - 1, 4)`` and ``(len(valid_clusters),
    len(valid_clusters))`` respectively — NOT ``n_clusters``. Leaf index ``i``
    in the dendrogram corresponds to the original cluster
    ``valid_clusters[i]``. ``prototypes`` still lists all ``n_clusters``
    prototypes (unfiltered) for reference visualisation.

    Downstream merge analyses read ``valid_clusters`` from this artifact
    rather than re-deriving it, so hierarchy and merge filter on the same
    set.
    """

    prototypes: list[Array]
    linkage_matrix: NDArray[np.float64]
    distance_matrix: NDArray[np.float64]
    valid_clusters: Array
    """Indices (into ``prototypes``) of clusters included in the hierarchy."""


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

    # NaN/Inf in distance matrix indicates degenerate model parameters
    if not np.all(np.isfinite(dist_np)):
        raise DivergentTrainingError("Non-finite values in distance matrix")

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

    The hierarchy's ``linkage_matrix`` is defined over ``valid_clusters``
    (indices 0..len(valid_clusters)-1). To label leaves and fetch
    prototypes correctly, leaf index ``i`` must be translated to the
    original cluster ``valid_clusters[i]``.

    Args:
        hierarchy: Cluster hierarchy artifact
        dataset: Dataset for prototype visualization
        metric_label: Label for the x-axis (e.g., "KL Divergence", "1 - Co-assignment")
        title: Figure title
    """
    valid_clusters = np.asarray(hierarchy.valid_clusters)
    n_valid = len(valid_clusters)
    prototype_shape = dataset.observable_shape

    # Compute figure dimensions
    dendrogram_width = 6
    height, width = prototype_shape
    # Cap prototype panel width to avoid enormous figures for wide observable shapes
    prototype_width = min(width / max(height, 1) * dendrogram_width, 12)
    spacing = 4

    fig_width = dendrogram_width + spacing + prototype_width
    cluster_height = 1.0
    fig_height = min(n_valid * cluster_height, 60)

    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.suptitle(title, fontsize=12)

    # Create grid with two columns
    gs = GridSpec(
        n_valid,
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
        leaf_label_func=lambda x: f"Cluster {int(valid_clusters[x])}",
    )
    dendrogram_ax.set_xlabel(metric_label)

    leaf_order = dendrogram_results["leaves"]
    if leaf_order is None:
        raise ValueError("Failed to get leaf order from dendrogram.")
    leaf_order = leaf_order[::-1]

    # Plot prototypes aligned with dendrogram leaves, mapping filtered
    # leaf indices back to original cluster ids via valid_clusters.
    for i, leaf_idx in enumerate(leaf_order):
        original_cluster = int(valid_clusters[leaf_idx])
        prototype_ax = fig.add_subplot(gs[i, 1])
        dataset.paint_observable(hierarchy.prototypes[original_cluster], prototype_ax)

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
    denom = diag_sqrt[:, None] * diag_sqrt[None, :]
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
    - ClusteringModel (n_clusters)

    Dead clusters (with total responsibility below ``min_cluster_size``
    of training data) are pruned before the co-assignment matrix is
    built. This avoids the sqrt-of-zero normalization blowing up into
    NaN when a cluster receives no probability mass.
    """

    filter_empty_clusters: bool = True
    min_cluster_size: float = 0.0005

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
        """Generate hierarchy from posterior co-assignments over valid clusters."""
        responsibilities = model.posterior_soft_assignments(params, dataset.train_data)
        assignments = jnp.argmax(responsibilities, axis=1)

        valid_clusters = get_valid_clusters(
            assignments,
            model.n_clusters,
            dataset.n_classes if dataset.has_labels else model.n_clusters,
            self.filter_empty_clusters,
            self.min_cluster_size,
            len(dataset.train_data),
        )

        # Restrict to valid clusters before normalization — prevents
        # sqrt(0) / 0 division on dead clusters.
        sub_resp = responsibilities[:, valid_clusters]
        similarity_matrix = compute_co_assignment_matrix(sub_resp)
        distance_matrix = 1.0 - similarity_matrix

        linkage_matrix, cleaned_distance = build_hierarchy_from_distance(
            distance_matrix
        )

        prototypes = model.compute_cluster_prototypes(params)

        return CoAssignmentHierarchy(
            prototypes=prototypes,
            linkage_matrix=linkage_matrix,
            distance_matrix=cleaned_distance,
            valid_clusters=valid_clusters,
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
            valid_clusters=artifact.valid_clusters,
        )

        return plot_hierarchy_dendrogram(
            plot_hierarchy,
            dataset,
            metric_label=metric_label,
            title="Hierarchical Clustering (Co-assignment)",
        )
