"""Cluster merging analyses for clustering models.

Provides analyses that merge learned clusters into class predictions using
different strategies (optimal assignment, co-assignment hierarchy).
"""

from __future__ import annotations

from abc import ABC
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
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from ....runtime import STATS_NUM, Artifact, MetricDict, RunHandler
from ...analysis import Analysis
from ..dataset import ClusteringDataset
from .hierarchy import ClusterHierarchy, CoAssignmentHierarchy

STATS_LEVEL = jnp.array(STATS_NUM)


### Utility Functions ###


def compute_optimal_mapping(
    cluster_probs: Array,
    true_labels: Array,
    n_classes: int,
) -> NDArray[np.int32]:
    """Compute optimal mapping from clusters to classes using Hungarian algorithm.

    Args:
        cluster_probs: Cluster assignment probabilities (n_samples, n_clusters)
        true_labels: True class labels (n_samples,)
        n_classes: Number of classes

    Returns:
        Binary mapping matrix (n_clusters, n_classes)
    """
    cluster_assignments = jnp.argmax(cluster_probs, axis=1)
    n_clusters = cluster_probs.shape[1]

    # Create contingency matrix
    contingency = np.zeros((n_clusters, n_classes))
    for i in range(len(true_labels)):
        label = int(true_labels[i])
        cluster = int(cluster_assignments[i])
        contingency[cluster, label] += 1

    # Use Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(-contingency)

    # Create mapping matrix
    mapping = np.zeros((n_clusters, n_classes), dtype=np.int32)
    for i, j in zip(row_ind, col_ind):
        mapping[i, j] = 1

    return mapping


def get_valid_clusters(
    train_assignments: Array,
    n_clusters: int,
    n_classes: int,
    filter_empty_clusters: bool,
    min_cluster_size: float,
    n_train_samples: int,
) -> Array:
    """Get valid cluster indices, filtering empty/small clusters if requested."""
    valid_clusters = jnp.arange(n_clusters)

    if filter_empty_clusters:
        cluster_counts = jnp.zeros(n_clusters)
        for i in range(n_clusters):
            cluster_counts = cluster_counts.at[i].set(jnp.sum(train_assignments == i))

        min_count = max(1, int(min_cluster_size * n_train_samples))
        valid_clusters = jnp.where(cluster_counts >= min_count)[0]

        # Ensure we have at least min(n_clusters, n_classes) valid clusters
        min_required = min(n_clusters, n_classes)
        if len(valid_clusters) < min_required:
            valid_clusters = jnp.argsort(-cluster_counts)[:min_required]

    return valid_clusters


def hierarchy_to_mapping(
    hierarchy: ClusterHierarchy, valid_clusters: Array, n_classes: int
) -> NDArray[np.int32]:
    """Convert cluster hierarchy to mapping using hierarchical clustering."""
    # Extract submatrix for valid clusters only
    filtered_dist = hierarchy.distance_matrix[valid_clusters][:, valid_clusters]

    # Convert to condensed form and run hierarchical clustering
    condensed_distances = scipy.spatial.distance.squareform(filtered_dist)
    filtered_linkage = scipy.cluster.hierarchy.linkage(
        condensed_distances, method="average"
    )

    # Cut the tree to get n_classes clusters
    cluster_ids = scipy.cluster.hierarchy.fcluster(
        filtered_linkage, n_classes, criterion="maxclust"
    )

    # Convert cluster IDs to one-hot mapping matrix
    mapping = np.zeros((len(valid_clusters), n_classes), dtype=np.int32)
    for i in range(len(valid_clusters)):
        class_id = cluster_ids[i] - 1  # fcluster returns 1-indexed
        mapping[i, class_id] = 1

    return mapping


def compute_merge_metrics(
    mapping: Array, probs: Array, labels: Array
) -> tuple[float, float, float]:
    """Compute accuracy, NMI, and ARI for given mapping."""
    from ..metrics import cluster_accuracy

    merged_probs = jnp.matmul(probs, mapping)
    merged_assignments = jnp.argmax(merged_probs, axis=1)
    accuracy = float(cluster_accuracy(labels, merged_assignments))
    nmi = float(normalized_mutual_info_score(np.array(labels), np.array(merged_assignments)))
    ari = float(adjusted_rand_score(np.array(labels), np.array(merged_assignments)))
    return accuracy, nmi, ari


### Artifacts ###


@dataclass(frozen=True)
class MergeResults(Artifact):
    """Base artifact containing merge results."""

    prototypes: list[Array]
    mapping: NDArray[np.int32]
    train_accuracy: float
    train_nmi_score: float
    train_ari_score: float
    test_accuracy: float
    test_nmi_score: float
    test_ari_score: float
    valid_clusters: Array
    similarity_type: str


@dataclass(frozen=True)
class OptimalMergeResults(MergeResults):
    """Optimal assignment merge results."""


@dataclass(frozen=True)
class CoAssignmentMergeResults(MergeResults):
    """Co-assignment-based merge results."""


### Plotting ###


def plot_merge_results(results: MergeResults, dataset: ClusteringDataset) -> Figure:
    """Plot merge results showing clusters grouped by class."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    n_classes = results.mapping.shape[1]
    n_clusters = len(results.prototypes)

    # Group clusters by class (using only valid clusters)
    cluster_by_class: list[list[int]] = [[] for _ in range(n_classes)]
    for cluster_idx in results.valid_clusters:
        class_idx = int(np.argmax(results.mapping[cluster_idx]))
        cluster_by_class[class_idx].append(int(cluster_idx))

    # Calculate the maximum number of clusters in any class
    max_clusters_per_class = max(1, max(len(clusters) for clusters in cluster_by_class))

    # Create figure dimensions
    height, width = dataset.observable_shape
    ratio = height / width if height > 0 and width > 0 else 1

    fig_width = min(20, max(10, max_clusters_per_class * 1.5))
    fig_height = max(6, n_classes * 1.5 * ratio)

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(n_classes, 1, figure=fig, hspace=0.5)

    for class_idx in range(n_classes):
        class_clusters = cluster_by_class[class_idx]
        n_class_clusters = len(class_clusters)

        class_ax = fig.add_subplot(gs[class_idx])
        class_ax.set_axis_off()
        class_ax.set_title(
            f"Class {class_idx}: {n_class_clusters} clusters",
            fontsize=14,
            loc="left",
        )

        if n_class_clusters == 0:
            continue

        class_gs = GridSpecFromSubplotSpec(
            1, max_clusters_per_class, subplot_spec=gs[class_idx], wspace=0.1
        )

        for i, cluster_idx in enumerate(class_clusters):
            cluster_ax = fig.add_subplot(class_gs[0, i])
            dataset.paint_observable(results.prototypes[cluster_idx], cluster_ax)

    title = f"Merge Strategy: {results.similarity_type.capitalize()}\n"
    title += f"Train Accuracy: {results.train_accuracy:.3f}, Train NMI: {results.train_nmi_score:.3f}, Train ARI: {results.train_ari_score:.3f}\n"
    title += f"Test Accuracy: {results.test_accuracy:.3f}, Test NMI: {results.test_nmi_score:.3f}, Test ARI: {results.test_ari_score:.3f}\n"
    title += f"Using {len(results.valid_clusters)}/{n_clusters} valid clusters"

    fig.suptitle(title, fontsize=14)
    return fig


### Analysis Classes ###


@dataclass(frozen=True)
class MergeAnalysis[T: MergeResults](Analysis[ClusteringDataset, Any, T], ABC):
    """Base class for cluster merging analyses.

    Works with any model implementing:
    - HasSoftAssignments (posterior_soft_assignments)
    - CanComputePrototypes (compute_cluster_prototypes)
    - n_clusters property
    """

    filter_empty_clusters: bool = True
    min_cluster_size: float = 0.0005

    @override
    def plot(self, artifact: T, dataset: ClusteringDataset) -> Figure:
        return plot_merge_results(artifact, dataset)


@dataclass(frozen=True)
class OptimalMergeAnalysis(MergeAnalysis[OptimalMergeResults]):
    """Merge clusters using optimal (Hungarian) assignment to classes."""

    @property
    @override
    def artifact_type(self) -> type[OptimalMergeResults]:
        return OptimalMergeResults

    @override
    def generate(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: Any,
        epoch: int,
        params: Array,
    ) -> OptimalMergeResults:
        prototypes = model.compute_cluster_prototypes(params)
        train_probs = model.posterior_soft_assignments(params, dataset.train_data)
        train_assignments = jnp.argmax(train_probs, axis=1)

        n_clusters = model.n_clusters
        n_classes = dataset.n_classes

        valid_clusters = get_valid_clusters(
            train_assignments, n_clusters, n_classes,
            self.filter_empty_clusters, self.min_cluster_size, len(dataset.train_data),
        )

        filtered_train_probs = train_probs[:, valid_clusters]
        filtered_mapping = compute_optimal_mapping(
            filtered_train_probs, dataset.train_labels, n_classes
        )

        full_mapping = jnp.zeros((n_clusters, n_classes), dtype=jnp.int32)
        for i, cluster_idx in enumerate(valid_clusters):
            full_mapping = full_mapping.at[cluster_idx].set(filtered_mapping[i])

        train_metrics = compute_merge_metrics(full_mapping, train_probs, dataset.train_labels)
        test_probs = model.posterior_soft_assignments(params, dataset.test_data)
        test_metrics = compute_merge_metrics(full_mapping, test_probs, dataset.test_labels)

        return OptimalMergeResults(
            prototypes=prototypes,
            mapping=np.array(full_mapping, dtype=np.int32),
            train_accuracy=train_metrics[0],
            train_nmi_score=train_metrics[1],
            train_ari_score=train_metrics[2],
            test_accuracy=test_metrics[0],
            test_nmi_score=test_metrics[1],
            test_ari_score=test_metrics[2],
            valid_clusters=valid_clusters,
            similarity_type="optimal",
        )

    @override
    def metrics(self, artifact: OptimalMergeResults) -> MetricDict:
        return {
            "Merging/Optimal Train Accuracy": (STATS_LEVEL, jnp.array(artifact.train_accuracy)),
            "Merging/Optimal Train NMI": (STATS_LEVEL, jnp.array(artifact.train_nmi_score)),
            "Merging/Optimal Train ARI": (STATS_LEVEL, jnp.array(artifact.train_ari_score)),
            "Merging/Optimal Test Accuracy": (STATS_LEVEL, jnp.array(artifact.test_accuracy)),
            "Merging/Optimal Test NMI": (STATS_LEVEL, jnp.array(artifact.test_nmi_score)),
            "Merging/Optimal Test ARI": (STATS_LEVEL, jnp.array(artifact.test_ari_score)),
        }


@dataclass(frozen=True)
class CoAssignmentMergeAnalysis(MergeAnalysis[CoAssignmentMergeResults]):
    """Merge clusters using co-assignment hierarchy."""

    @property
    @override
    def artifact_type(self) -> type[CoAssignmentMergeResults]:
        return CoAssignmentMergeResults

    @override
    def generate(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: Any,
        epoch: int,
        params: Array,
    ) -> CoAssignmentMergeResults:
        prototypes = model.compute_cluster_prototypes(params)
        train_probs = model.posterior_soft_assignments(params, dataset.train_data)
        train_assignments = jnp.argmax(train_probs, axis=1)

        n_clusters = model.n_clusters
        n_classes = dataset.n_classes

        valid_clusters = get_valid_clusters(
            train_assignments, n_clusters, n_classes,
            self.filter_empty_clusters, self.min_cluster_size, len(dataset.train_data),
        )

        # Load co-assignment hierarchy and compute mapping
        hierarchy = handler.load_artifact(epoch, CoAssignmentHierarchy)
        filtered_mapping = hierarchy_to_mapping(hierarchy, valid_clusters, n_classes)

        full_mapping = jnp.zeros((n_clusters, n_classes), dtype=jnp.int32)
        for i, cluster_idx in enumerate(valid_clusters):
            full_mapping = full_mapping.at[cluster_idx].set(filtered_mapping[i])

        train_metrics = compute_merge_metrics(full_mapping, train_probs, dataset.train_labels)
        test_probs = model.posterior_soft_assignments(params, dataset.test_data)
        test_metrics = compute_merge_metrics(full_mapping, test_probs, dataset.test_labels)

        return CoAssignmentMergeResults(
            prototypes=prototypes,
            mapping=np.array(full_mapping, dtype=np.int32),
            train_accuracy=train_metrics[0],
            train_nmi_score=train_metrics[1],
            train_ari_score=train_metrics[2],
            test_accuracy=test_metrics[0],
            test_nmi_score=test_metrics[1],
            test_ari_score=test_metrics[2],
            valid_clusters=valid_clusters,
            similarity_type="coassignment",
        )

    @override
    def metrics(self, artifact: CoAssignmentMergeResults) -> MetricDict:
        return {
            "Merging/CoAssignment Train Accuracy": (STATS_LEVEL, jnp.array(artifact.train_accuracy)),
            "Merging/CoAssignment Train NMI": (STATS_LEVEL, jnp.array(artifact.train_nmi_score)),
            "Merging/CoAssignment Train ARI": (STATS_LEVEL, jnp.array(artifact.train_ari_score)),
            "Merging/CoAssignment Test Accuracy": (STATS_LEVEL, jnp.array(artifact.test_accuracy)),
            "Merging/CoAssignment Test NMI": (STATS_LEVEL, jnp.array(artifact.test_nmi_score)),
            "Merging/CoAssignment Test ARI": (STATS_LEVEL, jnp.array(artifact.test_ari_score)),
        }
