"""Base class for DifferentiableHMoG implementations."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, override

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy
import scipy.spatial.distance
from goal.models import DifferentiableHMoG
from jax import Array
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from apps.interface import (
    Analysis,
    ClusteringDataset,
)
from apps.interface.clustering.analyses import CoAssignmentHierarchy
from apps.runtime import Artifact, MetricDict, RunHandler

from .base import (
    STATS_LEVEL,
    cluster_accuracy,
    cluster_probabilities,
    get_component_prototypes,
)
from .hierarchy import KLClusterHierarchy

### Analyses ###


def _compute_optimal_mapping(
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
    # Get hard assignments
    cluster_assignments = jnp.argmax(cluster_probs, axis=1)

    # Number of clusters
    n_clusters = cluster_probs.shape[1]

    # Create contingency matrix
    contingency = np.zeros((n_clusters, n_classes))

    # Fill contingency matrix
    for i in range(len(true_labels)):
        label = int(true_labels[i])
        cluster = int(cluster_assignments[i])
        contingency[cluster, label] += 1

    # Use Hungarian algorithm for optimal assignment
    # Negate contingency matrix for maximization
    row_ind, col_ind = linear_sum_assignment(-contingency)

    # Create mapping matrix
    mapping = np.zeros((n_clusters, n_classes), dtype=np.int32)
    for i, j in zip(row_ind, col_ind):
        mapping[i, j] = 1

    return mapping


def _get_valid_clusters(
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


def distance_matrix_to_mapping(
    distance_matrix: NDArray[np.float64], valid_clusters: Array, n_classes: int
) -> NDArray[np.int32]:
    """Convert distance matrix to cluster mapping using hierarchical clustering on filtered subset."""

    # Extract submatrix for valid clusters only
    filtered_dist = distance_matrix[valid_clusters][:, valid_clusters]

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


def _compute_metrics(
    mapping: Array, probs: Array, labels: Array
) -> tuple[Array, float, float]:
    """Compute accuracy, NMI, and ARI for given mapping."""

    merged_probs = jnp.matmul(probs, mapping)
    merged_assignments = jnp.argmax(merged_probs, axis=1)
    accuracy = cluster_accuracy(labels, merged_assignments)
    nmi = normalized_mutual_info_score(np.array(labels), np.array(merged_assignments))
    ari = adjusted_rand_score(np.array(labels), np.array(merged_assignments))
    return accuracy, float(nmi), float(ari)


### Merge Results ###


@dataclass(frozen=True)
class MergeResults(Artifact):
    """Base artifact containing merge results."""

    prototypes: list[Array]  # Original prototypes
    mapping: NDArray[np.int32]  # Mapping from clusters to classes
    train_accuracy: float  # Training accuracy after mapping
    train_nmi_score: float  # NMI score on training data
    train_ari_score: float  # ARI score on training data
    test_accuracy: float  # Test accuracy after mapping
    test_nmi_score: float  # NMI score on test data
    test_ari_score: float  # ARI score on test data
    valid_clusters: Array  # Indices of valid clusters used in mapping
    similarity_type: str  # Similarity metric used


@dataclass(frozen=True)
class KLMergeResults(MergeResults):
    """KL divergence-based merge results."""


@dataclass(frozen=True)
class CoAssignmentMergeResults(MergeResults):
    """Co-assignment-based merge results."""


@dataclass(frozen=True)
class OptimalMergeResults(MergeResults):
    """Optimal assignment merge results."""


def generate_merge_results[MR: MergeResults](
    handler: RunHandler,
    dataset: ClusteringDataset,
    model: DifferentiableHMoG,
    epoch: int,
    params: Array,
    merge_type: type[MR],
    filter_empty_clusters: bool = True,
    min_cluster_size: float = 0.0005,
) -> MR:
    """Generate merge results using specified merge strategy."""

    prototypes = get_component_prototypes(model, params)
    train_probs = cluster_probabilities(model, params, dataset.train_data)
    train_assignments = jnp.argmax(train_probs, axis=1)

    n_clusters = model.pst_man.n_categories
    n_classes = dataset.n_classes

    # Filter empty clusters if requested
    valid_clusters = _get_valid_clusters(
        train_assignments,
        n_clusters,
        n_classes,
        filter_empty_clusters,
        min_cluster_size,
        len(dataset.train_data),
    )

    # Filter train probabilities to valid clusters only
    filtered_train_probs = train_probs[:, valid_clusters]

    # Create full mapping matrix
    full_mapping = jnp.zeros((n_clusters, n_classes), dtype=jnp.int32)

    # Determine mapping based on merge strategy
    if merge_type == KLMergeResults:
        kl_hierarchy = handler.load_artifact(epoch, KLClusterHierarchy)
        filtered_mapping = distance_matrix_to_mapping(
            kl_hierarchy.distance_matrix, valid_clusters, n_classes
        )
        similarity_type = "kl"

    elif merge_type == CoAssignmentMergeResults:
        coassign_hierarchy = handler.load_artifact(epoch, CoAssignmentHierarchy)
        filtered_mapping = distance_matrix_to_mapping(
            coassign_hierarchy.distance_matrix, valid_clusters, n_classes
        )
        similarity_type = "coassignment"

    elif merge_type == OptimalMergeResults:
        filtered_mapping = _compute_optimal_mapping(
            filtered_train_probs, dataset.train_labels, n_classes
        )
        similarity_type = "optimal"
    else:
        raise TypeError(f"Unsupported merge type: {merge_type}")

    # Place filtered mapping into full mapping matrix
    for i, cluster_idx in enumerate(valid_clusters):
        full_mapping = full_mapping.at[cluster_idx].set(filtered_mapping[i])

    # Compute metrics
    train_metrics = _compute_metrics(full_mapping, train_probs, dataset.train_labels)
    test_probs = cluster_probabilities(model, params, dataset.test_data)
    test_metrics = _compute_metrics(full_mapping, test_probs, dataset.test_labels)

    return merge_type(
        prototypes=prototypes,
        mapping=np.array(full_mapping, dtype=np.int32),
        train_accuracy=float(train_metrics[0]),
        train_nmi_score=float(train_metrics[1]),
        train_ari_score=float(train_metrics[2]),
        test_accuracy=float(test_metrics[0]),
        test_nmi_score=float(test_metrics[1]),
        test_ari_score=float(test_metrics[2]),
        valid_clusters=valid_clusters,
        similarity_type=similarity_type,
    )


def merge_results_plotter(
    dataset: ClusteringDataset,
) -> Callable[[MergeResults], Figure]:
    """Create a visualization of merge results."""

    def plot_merge_results(results: MergeResults) -> Figure:
        """Plot merge results showing clusters grouped by class."""
        from matplotlib.gridspec import GridSpecFromSubplotSpec

        n_classes = results.mapping.shape[1]
        n_clusters = len(results.prototypes)

        # Group clusters by class (using only valid clusters)
        cluster_by_class = [[] for _ in range(n_classes)]
        for cluster_idx in results.valid_clusters:
            class_idx = np.argmax(results.mapping[cluster_idx])
            cluster_by_class[class_idx].append(cluster_idx)

        # Calculate the maximum number of clusters in any class
        max_clusters_per_class = max(
            1, max(len(clusters) for clusters in cluster_by_class)
        )

        # Create figure dimensions
        height, width = dataset.observable_shape
        ratio = height / width if height > 0 and width > 0 else 1

        # Make figure width based on max_clusters_per_class
        fig_width = min(20, max(10, max_clusters_per_class * 1.5))
        # Height based on number of classes
        fig_height = max(6, n_classes * 1.5 * ratio)

        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = GridSpec(n_classes, 1, figure=fig, hspace=0.5)

        # Plot each class row
        for class_idx in range(n_classes):
            class_clusters = cluster_by_class[class_idx]
            n_class_clusters = len(class_clusters)

            # Create row for this class
            class_ax = fig.add_subplot(gs[class_idx])
            class_ax.set_axis_off()

            # Set title, larger and left-justified
            class_ax.set_title(
                f"Class {class_idx}: {n_class_clusters} clusters",
                fontsize=14,
                loc="left",
            )

            if n_class_clusters == 0:
                # Skip creating subplots if no clusters
                continue

            # Create sub-grid for clusters in this class
            class_gs = GridSpecFromSubplotSpec(
                1, max_clusters_per_class, subplot_spec=gs[class_idx], wspace=0.1
            )

            # Plot each cluster in this class using clean paint_observable
            for i, cluster_idx in enumerate(class_clusters):
                cluster_ax = fig.add_subplot(class_gs[0, i])
                dataset.paint_observable(results.prototypes[cluster_idx], cluster_ax)

        # Add overall title with metrics
        title = f"Merge Strategy: {results.similarity_type.capitalize()}\n"
        title += f"Train Accuracy: {results.train_accuracy:.3f}, Train NMI: {results.train_nmi_score:.3f}, Train ARI: {results.train_ari_score:.3f}\n"
        title += f"Test Accuracy: {results.test_accuracy:.3f}, Test NMI: {results.test_nmi_score:.3f}, Test ARI: {results.test_ari_score:.3f}\n"
        title += f"Using {len(results.valid_clusters)}/{n_clusters} valid clusters"

        fig.suptitle(title, fontsize=14)
        return fig

    return plot_merge_results


### Analysis Classes ###


@dataclass(frozen=True)
class MergeAnalysis[T: MergeResults](Analysis[ClusteringDataset, Any, T], ABC):
    """Cluster merging analysis base class."""

    filter_empty_clusters: bool
    min_cluster_size: float

    @override
    def generate(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: Any,
        epoch: int,
        params: Array,
    ) -> T:
        return generate_merge_results(
            handler,
            dataset,
            model.manifold,
            epoch,
            params,
            self.artifact_type,
            self.filter_empty_clusters,
            self.min_cluster_size,
        )

    @override
    def plot(self, artifact: T, dataset: ClusteringDataset) -> Figure:
        plotter = merge_results_plotter(dataset)
        return plotter(artifact)


@dataclass(frozen=True)
class OptimalMergeAnalysis(MergeAnalysis[OptimalMergeResults]):
    @property
    @override
    def artifact_type(self) -> type[OptimalMergeResults]:
        return OptimalMergeResults

    @override
    def metrics(self, artifact: OptimalMergeResults) -> MetricDict:
        """Return metrics for optimal merge results."""
        return {
            "Merging/Optimal Train Accuracy": (
                STATS_LEVEL,
                jnp.array(artifact.train_accuracy),
            ),
            "Merging/Optimal Train NMI": (
                STATS_LEVEL,
                jnp.array(artifact.train_nmi_score),
            ),
            "Merging/Optimal Train ARI": (
                STATS_LEVEL,
                jnp.array(artifact.train_ari_score),
            ),
            "Merging/Optimal Test Accuracy": (
                STATS_LEVEL,
                jnp.array(artifact.test_accuracy),
            ),
            "Merging/Optimal Test NMI": (
                STATS_LEVEL,
                jnp.array(artifact.test_nmi_score),
            ),
            "Merging/Optimal Test ARI": (
                STATS_LEVEL,
                jnp.array(artifact.test_ari_score),
            ),
        }


@dataclass(frozen=True)
class KLMergeAnalysis(MergeAnalysis[KLMergeResults]):
    @property
    @override
    def artifact_type(self) -> type[KLMergeResults]:
        return KLMergeResults

    @override
    def metrics(self, artifact: KLMergeResults) -> MetricDict:
        """Return metrics for KL merge results."""
        return {
            "Merging/KL Train Accuracy": (
                STATS_LEVEL,
                jnp.array(artifact.train_accuracy),
            ),
            "Merging/KL Train NMI": (
                STATS_LEVEL,
                jnp.array(artifact.train_nmi_score),
            ),
            "Merging/KL Train ARI": (
                STATS_LEVEL,
                jnp.array(artifact.train_ari_score),
            ),
            "Merging/KL Test Accuracy": (
                STATS_LEVEL,
                jnp.array(artifact.test_accuracy),
            ),
            "Merging/KL Test NMI": (
                STATS_LEVEL,
                jnp.array(artifact.test_nmi_score),
            ),
            "Merging/KL Test ARI": (
                STATS_LEVEL,
                jnp.array(artifact.test_ari_score),
            ),
        }


@dataclass(frozen=True)
class CoAssignmentMergeAnalysis(MergeAnalysis[CoAssignmentMergeResults]):
    @property
    @override
    def artifact_type(self) -> type[CoAssignmentMergeResults]:
        return CoAssignmentMergeResults

    @override
    def metrics(self, artifact: CoAssignmentMergeResults) -> MetricDict:
        """Return metrics for co-assignment merge results."""
        return {
            "Merging/CoAssignment Train Accuracy": (
                STATS_LEVEL,
                jnp.array(artifact.train_accuracy),
            ),
            "Merging/CoAssignment Train NMI": (
                STATS_LEVEL,
                jnp.array(artifact.train_nmi_score),
            ),
            "Merging/CoAssignment Train ARI": (
                STATS_LEVEL,
                jnp.array(artifact.train_ari_score),
            ),
            "Merging/CoAssignment Test Accuracy": (
                STATS_LEVEL,
                jnp.array(artifact.test_accuracy),
            ),
            "Merging/CoAssignment Test NMI": (
                STATS_LEVEL,
                jnp.array(artifact.test_nmi_score),
            ),
            "Merging/CoAssignment Test ARI": (
                STATS_LEVEL,
                jnp.array(artifact.test_ari_score),
            ),
        }
