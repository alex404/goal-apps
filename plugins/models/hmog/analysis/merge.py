"""Base class for HMoG implementations."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Callable, override

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
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
from apps.runtime.handler import Artifact, MetricDict

from ..base import HMoG
from .base import (
    STATS_LEVEL,
    cluster_accuracy,
    cluster_probabilities,
    get_component_prototypes,
)
from .hierarchy import (
    CoAssignmentClusterHierarchy,
    KLClusterHierarchy,
    get_cluster_hierarchy,
)

### Analyses ###


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
    from scipy.optimize import linear_sum_assignment

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


def hierarchy_to_mapping(
    linkage_matrix: NDArray[np.float64],
    n_clusters: int,
    n_classes: int,
) -> NDArray[np.int32]:
    """Convert hierarchical clustering to mapping matrix.

    Args:
        linkage_matrix: Hierarchical clustering linkage matrix
        n_clusters: Number of original clusters
        n_classes: Number of target classes

    Returns:
        Binary mapping matrix of shape (n_clusters, n_classes)
    """
    import scipy.cluster.hierarchy

    # Cut the tree to get n_classes
    cluster_ids = scipy.cluster.hierarchy.fcluster(
        linkage_matrix, n_classes, criterion="maxclust"
    )

    # Convert cluster IDs to one-hot mapping
    mapping = np.zeros((n_clusters, n_classes), dtype=np.int32)
    for i in range(n_clusters):
        # Cluster IDs from fcluster are 1-indexed
        class_id = cluster_ids[i] - 1
        mapping[i, class_id] = 1

    return mapping


### Merge Results ###


@dataclass(frozen=True)
class MergeResults(Artifact):
    """Base artifact containing merge results."""

    prototypes: list[Array]  # Original prototypes
    mapping: NDArray[np.int32]  # Mapping from clusters to classes
    train_accuracy: float  # Training accuracy after mapping
    train_nmi_score: float  # NMI score on training data
    test_accuracy: float  # Test accuracy after mapping
    test_nmi_score: float  # NMI score on test data
    valid_clusters: Array  # Indices of valid clusters used in mapping
    similarity_type: str  # Similarity metric used

    @override
    def save_to_hdf5(self, file: File) -> None:
        """Save merge results to HDF5 file."""
        # Save prototypes
        proto_group = file.create_group("prototypes")
        for i, proto in enumerate(self.prototypes):
            proto_group.create_dataset(f"{i}", data=np.array(proto))

        # Save mapping and valid clusters
        file.create_dataset("mapping", data=self.mapping)
        file.create_dataset("valid_clusters", data=np.array(self.valid_clusters))

        # Save metrics
        file.attrs["train_accuracy"] = self.train_accuracy
        file.attrs["train_nmi_score"] = self.train_nmi_score
        file.attrs["test_accuracy"] = self.test_accuracy
        file.attrs["test_nmi_score"] = self.test_nmi_score
        file.attrs["similarity_type"] = self.similarity_type

    @classmethod
    @override
    def load_from_hdf5(cls, file: File) -> MergeResults:
        """Load merge results from HDF5 file."""
        # Load prototypes
        proto_group = file["prototypes"]
        assert isinstance(proto_group, Group)
        n_protos = len(proto_group)
        prototypes = [jnp.array(proto_group[f"{i}"]) for i in range(n_protos)]

        # Load mapping and valid clusters
        mapping = np.array(file["mapping"][()])  # pyright: ignore[reportIndexIssue,reportArgumentType]
        valid_clusters = jnp.array(file["valid_clusters"][()])  # pyright: ignore[reportIndexIssue,reportArgumentType]

        # Load metrics
        train_accuracy = float(file.attrs["train_accuracy"])  # pyright: ignore[reportArgumentType]
        train_nmi_score = float(file.attrs["train_nmi_score"])  # pyright: ignore[reportArgumentType]
        test_accuracy = float(file.attrs["test_accuracy"])  # pyright: ignore[reportArgumentType]
        test_nmi_score = float(file.attrs["test_nmi_score"])  # pyright: ignore[reportArgumentType]
        similarity_type = str(file.attrs["similarity_type"])

        return cls(
            prototypes=prototypes,
            mapping=mapping,
            train_accuracy=train_accuracy,
            train_nmi_score=train_nmi_score,
            test_accuracy=test_accuracy,
            test_nmi_score=test_nmi_score,
            valid_clusters=valid_clusters,
            similarity_type=similarity_type,
        )


@dataclass(frozen=True)
class KLMergeResults(MergeResults):
    """KL divergence-based merge results."""


@dataclass(frozen=True)
class CoAssignmentMergeResults(MergeResults):
    """Co-assignment-based merge results."""


@dataclass(frozen=True)
class OptimalMergeResults(MergeResults):
    """Optimal assignment merge results."""


def get_merge_results[M: HMoG, MR: MergeResults](
    model: M,
    params: Point[Natural, M],
    dataset: ClusteringDataset,
    merge_type: type[MR],
    filter_empty_clusters: bool = True,
    min_cluster_size: float = 0.0005,
) -> MR:
    """Generate merge results using specified merge strategy."""
    import numpy as np
    import scipy.spatial.distance
    from sklearn.metrics import normalized_mutual_info_score

    prototypes = get_component_prototypes(model, params)

    # Get cluster probabilities for training data
    train_probs = cluster_probabilities(model, params.array, dataset.train_data)
    train_assignments = jnp.argmax(train_probs, axis=1)

    n_clusters = model.upr_hrm.n_categories
    n_classes = dataset.n_classes

    # Filter empty/near-empty clusters if requested
    valid_clusters = jnp.arange(n_clusters)
    if filter_empty_clusters:
        # Count assignments to each cluster
        cluster_counts = jnp.zeros(n_clusters)
        for i in range(n_clusters):
            cluster_counts = cluster_counts.at[i].set(jnp.sum(train_assignments == i))

        # Find clusters with sufficient data points
        min_count = max(1, int(min_cluster_size * len(dataset.train_data)))
        valid_clusters = jnp.where(cluster_counts >= min_count)[0]

        # Ensure we have at least min(n_clusters, n_classes) valid clusters
        min_required = min(n_clusters, n_classes)
        if len(valid_clusters) < min_required:
            valid_clusters = jnp.argsort(-cluster_counts)[:min_required]

    # Create a full mapping matrix of the original size
    full_mapping = jnp.zeros((n_clusters, n_classes), dtype=jnp.int32)

    # Filter train probabilities to valid clusters only
    filtered_train_probs = train_probs[:, valid_clusters]

    # Determine similarity type and compute mapping for valid clusters only
    if merge_type == KLMergeResults:
        # Get the KL-based hierarchy
        kl_hierarchy = get_cluster_hierarchy(
            model, params, KLClusterHierarchy, dataset.train_data
        )
        full_sim_matrix = np.array(kl_hierarchy.similarity_matrix)

        # Extract submatrix for valid clusters
        filtered_dist_matrix = full_sim_matrix[valid_clusters][:, valid_clusters]

        # Ensure diagonal is zero
        np.fill_diagonal(filtered_dist_matrix, 0)

        # Convert square distance matrix to condensed form
        n = filtered_dist_matrix.shape[0]
        condensed_distances = np.zeros(n * (n - 1) // 2)
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                condensed_distances[k] = filtered_dist_matrix[i, j]
                k += 1

        # Run hierarchical clustering
        filtered_linkage = scipy.cluster.hierarchy.linkage(
            condensed_distances, method="average"
        )

        # Get mapping for valid clusters
        filtered_mapping = hierarchy_to_mapping(
            filtered_linkage,  # pyright: ignore[reportArgumentType]
            len(valid_clusters),
            n_classes,
        )

        similarity_type = "kl"

    elif merge_type == CoAssignmentMergeResults:
        # Get the co-assignment hierarchy
        coassign_hierarchy = get_cluster_hierarchy(
            model, params, CoAssignmentClusterHierarchy, dataset.train_data
        )
        full_sim_matrix = np.array(coassign_hierarchy.similarity_matrix)

        # Extract submatrix for valid clusters
        filtered_sim_matrix = full_sim_matrix[valid_clusters][:, valid_clusters]

        # Convert similarity to distance
        filtered_dist_matrix = 1.0 - filtered_sim_matrix

        # Ensure diagonal is exactly zero
        np.fill_diagonal(filtered_dist_matrix, 0)

        # Convert to condensed form
        condensed_distances = scipy.spatial.distance.squareform(filtered_dist_matrix)

        # Compute linkage
        filtered_linkage = scipy.cluster.hierarchy.linkage(
            condensed_distances, method="average"
        )

        # Get mapping for valid clusters
        filtered_mapping = hierarchy_to_mapping(
            filtered_linkage,  # pyright: ignore[reportArgumentType]
            len(valid_clusters),
            n_classes,
        )

        similarity_type = "coassignment"

    elif merge_type == OptimalMergeResults:
        # This was already correctly filtering probabilities
        filtered_mapping = compute_optimal_mapping(
            filtered_train_probs, dataset.train_labels, n_classes
        )
        similarity_type = "optimal"
    else:
        raise TypeError(f"Unsupported merge type: {merge_type}")

    # Place the filtered mapping into the full mapping matrix
    for i, cluster_idx in enumerate(valid_clusters):
        full_mapping = full_mapping.at[cluster_idx].set(filtered_mapping[i])

    # Compute metrics on training data
    train_merged_probs = jnp.matmul(train_probs, full_mapping)
    train_merged_assignments = jnp.argmax(train_merged_probs, axis=1)
    train_accuracy = cluster_accuracy(dataset.train_labels, train_merged_assignments)
    train_nmi = normalized_mutual_info_score(
        np.array(dataset.train_labels), np.array(train_merged_assignments)
    )

    # Compute metrics on test data
    test_probs = cluster_probabilities(model, params.array, dataset.test_data)
    test_merged_probs = jnp.matmul(test_probs, full_mapping)
    test_merged_assignments = jnp.argmax(test_merged_probs, axis=1)
    test_accuracy = cluster_accuracy(dataset.test_labels, test_merged_assignments)
    test_nmi = normalized_mutual_info_score(
        np.array(dataset.test_labels), np.array(test_merged_assignments)
    )
    full_mapping = np.array(full_mapping, dtype=np.int32)

    return merge_type(
        prototypes=prototypes,
        mapping=full_mapping,
        train_accuracy=float(train_accuracy),
        train_nmi_score=float(train_nmi),
        test_accuracy=float(test_accuracy),
        test_nmi_score=float(test_nmi),
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

            # Plot each cluster in this class, left-justified
            for i, cluster_idx in enumerate(class_clusters):
                cluster_ax = fig.add_subplot(class_gs[0, i])
                dataset.paint_observable(results.prototypes[cluster_idx], cluster_ax)

        # Add overall title with metrics
        title = f"Merge Strategy: {results.similarity_type.capitalize()}\n"
        title += f"Train Accuracy: {results.train_accuracy:.3f}, Train NMI: {results.train_nmi_score:.3f}\n"
        title += f"Test Accuracy: {results.test_accuracy:.3f}, Test NMI: {results.test_nmi_score:.3f}\n"
        title += f"Using {len(results.valid_clusters)}/{n_clusters} valid clusters"

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig

    return plot_merge_results


### Analysis Classes ###


@dataclass(frozen=True)
class MergeAnalysis[T: MergeResults](Analysis[ClusteringDataset, HMoG, T], ABC):
    """KL divergence-based cluster merging analysis."""

    filter_empty_clusters: bool
    min_cluster_size: float

    @override
    def generate(
        self,
        model: HMoG,
        params: Array,
        dataset: ClusteringDataset,
        key: Array | None = None,
    ) -> T:
        typed_params = model.natural_point(params)
        return get_merge_results(
            model,
            typed_params,
            dataset,
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
            "Merging/Optimal Test Accuracy": (
                STATS_LEVEL,
                jnp.array(artifact.test_accuracy),
            ),
            "Merging/Optimal Test NMI": (
                STATS_LEVEL,
                jnp.array(artifact.test_nmi_score),
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
            "Merging/KL Test Accuracy": (
                STATS_LEVEL,
                jnp.array(artifact.test_accuracy),
            ),
            "Merging/KL Test NMI": (
                STATS_LEVEL,
                jnp.array(artifact.test_nmi_score),
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
            "Merging/CoAssignment Test Accuracy": (
                STATS_LEVEL,
                jnp.array(artifact.test_accuracy),
            ),
            "Merging/CoAssignment Test NMI": (
                STATS_LEVEL,
                jnp.array(artifact.test_nmi_score),
            ),
        }
