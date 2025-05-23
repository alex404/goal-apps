"""Base class for HMoG implementations."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, override

import jax
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

from apps.configs import STATS_NUM
from apps.plugins import (
    ClusteringDataset,
)
from apps.runtime.handler import Artifact, MetricDict, RunHandler
from apps.runtime.logger import JaxLogger

from .analysis import (
    cluster_accuracy,
    cluster_assignments,
    cluster_probabilities,
    compute_optimal_mapping,
    get_component_prototypes,
    hierarchy_to_mapping,
    posterior_co_assignment_matrix,
    symmetric_kl_matrix,
)
from .base import HMoG

### Analysis Args ###


@dataclass(frozen=True)
class AnalysisArgs:
    """Arguments for HMoG analysis."""

    from_scratch: bool
    epoch: int | None


STATS_LEVEL = jnp.array(STATS_NUM)
INFO_LEVEL = jnp.array(logging.INFO)


### Generative Examples ###


@dataclass(frozen=True)
class GenerativeExamples(Artifact):
    """Collection of generated samples from the model."""

    samples: Array  # Array of shape (n_samples, data_dim)

    @override
    def save_to_hdf5(self, file: File) -> None:
        """Save generated samples to HDF5 file."""
        file.create_dataset("samples", data=np.array(self.samples))

    @classmethod
    @override
    def load_from_hdf5(cls, file: File) -> GenerativeExamples:
        """Load generated samples from HDF5 file."""
        samples = jnp.array(file["samples"][()])
        return cls(samples=samples)


def generate_examples[M: HMoG](
    model: M,
    params: Point[Natural, M],
    n_samples: int,
    key: Array,
) -> GenerativeExamples:
    """Generate sample examples from the model.

    Args:
        model: HMoG model
        params: Model parameters
        n_samples: Number of samples to generate
        key: Random key for sampling

    Returns:
        Collection of generated samples
    """
    samples = model.observable_sample(key, params, n_samples)
    return GenerativeExamples(samples=samples)


def generative_examples_plotter(
    dataset: ClusteringDataset,
) -> Callable[[GenerativeExamples], Figure]:
    """Create a grid of generated samples visualizations."""

    def plot_generative_examples(examples: GenerativeExamples) -> Figure:
        n_samples = min(
            36, examples.samples.shape[0]
        )  # Limit to 36 samples for display

        # Calculate grid dimensions (approximate square grid)
        grid_size = int(np.ceil(np.sqrt(n_samples)))
        height, width = dataset.observable_shape

        # Scale figure size based on observable shape
        fig_width = 2 * grid_size * (width / max(height, width))
        fig_height = 2 * grid_size * (height / max(height, width))

        # Create figure with subplots
        fig, axes = plt.subplots(
            grid_size, grid_size, figsize=(fig_width, fig_height), squeeze=False
        )

        # Plot each sample
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                ax = axes[i, j]
                if idx < n_samples:
                    dataset.paint_observable(examples.samples[idx], ax)
                else:
                    ax.axis("off")  # Hide empty plots

        plt.suptitle("Generated Samples from HMoG Model", fontsize=14)
        plt.tight_layout()
        return fig

    return plot_generative_examples


### ClusterCollection ###


@dataclass(frozen=True)
class ClusterStatistics(Artifact):
    """Collection of cluster prototypes with their members."""

    prototypes: list[Array]  # list of prototypes
    members: list[Array]  # list of (n_members, data_dim)

    @override
    def save_to_hdf5(self, file: File) -> None:
        """Save cluster collection to HDF5 file."""

        # Create groups for prototypes and members
        proto_group = file.create_group("prototypes")
        members_group = file.create_group("members")

        # Save each prototype
        for i, proto in enumerate(self.prototypes):
            proto_group.create_dataset(f"{i}", data=np.array(proto))

        # Save each member array
        for i, member_array in enumerate(self.members):
            members_group.create_dataset(f"{i}", data=np.array(member_array))

    @classmethod
    @override
    def load_from_hdf5(cls, file: File) -> ClusterStatistics:
        """Load cluster collection from HDF5 file."""

        # Load prototypes
        proto_group = file["prototypes"]
        assert isinstance(proto_group, Group)
        n_clusters = len(proto_group)

        prototypes = [jnp.array(proto_group[f"{i}"]) for i in range(n_clusters)]

        # Load members
        members_group = file["members"]
        assert isinstance(members_group, Group)

        members = [jnp.array(members_group[f"{i}"]) for i in range(n_clusters)]

        return cls(prototypes=prototypes, members=members)


def get_cluster_statistics[
    M: HMoG,
](
    model: M,
    dataset: ClusteringDataset,
    params: Point[Natural, M],
) -> ClusterStatistics:
    """Generate collection of clusters with their members."""

    train_data = dataset.train_data
    assignments = cluster_assignments(model, params.array, train_data)
    prototypes = get_component_prototypes(model, params)

    # Create cluster collections
    cluster_members = []

    for i in range(model.upr_hrm.n_categories):
        # Get members for this cluster
        cluster_mask = assignments == i
        members = train_data[cluster_mask]

        # Limit number of members if needed
        cluster_members.append(members)

    return ClusterStatistics(
        prototypes=prototypes,
        members=cluster_members,
    )


def cluster_statistics_plotter(
    dataset: ClusteringDataset,
) -> Callable[[ClusterStatistics], Figure]:
    """Create a grid of cluster prototype visualizations."""

    def plot_cluster_statistics(collection: ClusterStatistics) -> Figure:
        n_clusters = len(collection.prototypes)

        grid_shape = int(np.ceil(np.sqrt(n_clusters)))
        cluster_rows, cluster_cols = dataset.cluster_shape
        # normalize cluster shape
        cluster_rows /= np.max([cluster_rows, cluster_cols])
        cluster_cols /= np.max([cluster_rows, cluster_cols])
        scl = 5
        # Create figure
        fig = plt.figure(
            figsize=(scl * grid_shape * cluster_cols, scl * grid_shape * cluster_rows)
        )
        gs = GridSpec(grid_shape, grid_shape, figure=fig)

        # Plot each cluster
        for i in range(n_clusters):
            ax = fig.add_subplot(gs[i // grid_shape, i % grid_shape])
            dataset.paint_cluster(
                cluster_id=i,
                prototype=collection.prototypes[i],
                members=collection.members[i],
                axes=ax,
            )

        plt.tight_layout()
        return fig

    return plot_cluster_statistics


### Cluster Hierarchies ###


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
        linkage_matrix = np.array(file["linkage_matrix"][()])  # pyright: ignore[reportIndexIssue]
        similarity_matrix = jnp.array(file["similarity_matrix"][()])  # pyright: ignore[reportIndexIssue]

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


def get_cluster_hierarchy[M: HMoG, C: ClusterHierarchy](
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
    elif hierarchy_type == CoAssignmentClusterHierarchy:
        similarity_matrix = posterior_co_assignment_matrix(model, params, data)
    else:
        raise TypeError(f"Unsupported hierarchy type: {hierarchy_type}")

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
        linkage_matrix=linkage_matrix,
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

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        return fig

    return plot_cluster_hierarchy


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
        mapping = np.array(file["mapping"][()])
        valid_clusters = jnp.array(file["valid_clusters"][()])

        # Load metrics
        train_accuracy = float(file.attrs["train_accuracy"])
        train_nmi_score = float(file.attrs["train_nmi_score"])
        test_accuracy = float(file.attrs["test_accuracy"])
        test_nmi_score = float(file.attrs["test_nmi_score"])
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
    import scipy.cluster.hierarchy
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
            filtered_linkage, len(valid_clusters), n_classes
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
            filtered_linkage, len(valid_clusters), n_classes
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


### Loading Matrix Artifacts ###


@dataclass(frozen=True)
class LoadingMatrixArtifact(Artifact):
    """Loading matrix visualizations in both natural and mean parameterizations."""

    natural_loadings: Array  # Shape: (data_dim, latent_dim)
    mean_loadings: Array  # Shape: (data_dim, latent_dim)

    @override
    def save_to_hdf5(self, file: File) -> None:
        """Save loading matrices to HDF5 file."""
        file.create_dataset("natural_loadings", data=np.array(self.natural_loadings))
        file.create_dataset("mean_loadings", data=np.array(self.mean_loadings))

    @classmethod
    @override
    def load_from_hdf5(cls, file: File) -> LoadingMatrixArtifact:
        """Load loading matrices from HDF5 file."""
        natural_loadings = jnp.array(file["natural_loadings"][()])
        mean_loadings = jnp.array(file["mean_loadings"][()])
        return cls(natural_loadings=natural_loadings, mean_loadings=mean_loadings)


def get_loading_matrices[M: HMoG](
    model: M,
    params: Point[Natural, M],
) -> LoadingMatrixArtifact:
    """Extract loading matrices in both natural and mean parameterizations."""
    # Extract the interaction parameters
    obs_params, int_params, _ = model.split_params(params)

    # Get natural loadings (directly from the parameters)
    natural_loadings = model.int_man.to_dense(int_params)

    # Convert to mean parameterization for interpretability
    obs_loc, obs_prs = model.obs_man.split_params(obs_params)
    obs_prs_dense = model.obs_man.cov_man.to_dense(obs_prs)

    # In mean coordinates, the loading matrix is Σ_x * Λ
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
        data_dim, latent_dim = artifact.mean_loadings.shape

        # Create figure for all latent dimensions
        # We'll show natural and mean parameterizations side by side
        fig_width = 5 * min(6, latent_dim)  # Limit width for many latent dims
        fig_height = 2 * math.ceil(
            latent_dim / 3
        )  # Adjust height based on num dimensions

        fig = plt.figure(figsize=(fig_width, fig_height))

        # Create grid layout
        grid = GridSpec(math.ceil(latent_dim / 3), 6, figure=fig)

        # Add title
        fig.suptitle("Loading Matrix Visualization", fontsize=16)

        # Plot each latent dimension
        for i in range(latent_dim):
            row = i // 3
            col = (i % 3) * 2

            # Create axes for this latent dimension
            ax_natural = fig.add_subplot(grid[row, col])
            ax_mean = fig.add_subplot(grid[row, col + 1])

            # Extract patterns for this latent dimension
            natural_pattern = artifact.natural_loadings[:, i]
            mean_pattern = artifact.mean_loadings[:, i]

            # Use dataset visualization to plot
            dataset.paint_observable(natural_pattern, ax_natural)
            dataset.paint_observable(mean_pattern, ax_mean)

            # Add titles
            ax_natural.set_title(f"Natural Z{i + 1}")
            ax_mean.set_title(f"Mean Z{i + 1}")

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        return fig

    return plot_loading_matrices


### Log Artifacts ###


def log_artifacts[M: HMoG](
    handler: RunHandler,
    dataset: ClusteringDataset,
    logger: JaxLogger,
    model: M,
    epoch: int,
    params: Point[Natural, M] | None = None,
    key: Array | None = None,
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

    if key is None:
        key = jax.random.PRNGKey(42)

    # from_scratch if params is provided
    if params is not None:
        handler.save_params(params.array, epoch)
        cluster_statistics = get_cluster_statistics(model, dataset, params)
        kl_hierarchy = get_cluster_hierarchy(
            model, params, KLClusterHierarchy, dataset.train_data
        )
        co_hierarchy = get_cluster_hierarchy(
            model, params, CoAssignmentClusterHierarchy, dataset.train_data
        )
        gen_examples = generate_examples(model, params, 25, key)
        loading_matrices = get_loading_matrices(model, params)
    else:
        cluster_statistics = handler.load_artifact(epoch, ClusterStatistics)
        kl_hierarchy = handler.load_artifact(epoch, KLClusterHierarchy)
        co_hierarchy = handler.load_artifact(epoch, CoAssignmentClusterHierarchy)
        gen_examples = handler.load_artifact(epoch, GenerativeExamples)
        loading_matrices = handler.load_artifact(epoch, LoadingMatrixArtifact)

    # Plot and save
    plot_clusters_statistics = cluster_statistics_plotter(dataset)
    plot_hierarchy = hierarchy_plotter(dataset)
    plot_examples = generative_examples_plotter(dataset)
    plot_loadings = loading_matrix_plotter(dataset)

    logger.log_artifact(handler, epoch, cluster_statistics, plot_clusters_statistics)
    logger.log_artifact(handler, epoch, kl_hierarchy, plot_hierarchy)
    logger.log_artifact(handler, epoch, co_hierarchy, plot_hierarchy)
    logger.log_artifact(handler, epoch, gen_examples, plot_examples)
    logger.log_artifact(handler, epoch, loading_matrices, plot_loadings)

    if dataset.has_labels:
        if params is not None:
            kl_merge_results = get_merge_results(model, params, dataset, KLMergeResults)
            co_merge_results = get_merge_results(
                model, params, dataset, CoAssignmentMergeResults
            )
            op_merge_results = get_merge_results(
                model, params, dataset, OptimalMergeResults
            )
        else:
            kl_merge_results = handler.load_artifact(epoch, KLMergeResults)
            co_merge_results = handler.load_artifact(epoch, CoAssignmentMergeResults)
            op_merge_results = handler.load_artifact(epoch, OptimalMergeResults)

        plot_merge_results = merge_results_plotter(dataset)
        logger.log_artifact(handler, epoch, kl_merge_results, plot_merge_results)
        logger.log_artifact(handler, epoch, co_merge_results, plot_merge_results)
        logger.log_artifact(handler, epoch, op_merge_results, plot_merge_results)
        # Log merge metrics
        metrics: MetricDict = {
            "Merging/KL Train Accuracy": (
                STATS_LEVEL,
                jnp.array(kl_merge_results.train_accuracy),
            ),
            "Merging/KL Train NMI": (
                STATS_LEVEL,
                jnp.array(kl_merge_results.train_nmi_score),
            ),
            "Merging/KL Test Accuracy": (
                STATS_LEVEL,
                jnp.array(kl_merge_results.test_accuracy),
            ),
            "Merging/KL Test NMI": (
                STATS_LEVEL,
                jnp.array(kl_merge_results.test_nmi_score),
            ),
            "Merging/CoAssignment Train Accuracy": (
                STATS_LEVEL,
                jnp.array(co_merge_results.train_accuracy),
            ),
            "Merging/CoAssignment Train NMI": (
                STATS_LEVEL,
                jnp.array(co_merge_results.train_nmi_score),
            ),
            "Merging/CoAssignment Test Accuracy": (
                STATS_LEVEL,
                jnp.array(co_merge_results.test_accuracy),
            ),
            "Merging/CoAssignment Test NMI": (
                STATS_LEVEL,
                jnp.array(co_merge_results.test_nmi_score),
            ),
            "Merging/Optimal Train Accuracy": (
                STATS_LEVEL,
                jnp.array(op_merge_results.train_accuracy),
            ),
            "Merging/Optimal Train NMI": (
                STATS_LEVEL,
                jnp.array(op_merge_results.train_nmi_score),
            ),
            "Merging/Optimal Test Accuracy": (
                STATS_LEVEL,
                jnp.array(op_merge_results.test_accuracy),
            ),
            "Merging/Optimal Test NMI": (
                STATS_LEVEL,
                jnp.array(op_merge_results.test_nmi_score),
            ),
        }
        logger.log_metrics(metrics, jnp.array(epoch))
