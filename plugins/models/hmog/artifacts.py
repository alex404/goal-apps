"""Base class for HMoG implementations."""

from __future__ import annotations

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
    PositiveDefinite,
)
from goal.models import (
    AnalyticLinearGaussianModel,
    DifferentiableHMoG,
)
from h5py import File, Group
from jax import Array
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray

from apps.plugins import (
    ClusteringDataset,
)
from apps.runtime.handler import Artifact, RunHandler
from apps.runtime.logger import JaxLogger

### Analysis Args ###


@dataclass(frozen=True)
class AnalysisArgs:
    """Arguments for HMoG analysis."""

    from_scratch: bool
    epoch: int | None


### Helpers ###


def cluster_assignments[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    model: DifferentiableHMoG[ObsRep, LatRep], params: Array, data: Array
) -> Array:
    def data_point_cluster(x: Array) -> Array:
        cat_pst = model.pst_lat_man.prior(
            model.posterior_at(model.natural_point(params), x)
        )
        with model.lat_man.lat_man as lm:
            probs = lm.to_probs(lm.to_mean(cat_pst))
        return jnp.argmax(probs, axis=-1)

    return jax.lax.map(data_point_cluster, data, batch_size=2048)


def get_component_prototypes[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    model: DifferentiableHMoG[ObsRep, LatRep],
    params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
) -> list[Array]:
    # Split into likelihood and mixture parameters
    lkl_params, mix_params = model.split_conjugated(params)

    # Extract components from mixture
    comp_lats, _ = model.upr_hrm.split_natural_mixture(mix_params)

    # For each component, compute the observable distribution and get its mean
    prototypes: list[Array] = []

    ana_lgm = AnalyticLinearGaussianModel(
        obs_dim=model.lwr_hrm.obs_dim,  # Original latent becomes observable
        obs_rep=PositiveDefinite,
        lat_dim=model.lwr_hrm.lat_dim,  # Original observable becomes latent
    )

    for i in range(comp_lats.shape[0]):
        # Get latent mean for this component
        comp_lat_params = model.upr_hrm.cmp_man.get_replicate(comp_lats, jnp.asarray(i))
        lwr_hrm_params = ana_lgm.join_conjugated(lkl_params, comp_lat_params)
        lwr_hrm_means = ana_lgm.to_mean(lwr_hrm_params)
        lwr_hrm_obs = ana_lgm.split_params(lwr_hrm_means)[0]
        obs_means = ana_lgm.obs_man.split_mean_second_moment(lwr_hrm_obs)[0].array

        prototypes.append(obs_means)

    return prototypes


def compute_component_divergences[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    model: DifferentiableHMoG[ObsRep, LatRep],
    params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
) -> tuple[Array, Array, NDArray[np.float64]]:
    """Compute divergence statistics between mixture components.

    Returns:
        Tuple of (kl_matrix, symmetric_kl, linkage_matrix)
    """
    # Get raw KL divergences
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

    kl_matrix = jax.lax.map(kl_div_from_one_to_all, idxs)
    symmetric_kl = (kl_matrix + kl_matrix.T) / 2

    # Convert to numpy and handle distances
    dist_matrix = np.array(symmetric_kl, dtype=np.float64)

    # Ensure non-negative distances and exact zero diagonal
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

    return kl_matrix, symmetric_kl, linkage_matrix  # pyright: ignore[reportReturnType]


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


def get_cluster_statistics[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    model: DifferentiableHMoG[ObsRep, LatRep],
    dataset: ClusteringDataset,
    params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
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


### Cluster Hierarchy ###


@dataclass(frozen=True)
class ClusterHierarchy(Artifact):
    """Artifact containing clustering hierarchy analysis results."""

    prototypes: list[Array]
    linkage_matrix: NDArray[np.float64]

    @override
    def save_to_hdf5(self, file: File) -> None:
        """Save hierarchy data to HDF5 file."""
        # Save prototypes
        proto_group = file.create_group("prototypes")
        for i, proto in enumerate(self.prototypes):
            proto_group.create_dataset(f"{i}", data=np.array(proto))

        # Save linkage matrix
        file.create_dataset("linkage_matrix", data=self.linkage_matrix)

    @classmethod
    @override
    def load_from_hdf5(cls, file: File) -> ClusterHierarchy:
        """Load hierarchy data from HDF5 file."""
        # Load prototypes
        proto_group = file["prototypes"]
        assert isinstance(proto_group, Group)
        n_protos = len(proto_group)
        assert isinstance(proto_group, Group)
        prototypes = [jnp.array(proto_group[f"{i}"]) for i in range(n_protos)]

        # Load linkage matrix
        linkage_matrix = np.array(file["linkage_matrix"][()])  # pyright: ignore[reportIndexIssue]

        return cls(prototypes=prototypes, linkage_matrix=linkage_matrix)


def get_cluster_hierarchy[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    model: DifferentiableHMoG[ObsRep, LatRep],
    params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
) -> ClusterHierarchy:
    """Generate clustering hierarchy analysis."""
    prototypes = get_component_prototypes(model, params)
    _, _, linkage_matrix = compute_component_divergences(model, params)
    return ClusterHierarchy(prototypes=prototypes, linkage_matrix=linkage_matrix)


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
            hierarchy.linkage_matrix,
            orientation="left",
            ax=dendrogram_ax,
            leaf_font_size=10,
            leaf_label_func=lambda x: f"Cluster {x}",
        )
        dendrogram_ax.set_xlabel("Relative Entropy")

        leaf_order = dendrogram_results["leaves"]

        if leaf_order is None:
            raise ValueError("Failed to get leaf order from dendrogram.")

        leaf_order = leaf_order[::-1]

        for i, leaf_idx in enumerate(leaf_order):
            prototype_ax = fig.add_subplot(gs[i, 1])
            dataset.paint_observable(hierarchy.prototypes[leaf_idx], prototype_ax)
            # prototype_ax.set_title(f"Cluster {leaf_idx}", fontsize=8, y=1.0, pad=8)

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
        handler.save_params(epoch, params.array)
        clusters = get_cluster_hierarchy(model, params)
        cluster_statistics = get_cluster_statistics(model, dataset, params)
    else:
        clusters = handler.load_artifact(epoch, ClusterHierarchy)
        cluster_statistics = handler.load_artifact(epoch, ClusterStatistics)

    # Plot and save
    plot_hierarchy = hierarchy_plotter(dataset)
    plot_clusters = cluster_statistics_plotter(dataset)

    logger.log_artifact(handler, epoch, clusters, plot_hierarchy)
    logger.log_artifact(handler, epoch, cluster_statistics, plot_clusters)
