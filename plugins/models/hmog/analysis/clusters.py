"""Base class for HMoG implementations."""

from __future__ import annotations

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

from apps.plugins import (
    ClusteringDataset,
)
from apps.runtime.handler import Artifact

from ..base import HMoG
from .base import cluster_assignments, get_component_prototypes

### Cluster Statistics ###


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
