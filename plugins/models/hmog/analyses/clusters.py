"""Base class for DifferentiableHMoG implementations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, override

import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from apps.interface import (
    Analysis,
    ClusteringDataset,
)
from apps.runtime import Artifact, RunHandler

from .base import cluster_assignments, get_component_prototypes

### Helpers ###


### Cluster Statistics ###


@dataclass(frozen=True)
class ClusterStatistics(Artifact):
    """Collection of cluster prototypes with their members."""

    prototypes: list[Array]  # list of prototypes
    members: list[Array]  # list of (n_members, data_dim)


def generate_cluster_statistics(
    model: DifferentiableHMoG,
    dataset: ClusteringDataset,
    params: Array,
) -> ClusterStatistics:
    """Generate collection of clusters with their members."""

    train_data = dataset.train_data
    assignments = cluster_assignments(model, params, train_data)
    prototypes = get_component_prototypes(model, params)

    # Create cluster collections
    cluster_members = []

    for i in range(model.pst_man.n_categories):
        # Get members for this cluster
        cluster_mask = assignments == i
        members = train_data[cluster_mask]
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


### Cluster Analysis ###


@dataclass(frozen=True)
class ClusterStatisticsAnalysis(Analysis[ClusteringDataset, Any, ClusterStatistics]):
    """Analysis of cluster prototypes with their members."""

    @property
    @override
    def artifact_type(self) -> type[ClusterStatistics]:
        return ClusterStatistics

    @override
    def generate(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: Any,
        epoch: int,
        params: Array,
    ) -> ClusterStatistics:
        """Generate collection of clusters with their members."""
        return generate_cluster_statistics(model.manifold, dataset, params)

    @override
    def plot(self, artifact: ClusterStatistics, dataset: ClusteringDataset) -> Figure:
        """Create grid of cluster prototype visualizations."""
        plotter = cluster_statistics_plotter(dataset)
        return plotter(artifact)
