"""Base class for HMoG implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, override

import matplotlib.pyplot as plt
import numpy as np
from goal.geometry import (
    Natural,
    Point,
)
from jax import Array
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from apps.interface import (
    Analysis,
    ClusteringDataset,
)
from apps.runtime import Artifact, RunHandler

from ..base import HMoG
from .base import cluster_assignments, get_component_prototypes

### Helpers ###


### Cluster Statistics ###


@dataclass(frozen=True)
class ClusterStatistics(Artifact):
    """Collection of cluster prototypes with their members."""

    prototypes: list[Array]  # list of prototypes
    members: list[Array]  # list of (n_members, data_dim)


def generate_cluster_statistics[
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
class ClusterStatisticsAnalysis(Analysis[ClusteringDataset, HMoG, ClusterStatistics]):
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
        model: HMoG,
        epoch: int,
        params: Array,
    ) -> ClusterStatistics:
        """Generate collection of clusters with their members."""
        # Convert array to typed point for the model
        typed_params = model.natural_point(params)
        return generate_cluster_statistics(model, dataset, typed_params)

    @override
    def plot(self, artifact: ClusterStatistics, dataset: ClusteringDataset) -> Figure:
        """Create grid of cluster prototype visualizations."""
        plotter = cluster_statistics_plotter(dataset)
        return plotter(artifact)
