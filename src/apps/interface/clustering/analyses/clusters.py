"""Cluster statistics analysis for clustering models.

Requires models to implement:
- ClusteringModel (cluster_assignments, n_clusters)
- CanComputePrototypes (compute_cluster_prototypes)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override

import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from ....runtime import Artifact, RunHandler
from ...analysis import Analysis
from ..dataset import ClusteringDataset


@dataclass(frozen=True)
class ClusterStatistics(Artifact):
    """Collection of cluster prototypes with their members."""

    prototypes: list[Array]  # List of prototypes, one per cluster
    members: list[Array]  # List of (n_members, data_dim), one per cluster


@dataclass(frozen=True)
class ClusterStatisticsAnalysis(Analysis[ClusteringDataset, Any, ClusterStatistics]):
    """Analysis of cluster prototypes with their members.

    Works with any model implementing:
    - ClusteringModel (cluster_assignments, n_clusters)
    - CanComputePrototypes (compute_cluster_prototypes)
    """

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
        """Generate cluster statistics from model parameters."""
        train_data = dataset.train_data

        # Get cluster assignments and prototypes from model
        assignments = model.cluster_assignments(params, train_data)
        prototypes = model.compute_cluster_prototypes(params)

        # Gather members for each cluster
        cluster_members = []
        for i in range(model.n_clusters):
            cluster_mask = assignments == i
            members = train_data[cluster_mask]
            cluster_members.append(members)

        return ClusterStatistics(
            prototypes=prototypes,
            members=cluster_members,
        )

    @override
    def plot(self, artifact: ClusterStatistics, dataset: ClusteringDataset) -> Figure:
        """Create grid of cluster prototype visualizations."""
        n_clusters = len(artifact.prototypes)

        grid_shape = int(np.ceil(np.sqrt(n_clusters)))
        cluster_rows, cluster_cols = dataset.cluster_shape

        # Normalize cluster shape for aspect ratio
        cluster_rows_norm = cluster_rows / max(cluster_rows, cluster_cols)
        cluster_cols_norm = cluster_cols / max(cluster_rows, cluster_cols)
        scl = 5

        fig = plt.figure(
            figsize=(scl * grid_shape * cluster_cols_norm, scl * grid_shape * cluster_rows_norm)
        )
        gs = GridSpec(grid_shape, grid_shape, figure=fig)

        for i in range(n_clusters):
            ax = fig.add_subplot(gs[i // grid_shape, i % grid_shape])
            dataset.paint_cluster(
                cluster_id=i,
                prototype=artifact.prototypes[i],
                members=artifact.members[i],
                axes=ax,
            )

        plt.tight_layout()
        return fig
