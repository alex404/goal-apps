"""Cluster statistics analysis for MFA model."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array
from matplotlib.figure import Figure

from apps.interface import Analysis, ClusteringDataset
from apps.runtime import Artifact, RunHandler

from .base import get_responsibilities

if TYPE_CHECKING:
    from ..model import MFAModel


@dataclass(frozen=True)
class ClusterStatistics(Artifact):
    """Cluster assignments and prototypes."""

    prototypes: list[Array]
    """List of cluster centroids (prototypes), one per cluster."""

    members: list[Array]
    """List of arrays, where members[i] contains all data points in cluster i."""

    responsibilities: Array
    """Posterior probabilities of shape (n_samples, n_clusters)."""


@dataclass(frozen=True)
class ClusterStatisticsAnalysis(
    Analysis[ClusteringDataset, "MFAModel", ClusterStatistics]
):
    """Analysis for computing cluster statistics and prototypes."""

    @property
    @override
    def artifact_type(self) -> type[ClusterStatistics]:
        """Return artifact type."""
        return ClusterStatistics

    @override
    def generate(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: "MFAModel",
        epoch: int,
        params: Array,
    ) -> ClusterStatistics:
        """Generate cluster statistics from model parameters.

        Args:
            key: Random key
            handler: Run handler for loading params
            dataset: Dataset to analyze
            model: MFA model instance
            epoch: Epoch number
            params: Model parameters

        Returns:
            ClusterStatistics artifact
        """
        # Get responsibilities
        responsibilities = get_responsibilities(model.mfa, params, dataset.train_data)

        # Assign to clusters
        assignments = jnp.argmax(responsibilities, axis=1)

        # Compute prototypes (weighted means) and gather members
        prototypes = []
        members = []

        for k in range(model.n_clusters):
            cluster_mask = assignments == k
            cluster_data = dataset.train_data[cluster_mask]
            cluster_resp = responsibilities[cluster_mask, k]

            weighted_mean = jnp.average(cluster_data, axis=0, weights=cluster_resp)
            prototypes.append(weighted_mean)
            members.append(cluster_data)

        return ClusterStatistics(
            prototypes=prototypes,
            members=members,
            responsibilities=responsibilities,
        )

    @override
    def plot(
        self,
        artifact: ClusterStatistics,
        dataset: ClusteringDataset,
    ) -> Figure:
        """Visualize cluster prototypes as images.

        Args:
            artifact: Cluster statistics to visualize
            dataset: Dataset

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))

        for i, (ax, prototype) in enumerate(zip(axes.flat, artifact.prototypes)):
            img = prototype.reshape(28, 28)
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Cluster {i}")
            ax.axis("off")

        plt.tight_layout()
        return fig

    @override
    def metrics(self, artifact: ClusterStatistics) -> dict[str, tuple[Array, Array]]:
        """Compute cluster balance metrics.

        Args:
            artifact: Cluster statistics

        Returns:
            Dictionary of metrics in format {name: (log_level, value)}
        """
        cluster_sizes = [len(m) for m in artifact.members]
        return {
            "clusters/min_size": (
                jnp.array(logging.INFO),
                jnp.array(float(min(cluster_sizes))),
            ),
            "clusters/max_size": (
                jnp.array(logging.INFO),
                jnp.array(float(max(cluster_sizes))),
            ),
            "clusters/mean_size": (
                jnp.array(logging.INFO),
                jnp.array(float(sum(cluster_sizes) / len(cluster_sizes))),
            ),
        }
