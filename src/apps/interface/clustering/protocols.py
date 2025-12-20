"""Protocols defining capabilities for clustering models.

These protocols specify interfaces that clustering models can implement
to enable specific types of analyses. They serve as documentation of
model capabilities and provide type safety for analysis implementations.
"""

from __future__ import annotations

from typing import Protocol

from jax import Array

from ...runtime import RunHandler


class HasSoftAssignments(Protocol):
    """Protocol for models that provide soft cluster assignments (responsibilities).

    Models implementing this protocol can compute posterior probabilities
    that each data point belongs to each cluster, enabling weighted
    prototype computation and other soft-clustering analyses.
    """

    def posterior_soft_assignments(self, params: Array, data: Array) -> Array:
        """Compute posterior responsibilities p(z|x) for all data.

        Args:
            params: Model parameters
            data: Data array of shape (n_samples, data_dim)

        Returns:
            Responsibilities array of shape (n_samples, n_clusters) where
            responsibilities[i, k] = p(z=k | x=i, params)
        """
        ...


class HasClusterPrototypes(Protocol):
    """Protocol for models that can provide cluster prototypes and members.

    Models implementing this protocol store cluster statistics as artifacts
    during training, which can be loaded for visualization and analysis.
    """

    def get_cluster_prototypes(self, handler: RunHandler, epoch: int) -> list[Array]:
        """Get prototype/centroid for each cluster.

        Args:
            handler: RunHandler for loading artifacts
            epoch: Epoch number to load from

        Returns:
            List of prototype arrays, one per cluster
        """
        ...

    def get_cluster_members(self, handler: RunHandler, epoch: int) -> list[Array]:
        """Get cluster members by loading from artifact.

        Args:
            handler: RunHandler for loading artifacts
            epoch: Epoch number to load from

        Returns:
            List of arrays, where members[i] contains all members of cluster i
            with shape (n_members_i, data_dim)
        """
        ...


class HasClusterHierarchy(Protocol):
    """Protocol for models that can provide hierarchical cluster structure.

    Models implementing this protocol can compute or load a hierarchical
    clustering of their clusters, enabling dendrogram visualization.
    """

    def get_cluster_hierarchy(self, handler: RunHandler, epoch: int) -> Array:
        """Get hierarchical clustering of clusters.

        Args:
            handler: RunHandler for loading artifacts
            epoch: Epoch number to load from

        Returns:
            Scipy-compatible linkage matrix
        """
        ...
