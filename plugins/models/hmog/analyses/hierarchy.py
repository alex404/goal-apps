"""KL divergence based hierarchical clustering analysis for HMoG.

This is HMoG-specific because it requires computing KL divergence between
mixture components, which depends on the model's internal structure.

Reuses shared utilities from apps.interface.clustering.analyses.hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override

import jax.numpy as jnp
from jax import Array
from matplotlib.figure import Figure

from apps.interface import Analysis, ClusteringDataset
from apps.interface.clustering.analyses import (
    ClusterHierarchy,
    build_hierarchy_from_distance,
    plot_hierarchy_dendrogram,
)
from apps.interface.clustering.analyses.hierarchy import get_valid_clusters
from apps.runtime import RunHandler

from ..types import AnyHMoG
from .base import cluster_probabilities, get_component_prototypes, symmetric_kl_matrix


@dataclass(frozen=True)
class KLClusterHierarchy(ClusterHierarchy):
    """KL divergence-based clustering hierarchy."""

    pass


@dataclass(frozen=True)
class KLHierarchyAnalysis(Analysis[ClusteringDataset, Any, KLClusterHierarchy]):
    """KL divergence based hierarchical clustering analysis.

    HMoG-specific: requires access to model.manifold for computing
    KL divergence between mixture components.

    Dead components (those with insufficient hard-assignment mass on the
    training data) are pruned before the KL matrix is clustered, so the
    dendrogram reflects only components actually in use.
    """

    filter_empty_clusters: bool = True
    min_cluster_size: float = 0.0005

    @property
    @override
    def artifact_type(self) -> type[KLClusterHierarchy]:
        return KLClusterHierarchy

    @override
    def generate(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: Any,
        epoch: int,
        params: Array,
    ) -> KLClusterHierarchy:
        """Generate hierarchy from KL divergence between valid components."""
        manifold: AnyHMoG = model.manifold

        prototypes = get_component_prototypes(manifold, params)

        # Determine which components actually receive assignments on train.
        train_probs = cluster_probabilities(manifold, params, dataset.train_data)
        assignments = jnp.argmax(train_probs, axis=1)
        n_clusters = manifold.prr_man.n_categories
        valid_clusters = get_valid_clusters(
            assignments,
            n_clusters,
            dataset.n_classes if dataset.has_labels else n_clusters,
            self.filter_empty_clusters,
            self.min_cluster_size,
            len(dataset.train_data),
        )

        full_kl = symmetric_kl_matrix(manifold, params)
        kl_sub = full_kl[valid_clusters][:, valid_clusters]

        linkage_matrix, distance_matrix = build_hierarchy_from_distance(kl_sub)

        return KLClusterHierarchy(
            prototypes=prototypes,
            linkage_matrix=linkage_matrix,
            distance_matrix=distance_matrix,
            valid_clusters=valid_clusters,
        )

    @override
    def plot(self, artifact: KLClusterHierarchy, dataset: ClusteringDataset) -> Figure:
        return plot_hierarchy_dendrogram(
            artifact,
            dataset,
            metric_label="KL Divergence",
            title="Hierarchical Clustering (KL Divergence)",
        )
