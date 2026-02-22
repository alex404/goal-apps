"""KL divergence based hierarchical clustering analysis for HMoG.

This is HMoG-specific because it requires computing KL divergence between
mixture components, which depends on the model's internal structure.

Reuses shared utilities from apps.interface.clustering.analyses.hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override

from jax import Array

from ..types import AnyHMoG
from matplotlib.figure import Figure

from apps.interface import Analysis, ClusteringDataset
from apps.interface.clustering.analyses import (
    ClusterHierarchy,
    build_hierarchy_from_distance,
    plot_hierarchy_dendrogram,
)
from apps.runtime import RunHandler

from .base import get_component_prototypes, symmetric_kl_matrix


@dataclass(frozen=True)
class KLClusterHierarchy(ClusterHierarchy):
    """KL divergence-based clustering hierarchy."""

    pass


@dataclass(frozen=True)
class KLHierarchyAnalysis(Analysis[ClusteringDataset, Any, KLClusterHierarchy]):
    """KL divergence based hierarchical clustering analysis.

    HMoG-specific: requires access to model.manifold for computing
    KL divergence between mixture components.
    """

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
        """Generate hierarchy from KL divergence between components."""
        manifold: AnyHMoG = model.manifold

        # Get prototypes for visualization
        prototypes = get_component_prototypes(manifold, params)

        # Compute symmetric KL divergence matrix (already a distance)
        kl_matrix = symmetric_kl_matrix(manifold, params)

        # Build hierarchy using shared utility
        linkage_matrix, distance_matrix = build_hierarchy_from_distance(kl_matrix)

        return KLClusterHierarchy(
            prototypes=prototypes,
            linkage_matrix=linkage_matrix,
            distance_matrix=distance_matrix,
        )

    @override
    def plot(
        self, artifact: KLClusterHierarchy, dataset: ClusteringDataset
    ) -> Figure:
        return plot_hierarchy_dendrogram(
            artifact,
            dataset,
            metric_label="KL Divergence",
            title="Hierarchical Clustering (KL Divergence)",
        )
