"""KL divergence-based merge analysis for HMoG.

This module provides the KL-specific merge analysis. Generic merge analyses
(OptimalMergeAnalysis, CoAssignmentMergeAnalysis) are available from
apps.interface.clustering.analyses.merge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override

import jax.numpy as jnp
import numpy as np
from jax import Array
from matplotlib.figure import Figure

from apps.interface import ClusteringDataset
from apps.interface.clustering.analyses.merge import (
    MergeAnalysis,
    MergeResults,
    _fit_label_permutation,
    compute_merge_metrics,
    get_valid_clusters,
    hierarchy_to_mapping,
    plot_merge_results,
)
from apps.runtime import STATS_NUM, MetricDict, RunHandler

from .base import cluster_probabilities, get_component_prototypes
from .hierarchy import KLClusterHierarchy

STATS_LEVEL = jnp.array(STATS_NUM)


@dataclass(frozen=True)
class KLMergeResults(MergeResults):
    """KL divergence-based merge results."""


@dataclass(frozen=True)
class KLMergeAnalysis(MergeAnalysis[KLMergeResults]):
    """Merge clusters using KL divergence hierarchy.

    HMoG-specific: requires KLClusterHierarchy artifact which depends on
    computing KL divergence between mixture components.
    """

    @property
    @override
    def artifact_type(self) -> type[KLMergeResults]:
        return KLMergeResults

    @override
    def generate(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: Any,
        epoch: int,
        params: Array,
    ) -> KLMergeResults:
        manifold = model.manifold

        prototypes = get_component_prototypes(manifold, params)
        train_probs = cluster_probabilities(manifold, params, dataset.train_data)
        train_assignments = jnp.argmax(train_probs, axis=1)

        n_clusters = manifold.pst_man.n_categories
        n_classes = dataset.n_classes

        valid_clusters = get_valid_clusters(
            train_assignments, n_clusters, n_classes,
            self.filter_empty_clusters, self.min_cluster_size, len(dataset.train_data),
        )

        # Load KL hierarchy and compute mapping
        hierarchy = handler.load_artifact(epoch, KLClusterHierarchy)
        filtered_mapping = hierarchy_to_mapping(hierarchy, valid_clusters, n_classes)

        full_mapping = jnp.zeros((n_clusters, n_classes), dtype=jnp.int32)
        for i, cluster_idx in enumerate(valid_clusters):
            full_mapping = full_mapping.at[cluster_idx].set(filtered_mapping[i])

        label_permutation = _fit_label_permutation(full_mapping, train_probs, dataset.train_labels)
        train_metrics = compute_merge_metrics(full_mapping, train_probs, dataset.train_labels, label_permutation=label_permutation)
        test_probs = cluster_probabilities(manifold, params, dataset.test_data)
        test_metrics = compute_merge_metrics(full_mapping, test_probs, dataset.test_labels, label_permutation=label_permutation)

        return KLMergeResults(
            prototypes=prototypes,
            mapping=np.array(full_mapping, dtype=np.int32),
            train_accuracy=train_metrics[0],
            train_nmi_score=train_metrics[1],
            train_ari_score=train_metrics[2],
            test_accuracy=test_metrics[0],
            test_nmi_score=test_metrics[1],
            test_ari_score=test_metrics[2],
            valid_clusters=valid_clusters,
            similarity_type="kl",
        )

    @override
    def plot(self, artifact: KLMergeResults, dataset: ClusteringDataset) -> Figure:
        return plot_merge_results(artifact, dataset)

    @override
    def metrics(self, artifact: KLMergeResults) -> MetricDict:
        return {
            "Merging/KL Train Accuracy": (STATS_LEVEL, jnp.array(artifact.train_accuracy)),
            "Merging/KL Train NMI": (STATS_LEVEL, jnp.array(artifact.train_nmi_score)),
            "Merging/KL Train ARI": (STATS_LEVEL, jnp.array(artifact.train_ari_score)),
            "Merging/KL Test Accuracy": (STATS_LEVEL, jnp.array(artifact.test_accuracy)),
            "Merging/KL Test NMI": (STATS_LEVEL, jnp.array(artifact.test_nmi_score)),
            "Merging/KL Test ARI": (STATS_LEVEL, jnp.array(artifact.test_ari_score)),
        }
