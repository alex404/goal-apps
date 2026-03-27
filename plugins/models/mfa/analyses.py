"""KL divergence-based analyses for MFA models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from matplotlib.figure import Figure

from apps.interface import Analysis, ClusteringDataset
from apps.interface.clustering.analyses import (
    ClusterHierarchy,
    build_hierarchy_from_distance,
    plot_hierarchy_dendrogram,
)
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

from .types import MFA

STATS_LEVEL = jnp.array(STATS_NUM)


def symmetric_kl_matrix(mfa: MFA, params: Array) -> Array:
    """Compute symmetric KL divergence matrix between MFA latent components.

    Computes KL(p(z|k) || p(z|l)) between each pair of latent Gaussian
    components, then symmetrizes. After whitening, components have identity
    covariance so KL reduces to squared Euclidean distance in latent space.
    """
    mix_params = mfa.to_mixture_coords(params)
    comp_hrm_params, _ = mfa.mix_man.split_natural_mixture(mix_params)
    comp_hrm_2d = mfa.mix_man.cmp_man.to_2d(comp_hrm_params)

    lat_man = mfa.bas_hrm.lat_man

    def kl_div_between(i: Array, j: Array) -> Array:
        _, _, lat_i = mfa.bas_hrm.split_coords(comp_hrm_2d[i])
        lat_i_mean = lat_man.to_mean(lat_i)
        _, _, lat_j = mfa.bas_hrm.split_coords(comp_hrm_2d[j])
        return lat_man.relative_entropy(lat_i_mean, lat_j)

    idxs = jnp.arange(mfa.n_categories)

    def kl_from_one_to_all(i: Array) -> Array:
        return jax.vmap(kl_div_between, in_axes=(None, 0))(i, idxs)

    kl_matrix = jax.lax.map(kl_from_one_to_all, idxs)
    return (kl_matrix + kl_matrix.T) / 2


@dataclass(frozen=True)
class MFAKLClusterHierarchy(ClusterHierarchy):
    """KL divergence-based clustering hierarchy for MFA."""

    pass


@dataclass(frozen=True)
class MFAKLHierarchyAnalysis(Analysis[ClusteringDataset, Any, MFAKLClusterHierarchy]):
    """KL divergence based hierarchical clustering analysis for MFA."""

    @property
    @override
    def artifact_type(self) -> type[MFAKLClusterHierarchy]:
        return MFAKLClusterHierarchy

    @override
    def generate(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: Any,
        epoch: int,
        params: Array,
    ) -> MFAKLClusterHierarchy:
        kl_matrix = symmetric_kl_matrix(model.mfa, params)
        linkage_matrix, distance_matrix = build_hierarchy_from_distance(kl_matrix)
        prototypes = model.compute_cluster_prototypes(params)
        return MFAKLClusterHierarchy(
            prototypes=prototypes,
            linkage_matrix=linkage_matrix,
            distance_matrix=distance_matrix,
        )

    @override
    def plot(self, artifact: MFAKLClusterHierarchy, dataset: ClusteringDataset) -> Figure:
        return plot_hierarchy_dendrogram(
            artifact,
            dataset,
            metric_label="KL Divergence",
            title="Hierarchical Clustering (KL Divergence)",
        )


@dataclass(frozen=True)
class MFAKLMergeResults(MergeResults):
    """KL divergence-based merge results for MFA."""


@dataclass(frozen=True)
class MFAKLMergeAnalysis(MergeAnalysis[MFAKLMergeResults]):
    """Merge MFA clusters using KL divergence hierarchy."""

    @property
    @override
    def artifact_type(self) -> type[MFAKLMergeResults]:
        return MFAKLMergeResults

    @override
    def generate(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: Any,
        epoch: int,
        params: Array,
    ) -> MFAKLMergeResults:
        prototypes = model.compute_cluster_prototypes(params)
        train_probs = model.posterior_soft_assignments(params, dataset.train_data)
        train_assignments = jnp.argmax(train_probs, axis=1)

        n_clusters = model.n_clusters
        n_classes = dataset.n_classes

        valid_clusters = get_valid_clusters(
            train_assignments,
            n_clusters,
            n_classes,
            self.filter_empty_clusters,
            self.min_cluster_size,
            len(dataset.train_data),
        )

        hierarchy = handler.load_artifact(epoch, MFAKLClusterHierarchy)
        filtered_mapping = hierarchy_to_mapping(hierarchy, valid_clusters, n_classes)

        full_mapping = jnp.zeros((n_clusters, n_classes), dtype=jnp.int32)
        for i, cluster_idx in enumerate(valid_clusters):
            full_mapping = full_mapping.at[cluster_idx].set(filtered_mapping[i])

        label_permutation = _fit_label_permutation(
            full_mapping, train_probs, dataset.train_labels, n_classes
        )
        train_metrics = compute_merge_metrics(
            full_mapping, train_probs, dataset.train_labels, label_permutation=label_permutation
        )
        test_probs = model.posterior_soft_assignments(params, dataset.test_data)
        test_metrics = compute_merge_metrics(
            full_mapping, test_probs, dataset.test_labels, label_permutation=label_permutation
        )

        return MFAKLMergeResults(
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
    def plot(self, artifact: MFAKLMergeResults, dataset: ClusteringDataset) -> Figure:
        return plot_merge_results(artifact, dataset)

    @override
    def metrics(self, artifact: MFAKLMergeResults) -> MetricDict:
        return {
            "Merging/KL Train Accuracy": (STATS_LEVEL, jnp.array(artifact.train_accuracy)),
            "Merging/KL Train NMI": (STATS_LEVEL, jnp.array(artifact.train_nmi_score)),
            "Merging/KL Train ARI": (STATS_LEVEL, jnp.array(artifact.train_ari_score)),
            "Merging/KL Test Accuracy": (STATS_LEVEL, jnp.array(artifact.test_accuracy)),
            "Merging/KL Test NMI": (STATS_LEVEL, jnp.array(artifact.test_nmi_score)),
            "Merging/KL Test ARI": (STATS_LEVEL, jnp.array(artifact.test_ari_score)),
        }
