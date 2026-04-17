"""Configuration classes for clustering models and analyses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from ..config import RunConfig
from .dataset import ClusteringDatasetConfig
from .model import ClusteringModelConfig

# Analysis Configs


@dataclass
class AnalysisConfig:
    """Base configuration for analyses. All analyses can be disabled."""

    enabled: bool = True


@dataclass
class GenerativeSamplesConfig(AnalysisConfig):
    """Configuration for generative samples analysis."""

    n_samples: int = 100


@dataclass
class ClusterStatisticsConfig(AnalysisConfig):
    """Configuration for cluster statistics analysis."""

    pass


@dataclass
class CoAssignmentHierarchyConfig(AnalysisConfig):
    """Configuration for co-assignment based hierarchy analysis.

    Dead clusters (total responsibility below ``min_cluster_size`` of
    training data) are pruned before the co-assignment matrix is built,
    preventing sqrt-of-zero normalization blow-ups. ``valid_clusters`` is
    persisted in the resulting artifact so the downstream co-assignment
    merge analysis uses the same filter set.
    """

    filter_empty_clusters: bool = True
    min_cluster_size: float = 0.0005


@dataclass
class OptimalMergeConfig(AnalysisConfig):
    """Configuration for optimal (Hungarian) merge analysis."""

    enabled: bool = False  # Disabled by default (requires labels, uses cheating)
    filter_empty_clusters: bool = True
    min_cluster_size: float = 0.0005


@dataclass
class CoAssignmentMergeConfig(AnalysisConfig):
    """Configuration for co-assignment based merge analysis.

    Filter parameters live on ``CoAssignmentHierarchyConfig`` — the merge
    inherits whatever the hierarchy decided. There is nothing to configure
    here beyond enabling/disabling.
    """


@dataclass
class KLHierarchyConfig(AnalysisConfig):
    """Configuration for KL divergence based hierarchy analysis.

    Dead clusters (those with insufficient hard-assignment mass on the
    training data) are pruned before the KL distance matrix is
    clustered. ``valid_clusters`` is persisted in the resulting artifact
    so the downstream KL merge analysis uses the same filter set.
    """

    filter_empty_clusters: bool = True
    min_cluster_size: float = 0.0005


@dataclass
class KLMergeConfig(AnalysisConfig):
    """Configuration for KL divergence based merge analysis.

    Filter parameters live on ``KLHierarchyConfig`` — the merge inherits
    whatever the hierarchy decided.
    """


@dataclass
class ClusteringAnalysesConfig:
    """Configuration for all clustering analyses.

    Each field corresponds to an analysis that can be enabled/disabled
    and configured independently.
    """

    generative_samples: GenerativeSamplesConfig = field(
        default_factory=GenerativeSamplesConfig
    )
    cluster_statistics: ClusterStatisticsConfig = field(
        default_factory=ClusterStatisticsConfig
    )
    co_assignment_hierarchy: CoAssignmentHierarchyConfig = field(
        default_factory=CoAssignmentHierarchyConfig
    )
    optimal_merge: OptimalMergeConfig = field(default_factory=OptimalMergeConfig)
    co_assignment_merge: CoAssignmentMergeConfig = field(
        default_factory=CoAssignmentMergeConfig
    )
    kl_hierarchy: KLHierarchyConfig = field(default_factory=KLHierarchyConfig)
    kl_merge: KLMergeConfig = field(default_factory=KLMergeConfig)


# Run Configs


@dataclass
class ClusteringRunConfig(RunConfig):
    """Base configuration for clustering simulations."""

    dataset: ClusteringDatasetConfig = MISSING
    model: ClusteringModelConfig = MISSING
    defaults: list[Any] = field(
        default_factory=lambda: [{"model": MISSING}, {"dataset": MISSING}]
    )
