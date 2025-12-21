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
    """Configuration for co-assignment based hierarchy analysis."""

    pass


@dataclass
class OptimalMergeConfig(AnalysisConfig):
    """Configuration for optimal (Hungarian) merge analysis."""

    enabled: bool = False  # Disabled by default (requires labels, uses cheating)
    filter_empty_clusters: bool = True
    min_cluster_size: float = 0.0005


@dataclass
class CoAssignmentMergeConfig(AnalysisConfig):
    """Configuration for co-assignment based merge analysis."""

    filter_empty_clusters: bool = True
    min_cluster_size: float = 0.0005


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


# Run Configs

defaults: list[Any] = [
    {"model": MISSING},
    {"dataset": MISSING},
]


@dataclass
class ClusteringRunConfig(RunConfig):
    """Base configuration for clustering simulations."""

    dataset: ClusteringDatasetConfig = MISSING
    model: ClusteringModelConfig = MISSING
    defaults: list[Any] = field(default_factory=lambda: defaults)
