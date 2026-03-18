"""Clustering-specific abstractions for goal-apps.

This module provides clustering-specific extensions to the generic
apps.interface framework, including protocols for model capabilities
and reusable analysis implementations.
"""

from .config import (
    AnalysisConfig,
    ClusteringAnalysesConfig,
    ClusteringRunConfig,
    ClusterStatisticsConfig,
    CoAssignmentHierarchyConfig,
    CoAssignmentMergeConfig,
    GenerativeSamplesConfig,
    OptimalMergeConfig,
)
from .dataset import ClusteringDataset, ClusteringDatasetConfig
from .metrics import (
    add_clustering_metrics,
    cluster_accuracy,
    clustering_nmi,
    fit_cluster_mapping,
)
from .model import ClusteringModel, ClusteringModelConfig
from .protocols import (
    CanComputePrototypes,
    HasClusterHierarchy,
    HasClusterPrototypes,
    HasSoftAssignments,
)

__all__ = [
    "AnalysisConfig",
    "CanComputePrototypes",
    "ClusterStatisticsConfig",
    "ClusteringAnalysesConfig",
    "ClusteringDataset",
    "ClusteringDatasetConfig",
    "ClusteringModel",
    "ClusteringModelConfig",
    "ClusteringRunConfig",
    "CoAssignmentHierarchyConfig",
    "CoAssignmentMergeConfig",
    "GenerativeSamplesConfig",
    "HasClusterHierarchy",
    "HasClusterPrototypes",
    "HasSoftAssignments",
    "OptimalMergeConfig",
    "add_clustering_metrics",
    "cluster_accuracy",
    "clustering_nmi",
    "fit_cluster_mapping",
]
