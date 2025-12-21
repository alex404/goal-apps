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
from .metrics import cluster_accuracy, clustering_nmi
from .model import ClusteringModel, ClusteringModelConfig
from .protocols import (
    CanComputePrototypes,
    HasClusterHierarchy,
    HasClusterPrototypes,
    HasSoftAssignments,
)

__all__ = [
    # Config
    "AnalysisConfig",
    "ClusteringAnalysesConfig",
    "ClusteringRunConfig",
    "ClusterStatisticsConfig",
    "CoAssignmentHierarchyConfig",
    "CoAssignmentMergeConfig",
    "GenerativeSamplesConfig",
    "OptimalMergeConfig",
    # Dataset
    "ClusteringDataset",
    "ClusteringDatasetConfig",
    # Model
    "ClusteringModel",
    "ClusteringModelConfig",
    # Metrics
    "cluster_accuracy",
    "clustering_nmi",
    # Protocols
    "CanComputePrototypes",
    "HasClusterHierarchy",
    "HasClusterPrototypes",
    "HasSoftAssignments",
]
