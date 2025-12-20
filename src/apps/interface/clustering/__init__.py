"""Clustering-specific abstractions for goal-apps.

This module provides clustering-specific extensions to the generic
apps.interface framework, including protocols for model capabilities
and reusable analysis implementations.
"""

from .dataset import ClusteringDataset, ClusteringDatasetConfig
from .model import ClusteringModel, ClusteringModelConfig
from .protocols import (
    CanComputePrototypes,
    HasClusterHierarchy,
    HasClusterPrototypes,
    HasSoftAssignments,
)

__all__ = [
    # Dataset
    "ClusteringDataset",
    "ClusteringDatasetConfig",
    # Model
    "ClusteringModel",
    "ClusteringModelConfig",
    # Protocols
    "CanComputePrototypes",
    "HasClusterHierarchy",
    "HasClusterPrototypes",
    "HasSoftAssignments",
]
