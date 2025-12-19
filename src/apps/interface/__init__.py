from .analysis import Analysis
from .dataset import Dataset, DatasetConfig
from .model import Model, ModelConfig
from .protocols import HasLogLikelihood, IsGenerative

# Re-export clustering for backward compatibility
from .clustering import (
    ClusteringDataset,
    ClusteringDatasetConfig,
    ClusteringModel,
    ClusteringModelConfig,
)

__all__ = [
    # Generic
    "Analysis",
    "Dataset",
    "DatasetConfig",
    "Model",
    "ModelConfig",
    # Generic protocols
    "HasLogLikelihood",
    "IsGenerative",
    # Clustering (re-exported for convenience)
    "ClusteringDataset",
    "ClusteringDatasetConfig",
    "ClusteringModel",
    "ClusteringModelConfig",
]
