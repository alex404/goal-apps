"""Clustering model abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from jax import Array
from omegaconf import MISSING

from ..model import Model, ModelConfig
from .dataset import ClusteringDataset


@dataclass
class ClusteringModelConfig(ModelConfig):
    """Base configuration for clustering models."""

    _target_: str
    data_dim: int = MISSING
    n_clusters: int = MISSING


class ClusteringModel(Model[ClusteringDataset], ABC):
    """Abstract base class for clustering models.

    This is a minimal interface. Additional capabilities are provided
    via protocols (HasLogLikelihood, IsGenerative, HasSoftAssignments, etc.)
    """

    @property
    @abstractmethod
    def n_clusters(self) -> int:
        """Return the number of clusters in the model."""

    @abstractmethod
    def cluster_assignments(self, params: Array, data: Array) -> Array:
        """Assign data points to clusters.

        Args:
            params: Model parameters
            data: Data array of shape (n_samples, data_dim)

        Returns:
            Array of cluster assignments with shape (n_samples,)
        """
