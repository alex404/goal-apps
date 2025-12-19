"""Clustering dataset abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from jax import Array
from matplotlib.axes import Axes

from ..dataset import Dataset, DatasetConfig


@dataclass
class ClusteringDatasetConfig(DatasetConfig):
    """Base configuration for clustering datasets."""

    _target_: str


@dataclass(frozen=True)
class ClusteringDataset(Dataset, ABC):
    """Abstract base class for datasets used in clustering applications."""

    cache_dir: Path

    @property
    @abstractmethod
    def cluster_shape(self) -> tuple[int, int]:
        """Shape of the cluster prototypes visualization (used for plotting)."""

    @abstractmethod
    def paint_cluster(
        self, cluster_id: int, prototype: Array, members: Array, axes: Axes
    ):
        """Visualize a prototype and its cluster members.

        Args:
            cluster_id: The cluster index (scalar)
            prototype: The prototype for the cluster (data_dim,)
            members: The members of the cluster (n_members, data_dim)
            axes: Matplotlib axes to draw on
        """

    @property
    @abstractmethod
    def has_labels(self) -> bool:
        """Return True if the dataset has labels."""

    @property
    def n_classes(self) -> int:
        raise NotImplementedError("n_classes is not implemented for this dataset.")

    @property
    def train_labels(self) -> Array:
        """Training labels with shape (n_train,)."""
        raise NotImplementedError("train_labels is not implemented for this dataset.")

    @property
    def test_labels(self) -> Array:
        """Test labels with shape (n_test,)."""
        raise NotImplementedError("test_labels is not implemented for this dataset.")
