from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import override

from jax import Array
from matplotlib.axes import Axes

from .runtime.handler import RunHandler
from .runtime.logger import JaxLogger

### Interfaces ###


@dataclass(frozen=True)
class Dataset(ABC):
    """Root class for all datasets."""

    @property
    @abstractmethod
    def data_dim(self) -> int:
        """Dimensionality of each data point."""

    @property
    @abstractmethod
    def observable_shape(self) -> tuple[int, int]:
        """Shape of the observable data (used for plotting)."""

    @property
    @abstractmethod
    def train_data(self) -> Array:
        """Training data with shape (n_train, data_dim)."""

    @property
    @abstractmethod
    def test_data(self) -> Array:
        """Test data with shape (n_test, data_dim)."""

    @abstractmethod
    def paint_observable(self, observable: Array, axes: Axes):
        """A function for rendering an observation from the dataset.

        Args:
            observable: A single observation from the dataset (data_dim,)
            axes: Matplotlib axes to draw on
        """


class Model[D: Dataset](ABC):
    """Root class for all models."""

    @property
    @abstractmethod
    def n_epochs(self) -> int:
        """Dimensionality of each data point."""

    @abstractmethod
    def analyze(
        self,
        key: Array,
        handler: RunHandler,
        dataset: D,
        logger: JaxLogger,
    ) -> None:
        """Run analysis suite based on trained model."""

    @abstractmethod
    def train(
        self,
        key: Array,
        handler: RunHandler,
        dataset: D,
        logger: JaxLogger,
    ) -> None:
        """Evaluate model on dataset."""


### Clustering ###

# Datasets


@dataclass(frozen=True)
class ClusteringDataset(Dataset, ABC):
    """Abstract base class for datasets used in clustering applications."""

    cache_dir: Path

    @property
    @abstractmethod
    def cluster_shape(self) -> tuple[int, int]:
        """Shape of the cluster prototypes visaulization (used for plotting)."""

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


# Models


class ClusteringModel(Model[ClusteringDataset], ABC):
    """Abstract base class for all unsupervised models."""

    @property
    @abstractmethod
    def n_clusters(self) -> int:
        """Return the number of clusters in the model."""

    @abstractmethod
    def initialize_model(self, key: Array, data: Array) -> Array:
        """Initialize model parameters based on data dimensions and statistics."""

    @abstractmethod
    def generate(self, params: Array, key: Array, n_samples: int) -> Array:
        """Generate new samples using the trained model."""

    @abstractmethod
    @override
    def train(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        logger: JaxLogger,
    ) -> None:
        """Evaluate model on dataset."""
