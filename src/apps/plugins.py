from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, override

import jax.numpy as jnp
from jax import Array
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .runtime.handler import Artifact, MetricDict, RunHandler
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

    def get_dataset_analyses(self) -> dict[str, Analysis[Self, Any, Artifact]]:
        """Return dataset-specific analysis instances."""
        return {}


class Experiment[D: Dataset](ABC):
    """Root class for all models."""

    @property
    @abstractmethod
    def n_epochs(self) -> int:
        """Number of epochs used for training the model."""

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


### Analysis ###


@dataclass(frozen=True)
class Analysis[D: Dataset, M, T: Artifact](ABC):
    """Base class for analyses that produce artifacts and visualizations.

    This class standardizes the pattern of generating or loading artifacts
    and creating visualizations from them. Each analysis encapsulates:
    - The logic for generating an artifact from model parameters
    - The visualization function for that artifact type
    - The coordination of loading vs. regenerating artifacts
    """

    @abstractmethod
    def generate(
        self,
        key: Array,
        handler: RunHandler,
        dataset: D,
        model: M,
        epoch: int,
        params: Array,
    ) -> T:
        """Generate the analysis artifact from model parameters."""

    @abstractmethod
    def plot(self, artifact: T, dataset: D) -> Figure:
        """Create visualization from the artifact."""

    @property
    @abstractmethod
    def artifact_type(self) -> type[T]:
        """Return the artifact class for type checking and loading."""

    def metrics(self, artifact: T) -> MetricDict:
        """Return metrics collected during the analysis."""
        return {}

    def process(
        self,
        key: Array,
        handler: RunHandler,
        dataset: D,
        model: M,
        logger: JaxLogger,
        epoch: int,
        params: Array | None = None,
    ) -> None:
        """Process the analysis: generate or load artifact, then visualize and log."""
        if params is not None:
            artifact = self.generate(key, handler, dataset, model, epoch, params)
        else:
            artifact = handler.load_artifact(epoch, self.artifact_type)

        metrics = self.metrics(artifact)

        logger.log_metrics(metrics, jnp.array(epoch))
        logger.log_artifact(handler, epoch, artifact, lambda a: self.plot(a, dataset))


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


class ClusteringExperiment(Experiment[ClusteringDataset], ABC):
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
    def cluster_assignments(self, params: Array, data: Array) -> Array:
        """Assign data points to clusters."""

    @abstractmethod
    def get_cluster_prototypes(self, handler: RunHandler, epoch: int) -> list[Array]:
        """Get prototype/centroid for each cluster.

        Returns:
            Array of shape (n_clusters, data_dim) containing cluster prototypes
        """

    @abstractmethod
    def get_cluster_members(self, handler: RunHandler, epoch: int) -> list[Array]:
        """Get cluster members by loading from ClusterStatistics artifact.

        Returns:
            List of arrays, where members[i] contains all members of cluster i
            with shape (n_members_i, data_dim)
        """

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


class HierarchicalClusteringExperiment(ClusteringExperiment, ABC):
    """Clustering experiment that supports hierarchical analysis."""

    @abstractmethod
    def get_cluster_hierarchy(self, handler: RunHandler, epoch: int) -> Array:
        """Get hierarchical clustering of clusters.

        Returns:
            linkage_matrix: Scipy-compatible linkage matrix
        """
