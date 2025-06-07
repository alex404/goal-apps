from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

from jax import Array
from omegaconf import MISSING

from ..runtime import JaxLogger, RunHandler
from .dataset import ClusteringDataset, Dataset

### Generic Experiments ###


@dataclass
class ExperimentConfig:
    """Base configuration for models."""

    _target_: str


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


### Clustering Configs ###


@dataclass
class ClusteringExperimentConfig(ExperimentConfig):
    """Base configuration for clustering models."""

    _target_: str
    data_dim: int = MISSING
    n_clusters: int = MISSING


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
