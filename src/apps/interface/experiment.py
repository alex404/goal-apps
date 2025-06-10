import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, override

from jax import Array
from omegaconf import MISSING

from ..runtime import Artifact, JaxLogger, RunHandler
from .analysis import Analysis
from .dataset import ClusteringDataset, Dataset

log = logging.getLogger(__name__)

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

    @abstractmethod
    def initialize_model(self, key: Array, data: Array) -> Array:
        """Initialize model parameters based on data dimensions and statistics."""

    def prepare_model(self, key: Array, handler: RunHandler, data: Array) -> Array:
        """Initialize fresh parameters or load from a previous epoch.

        Args:
            key: Random key for initialization
            handler: RunHandler with from_epoch set
            data: Training data for initialization

        Returns:
            Model parameters as Array
        """
        if handler.from_epoch is None:
            # Fresh run - initialize
            log.info("Initializing model parameters")
            return self.initialize_model(key, data)
        # Continuation - load
        log.info(f"Loading parameters from epoch {handler.from_epoch}")
        return handler.load_params()

    @abstractmethod
    def get_analyses(self, dataset: D) -> list[Analysis[D, Any, Any]]:
        """Return a list of analyses to run after training.

        Each analysis should be an instance of Analysis with the appropriate
        dataset and model types.
        """
        pass

    def process_checkpoint[A: Artifact](
        self,
        key: Array,
        handler: RunHandler,
        logger: JaxLogger,
        dataset: D,
        model: Any,
        epoch: int,
        params: Array | None = None,
    ) -> None:
        """Complete epoch checkpointing: save params, run analyses, save metrics."""
        # 1. Save parameters
        if params is not None:
            handler.save_params(params, epoch)

        # 2. Run each analysis (generate artifacts + log metrics)
        for analysis in self.get_analyses(dataset):
            analysis.process(key, handler, dataset, model, logger, epoch, params)

        # 3. Save current metric state (periodic backup)
        current_metrics = logger.get_current_metrics()
        if current_metrics:  # Only save if we have metrics
            handler.save_metrics(current_metrics)


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
