import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, override

from jax import Array
from omegaconf import MISSING

from ..runtime import Artifact, Logger, RunHandler
from .analysis import Analysis
from .dataset import ClusteringDataset, Dataset

### Logging Setup ###

log = logging.getLogger(__name__)

### Generic Models ###


@dataclass
class ModelConfig:
    """Base configuration for models."""

    _target_: str


class Model[D: Dataset](ABC):
    """
    Base class for statistical models with their training procedures.

    In this library, a 'model' encompasses more than just the statistical model itself, but also the training algorithm optimized for that model's structure, and analysis methods specific to the model.
    """

    @property
    @abstractmethod
    def n_epochs(self) -> int:
        """Number of epochs used for training the model."""

    @abstractmethod
    def analyze(
        self,
        key: Array,
        handler: RunHandler,
        logger: Logger,
        dataset: D,
    ) -> None:
        """Run analysis suite based on trained model."""

    @abstractmethod
    def train(
        self,
        key: Array,
        handler: RunHandler,
        logger: Logger,
        dataset: D,
    ) -> None:
        """Train model on dataset."""

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
            Model parameters
        """
        if handler.resolve_epoch is None:
            # Fresh run - initialize
            log.info("Initializing model parameters")
            return self.initialize_model(key, data)
        # Continuation - load
        log.info(f"Loading parameters from epoch {handler.resolve_epoch}")
        return handler.load_params()

    @abstractmethod
    def get_analyses(self, dataset: D) -> list[Analysis[D, Any, Any]]:
        """Return a list of analyses to run after training.

        Each analysis should be an instance of Analysis with the appropriate dataset and model types.
        """
        pass

    def process_checkpoint[A: Artifact](
        self,
        key: Array,
        handler: RunHandler,
        logger: Logger,
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
            analysis.process(key, handler, logger, dataset, model, epoch, params)

        # 3. Save current metric state (periodic backup)
        handler.save_metrics(logger.get_metric_buffer())

        log.info(f"Epoch {epoch} checkpoint complete.")


### Clustering Configs ###


@dataclass
class ClusteringModelConfig(ModelConfig):
    """Base configuration for clustering models."""

    _target_: str
    data_dim: int = MISSING
    n_clusters: int = MISSING


class ClusteringModel(Model[ClusteringDataset], ABC):
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
        logger: Logger,
        dataset: ClusteringDataset,
    ) -> None:
        """Evaluate model on dataset."""


class HierarchicalClusteringModel(ClusteringModel, ABC):
    """Clustering model that supports hierarchical analysis."""

    @abstractmethod
    def get_cluster_hierarchy(self, handler: RunHandler, epoch: int) -> Array:
        """Get hierarchical clustering of clusters.

        Returns:
            linkage_matrix: Scipy-compatible linkage matrix
        """
