from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import override

from jax import Array

from ..plugins import Dataset, Model
from ..runtime.handler import RunHandler
from ..runtime.logger import ArtifactType, JaxLogger

### Datasets ###


class ClusteringDataset(Dataset, ABC):
    """Abstract base class for datasets used in clustering applications."""

    cache_dir: Path

    @abstractmethod
    def observable_to_artifact(self, obs: Array) -> tuple[Array, ArtifactType]:
        """Return the type of observable artifact for this dataset."""


### Models ###


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
    def cluster_assignments(self, params: Array, data: Array) -> Array:
        """Determine cluster assignments for each sample."""

    @abstractmethod
    @override
    def run_experiment(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        logger: JaxLogger,
    ) -> None:
        """Evaluate model on dataset."""

    @abstractmethod
    def get_component_prototypes(self, params: Array) -> Array:
        """Get a representative sample for each mixture component.

        Returns:
            Array with shape (n_components, *data_dims) containing a prototype
            for each mixture component in the observation space.
        """
