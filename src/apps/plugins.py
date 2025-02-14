from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import override

from jax import Array
from matplotlib.axes import Axes

from .runtime.handler import Artifact, JSONDict, RunHandler
from .runtime.logger import JaxLogger

### Interfaces ###


@dataclass(frozen=True)
class ObservableArtifact(Artifact):
    """Artifact wrapping a single observable."""

    obs: Array
    shape: tuple[int, int]

    @override
    def to_json(self) -> JSONDict:
        return {"obs": self.obs.tolist(), "shape": list(self.shape)}

    @classmethod
    @override
    def from_json(cls, json_dict: JSONDict) -> ObservableArtifact:  # pyright: ignore[reportIncompatibleMethodOverride]
        return cls(
            obs=Array(json_dict["obs"]),
            shape=tuple(json_dict["shape"]),
        )


class Dataset(ABC):
    """Root class for all datasets."""

    @property
    @abstractmethod
    def data_dim(self) -> int:
        """Dimensionality of each data point."""

    @property
    @abstractmethod
    def train_data(self) -> Array:
        """Training data with shape (n_train, data_dim)."""

    @property
    @abstractmethod
    def test_data(self) -> Array:
        """Test data with shape (n_test, data_dim)."""

    @abstractmethod
    def observable_artifact(self, observable: Array) -> ObservableArtifact:
        """Convert an observable to an artifact for logging."""

    @staticmethod
    @abstractmethod
    def paint_observable(observable: ObservableArtifact, axes: Axes):
        """A function for rendering an observation from the dataset."""

    pass


dataclass(frozen=True)


class Model[D: Dataset](ABC):
    """Root class for all models."""

    @property
    @abstractmethod
    def n_epochs(self) -> int:
        """Dimensionality of each data point."""

    @abstractmethod
    def run_analysis(
        self,
        key: Array,
        handler: RunHandler,
        dataset: D,
    ) -> None:
        """Evaluate model on dataset."""

    @abstractmethod
    def run_experiment(
        self,
        key: Array,
        handler: RunHandler,
        dataset: D,
        logger: JaxLogger,
    ) -> None:
        """Evaluate model on dataset."""


### Clustering ###

# Datasets


class ClusteringDataset(Dataset, ABC):
    """Abstract base class for datasets used in clustering applications."""

    cache_dir: Path


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
