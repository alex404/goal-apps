"""Generic dataset abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self

from jax import Array
from matplotlib.axes import Axes

from ..runtime import Artifact
from .analysis import Analysis


@dataclass
class DatasetConfig:
    """Base configuration for datasets."""

    _target_: str


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
        """Render an observation from the dataset.

        Args:
            observable: A single observation from the dataset (data_dim,)
            axes: Matplotlib axes to draw on
        """

    def get_dataset_analyses(self) -> dict[str, Analysis[Self, Any, Artifact]]:
        """Return dataset-specific analysis instances."""
        return {}
