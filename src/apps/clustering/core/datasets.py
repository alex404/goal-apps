from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

from jax import Array
from matplotlib.axes import Axes


@dataclass
class Dataset(ABC):
    """Abstract base class for datasets used in clustering applications."""

    cache_dir: Path

    @property
    @abstractmethod
    def train_images(self) -> Array:
        """Training data with shape (n_train, data_dim)."""

    @property
    @abstractmethod
    def test_images(self) -> Array:
        """Test data with shape (n_test, data_dim)."""

    @property
    @abstractmethod
    def data_dim(self) -> int:
        """Dimensionality of each data point."""

    @abstractmethod
    def __init__(self, cache_dir: Path) -> None:
        """Initialize dataset.

        Args:
            cache_dir: Directory for caching downloaded data
        """

    @abstractmethod
    def visualize_observable(
        self, observable: Array, ax: Axes | None = None, **kwargs: Any
    ) -> Axes:
        """Visualize a single observable (e.g. image, time series).

        Args:
            observable: Data point to visualize
            ax: Optional matplotlib axes to plot on
            **kwargs: Additional visualization parameters

        Returns:
            The matplotlib axes containing the visualization
        """


class SupervisedDataset(Dataset, ABC):
    """Abstract base class for datasets with labels."""

    @abstractmethod
    @override
    def __init__(self, cache_dir: Path) -> None:
        """Initialize dataset.

        Args:
            cache_dir: Directory for caching downloaded data
        """

    @property
    @abstractmethod
    def train_labels(self) -> Array:
        """Training labels with shape (n_train,)."""

    @property
    @abstractmethod
    def test_labels(self) -> Array:
        """Test labels with shape (n_test,)."""

    @property
    @abstractmethod
    def n_classes(self) -> int:
        """Number of distinct classes in labels."""
