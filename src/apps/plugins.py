from abc import ABC, abstractmethod

from jax import Array

from .runtime.handler import RunHandler
from .runtime.logger import JaxLogger


class Dataset(ABC):
    """Root class for all datasets."""

    @property
    @abstractmethod
    def data_dim(self) -> int:
        """Dimensionality of each data point."""

    pass


class Model[D: Dataset](ABC):
    """Root class for all models."""

    @property
    @abstractmethod
    def n_epochs(self) -> int:
        """Dimensionality of each data point."""

    @property
    @abstractmethod
    def metrics(self) -> list[str]:
        """Dimensionality of each data point."""

    @abstractmethod
    def run_experiment(
        self,
        key: Array,
        handler: RunHandler,
        dataset: D,
        logger: JaxLogger,
    ) -> None:
        """Evaluate model on dataset."""

    pass
