"""Generic model abstractions."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from jax import Array

from ..runtime import Artifact, Logger, RunHandler
from .analysis import Analysis
from .dataset import Dataset

log = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Base configuration for models."""

    _target_: str


class Model[D: Dataset](ABC):
    """Base class for statistical models with their training procedures.

    In this library, a 'model' encompasses more than just the statistical model
    itself, but also the training algorithm optimized for that model's structure,
    and analysis methods specific to the model.
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
            log.info("Initializing model parameters")
            return self.initialize_model(key, data)
        log.info(f"Loading parameters from epoch {handler.resolve_epoch}")
        return handler.load_params()

    @abstractmethod
    def get_analyses(self, dataset: D) -> list[Analysis[D, Any, Any]]:
        """Return a list of analyses to run after training."""

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
        if params is not None:
            handler.save_params(params, epoch)

        for analysis in self.get_analyses(dataset):
            analysis.process(key, handler, logger, dataset, model, epoch, params)

        handler.save_metrics(logger.get_metric_buffer())
        log.info(f"Epoch {epoch} checkpoint complete.")
