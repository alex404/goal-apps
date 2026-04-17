"""Generic model abstractions."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from jax import Array

from ..runtime import Logger, RunHandler
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
        """Total epochs the full training pipeline will run.

        For multi-phase models (e.g. HMoG), this is the sum across all
        phases. Used by the CLI for progress reporting and by Optuna
        pruners to know the horizon.
        """

    @property
    @abstractmethod
    def n_parameters(self) -> int:
        """Total number of free scalar parameters in the model.

        Used by ``add_ll_metrics`` to compute BIC. Count the flat vector
        size, not logical parameter objects.
        """

    @abstractmethod
    def analyze(
        self,
        key: Array,
        handler: RunHandler,
        logger: Logger,
        dataset: D,
    ) -> None:
        """Run analyses on a previously trained model.

        Typical implementation: load params via ``handler.load_params()``,
        then iterate ``get_analyses(dataset)`` calling
        ``analysis.process(...)``. Intended to be called from the ``analyze``
        CLI subcommand against an existing run directory; does not re-train.
        """

    @abstractmethod
    def train(
        self,
        key: Array,
        handler: RunHandler,
        logger: Logger,
        dataset: D,
    ) -> None:
        """Run the training pipeline end-to-end.

        Expected to:
          - call ``prepare_model(key, handler, data)`` to either initialize
            fresh params or resume from ``handler.resolve_epoch``;
          - drive a training loop, logging metrics via ``logger.log_metrics``;
          - invoke ``process_checkpoint(...)`` at checkpoint epochs (saves
            params, runs analyses, saves metrics, checks Optuna pruning);
          - call ``logger.finalize(handler)`` is handled by the caller —
            do NOT finalize here.
        """

    @abstractmethod
    def initialize_model(self, key: Array, data: Array) -> Array:
        """Produce a fresh parameter vector from ``data`` statistics.

        Called only on a fresh run (no checkpoint to resume from). Should be
        a pure JAX computation — no side effects. Returns a flat parameter
        array compatible with ``load_params``.
        """

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

    def metric_names(self, dataset: D) -> frozenset[str]:
        """Every metric key this model's pipeline can emit for ``dataset``.

        Default: union of ``metric_names`` across ``get_analyses(dataset)``.
        ``ClusteringModel`` extends this with LL/clustering/training keys
        auto-detected via protocols, so most clustering subclasses don't
        override this method — they just declare ``training_metric_keys``.

        Used by the optuna study-creation validator to check that the
        requested optimization metric is actually produced.
        """
        names: set[str] = set()
        for analysis in self.get_analyses(dataset):
            names |= analysis.metric_names
        return frozenset(names)

    def process_checkpoint(
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
        with logger.pause_timing():
            if params is not None:
                handler.save_params(params, epoch)

            for analysis in self.get_analyses(dataset):
                analysis.process(key, handler, logger, dataset, model, epoch, params)

            handler.save_metrics(logger.get_metric_buffer())

        logger.check_pruning(epoch)
        log.info(f"Epoch {epoch} checkpoint complete.")
