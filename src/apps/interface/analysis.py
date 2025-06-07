"""Shared utilities for GOAL examples."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from matplotlib.figure import Figure

from ..runtime import Artifact, JaxLogger, MetricDict, RunHandler

### Analysis Base Class ###


@dataclass(frozen=True)
class Analysis[D, M, T: Artifact](ABC):
    """Base class for analyses that produce artifacts and visualizations.

    This class standardizes the pattern of generating or loading artifacts
    and creating visualizations from them. Each analysis encapsulates:
    - The logic for generating an artifact from model parameters
    - The visualization function for that artifact type
    - The coordination of loading vs. regenerating artifacts
    """

    @abstractmethod
    def generate(
        self,
        key: Array,
        handler: RunHandler,
        dataset: D,
        model: M,
        epoch: int,
        params: Array,
    ) -> T:
        """Generate the analysis artifact from model parameters."""

    @abstractmethod
    def plot(self, artifact: T, dataset: D) -> Figure:
        """Create visualization from the artifact."""

    @property
    @abstractmethod
    def artifact_type(self) -> type[T]:
        """Return the artifact class for type checking and loading."""

    def metrics(self, artifact: T) -> MetricDict:
        """Return metrics collected during the analysis."""
        return {}

    def process(
        self,
        key: Array,
        handler: RunHandler,
        dataset: D,
        model: M,
        logger: JaxLogger,
        epoch: int,
        params: Array | None = None,
    ) -> None:
        """Process the analysis: generate or load artifact, then visualize and log."""
        if params is not None:
            artifact = self.generate(key, handler, dataset, model, epoch, params)
        else:
            artifact = handler.load_artifact(epoch, self.artifact_type)

        metrics = self.metrics(artifact)

        logger.log_metrics(metrics, jnp.array(epoch))
        logger.log_artifact(handler, epoch, artifact, lambda a: self.plot(a, dataset))
