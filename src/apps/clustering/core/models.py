"""Base protocols for clustering models."""

from abc import ABC, abstractmethod
from functools import partial
from typing import override

import jax
from jax import Array

from ...runtime import RunHandler
from .common import ProbabilisticResults, TwoStageResults
from .datasets import Dataset, SupervisedDataset


class Model[P](ABC):
    """Abstract base class for all unsupervised models."""

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Return the dimensionality of the latent space."""

    @property
    @abstractmethod
    def n_clusters(self) -> int:
        """Return the number of clusters in the model."""

    @abstractmethod
    def initialize(self, key: Array, data: Array) -> P:
        """Initialize model parameters based on data dimensions and statistics."""

    @partial(jax.jit, static_argnums=(0, 1, 2))
    @abstractmethod
    def fit(
        self,
        handler: RunHandler,
        dataset: Dataset,
        params0: P,
        train_sample: Array,
        test_sample: Array,
    ) -> tuple[P, Array, Array]:
        """Train model parameters, returning final parameters and training metrics."""

    @abstractmethod
    def generate(self, params: P, key: Array, n_samples: int) -> Array:
        """Generate new samples using the trained model."""

    @abstractmethod
    def cluster_assignments(self, params: P, data: Array) -> Array:
        """Determine cluster assignments for each sample."""

    @abstractmethod
    def evaluate(
        self, key: Array, handler: RunHandler, dataset: SupervisedDataset
    ) -> ProbabilisticResults | TwoStageResults:
        """Evaluate model on dataset."""

    @abstractmethod
    def get_component_prototypes(self, params: P) -> Array:
        """Get a representative sample for each mixture component.

        Returns:
            Array with shape (n_components, *data_dims) containing a prototype
            for each mixture component in the observation space.
        """


class ProbabilisticModel[P](Model[P], ABC):
    """Abstract base class for models that support likelihood computation."""

    @abstractmethod
    def log_likelihood(self, params: P, data: Array) -> Array:
        """Compute per-sample log likelihood under the model."""

    @abstractmethod
    @override
    def evaluate(
        self, key: Array, handler: RunHandler, dataset: SupervisedDataset
    ) -> ProbabilisticResults:
        """Evaluate probabilistic model on dataset."""


class TwoStageModel[P](Model[P], ABC):
    """Abstract base class for models that support two-stage training."""

    @abstractmethod
    def reconstruction_error(self, params: P, data: Array) -> Array:
        """Compute per-sample reconstruction error using encode/decode."""

    @abstractmethod
    @override
    def evaluate(
        self, key: Array, handler: RunHandler, dataset: SupervisedDataset
    ) -> TwoStageResults:
        """Evaluate two-stage model on dataset."""
