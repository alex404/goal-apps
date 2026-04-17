"""Generic protocols for probabilistic models.

These protocols define optional capabilities that any probabilistic model
can implement, enabling generic analyses across model types.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jax import Array


@runtime_checkable
class HasLogLikelihood(Protocol):
    """Protocol for models that can compute log-likelihood of data.

    Models implementing this protocol can use likelihood-based analyses
    and metrics. ``ClusteringModel.metric_names`` uses ``isinstance`` against
    this protocol to decide whether to include ``LL_METRIC_KEYS``, so a
    model that implements ``log_likelihood`` gets LL metrics declared
    automatically — no manual override needed.
    """

    def log_likelihood(self, params: Array, data: Array) -> float:
        """Compute average log-likelihood of data under the model.

        Args:
            params: Model parameters
            data: Data array of shape (n_samples, data_dim)

        Returns:
            Average log-likelihood across all samples
        """
        ...


class IsGenerative(Protocol):
    """Protocol for models that can generate new samples.

    Models implementing this protocol can be used with generative
    analyses to produce synthetic samples from the learned distribution.
    """

    def generate(self, params: Array, key: Array, n_samples: int) -> Array:
        """Generate samples from the model distribution.

        Args:
            params: Model parameters
            key: Random key for sampling
            n_samples: Number of samples to generate

        Returns:
            Generated samples array of shape (n_samples, data_dim)
        """
        ...
