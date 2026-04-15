"""Shared metric computation helpers for model training.

This module provides reusable metric computation functions that can be shared
across different models (HMOG, MFA, etc.).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypedDict, cast

import jax
import jax.numpy as jnp
from jax import Array

from .logger import Logger
from .util import STATS_NUM, MetricDict

log = logging.getLogger(__name__)

STATS_LEVEL = jnp.array(STATS_NUM)
INFO_LEVEL = jnp.array(logging.INFO)


LLMetrics = TypedDict(
    "LLMetrics",
    {
        "Log-Likelihood/Train": tuple[Array, Array],
        "Log-Likelihood/Test": tuple[Array, Array],
        "Log-Likelihood/Scaled BIC": tuple[Array, Array],
    },
)


def as_metric_dict(td: Any) -> MetricDict:
    """Widen a homogeneously-typed TypedDict to ``MetricDict``.

    Python's type system can't express that a TypedDict whose fields all
    have value type ``V`` is a ``dict[str, V]`` (``dict`` is invariant in
    ``V``, and TypedDicts model fields nominally). This function is the
    sole bridge: one place, one cast, documented.

    Preconditions (enforced only by caller discipline):
    - Every value type declared in the TypedDict must be
      ``tuple[Array, Array]``.
    """
    return cast(MetricDict, td)


def add_ll_metrics(
    metrics: MetricDict,
    model_dim: int,
    train_ll: Array,
    test_ll: Array,
    n_train_samples: int,
) -> MetricDict:
    """Add log-likelihood and BIC metrics.

    Args:
        metrics: Existing metrics dict to update
        model_dim: Number of model parameters
        train_ll: Average log-likelihood on training data
        test_ll: Average log-likelihood on test data
        n_train_samples: Number of training samples

    Returns:
        Updated metrics dict with:
        - Log-Likelihood/Train
        - Log-Likelihood/Test
        - Log-Likelihood/Scaled BIC
    """
    scaled_bic = -(model_dim * jnp.log(n_train_samples) / n_train_samples - 2 * train_ll) / 2
    ll: LLMetrics = {
        "Log-Likelihood/Train": (INFO_LEVEL, train_ll),
        "Log-Likelihood/Test": (INFO_LEVEL, test_ll),
        "Log-Likelihood/Scaled BIC": (INFO_LEVEL, scaled_bic),
    }
    metrics.update(as_metric_dict(ll))
    return metrics


def log_with_frequency(
    logger: Logger,
    epoch: Array,
    log_freq: int,
    compute_fn: Callable[[], MetricDict],
) -> None:
    """Log metrics at specified frequency using jax.lax.cond.

    This function is JIT-compatible and only computes/logs metrics
    when epoch is a multiple of log_freq.

    Args:
        logger: Logger instance for metric logging
        epoch: Current epoch number (0-indexed)
        log_freq: Log every log_freq epochs
        compute_fn: Function that computes and returns MetricDict
    """
    def do_log() -> None:
        metrics = compute_fn()
        logger.log_metrics(metrics, epoch + 1)

    def no_op() -> None:
        pass

    jax.lax.cond(epoch % log_freq == 0, do_log, no_op)
