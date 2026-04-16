"""Shared metric computation helpers for model training.

This module provides reusable metric computation functions that can be shared
across different models (HMOG, MFA, etc.).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

from .logger import Logger
from .util import INFO_LEVEL, STATS_LEVEL, MetricDict

log = logging.getLogger(__name__)


LL_METRIC_KEYS: frozenset[str] = frozenset({
    "Log-Likelihood/Train",
    "Log-Likelihood/Test",
    "Log-Likelihood/Scaled BIC",
})


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
    metrics.update({
        "Log-Likelihood/Train": (INFO_LEVEL, train_ll),
        "Log-Likelihood/Test": (INFO_LEVEL, test_ll),
        "Log-Likelihood/Scaled BIC": (INFO_LEVEL, scaled_bic),
    })
    return metrics


L1_L2_METRIC_KEYS: frozenset[str] = frozenset({
    "Regularization/L1 Norm",
    "Regularization/L1 Penalty",
    "Regularization/L2 Norm",
    "Regularization/L2 Penalty",
})


def l1_l2_regularizer(
    params: Array, int_params: Array, l1_reg: float, l2_reg: float
) -> tuple[Array, MetricDict]:
    """Compute L1 (on interactions) + L2 (on all params) regularization."""
    l1_norm = jnp.sum(jnp.abs(int_params))
    l1_loss = l1_reg * l1_norm
    l2_norm = jnp.sum(jnp.square(params))
    l2_loss = l2_reg * l2_norm
    metrics: MetricDict = {
        "Regularization/L1 Norm": (STATS_LEVEL, l1_norm),
        "Regularization/L1 Penalty": (STATS_LEVEL, l1_loss),
        "Regularization/L2 Norm": (STATS_LEVEL, l2_norm),
        "Regularization/L2 Penalty": (STATS_LEVEL, l2_loss),
    }
    return l1_loss + l2_loss, metrics


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
