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
from .util import STATS_NUM, MetricDict

log = logging.getLogger(__name__)

STATS_LEVEL = jnp.array(STATS_NUM)
INFO_LEVEL = jnp.array(logging.INFO)


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


def add_clustering_metrics(
    metrics: MetricDict,
    n_clusters: int,
    n_classes: int,
    train_labels: Array,
    test_labels: Array,
    train_clusters: Array,
    test_clusters: Array,
    clustering_nmi_fn: Callable[[int, int, Array, Array], Array],
) -> MetricDict:
    """Add clustering evaluation metrics.

    The cluster→class mapping is derived from the training split and applied
    to both splits, so test accuracy is a proper held-out evaluation.

    Args:
        metrics: Existing metrics dict to update
        n_clusters: Number of clusters in the model
        n_classes: Number of ground-truth classes
        train_labels: Ground-truth labels for training data
        test_labels: Ground-truth labels for test data
        train_clusters: Predicted cluster assignments for training data
        test_clusters: Predicted cluster assignments for test data
        clustering_nmi_fn: Function to compute normalized mutual information

    Returns:
        Updated metrics dict with:
        - Clustering/Train Accuracy
        - Clustering/Test Accuracy
        - Clustering/Train NMI
        - Clustering/Test NMI
    """
    # Fit cluster→class mapping on training data, apply to both splits
    # (lazy import avoids circular dependency between runtime and interface.clustering)
    from apps.interface.clustering.metrics import fit_cluster_mapping  # noqa: PLC0415
    train_mapping = fit_cluster_mapping(train_labels, train_clusters)
    train_acc = jnp.mean(train_mapping[jnp.clip(train_clusters, 0, 99)] == train_labels)
    test_acc = jnp.mean(train_mapping[jnp.clip(test_clusters, 0, 99)] == test_labels)

    train_nmi = clustering_nmi_fn(n_clusters, n_classes, train_clusters, train_labels)
    test_nmi = clustering_nmi_fn(n_clusters, n_classes, test_clusters, test_labels)

    metrics.update({
        "Clustering/Train Accuracy": (INFO_LEVEL, train_acc),
        "Clustering/Test Accuracy": (INFO_LEVEL, test_acc),
        "Clustering/Train NMI": (INFO_LEVEL, train_nmi),
        "Clustering/Test NMI": (INFO_LEVEL, test_nmi),
    })
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
