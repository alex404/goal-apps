"""Epoch metrics logging for MFA training."""

from __future__ import annotations

import logging

import jax.numpy as jnp
from goal.geometry import Replicated
from goal.models import Normal
from goal.models.graphical.mixture import CompleteMixtureOfSymmetric
from jax import Array

from apps.interface import ClusteringDataset
from apps.interface.clustering import cluster_accuracy, clustering_nmi
from apps.runtime import (
    STATS_NUM,
    Logger,
    MetricDict,
    add_clustering_metrics,
    add_ll_metrics,
    log_with_frequency,
    update_stats,
)

log = logging.getLogger(__name__)

# Type alias for MFA model
type MFA = CompleteMixtureOfSymmetric[Normal, Normal]

STATS_LEVEL = jnp.array(STATS_NUM)
INFO_LEVEL = jnp.array(logging.INFO)


def cluster_assignments(mfa: MFA, params: Array, data: Array) -> Array:
    """Compute hard cluster assignments for data points."""
    import jax

    return jax.lax.map(
        lambda x: mfa.posterior_hard_assignment(params, x),
        data,
        batch_size=2048,
    )


def log_epoch_metrics(
    dataset: ClusteringDataset,
    mfa: MFA,
    logger: Logger,
    params: Array,
    epoch: Array,
    initial_metrics: MetricDict,
    batch_grads: Array | None = None,
    log_freq: int = 1,
) -> None:
    """Log metrics during MFA training.

    Args:
        dataset: Dataset with train/test data and labels
        mfa: MFA model instance
        logger: Logger for metric output
        params: Current model parameters
        epoch: Current epoch number (0-indexed)
        initial_metrics: Metrics from regularization (passed from trainer)
        batch_grads: Optional gradient arrays for gradient norm logging
        log_freq: Log every log_freq epochs
    """
    train_data = dataset.train_data
    test_data = dataset.test_data

    def compute_metrics() -> MetricDict:
        metrics: MetricDict = dict(initial_metrics)

        # Log-likelihood metrics
        train_ll = mfa.average_log_observable_density(params, train_data)
        test_ll = mfa.average_log_observable_density(params, test_data)
        metrics = add_ll_metrics(metrics, mfa.dim, train_ll, test_ll, len(train_data))

        # Clustering metrics (if labels available)
        if dataset.has_labels:
            train_clusters = cluster_assignments(mfa, params, train_data)
            test_clusters = cluster_assignments(mfa, params, test_data)

            metrics = add_clustering_metrics(
                metrics,
                n_clusters=mfa.lat_man.n_categories,
                n_classes=dataset.n_classes,
                train_labels=dataset.train_labels,
                test_labels=dataset.test_labels,
                train_clusters=train_clusters,
                test_clusters=test_clusters,
                cluster_accuracy_fn=cluster_accuracy,
                clustering_nmi_fn=clustering_nmi,
            )

        # Parameter decomposition
        obs_params, int_params, lat_params = mfa.split_coords(params)
        obs_loc, obs_prs = mfa.bas_hrm.obs_man.split_coords(obs_params)

        # Parameter statistics
        metrics = update_stats("Params", "Obs Location", obs_loc, metrics)
        metrics = update_stats("Params", "Obs Precision", obs_prs, metrics)
        metrics = update_stats("Params", "Interaction", int_params, metrics)

        # Split latent mixture parameters
        lat_comp_params, lat_cat_params = mfa.lat_man.split_natural_mixture(lat_params)
        metrics = update_stats("Params", "Lat Components", lat_comp_params, metrics)
        metrics = update_stats("Params", "Categorical", lat_cat_params, metrics)

        # Mean statistics
        means = mfa.to_mean(params)
        obs_means, int_means, lat_means = mfa.split_coords(means)
        obs_mean, obs_cov = mfa.bas_hrm.obs_man.split_mean_covariance(obs_means)

        metrics = update_stats("Means", "Obs Mean", obs_mean, metrics)
        metrics = update_stats("Means", "Obs Cov", obs_cov, metrics)
        metrics = update_stats("Means", "Interaction", int_means, metrics)

        # Latent mixture means
        lat_comp_means, lat_cat_means = mfa.lat_man.split_mean_mixture(lat_means)
        metrics = update_stats("Means", "Lat Components", lat_comp_means, metrics)
        metrics = update_stats("Means", "Categorical", lat_cat_means, metrics)

        # Gradient norms (if available)
        if batch_grads is not None:
            def norm_grads(grad: Array) -> Array:
                obs_g, int_g, lat_g = mfa.split_coords(grad)
                obs_loc_g, obs_prs_g = mfa.bas_hrm.obs_man.split_coords(obs_g)
                return jnp.asarray([
                    jnp.linalg.norm(obs_loc_g),
                    jnp.linalg.norm(obs_prs_g),
                    jnp.linalg.norm(int_g),
                    jnp.linalg.norm(lat_g),
                ])

            batch_man = Replicated(mfa, batch_grads.shape[0])
            norms = batch_man.map(norm_grads, batch_grads).T
            for i, name in enumerate(["Obs Location", "Obs Precision", "Interaction", "Latent"]):
                metrics = update_stats("Grad Norms", name, norms[i], metrics)

        return metrics

    log_with_frequency(logger, epoch, log_freq, compute_metrics)
