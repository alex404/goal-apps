"""Configuration for DifferentiableHMoG implementations."""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
from goal.geometry import (
    Replicated,
)
from goal.models import DifferentiableHMoG, NormalLGM
from jax import Array

from apps.interface import (
    ClusteringDataset,
)
from apps.runtime import STATS_NUM, Logger, MetricDict

from .base import (
    analyze_component,
    cluster_accuracy,
    cluster_assignments,
    clustering_nmi,
    update_stats,
)

# Start logger
log = logging.getLogger(__name__)


STATS_LEVEL = jnp.array(STATS_NUM)
INFO_LEVEL = jnp.array(logging.INFO)


### Helpers ###


def pre_log_epoch_metrics(
    dataset: ClusteringDataset,
    model: NormalLGM,
    logger: Logger,
    params: Array,
    epoch: Array,
    initial_metrics: MetricDict,
    batch_grads: None | Array = None,
    log_freq: int = 1,
) -> None:
    """Log metrics for an epoch."""
    train_data = dataset.train_data
    test_data = dataset.test_data

    def compute_metrics() -> None:
        # Core Performance Metrics

        epoch_train_ll = model.average_log_observable_density(params, train_data)
        epoch_test_ll = model.average_log_observable_density(params, test_data)

        n_samps = train_data.shape[0]
        epoch_scaled_bic = (
            -(model.dim * jnp.log(n_samps) / n_samps - 2 * epoch_train_ll) / 2
        )

        # Start with initial metrics if provided
        metrics: MetricDict = dict(initial_metrics)

        # Add core metrics
        metrics.update(
            {
                "Log-Likelihood/Train": (INFO_LEVEL, epoch_train_ll),
                "Log-Likelihood/Test": (INFO_LEVEL, epoch_test_ll),
                "Log-Likelihood/Scaled BIC": (INFO_LEVEL, epoch_scaled_bic),
            }
        )

        obs_params, int_params, lat_params = model.split_coords(params)
        obs_loc_params, obs_prs_params = model.obs_man.split_coords(obs_params)
        lat_loc_params, lat_prs_params = model.pst_man.split_coords(lat_params)

        metrics = update_stats("Params", "Obs Location", obs_loc_params, metrics)
        metrics = update_stats("Params", "Obs Precision", obs_prs_params, metrics)
        metrics = update_stats("Params", "Obs Interaction", int_params, metrics)
        metrics = update_stats("Params", "Lat Location", lat_loc_params, metrics)
        metrics = update_stats("Params", "Lat Precision", lat_prs_params, metrics)

        # Regularization Metrics

        # Add latent distribution metrics
        metrics.update(
            {
                "Regularization/Loading Sparsity": (
                    STATS_LEVEL,
                    jnp.mean(jnp.abs(int_params) < 1e-6),
                ),
            }
        )

        # Prior statistics
        means = model.to_mean(params)

        obs_means, lwr_int_means, lat_means = model.split_coords(means)
        obs_mean, obs_cov = model.obs_man.split_mean_covariance(obs_means)
        lat_mean, lat_cov = model.pst_man.split_mean_covariance(lat_means)

        metrics = update_stats("Means", "Obs Mean", obs_mean, metrics)
        metrics = update_stats("Means", "Obs Cov", obs_cov, metrics)
        metrics = update_stats("Means", "Obs Interaction", lwr_int_means, metrics)
        metrics = update_stats("Means", "Lat Mean", lat_mean, metrics)
        metrics = update_stats("Means", "Lat Cov", lat_cov, metrics)

        ### Conjugation Stats ###

        lkl_params = model.lkl_fun_man.join_coords(obs_params, int_params)
        rho = model.conjugation_parameters(lkl_params)

        rho_stats = analyze_component(model.prr_man, rho)
        metrics.update(
            {
                "Conjugation/Location Norm": (STATS_LEVEL, rho_stats[0]),
                "Conjugation/Mean Norm": (STATS_LEVEL, rho_stats[1]),
                "Conjugation/Precision Cond": (STATS_LEVEL, rho_stats[2]),
                "Conjugation/Covariance Cond": (STATS_LEVEL, rho_stats[3]),
                "Conjugation/Precision LogDet": (STATS_LEVEL, rho_stats[4]),
                "Conjugation/Covariance LogDet": (STATS_LEVEL, rho_stats[5]),
            }
        )

        ### Grad Norms ###

        def norm_grads(grad: Array) -> Array:
            obs_grad, lwr_int_grad, lat_grad = model.split_coords(grad)
            obs_loc_grad, obs_prs_grad = model.obs_man.split_coords(obs_grad)
            lat_loc_grad, lat_prs_grad = model.pst_man.split_coords(lat_grad)
            return jnp.asarray(
                [
                    jnp.linalg.norm(grad)
                    for grad in [
                        obs_loc_grad,
                        obs_prs_grad,
                        lwr_int_grad,
                        lat_loc_grad,
                        lat_prs_grad,
                    ]
                ]
            )

        if batch_grads is not None:
            batch_man = Replicated(model, batch_grads.shape[0])
            grad_norms = batch_man.map(norm_grads, batch_grads).T

            metrics = update_stats("Grad Norms", "Obs Location", grad_norms[0], metrics)
            metrics = update_stats(
                "Grad Norms", "Obs Precision", grad_norms[1], metrics
            )
            metrics = update_stats(
                "Grad Norms", "Obs Interaction", grad_norms[2], metrics
            )
            metrics = update_stats("Grad Norms", "Lat Location", grad_norms[3], metrics)
            metrics = update_stats(
                "Grad Norms", "Lat Precision", grad_norms[4], metrics
            )

        logger.log_metrics(metrics, epoch + 1)

    def no_op() -> None:
        return None

    jax.lax.cond(epoch % log_freq == 0, compute_metrics, no_op)


def log_epoch_metrics(
    dataset: ClusteringDataset,
    model: DifferentiableHMoG,
    logger: Logger,
    params: Array,
    epoch: Array,
    initial_metrics: MetricDict,
    batch_grads: None | Array = None,
    log_freq: int = 1,
) -> None:
    """Log metrics for an epoch."""
    train_data = dataset.train_data
    test_data = dataset.test_data

    def compute_metrics() -> None:
        # Core Performance Metrics

        epoch_train_ll = model.average_log_observable_density(params, train_data)
        epoch_test_ll = model.average_log_observable_density(params, test_data)

        n_samps = train_data.shape[0]
        epoch_scaled_bic = (
            -(model.dim * jnp.log(n_samps) / n_samps - 2 * epoch_train_ll) / 2
        )
        metrics: MetricDict = dict(initial_metrics)

        metrics.update(
            {
                "Log-Likelihood/Train": (
                    INFO_LEVEL,
                    epoch_train_ll,
                ),
                "Log-Likelihood/Test": (
                    INFO_LEVEL,
                    epoch_test_ll,
                ),
                "Log-Likelihood/Scaled BIC": (
                    INFO_LEVEL,
                    epoch_scaled_bic,
                ),
            }
        )

        # Clustering metrics if dataset has labels
        if dataset.has_labels:
            # Get cluster assignments using the existing function
            train_clusters = cluster_assignments(model, params, train_data)
            test_clusters = cluster_assignments(model, params, test_data)

            # Compute accuracy
            train_acc = cluster_accuracy(dataset.train_labels, train_clusters)
            test_acc = cluster_accuracy(dataset.test_labels, test_clusters)

            # Compute NMI
            n_clusters = model.pst_man.n_categories
            n_classes = dataset.n_classes

            train_nmi = clustering_nmi(
                n_clusters, n_classes, train_clusters, dataset.train_labels
            )
            test_nmi = clustering_nmi(
                n_clusters, n_classes, test_clusters, dataset.test_labels
            )

            # Add to metrics dictionary
            metrics.update(
                {
                    "Clustering/Train Accuracy (Greedy)": (INFO_LEVEL, train_acc),
                    "Clustering/Test Accuracy (Greedy)": (INFO_LEVEL, test_acc),
                    "Clustering/Train NMI": (INFO_LEVEL, train_nmi),
                    "Clustering/Test NMI": (INFO_LEVEL, test_nmi),
                }
            )

        # Raw Parameter Statistics

        obs_params, lwr_int_params, upr_params = model.split_coords(params)
        obs_loc_params, obs_prs_params = model.obs_man.split_coords(obs_params)
        lat_params, upr_int_params, cat_params = model.pst_man.split_coords(upr_params)
        lat_loc_params, lat_prs_params = model.pst_man.obs_man.split_coords(lat_params)

        metrics = update_stats("Params", "Obs Location", obs_loc_params, metrics)
        metrics = update_stats("Params", "Obs Precision", obs_prs_params, metrics)
        metrics = update_stats("Params", "Obs Interaction", lwr_int_params, metrics)
        metrics = update_stats("Params", "Lat Location", lat_loc_params, metrics)
        metrics = update_stats("Params", "Lat Precision", lat_prs_params, metrics)
        metrics = update_stats("Params", "Lat Interaction", upr_int_params, metrics)
        metrics = update_stats("Params", "Categorical", cat_params, metrics)

        # Add latent distribution metrics
        metrics.update(
            {
                "Regularization/Loading Sparsity": (
                    STATS_LEVEL,
                    jnp.mean(jnp.abs(lwr_int_params) < 1e-6),
                ),
            }
        )

        ### Lower Harmonium Prior statistics ###

        means = model.to_mean(params)

        obs_means, lwr_int_means, lat_means = model.split_coords(means)
        obs_mean, obs_cov = model.obs_man.split_mean_covariance(obs_means)
        lat_obs_means, lat_int_means, lat_lat_means = model.pst_man.split_coords(
            lat_means
        )
        lat_mean, lat_cov = model.pst_man.obs_man.split_mean_covariance(lat_obs_means)

        metrics = update_stats("Means", "Obs Mean", obs_mean, metrics)
        metrics = update_stats("Means", "Obs Cov", obs_cov, metrics)
        metrics = update_stats("Means", "Obs Interaction", lwr_int_means, metrics)
        metrics = update_stats("Means", "Lat Mean", lat_mean, metrics)
        metrics = update_stats("Means", "Lat Cov", lat_cov, metrics)
        metrics = update_stats("Means", "Lat Interaction", lat_int_means, metrics)
        metrics = update_stats("Means", "Categorical", lat_lat_means, metrics)

        ### Conjugation and Latent Mixture statistics ###

        lkl_params, mix_params = model.split_conjugated(params)
        rho = model.lwr_hrm.conjugation_parameters(lkl_params)
        cmp_params, _ = model.prr_man.split_natural_mixture(mix_params)

        rho_stats = analyze_component(model.prr_man.obs_man, rho)
        metrics.update(
            {
                "Conjugation/Location Norm": (STATS_LEVEL, rho_stats[0]),
                "Conjugation/Mean Norm": (STATS_LEVEL, rho_stats[1]),
                "Conjugation/Precision Cond": (STATS_LEVEL, rho_stats[2]),
                "Conjugation/Covariance Cond": (STATS_LEVEL, rho_stats[3]),
                "Conjugation/Precision LogDet": (STATS_LEVEL, rho_stats[4]),
                "Conjugation/Covariance LogDet": (STATS_LEVEL, rho_stats[5]),
            }
        )

        cmp_stats = model.prr_man.cmp_man.map(
            lambda cmp: analyze_component(model.prr_man.obs_man, cmp), cmp_params
        ).T

        metrics = update_stats("Components", "Location Norm", cmp_stats[0], metrics)
        metrics = update_stats("Components", "Mean Norm", cmp_stats[1], metrics)
        metrics = update_stats("Components", "Precision Cond", cmp_stats[2], metrics)
        metrics = update_stats("Components", "Covariance Cond", cmp_stats[3], metrics)
        metrics = update_stats("Components", "Precision LogDet", cmp_stats[4], metrics)
        metrics = update_stats("Components", "Covariance LogDet", cmp_stats[5], metrics)

        ### Grad Norms ###

        def norm_grads(grad: Array) -> Array:
            obs_grad, lwr_int_grad, upr_grad = model.split_coords(grad)
            obs_loc_grad, obs_prs_grad = model.obs_man.split_coords(obs_grad)
            lat_grad, upr_int_grad, cat_grad = model.pst_man.split_coords(upr_grad)
            lat_loc_grad, lat_prs_grad = model.pst_man.obs_man.split_coords(lat_grad)
            grads = [
                obs_loc_grad,
                obs_prs_grad,
                lwr_int_grad,
                lat_loc_grad,
                lat_prs_grad,
                upr_int_grad,
                cat_grad,
            ]
            return jnp.asarray([jnp.linalg.norm(grad) for grad in grads])

        if batch_grads is not None:
            batch_man = Replicated(model, batch_grads.shape[0])
            grad_norms = batch_man.map(norm_grads, batch_grads).T

            metrics = update_stats("Grad Norms", "Obs Location", grad_norms[0], metrics)
            metrics = update_stats(
                "Grad Norms", "Obs Precision", grad_norms[1], metrics
            )
            metrics = update_stats(
                "Grad Norms", "Obs Interaction", grad_norms[2], metrics
            )
            metrics = update_stats("Grad Norms", "Lat Location", grad_norms[3], metrics)
            metrics = update_stats(
                "Grad Norms", "Lat Precision", grad_norms[4], metrics
            )
            metrics = update_stats(
                "Grad Norms", "Lat Interaction", grad_norms[5], metrics
            )
            metrics = update_stats("Grad Norms", "Categorical", grad_norms[6], metrics)

        logger.log_metrics(metrics, epoch + 1)

    def no_op() -> None:
        return None

    jax.lax.cond(epoch % log_freq == 0, compute_metrics, no_op)
