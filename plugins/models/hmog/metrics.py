"""Epoch metrics logging for HMoG training."""

from __future__ import annotations

import logging

import jax.numpy as jnp
from goal.geometry import Replicated
from goal.models import FullNormal
from jax import Array

from .types import DiagonalHMoG, DiagonalLGM

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

from .analyses.base import analyze_component, cluster_assignments

log = logging.getLogger(__name__)

STATS_LEVEL = jnp.array(STATS_NUM)
INFO_LEVEL = jnp.array(logging.INFO)


def add_conjugation_metrics(
    metrics: MetricDict, normal_man: FullNormal, rho: Array
) -> MetricDict:
    """Add conjugation parameter statistics."""
    rho_stats = analyze_component(normal_man, rho)
    metrics.update({
        "Conjugation/Location Norm": (STATS_LEVEL, rho_stats[0]),
        "Conjugation/Mean Norm": (STATS_LEVEL, rho_stats[1]),
        "Conjugation/Precision Cond": (STATS_LEVEL, rho_stats[2]),
        "Conjugation/Covariance Cond": (STATS_LEVEL, rho_stats[3]),
        "Conjugation/Precision LogDet": (STATS_LEVEL, rho_stats[4]),
        "Conjugation/Covariance LogDet": (STATS_LEVEL, rho_stats[5]),
    })
    return metrics


### LGM Pretraining Metrics ###


def pre_log_epoch_metrics(
    dataset: ClusteringDataset,
    model: DiagonalLGM,
    logger: Logger,
    params: Array,
    epoch: Array,
    initial_metrics: MetricDict,
    batch_grads: Array | None = None,
    log_freq: int = 1,
) -> None:
    """Log metrics during LGM pretraining phase."""
    train_data = dataset.train_data
    test_data = dataset.test_data

    def compute_metrics() -> MetricDict:
        metrics: MetricDict = dict(initial_metrics)

        # Log-likelihood metrics
        train_ll = model.average_log_observable_density(params, train_data)
        test_ll = model.average_log_observable_density(params, test_data)
        metrics = add_ll_metrics(metrics, model.dim, train_ll, test_ll, len(train_data))

        # Parameter decomposition
        obs_params, int_params, lat_params = model.split_coords(params)
        obs_loc, obs_prs = model.obs_man.split_coords(obs_params)
        lat_loc, lat_prs = model.pst_man.split_coords(lat_params)

        # Parameter statistics
        metrics = update_stats("Params", "Obs Location", obs_loc, metrics)
        metrics = update_stats("Params", "Obs Precision", obs_prs, metrics)
        metrics = update_stats("Params", "Obs Interaction", int_params, metrics)
        metrics = update_stats("Params", "Lat Location", lat_loc, metrics)
        metrics = update_stats("Params", "Lat Precision", lat_prs, metrics)

        # Regularization
        metrics["Regularization/Loading Sparsity"] = (
            STATS_LEVEL,
            jnp.mean(jnp.abs(int_params) < 1e-6),
        )

        # Mean statistics
        means = model.to_mean(params)
        obs_means, int_means, lat_means = model.split_coords(means)
        obs_mean, obs_cov = model.obs_man.split_mean_covariance(obs_means)
        lat_mean, lat_cov = model.pst_man.split_mean_covariance(lat_means)

        metrics = update_stats("Means", "Obs Mean", obs_mean, metrics)
        metrics = update_stats("Means", "Obs Cov", obs_cov, metrics)
        metrics = update_stats("Means", "Obs Interaction", int_means, metrics)
        metrics = update_stats("Means", "Lat Mean", lat_mean, metrics)
        metrics = update_stats("Means", "Lat Cov", lat_cov, metrics)

        # Conjugation statistics
        lkl_params = model.lkl_fun_man.join_coords(obs_params, int_params)
        rho = model.conjugation_parameters(lkl_params)
        metrics = add_conjugation_metrics(metrics, model.prr_man, rho)

        # Gradient norms
        if batch_grads is not None:
            def norm_grads(grad: Array) -> Array:
                obs_g, int_g, lat_g = model.split_coords(grad)
                obs_loc_g, obs_prs_g = model.obs_man.split_coords(obs_g)
                lat_loc_g, lat_prs_g = model.pst_man.split_coords(lat_g)
                return jnp.asarray([jnp.linalg.norm(g) for g in [
                    obs_loc_g, obs_prs_g, int_g, lat_loc_g, lat_prs_g
                ]])

            batch_man = Replicated(model, batch_grads.shape[0])
            norms = batch_man.map(norm_grads, batch_grads).T
            for i, name in enumerate(["Obs Location", "Obs Precision", "Obs Interaction",
                                       "Lat Location", "Lat Precision"]):
                metrics = update_stats("Grad Norms", name, norms[i], metrics)

        return metrics

    log_with_frequency(logger, epoch, log_freq, compute_metrics)


### Full HMoG Metrics ###


def log_epoch_metrics(
    dataset: ClusteringDataset,
    model: DiagonalHMoG,
    logger: Logger,
    params: Array,
    epoch: Array,
    initial_metrics: MetricDict,
    batch_grads: Array | None = None,
    log_freq: int = 1,
) -> None:
    """Log metrics during full HMoG training."""
    train_data = dataset.train_data
    test_data = dataset.test_data

    def compute_metrics() -> MetricDict:
        metrics: MetricDict = dict(initial_metrics)

        # Log-likelihood metrics
        train_ll = model.average_log_observable_density(params, train_data)
        test_ll = model.average_log_observable_density(params, test_data)
        metrics = add_ll_metrics(metrics, model.dim, train_ll, test_ll, len(train_data))

        # Clustering metrics (if labels available)
        if dataset.has_labels:
            train_clusters = cluster_assignments(model, params, train_data)
            test_clusters = cluster_assignments(model, params, test_data)

            metrics = add_clustering_metrics(
                metrics,
                n_clusters=model.pst_man.n_categories,
                n_classes=dataset.n_classes,
                train_labels=dataset.train_labels,
                test_labels=dataset.test_labels,
                train_clusters=train_clusters,
                test_clusters=test_clusters,
                cluster_accuracy_fn=cluster_accuracy,
                clustering_nmi_fn=clustering_nmi,
            )

        # Parameter decomposition (HMoG has more components than LGM)
        obs_params, lwr_int, upr_params = model.split_coords(params)
        obs_loc, obs_prs = model.obs_man.split_coords(obs_params)
        lat_params, upr_int, cat_params = model.pst_man.split_coords(upr_params)
        lat_loc, lat_prs = model.pst_man.obs_man.split_coords(lat_params)

        # Parameter statistics
        metrics = update_stats("Params", "Obs Location", obs_loc, metrics)
        metrics = update_stats("Params", "Obs Precision", obs_prs, metrics)
        metrics = update_stats("Params", "Obs Interaction", lwr_int, metrics)
        metrics = update_stats("Params", "Lat Location", lat_loc, metrics)
        metrics = update_stats("Params", "Lat Precision", lat_prs, metrics)
        metrics = update_stats("Params", "Lat Interaction", upr_int, metrics)
        metrics = update_stats("Params", "Categorical", cat_params, metrics)

        # Regularization
        metrics["Regularization/Loading Sparsity"] = (
            STATS_LEVEL,
            jnp.mean(jnp.abs(lwr_int) < 1e-6),
        )

        # Mean statistics
        means = model.to_mean(params)
        obs_means, lwr_int_means, lat_means = model.split_coords(means)
        obs_mean, obs_cov = model.obs_man.split_mean_covariance(obs_means)
        lat_obs_means, lat_int_means, cat_means = model.pst_man.split_coords(lat_means)
        lat_mean, lat_cov = model.pst_man.obs_man.split_mean_covariance(lat_obs_means)

        metrics = update_stats("Means", "Obs Mean", obs_mean, metrics)
        metrics = update_stats("Means", "Obs Cov", obs_cov, metrics)
        metrics = update_stats("Means", "Obs Interaction", lwr_int_means, metrics)
        metrics = update_stats("Means", "Lat Mean", lat_mean, metrics)
        metrics = update_stats("Means", "Lat Cov", lat_cov, metrics)
        metrics = update_stats("Means", "Lat Interaction", lat_int_means, metrics)
        metrics = update_stats("Means", "Categorical", cat_means, metrics)

        # Conjugation and component statistics
        lkl_params, mix_params = model.split_conjugated(params)
        rho = model.lwr_hrm.conjugation_parameters(lkl_params)
        metrics = add_conjugation_metrics(metrics, model.prr_man.obs_man, rho)

        cmp_params, _ = model.prr_man.split_natural_mixture(mix_params)
        cmp_stats = model.prr_man.cmp_man.map(
            lambda cmp: analyze_component(model.prr_man.obs_man, cmp), cmp_params
        ).T
        for i, name in enumerate(["Location Norm", "Mean Norm", "Precision Cond",
                                   "Covariance Cond", "Precision LogDet", "Covariance LogDet"]):
            metrics = update_stats("Components", name, cmp_stats[i], metrics)

        # Gradient norms
        if batch_grads is not None:
            def norm_grads(grad: Array) -> Array:
                obs_g, lwr_int_g, upr_g = model.split_coords(grad)
                obs_loc_g, obs_prs_g = model.obs_man.split_coords(obs_g)
                lat_g, upr_int_g, cat_g = model.pst_man.split_coords(upr_g)
                lat_loc_g, lat_prs_g = model.pst_man.obs_man.split_coords(lat_g)
                return jnp.asarray([jnp.linalg.norm(g) for g in [
                    obs_loc_g, obs_prs_g, lwr_int_g, lat_loc_g, lat_prs_g, upr_int_g, cat_g
                ]])

            batch_man = Replicated(model, batch_grads.shape[0])
            norms = batch_man.map(norm_grads, batch_grads).T
            for i, name in enumerate(["Obs Location", "Obs Precision", "Obs Interaction",
                                       "Lat Location", "Lat Precision", "Lat Interaction", "Categorical"]):
                metrics = update_stats("Grad Norms", name, norms[i], metrics)

        return metrics

    log_with_frequency(logger, epoch, log_freq, compute_metrics)
