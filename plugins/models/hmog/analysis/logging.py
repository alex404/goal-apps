"""Configuration for HMoG implementations."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from goal.geometry import (
    Mean,
    Natural,
    Point,
    Replicated,
)
from jax import Array

from apps.interface import (
    ClusteringDataset,
    ClusteringExperiment,
)
from apps.runtime import STATS_NUM, JaxLogger, MetricDict, RunHandler

from ..base import LGM, HMoG
from .base import (
    analyze_component,
    cluster_accuracy,
    cluster_assignments,
    clustering_nmi,
    update_stats,
)
from .clusters import ClusterStatisticsAnalysis
from .generative import GenerativeExamplesAnalysis
from .hierarchy import (
    CoAssignmentHierarchyAnalysis,
    KLHierarchyAnalysis,
)
from .loadings import LoadingMatrixAnalysis
from .merge import (
    CoAssignmentMergeAnalysis,
    KLMergeAnalysis,
    OptimalMergeAnalysis,
)

# Start logger
log = logging.getLogger(__name__)


STATS_LEVEL = jnp.array(STATS_NUM)
INFO_LEVEL = jnp.array(logging.INFO)


### Analysis Args ###


@dataclass(frozen=True)
class AnalysisArgs:
    """Arguments for HMoG analysis."""

    from_scratch: bool
    epoch: int | None


### Helpers ###


def pre_log_epoch_metrics[H: LGM](
    dataset: ClusteringDataset,
    model: H,
    logger: JaxLogger,
    params: Point[Natural, H],
    epoch: Array,
    initial_metrics: MetricDict,
    batch_grads: None | Point[Mean, Replicated[H]] = None,
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

        obs_params, int_params, lat_params = model.split_params(params)
        obs_loc_params, obs_prs_params = model.obs_man.split_params(obs_params)
        lat_loc_params, lat_prs_params = model.lat_man.split_params(lat_params)

        metrics = update_stats("Params", "Obs Location", obs_loc_params.array, metrics)
        metrics = update_stats("Params", "Obs Precision", obs_prs_params.array, metrics)
        metrics = update_stats("Params", "Obs Interaction", int_params.array, metrics)
        metrics = update_stats("Params", "Lat Location", lat_loc_params.array, metrics)
        metrics = update_stats("Params", "Lat Precision", lat_prs_params.array, metrics)

        # Regularization Metrics

        # Add latent distribution metrics
        metrics.update(
            {
                "Regularization/Loading Sparsity": (
                    STATS_LEVEL,
                    jnp.mean(jnp.abs(int_params.array) < 1e-6),
                ),
            }
        )

        # Prior statistics
        means = model.to_mean(params)

        obs_means, lwr_int_means, lat_means = model.split_params(means)
        obs_mean, obs_cov = model.obs_man.split_mean_covariance(obs_means)
        lat_mean, lat_cov = model.lat_man.split_mean_covariance(lat_means)

        metrics = update_stats("Means", "Obs Mean", obs_mean.array, metrics)
        metrics = update_stats("Means", "Obs Cov", obs_cov.array, metrics)
        metrics = update_stats("Means", "Obs Interaction", lwr_int_means.array, metrics)
        metrics = update_stats("Means", "Lat Mean", lat_mean.array, metrics)
        metrics = update_stats("Means", "Lat Cov", lat_cov.array, metrics)

        ### Conjugation Stats ###

        lkl_params = model.lkl_man.join_params(obs_params, int_params)
        rho = model.conjugation_parameters(lkl_params)

        rho_stats = analyze_component(model.con_lat_man, rho)
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

        def norm_grads(grad: Point[Mean, H]) -> Array:
            obs_grad, lwr_int_grad, lat_grad = model.split_params(grad)
            obs_loc_grad, obs_prs_grad = model.obs_man.split_params(obs_grad)
            lat_loc_grad, lat_prs_grad = model.lat_man.split_params(lat_grad)
            return jnp.asarray(
                [
                    jnp.linalg.norm(grad.array)
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
            batch_man: Replicated[H] = Replicated(model, batch_grads.shape[0])
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


def log_epoch_metrics[H: HMoG](
    dataset: ClusteringDataset,
    model: H,
    logger: JaxLogger,
    params: Point[Natural, H],
    epoch: Array,
    initial_metrics: MetricDict,
    batch_grads: None | Point[Mean, Replicated[H]] = None,
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
            train_clusters = cluster_assignments(model, params.array, train_data)
            test_clusters = cluster_assignments(model, params.array, test_data)

            # Compute accuracy
            train_acc = cluster_accuracy(dataset.train_labels, train_clusters)
            test_acc = cluster_accuracy(dataset.test_labels, test_clusters)

            # Compute NMI
            n_clusters = model.upr_hrm.n_categories
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

        obs_params, lwr_int_params, upr_params = model.split_params(params)
        obs_loc_params, obs_prs_params = model.obs_man.split_params(obs_params)
        lat_params, upr_int_params, cat_params = model.upr_hrm.split_params(upr_params)
        lat_loc_params, lat_prs_params = model.upr_hrm.obs_man.split_params(lat_params)

        metrics = update_stats("Params", "Obs Location", obs_loc_params.array, metrics)
        metrics = update_stats("Params", "Obs Precision", obs_prs_params.array, metrics)
        metrics = update_stats(
            "Params", "Obs Interaction", lwr_int_params.array, metrics
        )
        metrics = update_stats("Params", "Lat Location", lat_loc_params.array, metrics)
        metrics = update_stats("Params", "Lat Precision", lat_prs_params.array, metrics)
        metrics = update_stats(
            "Params", "Lat Interaction", upr_int_params.array, metrics
        )
        metrics = update_stats("Params", "Categorical", cat_params.array, metrics)

        # Add latent distribution metrics
        metrics.update(
            {
                "Regularization/Loading Sparsity": (
                    STATS_LEVEL,
                    jnp.mean(jnp.abs(lwr_int_params.array) < 1e-6),
                ),
            }
        )

        ### Lower Harmonium Prior statistics ###

        means = model.to_mean(params)

        obs_means, lwr_int_means, lat_means = model.split_params(means)
        obs_mean, obs_cov = model.obs_man.split_mean_covariance(obs_means)
        lat_obs_means, lat_int_means, lat_lat_means = model.lat_man.split_params(
            lat_means
        )
        lat_mean, lat_cov = model.lat_man.obs_man.split_mean_covariance(lat_obs_means)

        metrics = update_stats("Means", "Obs Mean", obs_mean.array, metrics)
        metrics = update_stats("Means", "Obs Cov", obs_cov.array, metrics)
        metrics = update_stats("Means", "Obs Interaction", lwr_int_means.array, metrics)
        metrics = update_stats("Means", "Lat Mean", lat_mean.array, metrics)
        metrics = update_stats("Means", "Lat Cov", lat_cov.array, metrics)
        metrics = update_stats("Means", "Lat Interaction", lat_int_means.array, metrics)
        metrics = update_stats("Means", "Categorical", lat_lat_means.array, metrics)

        ### Conjugation and Latent Mixture statistics ###

        lkl_params, mix_params = model.split_conjugated(params)
        rho = model.lwr_hrm.conjugation_parameters(lkl_params)
        cmp_params, _ = model.con_upr_hrm.split_natural_mixture(mix_params)

        rho_stats = analyze_component(model.con_upr_hrm.obs_man, rho)
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

        cmp_stats = model.con_upr_hrm.cmp_man.map(
            lambda cmp: analyze_component(model.con_upr_hrm.obs_man, cmp), cmp_params
        ).T

        metrics = update_stats("Components", "Location Norm", cmp_stats[0], metrics)
        metrics = update_stats("Components", "Mean Norm", cmp_stats[1], metrics)
        metrics = update_stats("Components", "Precision Cond", cmp_stats[2], metrics)
        metrics = update_stats("Components", "Covariance Cond", cmp_stats[3], metrics)
        metrics = update_stats("Components", "Precision LogDet", cmp_stats[4], metrics)
        metrics = update_stats("Components", "Covariance LogDet", cmp_stats[5], metrics)

        ### Grad Norms ###

        def norm_grads(grad: Point[Mean, H]) -> Array:
            obs_grad, lwr_int_grad, upr_grad = model.split_params(grad)
            obs_loc_grad, obs_prs_grad = model.obs_man.split_params(obs_grad)
            lat_grad, upr_int_grad, cat_grad = model.upr_hrm.split_params(upr_grad)
            lat_loc_grad, lat_prs_grad = model.upr_hrm.obs_man.split_params(lat_grad)
            grads = [
                obs_loc_grad,
                obs_prs_grad,
                lwr_int_grad,
                lat_loc_grad,
                lat_prs_grad,
                upr_int_grad,
                cat_grad,
            ]
            return jnp.asarray([jnp.linalg.norm(grad.array) for grad in grads])

        if batch_grads is not None:
            batch_man: Replicated[H] = Replicated(model, batch_grads.shape[0])
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


### Log Artifacts ###


# In plugins/models/hmog/analysis/logging.py


def log_artifacts[M: HMoG](
    handler: RunHandler,
    dataset: ClusteringDataset,
    logger: JaxLogger,
    experiment: ClusteringExperiment,
    model: M,
    epoch: int,
    params: Point[Natural, M] | None = None,
    key: Array | None = None,
) -> None:
    """Generate and save plots from artifacts."""

    if key is None:
        key = jax.random.PRNGKey(42)

    if params is not None:
        handler.save_params(params.array, epoch)

    # Convert params to array if provided
    params_array = params.array if params is not None else None

    analyses = [
        ClusterStatisticsAnalysis(),
        KLHierarchyAnalysis(),
        CoAssignmentHierarchyAnalysis(),
        GenerativeExamplesAnalysis(n_samples=1000),
        LoadingMatrixAnalysis(),
    ]

    for analysis in analyses:
        analysis.process(key, handler, dataset, model, logger, epoch, params_array)

    # Conditional analyses for labeled datasets
    if dataset.has_labels:
        merge_analyses = [
            KLMergeAnalysis(True, 0.0005),
            CoAssignmentMergeAnalysis(True, 0.0005),
            OptimalMergeAnalysis(True, 0.0005),
        ]

        for analysis in merge_analyses:
            analysis.process(key, handler, dataset, model, logger, epoch, params_array)

    # Dataset-specific analyses
    specialized_analyses = dataset.get_dataset_analyses()
    for name, analysis in specialized_analyses.items():
        # For dataset-specific analyses, we might need to pass cluster assignments
        # This would be handled through the dataset's interface
        analysis.process(key, handler, dataset, experiment, logger, epoch, params_array)
