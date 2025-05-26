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

from apps.configs import STATS_NUM
from apps.plugins import (
    ClusteringDataset,
)
from apps.runtime.handler import MetricDict, RunHandler
from apps.runtime.logger import JaxLogger

from ..base import LGM, HMoG
from .base import (
    analyze_component,
    cluster_accuracy,
    cluster_assignments,
    clustering_nmi,
    update_stats,
)
from .clusters import (
    ClusterStatistics,
    cluster_statistics_plotter,
    get_cluster_statistics,
)
from .generative import (
    GenerativeExamples,
    generate_examples,
    generative_examples_plotter,
)
from .hierarchy import (
    CoAssignmentClusterHierarchy,
    KLClusterHierarchy,
    get_cluster_hierarchy,
    hierarchy_plotter,
)
from .loadings import (
    LoadingMatrixArtifact,
    get_loading_matrices,
    loading_matrix_plotter,
)
from .merge import (
    CoAssignmentMergeResults,
    KLMergeResults,
    OptimalMergeResults,
    get_merge_results,
    merge_results_plotter,
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


def log_artifacts[M: HMoG](
    handler: RunHandler,
    dataset: ClusteringDataset,
    logger: JaxLogger,
    model: M,
    epoch: int,
    params: Point[Natural, M] | None = None,
    key: Array | None = None,
) -> None:
    """Generate and save plots from artifacts.

    Args:
        handler: Run handler containing saved artifacts
        dataset: Dataset used for visualization
        logger: Logger for saving artifacts and figures
        model: Model used for analysis and artifact generation
        params: If provided, generate new artifacts from these parameters
        epoch: Specific epoch to analyze, defaults to latest
    """

    if key is None:
        key = jax.random.PRNGKey(42)

    # from_scratch if params is provided
    if params is not None:
        handler.save_params(params.array, epoch)
        cluster_statistics = get_cluster_statistics(model, dataset, params)
        kl_hierarchy = get_cluster_hierarchy(
            model, params, KLClusterHierarchy, dataset.train_data
        )
        co_hierarchy = get_cluster_hierarchy(
            model, params, CoAssignmentClusterHierarchy, dataset.train_data
        )
        gen_examples = generate_examples(model, params, 25, key)
        loading_matrices = get_loading_matrices(model, params)
    else:
        cluster_statistics = handler.load_artifact(epoch, ClusterStatistics)
        kl_hierarchy = handler.load_artifact(epoch, KLClusterHierarchy)
        co_hierarchy = handler.load_artifact(epoch, CoAssignmentClusterHierarchy)
        gen_examples = handler.load_artifact(epoch, GenerativeExamples)
        loading_matrices = handler.load_artifact(epoch, LoadingMatrixArtifact)

    # Plot and save
    plot_clusters_statistics = cluster_statistics_plotter(dataset)
    plot_hierarchy = hierarchy_plotter(dataset)
    plot_examples = generative_examples_plotter(dataset)
    plot_loadings = loading_matrix_plotter(dataset)

    logger.log_artifact(handler, epoch, cluster_statistics, plot_clusters_statistics)
    logger.log_artifact(handler, epoch, kl_hierarchy, plot_hierarchy)
    logger.log_artifact(handler, epoch, co_hierarchy, plot_hierarchy)
    logger.log_artifact(handler, epoch, gen_examples, plot_examples)
    logger.log_artifact(handler, epoch, loading_matrices, plot_loadings)

    if dataset.has_labels:
        if params is not None:
            kl_merge_results = get_merge_results(model, params, dataset, KLMergeResults)
            co_merge_results = get_merge_results(
                model, params, dataset, CoAssignmentMergeResults
            )
            op_merge_results = get_merge_results(
                model, params, dataset, OptimalMergeResults
            )
        else:
            kl_merge_results = handler.load_artifact(epoch, KLMergeResults)
            co_merge_results = handler.load_artifact(epoch, CoAssignmentMergeResults)
            op_merge_results = handler.load_artifact(epoch, OptimalMergeResults)

        plot_merge_results = merge_results_plotter(dataset)
        logger.log_artifact(handler, epoch, kl_merge_results, plot_merge_results)
        logger.log_artifact(handler, epoch, co_merge_results, plot_merge_results)
        logger.log_artifact(handler, epoch, op_merge_results, plot_merge_results)
        # Log merge metrics
        metrics: MetricDict = {
            "Merging/KL Train Accuracy": (
                STATS_LEVEL,
                jnp.array(kl_merge_results.train_accuracy),
            ),
            "Merging/KL Train NMI": (
                STATS_LEVEL,
                jnp.array(kl_merge_results.train_nmi_score),
            ),
            "Merging/KL Test Accuracy": (
                STATS_LEVEL,
                jnp.array(kl_merge_results.test_accuracy),
            ),
            "Merging/KL Test NMI": (
                STATS_LEVEL,
                jnp.array(kl_merge_results.test_nmi_score),
            ),
            "Merging/CoAssignment Train Accuracy": (
                STATS_LEVEL,
                jnp.array(co_merge_results.train_accuracy),
            ),
            "Merging/CoAssignment Train NMI": (
                STATS_LEVEL,
                jnp.array(co_merge_results.train_nmi_score),
            ),
            "Merging/CoAssignment Test Accuracy": (
                STATS_LEVEL,
                jnp.array(co_merge_results.test_accuracy),
            ),
            "Merging/CoAssignment Test NMI": (
                STATS_LEVEL,
                jnp.array(co_merge_results.test_nmi_score),
            ),
            "Merging/Optimal Train Accuracy": (
                STATS_LEVEL,
                jnp.array(op_merge_results.train_accuracy),
            ),
            "Merging/Optimal Train NMI": (
                STATS_LEVEL,
                jnp.array(op_merge_results.train_nmi_score),
            ),
            "Merging/Optimal Test Accuracy": (
                STATS_LEVEL,
                jnp.array(op_merge_results.test_accuracy),
            ),
            "Merging/Optimal Test NMI": (
                STATS_LEVEL,
                jnp.array(op_merge_results.test_nmi_score),
            ),
        }
        logger.log_metrics(metrics, jnp.array(epoch))
