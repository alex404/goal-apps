"""Configuration for HMoG implementations."""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
from goal.geometry import (
    AffineMap,
    Manifold,
    Mean,
    Natural,
    Point,
    PositiveDefinite,
    Rectangular,
    Replicated,
)
from goal.models import (
    AnalyticLinearGaussianModel,
    DifferentiableLinearGaussianModel,
    Euclidean,
    FullNormal,
    Normal,
)
from jax import Array
from numpy.typing import NDArray

from apps.configs import STATS_NUM
from apps.plugins import (
    ClusteringDataset,
)
from apps.runtime.handler import MetricDict
from apps.runtime.logger import JaxLogger

from .base import LGM, HMoG

# Start logger
log = logging.getLogger(__name__)


### Helpers ###


# Add to analysis.py


def hierarchy_to_mapping(
    linkage_matrix: NDArray[np.float64],
    n_clusters: int,
    n_classes: int,
) -> NDArray[np.int32]:
    """Convert hierarchical clustering to mapping matrix.

    Args:
        linkage_matrix: Hierarchical clustering linkage matrix
        n_clusters: Number of original clusters
        n_classes: Number of target classes

    Returns:
        Binary mapping matrix of shape (n_clusters, n_classes)
    """
    import scipy.cluster.hierarchy

    # Cut the tree to get n_classes
    cluster_ids = scipy.cluster.hierarchy.fcluster(
        linkage_matrix, n_classes, criterion="maxclust"
    )

    # Convert cluster IDs to one-hot mapping
    mapping = np.zeros((n_clusters, n_classes), dtype=np.int32)
    for i in range(n_clusters):
        # Cluster IDs from fcluster are 1-indexed
        class_id = cluster_ids[i] - 1
        mapping[i, class_id] = 1

    return mapping


def compute_optimal_mapping(
    cluster_probs: Array,
    true_labels: Array,
    n_classes: int,
) -> NDArray[np.int32]:
    """Compute optimal mapping from clusters to classes using Hungarian algorithm.

    Args:
        cluster_probs: Cluster assignment probabilities (n_samples, n_clusters)
        true_labels: True class labels (n_samples,)
        n_classes: Number of classes

    Returns:
        Binary mapping matrix (n_clusters, n_classes)
    """
    from scipy.optimize import linear_sum_assignment

    # Get hard assignments
    cluster_assignments = jnp.argmax(cluster_probs, axis=1)

    # Number of clusters
    n_clusters = cluster_probs.shape[1]

    # Create contingency matrix
    contingency = np.zeros((n_clusters, n_classes))

    # Fill contingency matrix
    for i in range(len(true_labels)):
        label = int(true_labels[i])
        cluster = int(cluster_assignments[i])
        contingency[cluster, label] += 1

    # Use Hungarian algorithm for optimal assignment
    # Negate contingency matrix for maximization
    row_ind, col_ind = linear_sum_assignment(-contingency)

    # Create mapping matrix
    mapping = np.zeros((n_clusters, n_classes), dtype=np.int32)
    for i, j in zip(row_ind, col_ind):
        mapping[i, j] = 1

    return mapping


def cluster_probabilities(model: HMoG, params: Array, data: Array) -> Array:
    """Get cluster probability distributions for each data point.

    Args:
        model: HMoG model
        params: Model parameters
        data: Data points to get probabilities for

    Returns:
        Array of shape (n_samples, n_clusters) with probability distributions
    """

    def data_point_probs(x: Array) -> Array:
        cat_pst = model.upr_hrm.prior(
            model.posterior_at(model.natural_point(params), x)
        )
        with model.upr_hrm.lat_man as lm:
            return lm.to_probs(lm.to_mean(cat_pst))

    return jax.lax.map(data_point_probs, data, batch_size=2048)


def cluster_assignments(model: HMoG, params: Array, data: Array) -> Array:
    """Assign data points to clusters using the model.

    Args:
        model: HMoG model
        params: Model parameters
        data: Data points to assign

    Returns:
        Array of cluster assignments
    """
    probs = cluster_probabilities(model, params, data)
    return jnp.argmax(probs, axis=-1)


def symmetric_kl_matrix[
    M: HMoG,
](
    model: M,
    params: Point[Natural, M],
) -> Array:
    mix_params = model.prior(params)
    with model.con_upr_hrm as ch:
        comp_lats, _ = ch.split_natural_mixture(mix_params)

        def kl_div_between_components(i: Array, j: Array) -> Array:
            comp_i = ch.cmp_man.get_replicate(comp_lats, i)
            comp_i_mean = ch.obs_man.to_mean(comp_i)
            comp_j = ch.cmp_man.get_replicate(comp_lats, j)
            return ch.obs_man.relative_entropy(comp_i_mean, comp_j)

        idxs = jnp.arange(ch.n_categories)

    def kl_div_from_one_to_all(i: Array) -> Array:
        return jax.vmap(kl_div_between_components, in_axes=(None, 0))(i, idxs)

    kl_matrix = jax.lax.map(kl_div_from_one_to_all, idxs)
    return (kl_matrix + kl_matrix.T) / 2


def get_component_prototypes[
    M: HMoG,
](
    model: M,
    params: Point[Natural, M],
) -> list[Array]:
    # Split into likelihood and mixture parameters
    lkl_params, mix_params = model.split_conjugated(params)

    # Extract components from mixture
    comp_lats, _ = model.con_upr_hrm.split_natural_mixture(mix_params)

    # For each component, compute the observable distribution and get its mean
    prototypes: list[Array] = []

    ana_lgm = AnalyticLinearGaussianModel(
        obs_dim=model.lwr_hrm.obs_dim,  # Original latent becomes observable
        obs_rep=model.lwr_hrm.obs_rep,
        lat_dim=model.lwr_hrm.lat_dim,  # Original observable becomes latent
    )

    for i in range(comp_lats.shape[0]):
        # Get latent mean for this component
        comp_lat_params = model.con_upr_hrm.cmp_man.get_replicate(
            comp_lats, jnp.asarray(i)
        )
        lwr_hrm_params = ana_lgm.join_conjugated(lkl_params, comp_lat_params)
        lwr_hrm_means = ana_lgm.to_mean(lwr_hrm_params)
        lwr_hrm_obs = ana_lgm.split_params(lwr_hrm_means)[0]
        obs_means = ana_lgm.obs_man.split_mean_second_moment(lwr_hrm_obs)[0].array

        prototypes.append(obs_means)

    return prototypes


def posterior_co_assignment_matrix[M: HMoG](
    model: M,
    params: Point[Natural, M],
    data: Array,
) -> Array:
    """Compute posterior co-assignment matrix between components.

    This computes how often two clusters are assigned to the same data points,
    based on their posterior probabilities.

    Args:
        model: HMoG model
        params: Model parameters
        data: Data points for calculating empirical similarities

    Returns:
        Co-assignment similarity matrix (higher values = more similar)
    """
    # Get cluster probabilities for each data point
    probs = cluster_probabilities(model, params.array, data)

    # Compute co-assignment matrix efficiently through matrix multiplication
    # co_assignment[i,j] = sum_x p(x,i) * p(x,j)
    co_assignment = probs.T @ probs

    # Normalize to get correlation-like measure
    diag_sqrt = jnp.sqrt(jnp.diag(co_assignment))
    # Build outer product of sqrt-diag entries, add small epsilon for stability
    denom = diag_sqrt[:, None] * diag_sqrt[None, :] + 1e-12
    normalized_co_assignment = co_assignment / denom

    # Ensure perfect symmetry by averaging with transpose
    normalized_co_assignment = 0.5 * (
        normalized_co_assignment + normalized_co_assignment.T
    )

    return normalized_co_assignment


def clustering_nmi(
    n_clusters: int, n_classes: int, assignments: Array, true_labels: Array
) -> Array:
    """
    Compute Normalized Mutual Information (NMI) between cluster assignments and true labels.

    Fully JAX-compatible implementation that can be used with jax.jit.

    Args:
        assignments: Array of cluster assignments
        true_labels: Array of true class labels

    Returns:
        NMI score (0-1, higher is better)
    """
    # Get number of clusters and classes
    n_samples = assignments.shape[0]

    # Create indices for counting
    cluster_indices = assignments
    class_indices = true_labels

    # Compute cluster and class counts
    cluster_counts = jnp.zeros(n_clusters).at[cluster_indices].add(1.0)
    class_counts = jnp.zeros(n_classes).at[class_indices].add(1.0)

    # Initialize contingency matrix
    contingency = jnp.zeros((n_clusters, n_classes))

    # Build contingency matrix using scatter_add approach
    idx_matrix = jnp.stack([assignments, true_labels], axis=1)
    values = jnp.ones(n_samples)

    # Use a non-python loop to build the contingency table
    def update_contingency(i, cont):
        idx = idx_matrix[i]
        val = values[i]
        return cont.at[idx[0], idx[1]].add(val)

    contingency = jax.lax.fori_loop(0, n_samples, update_contingency, contingency)

    # Compute entropy for clusters
    cluster_probs = cluster_counts / n_samples
    cluster_entropy = -jnp.sum(
        jnp.where(cluster_probs > 0, cluster_probs * jnp.log(cluster_probs), 0.0)
    )

    # Compute entropy for classes
    class_probs = class_counts / n_samples
    class_entropy = -jnp.sum(
        jnp.where(class_probs > 0, class_probs * jnp.log(class_probs), 0.0)
    )

    # Compute mutual information
    joint_probs = contingency / n_samples
    outer_probs = jnp.outer(cluster_probs, class_probs)

    # Avoid log(0) by masking
    log_ratio = jnp.where(joint_probs > 0, jnp.log(joint_probs / outer_probs), 0.0)

    mutual_info = jnp.sum(joint_probs * log_ratio)

    # Compute NMI with small epsilon to avoid division by zero
    epsilon = 1e-10
    nmi = 2.0 * mutual_info / (cluster_entropy + class_entropy + epsilon)

    # Ensure NMI is in [0, 1]
    return jnp.clip(nmi, 0.0, 1.0)


def relative_entropy_regularization_full[
    ObsRep: PositiveDefinite,
    PostRep: PositiveDefinite,
](
    lgm: DifferentiableLinearGaussianModel[ObsRep, PostRep],
    batch: Array,
    lkl_params: Point[
        Natural, AffineMap[Rectangular, Euclidean, Euclidean, Normal[ObsRep]]
    ],
) -> tuple[AnalyticLinearGaussianModel[ObsRep], Array, Point[Mean, FullNormal]]:
    # Relative entropy regularization
    ana_lgm = AnalyticLinearGaussianModel(lgm.obs_dim, lgm.obs_rep, lgm.lat_dim)
    z = ana_lgm.lat_man.to_natural(ana_lgm.lat_man.standard_normal())
    lgm_params = ana_lgm.join_conjugated(lkl_params, z)
    lgm_means = ana_lgm.mean_posterior_statistics(lgm_params, batch)
    lgm_lat_means = ana_lgm.split_params(lgm_means)[2]
    re_loss = ana_lgm.lat_man.relative_entropy(lgm_lat_means, z)
    return ana_lgm, re_loss, lgm_lat_means


def cluster_accuracy(true_labels: Array, pred_clusters: Array) -> Array:
    """Compute clustering accuracy with optimal label assignment.

    Uses a fixed-size contingency matrix approach to be JIT-compatible.

    Args:
        true_labels: Ground truth labels
        pred_clusters: Predicted cluster assignments

    Returns:
        Clustering accuracy after optimal label assignment
    """
    # Use a fixed max size for JIT compatibility
    # If your model has more clusters, adjust this value
    max_clusters = 100

    # Create a fixed-size contingency matrix
    contingency = jnp.zeros((max_clusters, max_clusters))

    # Fill the contingency matrix
    def body_fun(i, cont):
        true_label = jnp.clip(true_labels[i], 0, max_clusters - 1)
        pred_cluster = jnp.clip(pred_clusters[i], 0, max_clusters - 1)
        return cont.at[pred_cluster, true_label].add(1)

    contingency = jax.lax.fori_loop(0, true_labels.shape[0], body_fun, contingency)

    # Find the best cluster-to-label assignment
    cluster_to_label = jnp.argmax(contingency, axis=1)

    # Map each point's cluster to its best label
    def map_fn(i):
        cluster = jnp.clip(pred_clusters[i], 0, max_clusters - 1)
        return cluster_to_label[cluster]

    mapped_preds = jax.vmap(map_fn)(jnp.arange(true_labels.shape[0]))

    # Compute accuracy
    return jnp.mean(mapped_preds == true_labels)


### Logging ###

STATS_LEVEL = jnp.array(STATS_NUM)
INFO_LEVEL = jnp.array(logging.INFO)


def update_stats[M: Manifold](
    group: str, name: str, stats: Array, metrics: MetricDict
) -> MetricDict:
    metrics.update(
        {
            f"{group}/{name} Min": (
                STATS_LEVEL,
                jnp.min(stats),
            ),
            f"{group}/{name} Median": (
                STATS_LEVEL,
                jnp.median(stats),
            ),
            f"{group}/{name} Max": (
                STATS_LEVEL,
                jnp.max(stats),
            ),
        }
    )
    return metrics


def analyze_component(
    nor_man: FullNormal, nrm_params: Point[Natural, FullNormal]
) -> Array:
    nrm_means = nor_man.to_mean(nrm_params)
    loc, prs = nor_man.split_location_precision(nrm_params)
    mean, cov = nor_man.split_mean_covariance(nrm_means)
    dns_prs = nor_man.cov_man.to_dense(prs)
    dns_cov = nor_man.cov_man.to_dense(cov)
    loc_nrm = jnp.linalg.norm(loc.array)
    mean_nrm = jnp.linalg.norm(mean.array)
    prs_cond = jnp.linalg.cond(dns_prs)
    cov_cond = jnp.linalg.cond(dns_cov)
    prs_ldet = jnp.linalg.slogdet(dns_prs)[1]
    cov_ldet = jnp.linalg.slogdet(dns_cov)[1]
    return jnp.asarray(
        [
            loc_nrm,
            mean_nrm,
            prs_cond,
            cov_cond,
            prs_ldet,
            cov_ldet,
        ]
    )


def pre_log_epoch_metrics[H: LGM](
    dataset: ClusteringDataset,
    model: H,
    logger: JaxLogger,
    params: Point[Natural, H],
    epoch: Array,
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

        metrics: MetricDict = {
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
        metrics: MetricDict = {
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
