"""Configuration for HMoG implementations."""

from __future__ import annotations

import logging
from typing import Callable

import jax
import jax.numpy as jnp
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

from apps.configs import STATS_LEVEL
from apps.plugins import (
    ClusteringDataset,
)
from apps.runtime.handler import MetricDict
from apps.runtime.logger import JaxLogger

from .base import HMoG

# Start logger
log = logging.getLogger(__name__)


### Helpers ###


def fori[X](lower: int, upper: int, body_fun: Callable[[Array, X], X], init: X) -> X:
    return jax.lax.fori_loop(lower, upper, body_fun, init)  # pyright: ignore[reportUnknownVariableType]


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


def log_epoch_metrics[H: HMoG](
    dataset: ClusteringDataset,
    hmog_model: H,
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

        epoch_train_ll = hmog_model.average_log_observable_density(params, train_data)
        epoch_test_ll = hmog_model.average_log_observable_density(params, test_data)

        n_samps = train_data.shape[0]
        epoch_scaled_bic = (
            -(hmog_model.dim * jnp.log(n_samps) / n_samps - 2 * epoch_train_ll) / 2
        )
        info = jnp.array(logging.INFO)

        metrics: MetricDict = {
            "Performance/Train Log-Likelihood": (
                info,
                epoch_train_ll,
            ),
            "Performance/Test Log-Likelihood": (
                info,
                epoch_test_ll,
            ),
            "Performance/Scaled BIC": (
                info,
                epoch_scaled_bic,
            ),
        }

        # Clustering metrics if dataset has labels
        if dataset.has_labels:
            # Get cluster assignments using the existing function
            train_clusters = cluster_assignments(hmog_model, params.array, train_data)
            test_clusters = cluster_assignments(hmog_model, params.array, test_data)

            # Compute accuracy
            train_acc = cluster_accuracy(dataset.train_labels, train_clusters)
            test_acc = cluster_accuracy(dataset.test_labels, test_clusters)

            # Compute NMI
            n_clusters = hmog_model.upr_hrm.n_categories
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
                    "Clustering/Train Accuracy": (info, train_acc),
                    "Clustering/Test Accuracy": (info, test_acc),
                    "Clustering/Train NMI": (info, train_nmi),
                    "Clustering/Test NMI": (info, test_nmi),
                }
            )
            # Get cluster assignments using the existing function
            # train_probs = cluster_probabilities(hmog_model, params.array, train_data)
            # test_probs = cluster_probabilities(hmog_model, params.array, test_data)
            #
            # n_classes = dataset.n_classes
            # merge_mapping = cluster_merge_mapping(hmog_model, params, n_classes)
            # merged_train_acc = mapped_cluster_accuracy(
            #     dataset.train_labels, train_probs, merge_mapping
            # )
            # merged_test_acc = mapped_cluster_accuracy(
            #     dataset.test_labels, test_probs, merge_mapping
            # )
            #
            # # Add to metrics dictionary
            # metrics.update(
            #     {
            #         "Clustering/Train Accuracy": (info, merged_train_acc),
            #         "Clustering/Test Accuracy": (info, merged_test_acc),
            #     }
            # )

        # Raw Parameter Statistics

        stats = jnp.array(STATS_LEVEL)

        def update_parameter_stats[M: Manifold](
            name: str, params: Point[Natural, M]
        ) -> None:
            array = params.array
            metrics.update(
                {
                    f"Params/{name} Min": (
                        stats,
                        jnp.min(array),
                    ),
                    f"Params/{name} Median": (
                        stats,
                        jnp.median(array),
                    ),
                    f"Params/{name} Max": (
                        stats,
                        jnp.max(array),
                    ),
                }
            )

        obs_params, lwr_int_params, upr_params = hmog_model.split_params(params)
        obs_loc_params, obs_prs_params = hmog_model.obs_man.split_params(obs_params)
        lat_params, upr_int_params, cat_params = hmog_model.upr_hrm.split_params(
            upr_params
        )
        lat_loc_params, lat_prs_params = hmog_model.upr_hrm.obs_man.split_params(
            lat_params
        )

        update_parameter_stats("Obs Location", obs_loc_params)
        update_parameter_stats("Obs Precision", obs_prs_params)
        update_parameter_stats("Obs Interaction", lwr_int_params)
        update_parameter_stats("Lat Location", lat_loc_params)
        update_parameter_stats("Lat Precision", lat_prs_params)
        update_parameter_stats("Lat Interaction", upr_int_params)
        update_parameter_stats("Categorical", cat_params)

        # Regularization Metrics

        # Compute latent distribution statistics
        # Get latent distribution in mean coordinates for analysis
        lgm = hmog_model.lwr_hrm
        lkl_params = lgm.lkl_man.join_params(obs_params, lwr_int_params)
        ana_lgm, re_loss, lgm_lat_means = relative_entropy_regularization_full(
            lgm, train_data, lkl_params
        )
        lat_mean, lat_cov = ana_lgm.lat_man.split_mean_covariance(lgm_lat_means)
        lat_mean_array = lat_mean.array
        lat_cov_array = ana_lgm.lat_man.cov_man.to_dense(lat_cov)

        # Add latent distribution metrics
        metrics.update(
            {
                "Regularization/Loading Sparsity": (
                    stats,
                    jnp.mean(jnp.abs(lwr_int_params.array) < 1e-6),
                ),
                "Regularization/Latent KL to Z": (
                    stats,
                    re_loss,
                ),
                # Mean vector summary
                "Regularization/Latent Mean Norm": (
                    stats,
                    jnp.linalg.norm(lat_mean_array),
                ),
                "Regularization/Latent Mean Min": (
                    stats,
                    jnp.min(lat_mean_array),
                ),
                "Regularization/Latent Mean Max": (
                    stats,
                    jnp.max(lat_mean_array),
                ),
                # Variance summary
                "Regularization/Latent Var Min": (
                    stats,
                    jnp.min(jnp.diag(lat_cov_array)),
                ),
                "Regularization/Latent Var Max": (
                    stats,
                    jnp.max(jnp.diag(lat_cov_array)),
                ),
                "Regularization/Latent Var Mean": (
                    stats,
                    jnp.mean(jnp.diag(lat_cov_array)),
                ),
                # Eigenvalue analysis
                "Regularization/Latent Eigenvalue Min": (
                    stats,
                    jnp.min(jnp.linalg.eigvalsh(lat_cov_array)),
                ),
                # Eigenvalue analysis
                "Regularization/Latent Eigenvalue Median": (
                    stats,
                    jnp.median(jnp.linalg.eigvalsh(lat_cov_array)),
                ),
                "Regularization/Latent Eigenvalue Max": (
                    stats,
                    jnp.max(jnp.linalg.eigvalsh(lat_cov_array)),
                ),
                # Structure summary
                "Regularization/Latent Off-Diag Magnitude": (
                    stats,
                    jnp.linalg.norm(
                        lat_cov_array - jnp.diag(jnp.diag(lat_cov_array)), "fro"
                    ),
                ),
                "Regularization/Latent Condition Number": (
                    stats,
                    jnp.linalg.cond(
                        lat_cov_array + jnp.eye(lat_cov_array.shape[0])
                    ),  # Small epsilon for stability
                ),
                "Regularization/Latent Effective Rank": (
                    stats,
                    jnp.sum(jnp.linalg.eigvalsh(lat_cov_array) > 1e-6),
                ),
            }
        )

        ### Grad Norms ###

        def update_grad_stats[M: Manifold](name: str, grad_norms: Array) -> None:
            metrics.update(
                {
                    f"Grad Norms/{name} Min": (
                        stats,
                        jnp.min(grad_norms),
                    ),
                    f"Grad Norms/{name} Median": (
                        stats,
                        jnp.median(grad_norms),
                    ),
                    f"Grad Norms/{name} Max": (
                        stats,
                        jnp.max(grad_norms),
                    ),
                }
            )

        def norm_grads(grad: Point[Mean, H]) -> Array:
            obs_grad, lwr_int_grad, upr_grad = hmog_model.split_params(grad)
            obs_loc_grad, obs_prs_grad = hmog_model.obs_man.split_params(obs_grad)
            lat_grad, upr_int_grad, cat_grad = hmog_model.upr_hrm.split_params(upr_grad)
            lat_loc_grad, lat_prs_grad = hmog_model.upr_hrm.obs_man.split_params(
                lat_grad
            )
            return jnp.asarray(
                [
                    jnp.linalg.norm(grad.array)
                    for grad in [
                        obs_loc_grad,
                        obs_prs_grad,
                        lwr_int_grad,
                        lat_loc_grad,
                        lat_prs_grad,
                        upr_int_grad,
                        cat_grad,
                    ]
                ]
            )

        if batch_grads is not None:
            batch_man: Replicated[H] = Replicated(hmog_model, batch_grads.shape[0])
            grad_norms = batch_man.map(norm_grads, batch_grads).T

            update_grad_stats("Obs Location", grad_norms[0])
            update_grad_stats("Obs Precision", grad_norms[1])
            update_grad_stats("Obs Interaction", grad_norms[2])
            update_grad_stats("Lat Location", grad_norms[3])
            update_grad_stats("Lat Precision", grad_norms[4])
            update_grad_stats("Lat Interaction", grad_norms[5])
            update_grad_stats("Categorical", grad_norms[6])

        logger.log_metrics(metrics, epoch + 1)

    def no_op() -> None:
        return None

    jax.lax.cond(epoch % log_freq == 0, compute_metrics, no_op)


### Graveyard ###

# NB: This won't work in JIT mode for now...
#
# def cluster_merge_mapping[M: HMoG](
#     model: M, params: Point[Natural, M], n_classes: Array
# ) -> Array:
#     """Compute a mapping matrix from original clusters to merged clusters.
#
#     Args:
#         model: HMoG model
#         params: Model parameters
#         n_target_clusters: Target number of clusters after merging
#
#     Returns:
#         Binary mapping matrix of shape (n_original_clusters, n_target_clusters)
#         where M[i,j] = 1 if original cluster i maps to merged cluster j
#     """
#     # Get symmetrized KL divergence matrix
#     sym_kl = symmetric_kl_matrix(model, params)
#
#     # Number of original clusters
#     n_original_clusters = sym_kl.shape[0]
#
#     # Start with each cluster in its own group
#     # cluster_group[i] = j means cluster i is in group j
#     cluster_group = jnp.arange(n_original_clusters)
#
#     # Iteratively merge the most similar clusters
#     def merge_step(groups: Array, _):
#         # Create a mask for pairs in different groups
#         # Create indices for all pairs
#         row_idx = jnp.arange(n_original_clusters)[:, None]
#         col_idx = jnp.arange(n_original_clusters)[None, :]
#
#         # Check if pairs belong to different groups (and aren't the same cluster)
#         groups_i = groups[row_idx]
#         groups_j = groups[col_idx]
#         diff_groups = (groups_i != groups_j) & (row_idx != col_idx)
#
#         # Mask KL matrix
#         masked_kl = jnp.where(diff_groups, sym_kl, jnp.inf)
#
#         # Find indices of minimum KL
#         flat_idx = jnp.argmin(masked_kl.ravel())
#         i, j = jnp.unravel_index(flat_idx, sym_kl.shape)
#
#         # Get the groups to merge
#         group_i = groups[i]
#         group_j = groups[j]
#
#         # Use the smaller group index as target
#         target_group = jnp.minimum(group_i, group_j)
#         source_group = jnp.maximum(group_i, group_j)
#
#         # Update all clusters in source_group to target_group
#         new_groups = jnp.where(groups == source_group, target_group, groups)
#
#         # Relabel groups to be consecutive
#         # Create a mapping from old group IDs to new consecutive IDs
#         unique_groups, indices = jnp.unique(new_groups, return_inverse=True)
#         relabeled_groups = indices
#
#         return relabeled_groups, None
#
#     # Calculate number of merges needed
#     n_merges = n_original_clusters - n_classes
#
#     # Apply merging steps
#     final_groups, _ = jax.lax.scan(merge_step, cluster_group, jnp.arange(n_merges))
#
#     return jax.nn.one_hot(final_groups, n_classes, dtype=jnp.int32)
#
#
#
# def mapped_cluster_accuracy(
#     true_labels: Array, cluster_probs: Array, mapping: Array
# ) -> Array:
#     """Compute clustering accuracy with optimal label assignment.
#
#     Args:
#         true_labels: Ground truth labels with shape (n_samples,)
#         cluster_probs: Cluster assignment probabilities with shape (n_samples, n_clusters)
#         mapping: Optional mapping matrix from original to merged clusters
#
#     Returns:
#         Clustering accuracy after optimal label assignment
#     """
#     # Apply mapping if provided
#     if mapping is not None:
#         cluster_probs = jnp.matmul(cluster_probs, mapping)
#
#     # Get hard assignments
#     cluster_assignments = jnp.argmax(cluster_probs, axis=1)
#
#     # Number of clusters and classes
#     n_clusters = mapping.shape[1]
#     n_classes = mapping.shape[0]
#
#     # Assuming n_clusters >= n_classes, create an appropriately sized contingency matrix
#     contingency = jnp.zeros((n_clusters, n_classes))
#
#     # Update contingency matrix - no need for clipping since dimensions are appropriate
#     def update_contingency(i, cont):
#         label = true_labels[i]
#         cluster = cluster_assignments[i]
#         return cont.at[cluster, label].add(1)
#
#     contingency = jax.lax.fori_loop(
#         0, true_labels.shape[0], update_contingency, contingency
#     )
#
#     # Find optimal assignment (max value in each row)
#     cluster_to_label = jnp.argmax(contingency, axis=1)
#
#     # Map each prediction to its optimal class - all clusters should have valid mappings
#     mapped_predictions = cluster_to_label[cluster_assignments]
#
#     # Compute accuracy
#     return jnp.mean(mapped_predictions == true_labels)
#
# def relative_entropy_regularization[
#     ObsRep: PositiveDefinite,
#     PostRep: PositiveDefinite,
# ](
#     lgm: DifferentiableLinearGaussianModel[ObsRep, PostRep],
#     batch: Array,
#     lkl_params: Point[
#         Natural, AffineMap[Rectangular, Euclidean, Euclidean, Normal[ObsRep]]
#     ],
# ) -> tuple[Array, Point[Mean, Normal[PostRep]]]:
#     z = lgm.lat_man.to_natural(lgm.lat_man.standard_normal())
#     rho = lgm.conjugation_parameters(lkl_params)
#     lat_params = lgm.pst_lat_emb.translate(-rho, z)
#     lat_loc, lat_prs = lgm.con_lat_man.split_location_precision(lat_params)
#     dns_prs = lgm.con_lat_man.cov_man.to_dense(lat_prs)
#     dia_prs = lgm.lat_man.cov_man.from_dense(dns_prs)
#     sub_lat_params = lgm.lat_man.join_location_precision(lat_loc, dia_prs)
#     obs_params, int_params = lgm.lkl_man.split_params(lkl_params)
#     lgm_params = lgm.join_params(obs_params, int_params, sub_lat_params)
#     lgm_means = lgm.mean_posterior_statistics(lgm_params, batch)
#     lgm_lat_means = lgm.split_params(lgm_means)[2]
#     return lgm.lat_man.relative_entropy(lgm_lat_means, z), lgm_lat_means
#
