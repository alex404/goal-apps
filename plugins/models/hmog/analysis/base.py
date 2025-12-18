"""Configuration for DifferentiableHMoG implementations."""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
from goal.geometry import (
    Manifold,
)
from goal.models import (
    DifferentiableHMoG,
    Normal,
    NormalAnalyticLGM,
)
from jax import Array

from apps.runtime import STATS_NUM, MetricDict

# Start logger
log = logging.getLogger(__name__)


STATS_LEVEL = jnp.array(STATS_NUM)
INFO_LEVEL = jnp.array(logging.INFO)


### Analysis ###


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


def cluster_assignments(model: DifferentiableHMoG, params: Array, data: Array) -> Array:
    """Assign data points to clusters using the model.

    Args:
        model: DifferentiableHMoG model
        params: Model parameters
        data: Data points to assign

    Returns:
        Array of cluster assignments
    """
    probs = cluster_probabilities(model, params, data)
    return jnp.argmax(probs, axis=-1)


def symmetric_kl_matrix(
    model: DifferentiableHMoG,
    params: Array,
) -> Array:
    mix_params = model.prior(params)
    with model.prr_man as ch:
        comp_lats, _ = ch.split_natural_mixture(mix_params)

        # Convert flat components to 2D for indexing inside vmap
        comp_lats_2d = ch.cmp_man.to_2d(comp_lats)

        def kl_div_between_components(i: Array, j: Array) -> Array:
            comp_i = comp_lats_2d[i]
            comp_i_mean = ch.obs_man.to_mean(comp_i)
            comp_j = comp_lats_2d[j]
            return ch.obs_man.relative_entropy(comp_i_mean, comp_j)

        idxs = jnp.arange(ch.n_categories)

    def kl_div_from_one_to_all(i: Array) -> Array:
        return jax.vmap(kl_div_between_components, in_axes=(None, 0))(i, idxs)

    kl_matrix = jax.lax.map(kl_div_from_one_to_all, idxs)
    return (kl_matrix + kl_matrix.T) / 2


def get_component_prototypes(
    model: DifferentiableHMoG,
    params: Array,
) -> list[Array]:
    # Split into likelihood and mixture parameters
    lkl_params, mix_params = model.split_conjugated(params)

    # Extract components from mixture
    comp_lats, _ = model.prr_man.split_natural_mixture(mix_params)

    # For each component, compute the observable distribution and get its mean
    prototypes: list[Array] = []

    ana_lgm = NormalAnalyticLGM(
        obs_dim=model.lwr_hrm.obs_dim,  # Original latent becomes observable
        obs_rep=model.lwr_hrm.obs_rep,
        lat_dim=model.lwr_hrm.lat_dim,  # Original observable becomes latent
    )

    for i in range(model.prr_man.cmp_man.n_reps):
        # Get latent params for this component (comp_lats is flat)
        comp_lat_params = model.prr_man.cmp_man.get_replicate(comp_lats, i)
        lwr_hrm_params = ana_lgm.join_conjugated(lkl_params, comp_lat_params)
        lwr_hrm_means = ana_lgm.to_mean(lwr_hrm_params)
        lwr_hrm_obs = ana_lgm.split_coords(lwr_hrm_means)[0]
        obs_means = ana_lgm.obs_man.split_mean_second_moment(lwr_hrm_obs)[0]

        prototypes.append(obs_means)

    return prototypes


def cluster_probabilities(
    model: DifferentiableHMoG, params: Array, data: Array
) -> Array:
    """Get cluster probability distributions for each data point.

    Args:
        model: DifferentiableHMoG model
        params: Model parameters
        data: Data points to get probabilities for

    Returns:
        Array of shape (n_samples, n_clusters) with probability distributions
    """

    def data_point_probs(x: Array) -> Array:
        cat_pst = model.pst_man.prior(model.posterior_at(params, x))
        with model.pst_man.lat_man as lm:
            return lm.to_probs(lm.to_mean(cat_pst))

    return jax.lax.map(data_point_probs, data, batch_size=2048)


### Logging ###


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


def analyze_component(nor_man: Normal, nrm_params: Array) -> Array:
    nrm_means = nor_man.to_mean(nrm_params)
    loc, prs = nor_man.split_location_precision(nrm_params)
    mean, cov = nor_man.split_mean_covariance(nrm_means)
    dns_prs = nor_man.cov_man.to_matrix(prs)
    dns_cov = nor_man.cov_man.to_matrix(cov)
    loc_nrm = jnp.linalg.norm(loc)
    mean_nrm = jnp.linalg.norm(mean)
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
