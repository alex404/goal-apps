"""Generic clustering metric utilities.

Provides JAX-compatible metric functions for evaluating clustering performance.
These can be used within jax.jit-compiled training loops.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def cluster_accuracy(true_labels: Array, pred_clusters: Array) -> Array:
    """Compute clustering accuracy with optimal label assignment.

    Uses a greedy assignment approach that is JIT-compatible. For each cluster,
    assigns it to the class that has the most samples in that cluster.

    Args:
        true_labels: Ground truth labels (n_samples,)
        pred_clusters: Predicted cluster assignments (n_samples,)

    Returns:
        Clustering accuracy after optimal label assignment
    """
    # Use a fixed max size for JIT compatibility
    max_clusters = 100

    # Create a fixed-size contingency matrix
    contingency = jnp.zeros((max_clusters, max_clusters))

    # Fill the contingency matrix
    def body_fun(i: Array, cont: Array) -> Array:
        true_label = jnp.clip(true_labels[i], 0, max_clusters - 1)
        pred_cluster = jnp.clip(pred_clusters[i], 0, max_clusters - 1)
        return cont.at[pred_cluster, true_label].add(1)

    contingency = jax.lax.fori_loop(0, true_labels.shape[0], body_fun, contingency)

    # Find the best cluster-to-label assignment (greedy)
    cluster_to_label = jnp.argmax(contingency, axis=1)

    # Map each point's cluster to its best label
    def map_fn(i: Array) -> Array:
        cluster = jnp.clip(pred_clusters[i], 0, max_clusters - 1)
        return cluster_to_label[cluster]

    mapped_preds = jax.vmap(map_fn)(jnp.arange(true_labels.shape[0]))

    return jnp.mean(mapped_preds == true_labels)


def clustering_nmi(
    n_clusters: int, n_classes: int, assignments: Array, true_labels: Array
) -> Array:
    """Compute Normalized Mutual Information (NMI) between cluster assignments and true labels.

    Fully JAX-compatible implementation that can be used within jax.jit.

    Args:
        n_clusters: Number of clusters
        n_classes: Number of classes
        assignments: Array of cluster assignments (n_samples,)
        true_labels: Array of true class labels (n_samples,)

    Returns:
        NMI score in [0, 1] (higher is better)
    """
    n_samples = assignments.shape[0]

    # Compute cluster and class counts
    cluster_counts = jnp.zeros(n_clusters).at[assignments].add(1.0)
    class_counts = jnp.zeros(n_classes).at[true_labels].add(1.0)

    # Initialize contingency matrix
    contingency = jnp.zeros((n_clusters, n_classes))

    # Build contingency matrix
    idx_matrix = jnp.stack([assignments, true_labels], axis=1)
    values = jnp.ones(n_samples)

    def update_contingency(i: Array, cont: Array) -> Array:
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
