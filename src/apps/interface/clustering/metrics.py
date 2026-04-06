"""Generic clustering metric utilities.

Provides JAX-compatible metric functions for evaluating clustering performance.
These can be used within jax.jit-compiled training loops.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

from apps.runtime.util import MetricDict

INFO_LEVEL = jnp.array(logging.INFO)


def fit_cluster_mapping(
    true_labels: Array, pred_clusters: Array, n_clusters: int, n_classes: int
) -> Array:
    """Derive greedy cluster→class mapping from labeled data.

    Builds a contingency matrix and assigns each cluster to its most frequent
    class. Returns the mapping array so it can be applied to a held-out split.

    Args:
        true_labels: Ground truth labels used to derive the mapping (n_samples,)
        pred_clusters: Predicted cluster assignments (n_samples,)
        n_clusters: Number of clusters in the model
        n_classes: Number of ground-truth classes

    Returns:
        cluster_to_label: Array of shape (n_clusters,) mapping cluster index → class label
    """
    contingency = jnp.zeros((n_clusters, n_classes))

    def body_fun(i: Array, cont: Array) -> Array:
        return cont.at[pred_clusters[i], true_labels[i]].add(1)

    contingency = jax.lax.fori_loop(0, true_labels.shape[0], body_fun, contingency)
    return jnp.argmax(contingency, axis=1)


def cluster_accuracy(
    true_labels: Array, pred_clusters: Array, n_clusters: int, n_classes: int
) -> Array:
    """Compute clustering accuracy with greedy label assignment.

    Fits the cluster→class mapping from the same data it evaluates on.
    Use fit_cluster_mapping + direct evaluation when you need to apply a
    mapping derived from a separate (training) split.

    Args:
        true_labels: Ground truth labels (n_samples,)
        pred_clusters: Predicted cluster assignments (n_samples,)
        n_clusters: Number of clusters in the model
        n_classes: Number of ground-truth classes

    Returns:
        Clustering accuracy after greedy label assignment
    """
    mapping = fit_cluster_mapping(true_labels, pred_clusters, n_clusters, n_classes)
    mapped_preds = mapping[pred_clusters]
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

    return 2.0 * mutual_info / (cluster_entropy + class_entropy)


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
    train_mapping = fit_cluster_mapping(
        train_labels, train_clusters, n_clusters, n_classes
    )
    train_acc = jnp.mean(train_mapping[train_clusters] == train_labels)
    test_acc = jnp.mean(train_mapping[test_clusters] == test_labels)

    train_nmi = clustering_nmi_fn(n_clusters, n_classes, train_clusters, train_labels)
    test_nmi = clustering_nmi_fn(n_clusters, n_classes, test_clusters, test_labels)

    metrics.update({
        "Clustering/Train Accuracy": (INFO_LEVEL, train_acc),
        "Clustering/Test Accuracy": (INFO_LEVEL, test_acc),
        "Clustering/Train NMI": (INFO_LEVEL, train_nmi),
        "Clustering/Test NMI": (INFO_LEVEL, test_nmi),
    })
    return metrics
