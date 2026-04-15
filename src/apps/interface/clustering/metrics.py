"""Generic clustering metric utilities.

Provides JAX-compatible metric functions for evaluating clustering performance.
These can be used within jax.jit-compiled training loops.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypedDict

import jax
import jax.numpy as jnp
from jax import Array

from apps.runtime.metrics import as_metric_dict
from apps.runtime.util import MetricDict

INFO_LEVEL = jnp.array(logging.INFO)


ClusteringMetrics = TypedDict(
    "ClusteringMetrics",
    {
        "Clustering/Train Accuracy": tuple[Array, Array],
        "Clustering/Test Accuracy": tuple[Array, Array],
        "Clustering/Train NMI": tuple[Array, Array],
        "Clustering/Test NMI": tuple[Array, Array],
        "Clustering/Train ARI": tuple[Array, Array],
        "Clustering/Test ARI": tuple[Array, Array],
    },
)


def _build_contingency(
    n_clusters: int, n_classes: int, assignments: Array, true_labels: Array
) -> Array:
    """Build contingency matrix from cluster assignments and true labels."""
    n_samples = assignments.shape[0]
    contingency = jnp.zeros((n_clusters, n_classes))
    idx_matrix = jnp.stack([assignments, true_labels], axis=1)

    def update_contingency(i: Array, cont: Array) -> Array:
        idx = idx_matrix[i]
        return cont.at[idx[0], idx[1]].add(1.0)

    return jax.lax.fori_loop(0, n_samples, update_contingency, contingency)


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
    contingency = _build_contingency(n_clusters, n_classes, pred_clusters, true_labels)
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


def clustering_ari(
    n_clusters: int, n_classes: int, assignments: Array, true_labels: Array
) -> Array:
    """Compute Adjusted Rand Index (ARI) between cluster assignments and true labels.

    Fully JAX-compatible implementation using the contingency matrix formulation.

    Args:
        n_clusters: Number of clusters
        n_classes: Number of classes
        assignments: Array of cluster assignments (n_samples,)
        true_labels: Array of true class labels (n_samples,)

    Returns:
        ARI score in [-1, 1] (higher is better, 1.0 is perfect)
    """
    n_samples = jnp.array(assignments.shape[0], dtype=jnp.float32)
    contingency = _build_contingency(n_clusters, n_classes, assignments, true_labels)

    def comb2(x: Array) -> Array:
        return x * (x - 1) / 2.0

    sum_comb_c = jnp.sum(comb2(contingency))
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)
    sum_comb_a = jnp.sum(comb2(row_sums))
    sum_comb_b = jnp.sum(comb2(col_sums))
    comb_n = comb2(n_samples)

    expected_index = sum_comb_a * sum_comb_b / comb_n
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    return (sum_comb_c - expected_index) / (max_index - expected_index + 1e-10)


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

    contingency = _build_contingency(n_clusters, n_classes, assignments, true_labels)

    # Compute cluster and class counts
    cluster_counts = contingency.sum(axis=1)
    class_counts = contingency.sum(axis=0)

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
        - Clustering/Train ARI
        - Clustering/Test ARI
    """
    train_mapping = fit_cluster_mapping(
        train_labels, train_clusters, n_clusters, n_classes
    )
    train_acc = jnp.mean(train_mapping[train_clusters] == train_labels)
    test_acc = jnp.mean(train_mapping[test_clusters] == test_labels)

    train_nmi = clustering_nmi_fn(n_clusters, n_classes, train_clusters, train_labels)
    test_nmi = clustering_nmi_fn(n_clusters, n_classes, test_clusters, test_labels)

    train_ari = clustering_ari(n_clusters, n_classes, train_clusters, train_labels)
    test_ari = clustering_ari(n_clusters, n_classes, test_clusters, test_labels)

    cm: ClusteringMetrics = {
        "Clustering/Train Accuracy": (INFO_LEVEL, train_acc),
        "Clustering/Test Accuracy": (INFO_LEVEL, test_acc),
        "Clustering/Train NMI": (INFO_LEVEL, train_nmi),
        "Clustering/Test NMI": (INFO_LEVEL, test_nmi),
        "Clustering/Train ARI": (INFO_LEVEL, train_ari),
        "Clustering/Test ARI": (INFO_LEVEL, test_ari),
    }
    metrics.update(as_metric_dict(cm))
    return metrics
