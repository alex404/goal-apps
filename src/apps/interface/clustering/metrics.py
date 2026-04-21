"""Generic clustering metric utilities.

Provides JAX-compatible metric functions for evaluating clustering performance.
These can be used within jax.jit-compiled training loops.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from goal.models import Categorical
from jax import Array

from apps.runtime import INFO_LEVEL, MetricDict

CLUSTERING_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "Clustering/Train Accuracy",
        "Clustering/Test Accuracy",
        "Clustering/Train NMI",
        "Clustering/Test NMI",
        "Clustering/Train ARI",
        "Clustering/Test ARI",
    }
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
    """Derive greedy cluster->class mapping from labeled data.

    Builds a contingency matrix and assigns each cluster to its most frequent
    class. Returns the mapping array so it can be applied to a held-out split.

    Args:
        true_labels: Ground truth labels used to derive the mapping (n_samples,)
        pred_clusters: Predicted cluster assignments (n_samples,)
        n_clusters: Number of clusters in the model
        n_classes: Number of ground-truth classes

    Returns:
        cluster_to_label: Array of shape (n_clusters,) mapping cluster index -> class label
    """
    contingency = _build_contingency(n_clusters, n_classes, pred_clusters, true_labels)
    return jnp.argmax(contingency, axis=1)


def cluster_accuracy(
    true_labels: Array, pred_clusters: Array, n_clusters: int, n_classes: int
) -> Array:
    """Compute clustering accuracy with greedy label assignment.

    Fits the cluster->class mapping from the same data it evaluates on.
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
    # Fail-fast: no epsilon guard. A zero denominator means one split is
    # perfectly uniform (single cluster or single class) — ARI is genuinely
    # undefined. Let it return NaN so the degenerate run surfaces.
    return (sum_comb_c - expected_index) / (max_index - expected_index)


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

    cluster_probs = contingency.sum(axis=1) / n_samples
    class_probs = contingency.sum(axis=0) / n_samples
    joint_probs = (contingency / n_samples).ravel()

    # Compute entropies via goal-jax's Categorical negative_entropy, which
    # consumes MEAN coordinates (K - 1 dims; the 0th probability is dropped
    # and reconstructed as 1 - sum(means) internally). ``from_probs`` converts
    # our full-length probability vector into mean coords. Natural coords
    # (log-odds via ``to_natural``) would give identical entropy but requires
    # extra softmax conversion — mean coords are what we already have.
    cat_k = Categorical(n_categories=n_clusters)
    cat_c = Categorical(n_categories=n_classes)
    cat_kc = Categorical(n_categories=n_clusters * n_classes)
    cluster_entropy = -cat_k.negative_entropy(cat_k.from_probs(cluster_probs))
    class_entropy = -cat_c.negative_entropy(cat_c.from_probs(class_probs))
    joint_entropy = -cat_kc.negative_entropy(cat_kc.from_probs(joint_probs))
    mutual_info = cluster_entropy + class_entropy - joint_entropy

    # Fail-fast: no guard on the denominator. If both splits are degenerate
    # (single cluster AND single class), NMI is undefined — let NaN surface.
    return 2.0 * mutual_info / (cluster_entropy + class_entropy)


def add_clustering_metrics(
    metrics: MetricDict,
    n_clusters: int,
    n_classes: int,
    train_labels: Array,
    test_labels: Array,
    train_clusters: Array,
    test_clusters: Array,
) -> MetricDict:
    """Add clustering evaluation metrics.

    The cluster->class mapping is derived from the training split and applied
    to both splits, so test accuracy is a proper held-out evaluation.

    Args:
        metrics: Existing metrics dict to update
        n_clusters: Number of clusters in the model
        n_classes: Number of ground-truth classes
        train_labels: Ground-truth labels for training data
        test_labels: Ground-truth labels for test data
        train_clusters: Predicted cluster assignments for training data
        test_clusters: Predicted cluster assignments for test data

    Returns:
        Updated metrics dict with clustering accuracy, NMI, and ARI.
    """
    train_mapping = fit_cluster_mapping(
        train_labels, train_clusters, n_clusters, n_classes
    )
    train_acc = jnp.mean(train_mapping[train_clusters] == train_labels)
    test_acc = jnp.mean(train_mapping[test_clusters] == test_labels)

    train_nmi = clustering_nmi(n_clusters, n_classes, train_clusters, train_labels)
    test_nmi = clustering_nmi(n_clusters, n_classes, test_clusters, test_labels)

    train_ari = clustering_ari(n_clusters, n_classes, train_clusters, train_labels)
    test_ari = clustering_ari(n_clusters, n_classes, test_clusters, test_labels)

    metrics.update(
        {
            "Clustering/Train Accuracy": (INFO_LEVEL, train_acc),
            "Clustering/Test Accuracy": (INFO_LEVEL, test_acc),
            "Clustering/Train NMI": (INFO_LEVEL, train_nmi),
            "Clustering/Test NMI": (INFO_LEVEL, test_nmi),
            "Clustering/Train ARI": (INFO_LEVEL, train_ari),
            "Clustering/Test ARI": (INFO_LEVEL, test_ari),
        }
    )
    return metrics
