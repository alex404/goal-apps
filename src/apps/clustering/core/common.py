from typing import TypedDict

import jax.numpy as jnp
import numpy as np
from jax import Array
from sklearn.metrics import accuracy_score


def evaluate_clustering(cluster_assignments: Array, true_labels: Array) -> float:
    """Evaluate clustering by finding optimal label assignment."""
    n_clusters = int(jnp.max(cluster_assignments)) + 1
    n_classes = int(jnp.max(true_labels)) + 1

    # Compute cluster-class frequency matrix
    freq_matrix = np.zeros((n_clusters, n_classes))
    for i in range(len(true_labels)):
        freq_matrix[int(cluster_assignments[i]), int(true_labels[i])] += 1

    # Assign each cluster to its most frequent class
    cluster_to_class = np.argmax(freq_matrix, axis=1)
    predicted_labels = jnp.array([cluster_to_class[i] for i in cluster_assignments])

    return float(accuracy_score(true_labels, predicted_labels))


class ProbabilisticResults(TypedDict):
    model_name: str
    test_log_likelihood: list[float]
    train_log_likelihood: list[float]
    final_train_log_likelihood: float
    final_test_log_likelihood: float
    train_accuracy: float
    test_accuracy: float
    latent_dim: int
    n_clusters: int
    n_parameters: int
    training_time: float
    prototypes: list[list[float]]


class TwoStageResults(TypedDict):
    model_name: str
    reconstruction_error: float
    train_accuracy: float
    test_accuracy: float
    latent_dim: int
    n_clusters: int
    training_time: float
