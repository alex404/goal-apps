"""Generic clustering evaluation metrics.

This module provides reusable analysis classes for computing standard
clustering metrics like accuracy, NMI, and log-likelihood. These work
with any clustering model that implements the required protocols.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import override

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array
from matplotlib.figure import Figure
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score

from apps.interface import Analysis, ClusteringDataset, ClusteringModel
from apps.runtime import Artifact, RunHandler


@dataclass(frozen=True)
class ClusteringMetrics(Artifact):
    """Standard clustering evaluation metrics."""

    train_log_likelihood: float
    test_log_likelihood: float
    accuracy: float | None
    nmi: float | None


@dataclass(frozen=True)
class ClusteringMetricsAnalysis[M: ClusteringModel](
    Analysis[ClusteringDataset, M, ClusteringMetrics]
):
    """Generic analysis for computing clustering evaluation metrics.

    This analysis works with any clustering model that implements HasLogLikelihood
    (log_likelihood method). The model must provide:
    - log_likelihood(params, data) -> float
    - cluster_assignments(params, data) -> Array
    - n_clusters property
    """

    @property
    @override
    def artifact_type(self) -> type[ClusteringMetrics]:
        """Return artifact type."""
        return ClusteringMetrics

    @override
    def generate(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: M,
        epoch: int,
        params: Array,
    ) -> ClusteringMetrics:
        """Compute clustering evaluation metrics.

        Args:
            key: Random key
            handler: Run handler for loading params
            dataset: Dataset to evaluate
            model: Clustering model instance
            epoch: Epoch number
            params: Model parameters

        Returns:
            ClusteringMetrics artifact with computed metrics
        """
        # Compute log-likelihoods using the HasLogLikelihood protocol
        train_ll = model.log_likelihood(params, dataset.train_data)
        test_ll = model.log_likelihood(params, dataset.test_data)

        # Compute classification metrics if labels available
        test_labels = getattr(dataset, "test_labels", None)
        assignments = model.cluster_assignments(params, dataset.test_data)

        nmi = (
            float(normalized_mutual_info_score(test_labels, assignments))
            if test_labels is not None
            else None
        )

        # Compute accuracy via Hungarian matching
        if test_labels is not None:
            n_classes = int(jnp.max(test_labels)) + 1
            confusion = jnp.zeros((model.n_clusters, n_classes))

            for c in range(model.n_clusters):
                for l in range(n_classes):
                    count = jnp.sum((assignments == c) & (test_labels == l))
                    confusion = confusion.at[c, l].set(count)

            row_ind, col_ind = linear_sum_assignment(-confusion)
            correct = sum(confusion[r, c] for r, c in zip(row_ind, col_ind))
            accuracy = float(correct / len(test_labels))
        else:
            accuracy = None

        return ClusteringMetrics(
            train_log_likelihood=train_ll,
            test_log_likelihood=test_ll,
            accuracy=accuracy,
            nmi=nmi,
        )

    @override
    def plot(
        self,
        artifact: ClusteringMetrics,
        dataset: ClusteringDataset,
    ) -> Figure:
        """Clustering metrics don't have a visual plot.

        Args:
            artifact: Clustering metrics
            dataset: Dataset

        Returns:
            Empty matplotlib figure
        """
        fig = plt.figure(figsize=(1, 1))
        plt.close(fig)
        return fig

    @override
    def metrics(self, artifact: ClusteringMetrics) -> dict[str, tuple[Array, Array]]:
        """Return metrics for logging.

        Args:
            artifact: Clustering metrics

        Returns:
            Dictionary of metrics in format {name: (log_level, value)}
        """
        result = {
            "eval/train_log_likelihood": (
                jnp.array(logging.INFO),
                jnp.array(artifact.train_log_likelihood),
            ),
            "eval/test_log_likelihood": (
                jnp.array(logging.INFO),
                jnp.array(artifact.test_log_likelihood),
            ),
        }

        if artifact.accuracy is not None:
            result["eval/accuracy"] = (
                jnp.array(logging.INFO),
                jnp.array(artifact.accuracy),
            )

        if artifact.nmi is not None:
            result["eval/nmi"] = (jnp.array(logging.INFO), jnp.array(artifact.nmi))

        return result
