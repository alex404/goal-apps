"""LDA (Latent Dirichlet Allocation) benchmark clustering model."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, override

import jax.numpy as jnp
import numpy as np
from hydra.core.config_store import ConfigStore
from jax import Array
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from apps.interface import (
    Analysis,
    ClusteringDataset,
    ClusteringModel,
    ClusteringModelConfig,
)
from apps.runtime import Logger, RunHandler

log = logging.getLogger(__name__)


@dataclass
class LDAConfig(ClusteringModelConfig):
    """Configuration for LDA topic modeling."""

    _target_: str = "plugins.models.lda.LDAModel"

    # LDA specific parameters
    random_state: int = 42
    max_iter: int = 50  # Reduced for speed on high-dim data
    learning_method: str = "online"  # Online is faster for large datasets
    alpha: float = 0.1  # Document-topic concentration
    beta: float = 0.01  # Topic-word concentration
    learning_decay: float = 0.7  # For online learning
    perp_tol: float = 1e-1  # Perplexity tolerance


# Register config
cs = ConfigStore.instance()
cs.store(group="model", name="lda", node=LDAConfig)


class LDAModel(ClusteringModel):
    """LDA topic modeling benchmark for clustering."""

    def __init__(
        self,
        n_clusters: int,
        data_dim: int,  # Required by base class
        random_state: int = 42,
        max_iter: int = 50,
        learning_method: str = "online",
        alpha: float = 0.1,
        beta: float = 0.01,
        learning_decay: float = 0.7,
        perp_tol: float = 1e-1,
    ):
        self._n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.learning_method = learning_method
        self.alpha = alpha
        self.beta = beta
        self.learning_decay = learning_decay
        self.perp_tol = perp_tol
        self._lda = None
        self._topic_distributions = None
        self._train_data = None

    @property
    @override
    def n_clusters(self) -> int:
        return self._n_clusters

    @property
    @override
    def n_epochs(self) -> int:
        """LDA trains in a single step."""
        return 1

    @override
    def initialize_model(self, key: Array, data: Array) -> Array:
        """Initialize LDA model - returns dummy params since sklearn handles this."""
        return jnp.array([0.0])

    @override
    def train(
        self,
        key: Array,
        handler: RunHandler,
        logger: Logger,
        dataset: ClusteringDataset,
    ) -> None:
        """Train LDA on the dataset."""
        log.info("Training LDA topic model")

        # Convert JAX arrays to numpy for sklearn
        train_data = np.array(dataset.train_data)
        self._train_data = train_data  # Store for get_cluster_members

        # Ensure non-negative values for LDA (required for topic modeling)
        if np.any(train_data < 0):
            log.warning("LDA requires non-negative input - taking absolute values")
            train_data = np.abs(train_data)

        # Initialize and fit LDA
        self._lda = LatentDirichletAllocation(
            n_components=self._n_clusters,
            random_state=self.random_state,
            max_iter=self.max_iter,
            learning_method=self.learning_method,
            doc_topic_prior=self.alpha,
            topic_word_prior=self.beta,
            learning_decay=self.learning_decay,
            perp_tol=self.perp_tol,
        )

        log.info(
            f"Fitting LDA with {self._n_clusters} topics on {train_data.shape} data"
        )
        self._lda.fit(train_data)

        # Get topic distributions for documents
        train_topic_dist = self._lda.transform(train_data)

        # Assign documents to dominant topics for clustering evaluation
        train_labels = np.argmax(train_topic_dist, axis=1)

        if dataset.has_labels:
            true_labels = np.array(dataset.train_labels)
            nmi = normalized_mutual_info_score(true_labels, train_labels)
            ari = adjusted_rand_score(true_labels, train_labels)

            # Compute accuracy with optimal cluster-to-class assignment (Hungarian algorithm)
            def cluster_accuracy(y_true, y_pred):
                # Create confusion matrix
                n_clusters = len(np.unique(y_pred))
                n_classes = len(np.unique(y_true))
                cost_matrix = np.zeros((n_clusters, n_classes))

                for i in range(n_clusters):
                    for j in range(n_classes):
                        # Cost is negative count (we want to maximize matches)
                        cost_matrix[i, j] = -np.sum((y_pred == i) & (y_true == j))

                # Solve assignment problem
                row_indices, col_indices = linear_sum_assignment(cost_matrix)

                # Calculate accuracy with optimal assignment
                total_correct = 0
                for i, j in zip(row_indices, col_indices):
                    total_correct += -cost_matrix[i, j]  # Convert back to positive

                return total_correct / len(y_true)

            accuracy = cluster_accuracy(true_labels, train_labels)

            logger.log_metrics({"LDA/Train NMI": (20, jnp.array(nmi))}, jnp.array(0))
            logger.log_metrics({"LDA/Train ARI": (20, jnp.array(ari))}, jnp.array(0))
            logger.log_metrics(
                {"LDA/Train Accuracy": (20, jnp.array(accuracy))}, jnp.array(0)
            )

            # Test metrics
            test_data = np.array(dataset.test_data)
            if np.any(test_data < 0):
                test_data = np.abs(test_data)

            test_topic_dist = self._lda.transform(test_data)
            test_labels = np.argmax(test_topic_dist, axis=1)
            test_true_labels = np.array(dataset.test_labels)

            test_nmi = normalized_mutual_info_score(test_true_labels, test_labels)
            test_ari = adjusted_rand_score(test_true_labels, test_labels)
            test_accuracy = cluster_accuracy(test_true_labels, test_labels)

            logger.log_metrics(
                {"LDA/Test NMI": (20, jnp.array(test_nmi))}, jnp.array(0)
            )
            logger.log_metrics(
                {"LDA/Test ARI": (20, jnp.array(test_ari))}, jnp.array(0)
            )
            logger.log_metrics(
                {"LDA/Test Accuracy": (20, jnp.array(test_accuracy))}, jnp.array(0)
            )

            log.info(
                f"LDA Train NMI: {nmi:.3f}, ARI: {ari:.3f}, Accuracy: {accuracy:.3f}"
            )
            log.info(
                f"LDA Test NMI: {test_nmi:.3f}, ARI: {test_ari:.3f}, Accuracy: {test_accuracy:.3f}"
            )

        # Log perplexity (lower is better for LDA)
        perplexity = self._lda.perplexity(train_data)
        logger.log_metrics(
            {"LDA/Train Perplexity": (20, jnp.array(perplexity))}, jnp.array(0)
        )
        log.info(f"LDA Train Perplexity: {perplexity:.3f}")

        # Save dummy parameters
        handler.save_params(jnp.array([0.0]), 0)
        log.info("LDA training completed")

    @override
    def generate(self, params: Array, key: Array, n_samples: int) -> Array:
        """Generate samples from LDA topics."""
        if self._lda is None:
            raise ValueError("Model must be trained before generating samples")

        # Generate samples by sampling from topics
        rng = np.random.RandomState(int(key[0]) if len(key) > 0 else 42)

        # Sample topic proportions for each document
        topic_proportions = rng.dirichlet(
            np.ones(self._n_clusters) * self.alpha, n_samples
        )

        # Sample words from topics weighted by proportions
        topic_word_dist = self._lda.components_  # shape: (n_topics, n_features)

        samples = np.zeros((n_samples, topic_word_dist.shape[1]))

        for i in range(n_samples):
            # For each document, sample words based on topic mixture
            doc_word_dist = np.dot(topic_proportions[i], topic_word_dist)
            doc_word_dist /= doc_word_dist.sum()  # Normalize

            # Sample some words (simple approach - could be more sophisticated)
            n_words = rng.poisson(50)  # Average document length
            word_samples = rng.multinomial(n_words, doc_word_dist)
            samples[i] = word_samples

        return jnp.array(samples)

    @override
    def cluster_assignments(self, params: Array, data: Array) -> Array:
        """Assign data points to dominant topics."""
        if self._lda is None:
            raise ValueError("Model must be trained before assigning clusters")

        data_np = np.array(data)
        if np.any(data_np < 0):
            data_np = np.abs(data_np)

        topic_distributions = self._lda.transform(data_np)
        assignments = np.argmax(topic_distributions, axis=1)
        return jnp.array(assignments)

    @override
    def get_cluster_prototypes(self, handler: RunHandler, epoch: int) -> list[Array]:
        """Get LDA topic-word distributions as prototypes."""
        if self._lda is None:
            raise ValueError("Model must be trained before getting prototypes")

        # Return topic-word distributions (each row is a topic)
        topic_word_dist = self._lda.components_
        return [jnp.array(topic) for topic in topic_word_dist]

    @override
    def get_cluster_members(self, handler: RunHandler, epoch: int) -> list[Array]:
        """Get cluster members for each topic."""
        if self._lda is None or self._train_data is None:
            raise ValueError("Model must be trained before getting cluster members")

        # Get topic assignments (dominant topic per document)
        topic_distributions = self._lda.transform(self._train_data)
        assignments = np.argmax(topic_distributions, axis=1)

        # Group documents by dominant topic
        cluster_members = []
        for topic_id in range(self._n_clusters):
            topic_mask = assignments == topic_id
            members = self._train_data[topic_mask]
            cluster_members.append(jnp.array(members))

        return cluster_members

    @override
    def analyze(
        self,
        key: Array,
        handler: RunHandler,
        logger: Logger,
        dataset: ClusteringDataset,
    ) -> None:
        """Run analysis suite for LDA (minimal for benchmark)."""
        log.info("LDA analysis completed (metrics logged during training)")

    @override
    def get_analyses(
        self, dataset: ClusteringDataset
    ) -> list[Analysis[ClusteringDataset, Any, Any]]:
        """Return list of analyses for LDA."""
        # For now, return empty list - could add topic analysis later
        return []
