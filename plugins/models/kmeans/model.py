"""K-means and PCA+K-means benchmark clustering models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, override

import jax.numpy as jnp
import numpy as np
from hydra.core.config_store import ConfigStore
from jax import Array
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score
from scipy.optimize import linear_sum_assignment
from sklearn.pipeline import Pipeline

from apps.interface import Analysis, ClusteringDataset, ClusteringModel, ClusteringModelConfig
from apps.runtime import Logger, RunHandler

log = logging.getLogger(__name__)


### Raw K-means ###

@dataclass
class KMeansConfig(ClusteringModelConfig):
    """Configuration for K-means clustering."""

    _target_: str = "plugins.models.kmeans.KMeansModel"
    
    # K-means specific parameters
    random_state: int = 42
    max_iter: int = 300
    n_init: int = 10
    tol: float = 1e-4
    algorithm: str = "lloyd"  # lloyd, elkan, auto, full


# Register config
cs = ConfigStore.instance()
cs.store(group="model", name="kmeans", node=KMeansConfig)


class KMeansModel(ClusteringModel):
    """K-means clustering benchmark model."""

    def __init__(
        self,
        n_clusters: int,
        data_dim: int,  # Required by base class
        random_state: int = 42,
        max_iter: int = 300,
        n_init: int = 10,
        tol: float = 1e-4,
        algorithm: str = "lloyd",
    ):
        self._n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.algorithm = algorithm
        self._kmeans = None
        self._cluster_centers = None
        self._train_data = None

    @property
    @override
    def n_clusters(self) -> int:
        return self._n_clusters

    @property
    @override
    def n_epochs(self) -> int:
        """K-means trains in a single step."""
        return 1

    @override
    def initialize_model(self, key: Array, data: Array) -> Array:
        """Initialize K-means model - returns dummy params since sklearn handles this."""
        # K-means doesn't need JAX parameters, return dummy array
        return jnp.array([0.0])

    @override
    def train(
        self,
        key: Array,
        handler: RunHandler,
        logger: Logger,
        dataset: ClusteringDataset,
    ) -> None:
        """Train K-means on the dataset."""
        log.info("Training K-means clustering model")
        
        # Convert JAX arrays to numpy for sklearn
        train_data = np.array(dataset.train_data)
        self._train_data = train_data  # Store for get_cluster_members
        
        # Initialize and fit K-means
        self._kmeans = KMeans(
            n_clusters=self._n_clusters,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init=self.n_init,
            tol=self.tol,
            algorithm=self.algorithm,
        )
        
        log.info(f"Fitting K-means with {self._n_clusters} clusters on {train_data.shape} data")
        self._kmeans.fit(train_data)
        self._cluster_centers = self._kmeans.cluster_centers_
        
        # Compute and log basic metrics
        train_labels = self._kmeans.predict(train_data)
        
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
            
            logger.log_metrics({"K-means/Train NMI": (20, jnp.array(nmi))}, jnp.array(0))
            logger.log_metrics({"K-means/Train ARI": (20, jnp.array(ari))}, jnp.array(0))
            logger.log_metrics({"K-means/Train Accuracy": (20, jnp.array(accuracy))}, jnp.array(0))
            
            # Test metrics
            test_data = np.array(dataset.test_data)
            test_labels = self._kmeans.predict(test_data)
            test_true_labels = np.array(dataset.test_labels)
            
            test_nmi = normalized_mutual_info_score(test_true_labels, test_labels)
            test_ari = adjusted_rand_score(test_true_labels, test_labels)
            test_accuracy = cluster_accuracy(test_true_labels, test_labels)
            
            logger.log_metrics({"K-means/Test NMI": (20, jnp.array(test_nmi))}, jnp.array(0))
            logger.log_metrics({"K-means/Test ARI": (20, jnp.array(test_ari))}, jnp.array(0))
            logger.log_metrics({"K-means/Test Accuracy": (20, jnp.array(test_accuracy))}, jnp.array(0))
            
            log.info(f"K-means Train NMI: {nmi:.3f}, ARI: {ari:.3f}, Accuracy: {accuracy:.3f}")
            log.info(f"K-means Test NMI: {test_nmi:.3f}, ARI: {test_ari:.3f}, Accuracy: {test_accuracy:.3f}")
        
        # Save dummy parameters (K-means stores state internally)
        handler.save_params(jnp.array([0.0]), 0)
        log.info("K-means training completed")

    def generate(self, params: Array, key: Array, n_samples: int) -> Array:
        """Generate samples from cluster centers (simple approach)."""
        if self._cluster_centers is None:
            raise ValueError("Model must be trained before generating samples")
        
        # Simple generation: sample from cluster centers with small noise
        rng = np.random.RandomState(int(key[0]) if len(key) > 0 else 42)
        
        # Randomly select cluster centers
        cluster_indices = rng.choice(self._n_clusters, size=n_samples)
        samples = self._cluster_centers[cluster_indices]
        
        # Add small amount of noise
        noise_scale = 0.1 * np.std(self._cluster_centers)
        noise = rng.normal(0, noise_scale, samples.shape)
        samples += noise
        
        return jnp.array(samples)

    @override
    def cluster_assignments(self, params: Array, data: Array) -> Array:
        """Assign data points to clusters."""
        if self._kmeans is None:
            raise ValueError("Model must be trained before assigning clusters")
        
        data_np = np.array(data)
        assignments = self._kmeans.predict(data_np)
        return jnp.array(assignments)

    def get_cluster_prototypes(self, handler: RunHandler, epoch: int) -> list[Array]:
        """Get K-means cluster centers as prototypes."""
        if self._cluster_centers is None:
            raise ValueError("Model must be trained before getting prototypes")

        return [jnp.array(center) for center in self._cluster_centers]

    def get_cluster_members(self, handler: RunHandler, epoch: int) -> list[Array]:
        """Get cluster members for each cluster."""
        if self._kmeans is None or self._train_data is None:
            raise ValueError("Model must be trained before getting cluster members")
        
        # Get cluster assignments for training data
        assignments = self._kmeans.predict(self._train_data)
        
        # Group data points by cluster
        cluster_members = []
        for cluster_id in range(self._n_clusters):
            # Find all points assigned to this cluster
            cluster_mask = assignments == cluster_id
            members = self._train_data[cluster_mask]
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
        """Run analysis suite for K-means (minimal for benchmark)."""
        # For benchmarking, we already logged metrics during training
        # Could add more sophisticated analysis here if needed
        log.info("K-means analysis completed (metrics logged during training)")

    @override
    def get_analyses(self, dataset: ClusteringDataset) -> list[Analysis[ClusteringDataset, Any, Any]]:
        """Return list of analyses for K-means."""
        # For now, return empty list - could add simple cluster statistics later
        return []


### PCA + K-means ###

@dataclass
class PCAKMeansConfig(ClusteringModelConfig):
    """Configuration for PCA + K-means clustering."""

    _target_: str = "plugins.models.kmeans.PCAKMeansModel"
    
    # PCA parameters
    n_components: int | float = 50  # Number of components or variance ratio
    pca_random_state: int = 42
    
    # K-means parameters
    random_state: int = 42
    max_iter: int = 300
    n_init: int = 10
    tol: float = 1e-4
    algorithm: str = "lloyd"


# Register config
cs.store(group="model", name="pca_kmeans", node=PCAKMeansConfig)


class PCAKMeansModel(ClusteringModel):
    """PCA + K-means clustering benchmark model."""

    def __init__(
        self,
        n_clusters: int,
        data_dim: int,  # Required by base class
        n_components: int | float = 50,
        pca_random_state: int = 42,
        random_state: int = 42,
        max_iter: int = 300,
        n_init: int = 10,
        tol: float = 1e-4,
        algorithm: str = "lloyd",
    ):
        self._n_clusters = n_clusters
        self.n_components = n_components
        self.pca_random_state = pca_random_state
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.algorithm = algorithm
        self._pipeline = None
        self._pca = None
        self._kmeans = None
        self._train_data = None

    @property
    @override
    def n_clusters(self) -> int:
        return self._n_clusters

    @property
    @override
    def n_epochs(self) -> int:
        """PCA + K-means trains in a single step."""
        return 1

    @override
    def initialize_model(self, key: Array, data: Array) -> Array:
        """Initialize PCA + K-means pipeline."""
        return jnp.array([0.0])

    @override
    def train(
        self,
        key: Array,
        handler: RunHandler,
        logger: Logger,
        dataset: ClusteringDataset,
    ) -> None:
        """Train PCA + K-means pipeline."""
        log.info("Training PCA + K-means clustering model")
        
        # Convert JAX arrays to numpy for sklearn
        train_data = np.array(dataset.train_data)
        self._train_data = train_data  # Store for get_cluster_members
        
        # Create PCA + K-means pipeline
        self._pca = PCA(
            n_components=self.n_components,
            random_state=self.pca_random_state,
        )
        
        self._kmeans = KMeans(
            n_clusters=self._n_clusters,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init=self.n_init,
            tol=self.tol,
            algorithm=self.algorithm,
        )
        
        self._pipeline = Pipeline([
            ('pca', self._pca),
            ('kmeans', self._kmeans)
        ])
        
        log.info(f"Fitting PCA({self.n_components}) + K-means({self._n_clusters}) on {train_data.shape} data")
        self._pipeline.fit(train_data)
        
        # Log PCA info
        if isinstance(self.n_components, int):
            explained_var = np.sum(self._pca.explained_variance_ratio_)
            log.info(f"PCA explains {explained_var:.3f} of variance with {self._pca.n_components_} components")
            logger.log_metrics({"PCA/Explained_Variance_Ratio": (20, jnp.array(explained_var))}, jnp.array(0))
            logger.log_metrics({"PCA/N_Components": (20, jnp.array(self._pca.n_components_))}, jnp.array(0))
        
        # Get cluster assignments
        train_labels = self._pipeline.predict(train_data)
        
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
            
            logger.log_metrics({"PCA+K-means/Train NMI": (20, jnp.array(nmi))}, jnp.array(0))
            logger.log_metrics({"PCA+K-means/Train ARI": (20, jnp.array(ari))}, jnp.array(0))
            logger.log_metrics({"PCA+K-means/Train Accuracy": (20, jnp.array(accuracy))}, jnp.array(0))
            
            # Test metrics
            test_data = np.array(dataset.test_data)
            test_labels = self._pipeline.predict(test_data)
            test_true_labels = np.array(dataset.test_labels)
            
            test_nmi = normalized_mutual_info_score(test_true_labels, test_labels)
            test_ari = adjusted_rand_score(test_true_labels, test_labels)
            test_accuracy = cluster_accuracy(test_true_labels, test_labels)
            
            logger.log_metrics({"PCA+K-means/Test NMI": (20, jnp.array(test_nmi))}, jnp.array(0))
            logger.log_metrics({"PCA+K-means/Test ARI": (20, jnp.array(test_ari))}, jnp.array(0))
            logger.log_metrics({"PCA+K-means/Test Accuracy": (20, jnp.array(test_accuracy))}, jnp.array(0))
            
            log.info(f"PCA+K-means Train NMI: {nmi:.3f}, ARI: {ari:.3f}, Accuracy: {accuracy:.3f}")
            log.info(f"PCA+K-means Test NMI: {test_nmi:.3f}, ARI: {test_ari:.3f}, Accuracy: {test_accuracy:.3f}")
        
        # Save dummy parameters
        handler.save_params(jnp.array([0.0]), 0)
        log.info("PCA + K-means training completed")

    def generate(self, params: Array, key: Array, n_samples: int) -> Array:
        """Generate samples by sampling from clusters and inverse PCA transform."""
        if self._pipeline is None:
            raise ValueError("Model must be trained before generating samples")
        
        rng = np.random.RandomState(int(key[0]) if len(key) > 0 else 42)
        
        # Get cluster centers in PCA space
        cluster_centers_pca = self._kmeans.cluster_centers_
        
        # Sample from clusters with noise in PCA space
        cluster_indices = rng.choice(self._n_clusters, size=n_samples)
        samples_pca = cluster_centers_pca[cluster_indices]
        
        # Add noise in PCA space
        noise_scale = 0.1 * np.std(cluster_centers_pca)
        noise = rng.normal(0, noise_scale, samples_pca.shape)
        samples_pca += noise
        
        # Inverse transform back to original space
        samples_original = self._pca.inverse_transform(samples_pca)
        
        return jnp.array(samples_original)

    @override
    def cluster_assignments(self, params: Array, data: Array) -> Array:
        """Assign data points to clusters using PCA + K-means pipeline."""
        if self._pipeline is None:
            raise ValueError("Model must be trained before assigning clusters")
        
        data_np = np.array(data)
        assignments = self._pipeline.predict(data_np)
        return jnp.array(assignments)

    def get_cluster_prototypes(self, handler: RunHandler, epoch: int) -> list[Array]:
        """Get cluster prototypes in original space (inverse transform of PCA centers)."""
        if self._pipeline is None:
            raise ValueError("Model must be trained before getting prototypes")

        cluster_centers_pca = self._kmeans.cluster_centers_
        cluster_centers_original = self._pca.inverse_transform(cluster_centers_pca)
        return [jnp.array(center) for center in cluster_centers_original]

    def get_cluster_members(self, handler: RunHandler, epoch: int) -> list[Array]:
        """Get cluster members for each cluster."""
        if self._pipeline is None or self._train_data is None:
            raise ValueError("Model must be trained before getting cluster members")
        
        # Get cluster assignments
        assignments = self._pipeline.predict(self._train_data)
        
        # Group data points by cluster
        cluster_members = []
        for cluster_id in range(self._n_clusters):
            cluster_mask = assignments == cluster_id
            members = self._train_data[cluster_mask]
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
        """Run analysis suite for PCA + K-means (minimal for benchmark)."""
        log.info("PCA + K-means analysis completed (metrics logged during training)")

    @override
    def get_analyses(self, dataset: ClusteringDataset) -> list[Analysis[ClusteringDataset, Any, Any]]:
        """Return list of analyses for PCA + K-means."""
        return []