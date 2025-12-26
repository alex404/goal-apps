"""MFA (Mixture of Factor Analyzers) model implementation."""

from __future__ import annotations

import logging
from typing import Any, override

import jax
import jax.numpy as jnp
import numpy as np
from goal.models import FactorAnalysis, Normal
from goal.models.graphical.mixture import MixtureOfConjugated
from jax import Array
from sklearn.cluster import KMeans

from apps.interface import Analysis, ClusteringDataset, ClusteringModel
from apps.interface.analyses import GenerativeSamplesAnalysis
from apps.interface.clustering.analyses import (
    ClusterStatistics,
    ClusterStatisticsAnalysis,
    CoAssignmentHierarchyAnalysis,
    CoAssignmentMergeAnalysis,
    OptimalMergeAnalysis,
)
from apps.interface.clustering.config import ClusteringAnalysesConfig
from apps.interface.clustering.protocols import (
    CanComputePrototypes,
    HasSoftAssignments,
)
from apps.interface.protocols import HasLogLikelihood, IsGenerative
from apps.runtime import Logger, RunHandler

from .trainers import GradientTrainer

log = logging.getLogger(__name__)

# Type alias for MFA model
type MFA = MixtureOfConjugated[Normal, Normal]


class MFAModel(
    ClusteringModel,
    HasLogLikelihood,
    IsGenerative,
    HasSoftAssignments,
    CanComputePrototypes,
):
    """Mixture of Factor Analyzers model for probabilistic clustering and classification.

    MFA combines factor analysis with mixture modeling, enabling both dimensionality reduction
    and clustering. Each mixture component has its own mean in factor space, but components
    share a common loading matrix and noise covariance structure.
    """

    def __init__(
        self,
        data_dim: int,
        latent_dim: int,
        n_clusters: int,
        trainer: GradientTrainer,
        analyses: ClusteringAnalysesConfig,
        init_scale: float = 0.01,
        min_var: float = 0.01,
    ):
        """Initialize MFA model.

        Args:
            data_dim: Dimension of observable data
            latent_dim: Dimension of latent factors
            n_clusters: Number of mixture components
            trainer: Trainer instance for optimization
            analyses: Configuration for analyses
            init_scale: Scale for parameter initialization (smaller for high-dim data)
            min_var: Minimum variance for regularization (prevents NaN for zero-variance pixels)
        """
        self.data_dim: int = data_dim
        self.latent_dim: int = latent_dim
        self.n_clusters_val: int = n_clusters
        self.trainer: GradientTrainer = trainer
        self.analyses_config: ClusteringAnalysesConfig = analyses
        self.init_scale: float = init_scale
        self.min_var: float = min_var

        # Create MFA model from goal-jax
        base_fa = FactorAnalysis(obs_dim=data_dim, lat_dim=latent_dim)
        self.mfa: MFA = MixtureOfConjugated(n_categories=n_clusters, hrm=base_fa)

    # Properties

    @property
    @override
    def n_epochs(self) -> int:
        """Number of training epochs."""
        return self.trainer.n_epochs

    @property
    @override
    def n_clusters(self) -> int:
        """Number of mixture components."""
        return self.n_clusters_val

    # Core methods

    @override
    def initialize_model(self, key: Array, data: Array) -> Array:
        """Initialize MFA parameters from k-means clustering.

        Each mixture component is initialized as an FA with:
        - Observable mean = k-means cluster center
        - Observable covariance = identity (scaled by data variance)
        - Interaction (loading) = small noise
        - Latent = standard normal N(0, I)

        This ensures that without training, the MFA achieves similar
        clustering performance to k-means.

        Args:
            key: Random key for initialization
            data: Training data for initialization

        Returns:
            Initial model parameters (natural coordinates)
        """
        keys = jax.random.split(key, 3)

        # Run k-means to get cluster centers
        log.info("Running k-means for initialization...")
        kmeans = KMeans(
            n_clusters=self.n_clusters, random_state=int(keys[0][0]), n_init="auto"
        )
        kmeans.fit(np.asarray(data))
        centers = jax.numpy.asarray(kmeans.cluster_centers_)
        log.info("K-means initialization complete")

        # Compute uniform variance for covariance initialization
        # Using mean variance across dimensions gives isotropic-like behavior
        # that matches k-means' Euclidean distance assumption
        data_var = jnp.var(data, axis=0)
        mean_var = jnp.maximum(jnp.mean(data_var), self.min_var)
        reg_var = jnp.full(data_var.shape, mean_var)

        # Build each FA component in natural coordinates
        obs_man_fa = self.mfa.hrm.obs_man  # Diagonal Normal for observables
        pst_man_fa = self.mfa.hrm.pst_man  # PositiveDefinite Normal for latents

        component_nat_list = []
        int_key = keys[1]
        for k in range(self.n_clusters):
            # Observable: mean=center, cov=diag(reg_var) -> convert to natural
            obs_cov_params = obs_man_fa.cov_man.from_matrix(jnp.diag(reg_var))
            obs_means = obs_man_fa.join_mean_covariance(centers[k], obs_cov_params)
            obs_params = obs_man_fa.to_natural(obs_means)

            # Interaction: small noise for loading matrix
            int_key, subkey = jax.random.split(int_key)
            int_params = self.init_scale * jax.random.normal(
                subkey, shape=(self.mfa.hrm.int_man.dim,)
            )

            # Latent: standard normal in natural coords
            lat_means = pst_man_fa.standard_normal()
            lat_params = pst_man_fa.to_natural(lat_means)

            # Join FA params
            fa_params = self.mfa.hrm.join_coords(obs_params, int_params, lat_params)
            component_nat_list.append(fa_params)

        components_nat = jnp.concatenate(component_nat_list)

        # Categorical: uniform weights (natural params are log-odds, all zero for uniform)
        cat_nat = jnp.zeros(self.n_clusters - 1)

        # Join into mixture natural params and convert to MFA params
        mix_params = self.mfa.mix_man.join_natural_mixture(components_nat, cat_nat)
        mfa_params = self.mfa.from_mixture_params(mix_params)

        return mfa_params

    @override
    def log_likelihood(self, params: Array, data: Array) -> float:
        """Compute average log-likelihood on data."""
        return float(self.mfa.average_log_observable_density(params, data))

    @override
    def generate(self, params: Array, key: Array, n_samples: int) -> Array:
        """Generate samples from the model."""
        return self.mfa.observable_sample(key, params, n_samples)

    @override
    def posterior_soft_assignments(self, params: Array, data: Array) -> Array:
        """Compute posterior responsibilities p(z|x) for all data."""
        return jax.lax.map(
            lambda x: self.mfa.posterior_soft_assignments(params, x),
            data,
            batch_size=2048,
        )

    @override
    def cluster_assignments(self, params: Array, data: Array) -> Array:
        """Assign data to clusters using posterior responsibilities."""
        return jax.lax.map(
            lambda x: self.mfa.posterior_hard_assignment(params, x),
            data,
            batch_size=2048,
        )

    @override
    def compute_cluster_prototypes(self, params: Array) -> list[Array]:
        """Compute model-derived prototypes from component observable distributions."""
        # Convert to mixture of harmoniums representation
        mix_params = self.mfa.to_mixture_params(params)

        # Split into component harmonium params (natural coordinates)
        comp_hrm_params, _ = self.mfa.mix_man.split_natural_mixture(mix_params)

        prototypes = []
        for k in range(self.n_clusters):
            # Get component k's harmonium params (natural coords)
            hrm_k = self.mfa.mix_man.cmp_man.get_replicate(comp_hrm_params, k)
            # Convert to mean coordinates FIRST (marginalizes out latent)
            hrm_k_mean = self.mfa.hrm.to_mean(hrm_k)
            # Then split to get observable mean params
            obs_k_mean, _, _ = self.mfa.hrm.split_coords(hrm_k_mean)
            # Extract mean from Normal mean coordinates (mean, second_moment)
            mean_k = self.mfa.hrm.obs_man.split_mean_second_moment(obs_k_mean)[0]
            prototypes.append(mean_k)

        return prototypes

    def get_cluster_prototypes(self, handler: RunHandler, epoch: int) -> list[Array]:
        """Load cluster prototypes from ClusterStatistics artifact."""
        artifact = handler.load_artifact(epoch, ClusterStatistics)
        return artifact.prototypes

    def get_cluster_members(self, handler: RunHandler, epoch: int) -> list[Array]:
        """Load cluster members from ClusterStatistics artifact."""
        artifact = handler.load_artifact(epoch, ClusterStatistics)
        return artifact.members

    # Training & Analysis

    @override
    def train(
        self,
        key: Array,
        handler: RunHandler,
        logger: Logger,
        dataset: ClusteringDataset,
    ) -> None:
        """Train the MFA model.

        Args:
            key: Random key
            handler: Run handler for saving/loading
            logger: Logger for metrics
            dataset: Dataset to train on
        """
        # Initialize or load parameters
        params = self.prepare_model(key, handler, dataset.train_data)
        epoch = handler.resolve_epoch or 0

        # Train with gradient descent
        log.info(f"Training MFA model for {self.trainer.n_epochs} epochs")
        params = self.trainer.train(
            mfa=self.mfa,
            data=dataset.train_data,
            logger=logger,
            epoch_offset=epoch,
            params0=params,
            key=key,
        )
        epoch += self.trainer.n_epochs

        # Final checkpoint
        log.info(f"Saving final checkpoint at epoch {epoch}")
        self.process_checkpoint(key, handler, logger, dataset, self, epoch, params)

    @override
    def analyze(
        self,
        key: Array,
        handler: RunHandler,
        logger: Logger,
        dataset: ClusteringDataset,
    ) -> None:
        """Run all analyses.

        Args:
            key: Random key
            handler: Run handler for loading artifacts
            logger: Logger for metrics
            dataset: Dataset to analyze
        """
        if handler.resolve_epoch is None:
            raise RuntimeError("No saved parameters found for analysis")
        epoch = handler.resolve_epoch
        log.info(f"Running analyses for epoch {epoch}")
        for analysis in self.get_analyses(dataset):
            analysis.process(key, handler, logger, dataset, self, epoch, None)

    @override
    def get_analyses(
        self, dataset: ClusteringDataset
    ) -> list[Analysis[ClusteringDataset, Any, Any]]:
        """Build analyses list from configuration."""
        analyses: list[Analysis[ClusteringDataset, Any, Any]] = []
        cfg = self.analyses_config

        if cfg.generative_samples.enabled:
            analyses.append(
                GenerativeSamplesAnalysis(n_samples=cfg.generative_samples.n_samples)
            )

        if cfg.cluster_statistics.enabled:
            analyses.append(ClusterStatisticsAnalysis())

        if cfg.co_assignment_hierarchy.enabled:
            analyses.append(CoAssignmentHierarchyAnalysis())

        if cfg.optimal_merge.enabled:
            analyses.append(
                OptimalMergeAnalysis(
                    filter_empty_clusters=cfg.optimal_merge.filter_empty_clusters,
                    min_cluster_size=cfg.optimal_merge.min_cluster_size,
                )
            )

        if cfg.co_assignment_merge.enabled:
            analyses.append(
                CoAssignmentMergeAnalysis(
                    filter_empty_clusters=cfg.co_assignment_merge.filter_empty_clusters,
                    min_cluster_size=cfg.co_assignment_merge.min_cluster_size,
                )
            )

        return analyses + list(dataset.get_dataset_analyses().values())
