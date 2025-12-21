"""MFA (Mixture of Factor Analyzers) model implementation."""

from __future__ import annotations

import logging
from typing import Any, override

import jax.numpy as jnp
from goal.models import FactorAnalysis
from goal.models.graphical.mixture import MixtureOfConjugated
from jax import Array

from apps.interface import Analysis, ClusteringDataset, ClusteringModel
from apps.interface.clustering.protocols import (
    HasClusterHierarchy,
    HasClusterPrototypes,
    HasSoftAssignments,
)
from apps.interface.protocols import HasLogLikelihood, IsGenerative
from apps.runtime import Logger, RunHandler

from .analysis.clusters import ClusterStatistics, ClusterStatisticsAnalysis
from .analysis.generative import GenerativeSamplesAnalysis
from .analysis.latent import LatentProjectionsAnalysis
from .base import MFA
from .trainers import GradientTrainer

log = logging.getLogger(__name__)


class MFAModel(
    ClusteringModel,
    HasLogLikelihood,
    IsGenerative,
    HasSoftAssignments,
    HasClusterPrototypes,
    HasClusterHierarchy,
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
    ):
        """Initialize MFA model.

        Args:
            data_dim: Dimension of observable data
            latent_dim: Dimension of latent factors
            n_clusters: Number of mixture components
            trainer: Trainer instance for optimization
        """
        self.data_dim: int = data_dim
        self.latent_dim: int = latent_dim
        self.n_clusters_val: int = n_clusters
        self.trainer: GradientTrainer = trainer

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
        """Initialize MFA parameters from data sample.

        Args:
            key: Random key for initialization
            data: Training data for initialization

        Returns:
            Initial model parameters
        """
        return self.mfa.initialize_from_sample(key, data)

    def log_likelihood(self, params: Array, data: Array) -> float:
        """Compute average log-likelihood on data."""
        return float(self.mfa.average_log_observable_density(params, data))

    def generate(self, params: Array, key: Array, n_samples: int) -> Array:
        """Generate samples from the model."""
        return self.mfa.observable_sample(key, params, n_samples)

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

    def encode(self, params: Array, data: Array) -> Array:
        """Encode data into latent space representation (MFA-specific)."""
        import jax

        def get_latent_mean(x: Array) -> Array:
            posterior_params = self.mfa.posterior_at(params, x)
            lat_obs, _, _ = self.mfa.lat_man.split_coords(posterior_params)
            return self.mfa.lat_man.obs_man.to_mean(lat_obs)

        return jax.vmap(get_latent_mean)(data)

    def get_cluster_prototypes(self, handler: RunHandler, epoch: int) -> list[Array]:
        """Load cluster prototypes from ClusterStatistics artifact."""
        artifact = handler.load_artifact(epoch, ClusterStatistics)
        return artifact.prototypes

    def get_cluster_members(self, handler: RunHandler, epoch: int) -> list[Array]:
        """Load cluster members from ClusterStatistics artifact."""
        artifact = handler.load_artifact(epoch, ClusterStatistics)
        return artifact.members

    def get_cluster_hierarchy(self, handler: RunHandler, epoch: int) -> Array:
        """Get cluster hierarchy (computed from cluster means)."""
        from scipy.cluster.hierarchy import linkage

        prototypes = self.get_cluster_prototypes(handler, epoch)
        prototype_matrix = jnp.stack(prototypes)
        return jnp.array(linkage(prototype_matrix, method="ward"))

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
            key,
            handler,
            dataset,
            self,
            logger,
            epoch_offset=epoch,
            params0=params,
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
        epoch: int = handler.resolve_epoch  # type: ignore[assignment]
        log.info(f"Running analyses for epoch {epoch}")
        for analysis in self.get_analyses(dataset):
            analysis.process(key, handler, logger, dataset, self, epoch, None)

    @override
    def get_analyses(self, dataset: ClusteringDataset) -> list[Analysis[ClusteringDataset, Any, Any]]:
        """Return list of analyses to run."""
        return [
            ClusterStatisticsAnalysis(),
            GenerativeSamplesAnalysis(n_samples=100),
            LatentProjectionsAnalysis(),
        ]
