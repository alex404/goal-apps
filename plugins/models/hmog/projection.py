"""Projection-based HMoG training.

Trains in two phases:
1. Learn a Factor Analysis / LGM on raw data
2. Project data to latent space via posterior means, then train a diagonal
   Gaussian mixture on the projected data
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, override

import jax
import jax.numpy as jnp
import optax
from goal.geometry import Diagonal
from goal.models import AnalyticMixture, Normal, differentiable_hmog
from jax import Array

from apps.interface import Analysis, ClusteringDataset, ClusteringModel
from apps.interface.analyses import GenerativeSamplesAnalysis
from apps.interface.clustering.analyses import (
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

from .metrics import log_epoch_metrics
from .trainers import LGMPreTrainer
from .types import DiagonalHMoG, DiagonalLGM

# Start logger
log = logging.getLogger(__name__)


def to_differentiable(
    lgm: DiagonalLGM,
    hmog: DiagonalHMoG,
    lgm_params: Array,
    mix_params: Array,
) -> Array:
    """Combine LGM params and trained mixture params into full DiagonalHMoG params.

    The LGM provides observable (obs) and interaction (int) parameters.
    The mixture provides the posterior (pst) parameters for the latent space.
    """
    obs_params, int_params, _ = lgm.split_coords(lgm_params)
    return hmog.join_coords(obs_params, int_params, mix_params)


@dataclass(frozen=True)
class ProjectionTrainer:
    """Trainer for projection-based training of mixture models.

    Projects data to the latent space using a pre-trained LGM,
    then trains a mixture model on the projected data.
    """

    # Training hyperparameters
    lr: float
    n_epochs: int
    batch_size: None | int = None
    batch_steps: int = 1000
    grad_clip: float = 1.0

    # Regularization parameters
    l1_reg: float = 0.0
    l2_reg: float = 0.0

    # Parameter bounds
    min_prob: float = 1e-4
    lat_min_var: float = 1e-6
    lat_jitter_var: float = 0.0

    def project_data(self, lgm: DiagonalLGM, params: Array, data: Array) -> Array:
        """Project data to the latent space using the LGM posterior means."""

        def posterior_mean(x: Array) -> Array:
            prms = lgm.posterior_at(params, x)
            with lgm.pst_man as lm:
                return lm.split_mean_covariance(lm.to_mean(prms))[0]

        return jax.lax.map(posterior_mean, data, batch_size=256)

    def bound_mixture_means(
        self, model: AnalyticMixture[Normal[Diagonal]], means: Array
    ) -> Array:
        """Apply bounds to posterior statistics for numerical stability."""
        comp_means, prob_means = model.split_mean_mixture(means)

        bounded_comp_means = model.cmp_man.map(
            lambda m: model.obs_man.regularize_covariance(
                m, self.lat_jitter_var, self.lat_min_var
            ),
            comp_means,
            flatten=True,
        )

        probs = model.lat_man.to_probs(prob_means)
        bounded_probs = jnp.clip(probs, self.min_prob, 1.0)
        bounded_probs = bounded_probs / jnp.sum(bounded_probs)
        bounded_prob_means = model.lat_man.from_probs(bounded_probs)

        return model.join_mean_mixture(bounded_comp_means, bounded_prob_means)

    def make_regularizer(self) -> Callable[[Array], Array]:
        """Create a regularizer for the mixture model."""

        def loss_fn(params: Array) -> Array:
            return self.l2_reg * jnp.sum(jnp.square(params))

        return loss_fn

    def train(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        lgm: DiagonalLGM,
        lgm_params: Array,
        logger: Logger,
        epoch_offset: int,
        mix: AnalyticMixture[Normal[Diagonal]],
        hmog: DiagonalHMoG,
        params0: Array,
    ) -> Array:
        """Train a mixture model on data projected to the latent space."""
        train_data = dataset.train_data
        projected_data = self.project_data(lgm, lgm_params, train_data)

        # Setup optimizer
        if self.grad_clip > 0.0:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.grad_clip),
                optax.adam(learning_rate=self.lr),
            )
        else:
            optimizer = optax.adam(learning_rate=self.lr)

        opt_state = optimizer.init(params0)

        # Determine batch size and number of batches
        if self.batch_size is None:
            batch_size = projected_data.shape[0]
            n_batches = 1
        else:
            batch_size = self.batch_size
            n_batches = projected_data.shape[0] // batch_size

        log.info(f"Training mixture on projected data for {self.n_epochs} epochs")

        regularizer = self.make_regularizer()

        def batch_step(
            carry: tuple[optax.OptState, Array],
            batch: Array,
        ) -> tuple[tuple[optax.OptState, Array], Array]:
            opt_state, params = carry

            posterior_stats = mix.mean_posterior_statistics(params, batch)
            bounded_posterior_stats = self.bound_mixture_means(mix, posterior_stats)

            def inner_step(
                carry: tuple[optax.OptState, Array],
                _: None,
            ) -> tuple[tuple[optax.OptState, Array], Array]:
                current_opt_state, current_params = carry

                prior_stats = mix.to_mean(current_params)
                grad = prior_stats - bounded_posterior_stats

                reg_grad = jax.grad(regularizer)(current_params)
                grad = grad + reg_grad

                updates, new_opt_state = optimizer.update(
                    grad, current_opt_state, current_params
                )
                new_params = optax.apply_updates(current_params, updates)

                logger.monitor_params(
                    {
                        "original_params": current_params,
                        "updated_params": new_params,
                        "batch": batch,
                        "posterior_stats": posterior_stats,
                        "bounded_posterior_stats": bounded_posterior_stats,
                        "prior_stats": prior_stats,
                        "grad": grad,
                    },
                    handler,
                    context="PROJECTION",
                )

                return (new_opt_state, new_params), grad

            (final_opt_state, final_params), all_grads = jax.lax.scan(
                inner_step,
                (opt_state, params),
                None,
                length=self.batch_steps,
            )

            return (final_opt_state, final_params), all_grads

        def epoch_step(
            epoch: Array,
            carry: tuple[optax.OptState, Array, Array],
        ) -> tuple[optax.OptState, Array, Array]:
            opt_state, params, epoch_key = carry

            shuffle_key, next_key = jax.random.split(epoch_key)

            shuffled_indices = jax.random.permutation(
                shuffle_key, projected_data.shape[0]
            )
            batched_indices = shuffled_indices[: n_batches * batch_size]
            batched_data = projected_data[batched_indices].reshape(
                (n_batches, batch_size, -1)
            )

            (opt_state, new_params), _ = jax.lax.scan(
                batch_step, (opt_state, params), batched_data
            )

            hmog_params = to_differentiable(lgm, hmog, lgm_params, new_params)

            log_epoch_metrics(
                dataset,
                hmog,
                logger,
                hmog_params,
                epoch + epoch_offset,
                {},
                None,
                log_freq=10,
            )

            return opt_state, new_params, next_key

        (_, params_final, _) = jax.lax.fori_loop(
            0, self.n_epochs, epoch_step, (opt_state, params0, key)
        )

        return params_final


class ProjectionHMoGModel(
    ClusteringModel,
    HasLogLikelihood,
    IsGenerative,
    HasSoftAssignments,
    CanComputePrototypes,
):
    """Model framework for projection-based DiagonalHMoG training.

    Trains in two phases:
    1. Use PreTrainer to learn a Linear Gaussian Model (LGM)
    2. Project data to latent space using the trained LGM
    3. Train a mixture model on the projected data
    """

    manifold: DiagonalHMoG
    pre: LGMPreTrainer
    projection: ProjectionTrainer
    lgm_noise_scale: float
    mix_noise_scale: float
    analyses_config: ClusteringAnalysesConfig

    def __init__(
        self,
        data_dim: int,
        latent_dim: int,
        n_clusters: int,
        pre: LGMPreTrainer,
        pro: ProjectionTrainer,
        lgm_noise_scale: float,
        mix_noise_scale: float,
        analyses: ClusteringAnalysesConfig,
    ) -> None:
        super().__init__()

        self.manifold = differentiable_hmog(
            obs_dim=data_dim,
            obs_rep=Diagonal(),
            lat_dim=latent_dim,
            pst_rep=Diagonal(),
            n_components=n_clusters,
        )

        self.pre = pre
        self.projection = pro
        self.lgm_noise_scale = lgm_noise_scale
        self.mix_noise_scale = mix_noise_scale
        self.analyses_config = analyses

        log.info(
            f"Initialized Projection DiagonalHMoG model with dimension {self.manifold.dim}."
        )

    # Properties

    @property
    @override
    def n_epochs(self) -> int:
        return self.pre.n_epochs + self.projection.n_epochs

    @property
    def latent_dim(self) -> int:
        return self.manifold.prr_man.obs_man.data_dim

    @property
    @override
    def n_clusters(self) -> int:
        return self.manifold.prr_man.lat_man.dim + 1

    @property
    @override
    def n_parameters(self) -> int:
        return self.manifold.dim

    # Initialization

    @override
    def initialize_model(self, key: Array, data: Array) -> Array:
        key_lgm, key_mix = jax.random.split(key)
        lgm_params = self._initialize_lgm(key_lgm, data)
        mix_params = self._initialize_mixture(key_mix)
        obs_params, int_params, _ = self.manifold.lwr_hrm.split_coords(lgm_params)
        return self.manifold.join_coords(obs_params, int_params, mix_params)

    def _initialize_lgm(self, key: Array, data: Array) -> Array:
        lgm = self.manifold.lwr_hrm
        obs_means = lgm.obs_man.average_sufficient_statistic(data)
        obs_means = lgm.obs_man.regularize_covariance(
            obs_means, self.pre.jitter_var, self.pre.min_var
        )
        obs_params = lgm.obs_man.to_natural(obs_means)

        int_noise = self.lgm_noise_scale * jax.random.normal(
            key, lgm.int_man.matrix_shape
        )
        int_params = lgm.int_man.rep.from_matrix(int_noise)

        lat_params = lgm.pst_man.to_natural(lgm.pst_man.standard_normal())

        return lgm.join_coords(obs_params, int_params, lat_params)

    def _initialize_mixture(self, key: Array) -> Array:
        return self.manifold.pst_man.initialize(key, shape=self.mix_noise_scale)

    # Protocol methods

    @override
    def log_likelihood(self, params: Array, data: Array) -> float:
        return float(self.manifold.average_log_observable_density(params, data))

    @override
    def posterior_soft_assignments(self, params: Array, data: Array) -> Array:
        return jax.lax.map(
            lambda x: self.manifold.posterior_soft_assignments(params, x),
            data,
            batch_size=2048,
        )

    @override
    def compute_cluster_prototypes(self, params: Array) -> list[Array]:
        from .analyses.base import get_component_prototypes

        return get_component_prototypes(self.manifold, params)

    @override
    def cluster_assignments(self, params: Array, data: Array) -> Array:
        return jax.lax.map(
            lambda x: self.manifold.posterior_hard_assignment(params, x),
            data,
            batch_size=2048,
        )

    @override
    def get_analyses(
        self, dataset: ClusteringDataset
    ) -> list[Analysis[ClusteringDataset, Any, Any]]:
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

        if dataset.has_labels:
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

    @override
    def generate(self, params: Array, key: Array, n_samples: int) -> Array:
        return self.manifold.observable_sample(key, params, n_samples)

    @override
    def analyze(
        self,
        key: Array,
        handler: RunHandler,
        logger: Logger,
        dataset: ClusteringDataset,
    ) -> None:
        if handler.resolve_epoch is None:
            raise RuntimeError("No saved parameters found for analysis")
        epoch = handler.resolve_epoch

        if handler.recompute_artifacts:
            log.info("Recomputing artifacts from scratch.")
            key_check, key_model = jax.random.split(key, 2)
            params_array = self.prepare_model(key_model, handler, dataset.train_data)
            self.process_checkpoint(
                key_check, handler, logger, dataset, self, epoch, params_array
            )
        else:
            log.info("Loading existing artifacts.")
            self.process_checkpoint(key, handler, logger, dataset, self, epoch)

    @override
    def train(
        self,
        key: Array,
        handler: RunHandler,
        logger: Logger,
        dataset: ClusteringDataset,
    ) -> None:
        key_init, key_pre, key_proj = jax.random.split(key, 3)

        lgm = self.manifold.lwr_hrm
        mixture = self.manifold.pst_man

        # Initialize LGM parameters
        lgm_params = self._initialize_lgm(key_init, dataset.train_data)

        epoch = 0

        # Phase 1: Train the LGM
        if self.pre.n_epochs > 0:
            log.info("Phase 1: Training LGM for projection")
            lgm_params = self.pre.train(
                key_pre,
                handler,
                dataset,
                lgm,
                logger,
                epoch,
                lgm_params,
            )
            epoch += self.pre.n_epochs
            # Save intermediate LGM as full model params
            mix_params = self._initialize_mixture(key_proj)
            obs_params, int_params, _ = lgm.split_coords(lgm_params)
            full_params = self.manifold.join_coords(obs_params, int_params, mix_params)
            handler.save_params(full_params, epoch)

        # Phase 2: Train the mixture model on projected data
        if self.projection.n_epochs > 0:
            log.info("Phase 2: Training mixture model on projected data")

            mix_params = self._initialize_mixture(key_proj)

            trained_mix_params = self.projection.train(
                key_proj,
                handler,
                dataset,
                lgm,
                lgm_params,
                logger,
                epoch,
                mixture,
                self.manifold,
                mix_params,
            )

            final_params = to_differentiable(
                lgm, self.manifold, lgm_params, trained_mix_params
            )

            final_epoch = epoch + self.projection.n_epochs

            self.process_checkpoint(
                key, handler, logger, dataset, self, final_epoch, final_params
            )

        log.info("Training complete")
