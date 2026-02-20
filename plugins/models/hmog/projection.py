"""Projection-based HMoG training.

Trains in two phases:
1. Learn a Factor Analysis / LGM on raw data
2. Project data to latent space via posterior means, then train a diagonal
   Gaussian mixture on the projected data
3. Assemble AnalyticHMoG params via join_conjugated for correct conjugation
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
from goal.models import (
    AnalyticHMoG,
    AnalyticMixture,
    FactorAnalysis,
    FullNormal,
    Normal,
    NormalCovarianceEmbedding,
    analytic_hmog,
)
from jax import Array

from apps.interface import Analysis, ClusteringDataset, ClusteringModel
from apps.interface.analyses import GenerativeSamplesAnalysis
from apps.interface.clustering import cluster_accuracy, clustering_nmi
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
from apps.runtime import Logger, RunHandler, add_clustering_metrics, add_ll_metrics, log_with_frequency
from apps.runtime.util import MetricDict

from .trainers import LGMPreTrainer

# Start logger
log = logging.getLogger(__name__)


def embed_to_full_mix(
    diag_mix: AnalyticMixture[Normal[Diagonal]],
    full_mix: AnalyticMixture[FullNormal],
    mix_params: Array,
) -> Array:
    """Embed diagonal GMM natural params into FullNormal GMM natural params.

    Converts AnalyticMixture[Normal[Diagonal]] natural params to
    AnalyticMixture[FullNormal] natural params. Off-diagonal precision entries
    are zero (diagonal covariance embedded into full covariance space).
    """
    cov_emb = NormalCovarianceEmbedding(diag_mix.obs_man, full_mix.obs_man)
    comp_params, cat_params = diag_mix.split_natural_mixture(mix_params)
    # Apply cov_emb to each component: (n_clusters, diag_dim) → (n_clusters, full_dim)
    comp_params_2d = diag_mix.cmp_man.to_2d(comp_params)
    full_comp_params = jax.vmap(cov_emb.embed)(comp_params_2d).ravel()
    return full_mix.join_natural_mixture(full_comp_params, cat_params)


def to_analytic_params(
    lgm: FactorAnalysis,
    diag_mix: AnalyticMixture[Normal[Diagonal]],
    analytic_manifold: AnalyticHMoG[Diagonal],
    lgm_params: Array,
    mix_params: Array,
) -> Array:
    """Assemble AnalyticHMoG params from LGM params and diagonal GMM params.

    Uses join_conjugated to correctly account for the conjugation correction,
    ensuring the effective prior in the HMoG equals mix_params (not mix_params + rho).
    The obs and int params from the Diagonal LGM are compatible with AnalyticHMoG's
    lkl_fun_man since both share the same obs/int manifold structure.
    """
    obs_params, int_params, _ = lgm.split_coords(lgm_params)
    lkl_params = jnp.concatenate([obs_params, int_params])
    full_mix_params = embed_to_full_mix(diag_mix, analytic_manifold.upr_hrm, mix_params)
    return analytic_manifold.join_conjugated(lkl_params, full_mix_params)


@dataclass(frozen=True)
class ProjectionTrainer:
    """Trainer for projection-based training of mixture models.

    Projects data to the latent space using a pre-trained LGM,
    then trains a standalone diagonal Gaussian mixture on the projected data.
    Conjugation correction is applied at assembly time via join_conjugated.
    """

    # Training hyperparameters
    lr: float
    n_epochs: int
    batch_size: None | int = None
    batch_steps: int = 1000
    grad_clip: float = 1.0

    # Regularization parameters
    l2_reg: float = 0.0

    # Parameter bounds
    min_prob: float = 1e-4
    lat_min_var: float = 1e-6
    lat_jitter_var: float = 0.0

    # Optimizer behaviour
    epoch_reset: bool = True
    """Reset Adam state at the start of each epoch (each new E-step).

    Prevents stale momentum from one E-step's gradient direction carrying over
    into the next E-step when the posterior landscape has shifted.  Particularly
    useful with large batch_steps, where Adam can accumulate significant momentum
    between E-step refreshes.
    """

    def project_data(self, lgm: FactorAnalysis, params: Array, data: Array) -> Array:
        """Project data to the latent space using the LGM posterior means."""

        def posterior_mean(x: Array) -> Array:
            prms = lgm.posterior_at(params, x)
            with lgm.lat_man as lm:
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

    def make_regularizer(self, mix_params: Array) -> Array:
        """Compute L2 regularization gradient."""
        return self.l2_reg * 2.0 * mix_params

    def train_on_projections(
        self,
        key: Array,
        latent_locations: Array,
        mix: AnalyticMixture[Normal[Diagonal]],
        params0: Array,
        n_epochs: int | None = None,
        log_callback: Callable[[Array, Array], None] | None = None,
        epoch_offset: int = 0,
    ) -> Array:
        """Run mixture EM on precomputed latent projections for n_epochs epochs.

        Core training loop — call this when latent_locations are already computed
        to avoid repeating the projection step. Uses standalone mixture EM with
        conjugation correction applied at assembly time via join_conjugated.
        """
        n_epochs_to_run = n_epochs if n_epochs is not None else self.n_epochs

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
            batch_size = latent_locations.shape[0]
            n_batches = 1
        else:
            batch_size = self.batch_size
            n_batches = latent_locations.shape[0] // batch_size

        def batch_step(
            carry: tuple[optax.OptState, Array],
            batch_locs: Array,
        ):
            opt_state, mix_params = carry

            # E-step: compute posterior sufficient statistics using standalone mixture.
            posterior_stats = mix.mean_posterior_statistics(mix_params, batch_locs)
            bounded_posterior_stats = self.bound_mixture_means(mix, posterior_stats)

            def inner_step(
                carry: tuple[optax.OptState, Array],
                _: None,
            ):
                current_opt_state, current_params = carry

                # M-step: natural gradient = prior mean params - posterior mean params
                grad = mix.to_mean(current_params) - bounded_posterior_stats
                reg_grad = self.make_regularizer(current_params)
                grad = grad + reg_grad

                updates, new_opt_state = optimizer.update(
                    grad, current_opt_state, current_params
                )
                new_params: Array = optax.apply_updates(current_params, updates)  # pyright: ignore[reportAssignmentType]

                return (new_opt_state, new_params), None

            (final_opt_state, final_params), _ = jax.lax.scan(
                inner_step,
                (opt_state, mix_params),
                None,
                length=self.batch_steps,
            )

            return (final_opt_state, final_params), None

        def epoch_step(
            epoch: Array,
            carry: tuple[optax.OptState, Array, Array],
        ) -> tuple[optax.OptState, Array, Array]:
            opt_state, mix_params, epoch_key = carry

            # Optionally reset optimizer state so stale momentum from the
            # previous E-step does not bias this epoch's M-steps.
            if self.epoch_reset:
                opt_state = optimizer.init(mix_params)

            shuffle_key, next_key = jax.random.split(epoch_key)

            shuffled_indices = jax.random.permutation(
                shuffle_key, latent_locations.shape[0]
            )
            batched_indices = shuffled_indices[: n_batches * batch_size]
            batched_locs = latent_locations[batched_indices].reshape(
                (n_batches, batch_size, -1)
            )

            (opt_state, new_mix_params), _ = jax.lax.scan(
                batch_step, (opt_state, mix_params), batched_locs
            )

            if log_callback is not None:
                log_callback(new_mix_params, epoch + epoch_offset)

            return opt_state, new_mix_params, next_key

        (_, mix_params_final, _) = jax.lax.fori_loop(
            0, n_epochs_to_run, epoch_step, (opt_state, params0, key)
        )

        return mix_params_final

    def train(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        lgm: FactorAnalysis,
        lgm_params: Array,
        logger: Logger,
        epoch_offset: int,
        mix: AnalyticMixture[Normal[Diagonal]],
        params0: Array,
    ) -> Array:
        """Train diagonal GMM params on projected latent data (LGM fixed).

        Convenience wrapper: precomputes latent projections then calls
        train_on_projections for self.n_epochs epochs.
        """
        log.info("Precomputing latent projections for all training data")
        latent_locations = self.project_data(lgm, lgm_params, dataset.train_data)
        log.info(
            f"Training mixture with conjugation-aware EM for {self.n_epochs} epochs"
        )
        return self.train_on_projections(key, latent_locations, mix, params0)


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
    2. Project data to latent space, train a standalone diagonal GMM
    3. Assemble into AnalyticHMoG via join_conjugated for correct conjugation

    Evaluations use the AnalyticHMoG which correctly accounts for conjugation.
    """

    lgm: FactorAnalysis
    mixture: AnalyticMixture[Normal[Diagonal]]
    analytic_manifold: AnalyticHMoG[Diagonal]
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

        self.lgm = FactorAnalysis(obs_dim=data_dim, lat_dim=latent_dim)
        self.mixture = AnalyticMixture(
            obs_man=Normal(latent_dim, Diagonal()),
            n_categories=n_clusters,
        )

        self.analytic_manifold = analytic_hmog(
            obs_dim=data_dim,
            obs_rep=Diagonal(),
            lat_dim=latent_dim,
            n_components=n_clusters,
        )

        self.pre = pre
        self.projection = pro
        self.lgm_noise_scale = lgm_noise_scale
        self.mix_noise_scale = mix_noise_scale
        self.analyses_config = analyses

        log.info(
            f"Initialized Projection DiagonalHMoG model with dimension {self.analytic_manifold.dim}."
        )

    # Properties

    @property
    @override
    def n_epochs(self) -> int:
        return self.pre.n_epochs + self.projection.n_epochs

    @property
    def latent_dim(self) -> int:
        return self.lgm.lat_dim

    @property
    @override
    def n_clusters(self) -> int:
        return self.mixture.n_categories

    @property
    @override
    def n_parameters(self) -> int:
        return self.lgm.dim - self.lgm.lat_man.dim + self.mixture.dim

    # Initialization

    @override
    def initialize_model(self, key: Array, data: Array) -> Array:
        key_lgm, key_mix = jax.random.split(key)
        lgm_params = self._initialize_lgm(key_lgm, data)
        mix_params = self._initialize_mixture(key_mix)
        return to_analytic_params(
            self.lgm,
            self.mixture,
            self.analytic_manifold,
            lgm_params,
            mix_params,
        )

    def _initialize_lgm(self, key: Array, data: Array) -> Array:
        lgm = self.lgm
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
        return self.mixture.initialize(key, shape=self.mix_noise_scale)

    # Protocol methods

    @override
    def log_likelihood(self, params: Array, data: Array) -> float:
        return float(
            self.analytic_manifold.average_log_observable_density(params, data)
        )

    @override
    def posterior_soft_assignments(self, params: Array, data: Array) -> Array:
        return jax.lax.map(
            lambda x: self.analytic_manifold.posterior_assignments(params, x),
            data,
            batch_size=2048,
        )

    @override
    def compute_cluster_prototypes(self, params: Array) -> list[Array]:
        # Recover actual prior mixture params (stored as prior - rho; add rho back)
        lkl_params, lat_stored = self.analytic_manifold.split_conjugated(params)
        rho = self.analytic_manifold.conjugation_parameters(lkl_params)
        full_mix_params = lat_stored + rho  # actual AnalyticMixture[FullNormal] params

        comp_lats, _ = self.analytic_manifold.upr_hrm.split_natural_mixture(
            full_mix_params
        )

        ana_lgm = self.analytic_manifold.lwr_hrm
        prototypes: list[Array] = []

        for i in range(self.analytic_manifold.upr_hrm.n_categories):
            comp_lat_params = self.analytic_manifold.upr_hrm.cmp_man.get_replicate(
                comp_lats, i
            )
            lwr_hrm_params = ana_lgm.join_conjugated(lkl_params, comp_lat_params)
            lwr_hrm_means = ana_lgm.to_mean(lwr_hrm_params)
            lwr_hrm_obs = ana_lgm.split_coords(lwr_hrm_means)[0]
            obs_means = ana_lgm.obs_man.split_mean_second_moment(lwr_hrm_obs)[0]
            prototypes.append(obs_means)

        return prototypes

    @override
    def cluster_assignments(self, params: Array, data: Array) -> Array:
        return jax.lax.map(
            lambda x: jnp.argmax(
                self.analytic_manifold.posterior_assignments(params, x)
            ),
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
        return self.analytic_manifold.observable_sample(key, params, n_samples)

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

        lgm = self.lgm
        mixture = self.mixture

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
            # Save intermediate state as AnalyticHMoG params
            mix_params = self._initialize_mixture(key_proj)
            full_params = to_analytic_params(
                lgm, mixture, self.analytic_manifold, lgm_params, mix_params
            )
            handler.save_params(full_params, epoch)

        # Phase 2: Train the mixture model on projected data
        if self.projection.n_epochs > 0:
            log.info("Phase 2: Training mixture model on projected data")

            key_proj, key_mix_init = jax.random.split(key_proj)
            mix_params = self._initialize_mixture(key_mix_init)

            # Precompute latent projections once
            log.info("Precomputing latent projections for all training data")
            latent_locations = self.projection.project_data(
                lgm, lgm_params, dataset.train_data
            )

            log.info(
                f"Training mixture with conjugation-aware EM for {self.projection.n_epochs} epochs"
            )

            def log_proj(mix_ps: Array, ep: Array) -> None:
                def compute_metrics() -> MetricDict:
                    hmog_params = to_analytic_params(
                        lgm, mixture, self.analytic_manifold, lgm_params, mix_ps
                    )
                    train_ll = self.analytic_manifold.average_log_observable_density(
                        hmog_params, dataset.train_data
                    )
                    test_ll = self.analytic_manifold.average_log_observable_density(
                        hmog_params, dataset.test_data
                    )
                    metrics: MetricDict = {}
                    add_ll_metrics(
                        metrics,
                        self.n_parameters,
                        train_ll,
                        test_ll,
                        dataset.train_data.shape[0],
                    )
                    if dataset.has_labels:
                        train_clusters = self.cluster_assignments(
                            hmog_params, dataset.train_data
                        )
                        test_clusters = self.cluster_assignments(
                            hmog_params, dataset.test_data
                        )
                        add_clustering_metrics(
                            metrics,
                            n_clusters=self.n_clusters,
                            n_classes=dataset.n_classes,
                            train_labels=dataset.train_labels,
                            test_labels=dataset.test_labels,
                            train_clusters=train_clusters,
                            test_clusters=test_clusters,
                            cluster_accuracy_fn=cluster_accuracy,
                            clustering_nmi_fn=clustering_nmi,
                        )
                    return metrics

                log_with_frequency(logger, ep, 10, compute_metrics)

            mix_params = self.projection.train_on_projections(
                key_proj,
                latent_locations,
                mixture,
                mix_params,
                log_callback=log_proj,
                epoch_offset=epoch,
            )

            final_params = to_analytic_params(
                lgm, mixture, self.analytic_manifold, lgm_params, mix_params
            )
            final_epoch = epoch + self.projection.n_epochs

            self.process_checkpoint(
                key, handler, logger, dataset, self, final_epoch, final_params
            )

        log.info("Training complete")
