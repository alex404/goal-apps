"""Base class for HMoG implementations."""

from __future__ import annotations

import logging
from abc import ABC
from typing import Any, override

import jax
import jax.numpy as jnp
from goal.geometry import Diagonal, PositiveDefinite
from goal.models import differentiable_hmog
from jax import Array

from apps.interface import Analysis, ClusteringDataset, ClusteringModel
from apps.interface.analyses import GenerativeSamplesAnalysis
from apps.interface.clustering import cycle_lr_schedule, run_cyclic_training
from apps.interface.clustering.analyses import (
    ClusterStatistics,
    ClusterStatisticsAnalysis,
    CoAssignmentHierarchy,
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
from apps.runtime import L1_L2_METRIC_KEYS, Logger, RunHandler

from .analyses.base import get_component_prototypes
from .analyses.hierarchy import KLHierarchyAnalysis
from .analyses.loadings import LoadingMatrixAnalysis
from .analyses.merge import KLMergeAnalysis
from .trainers import (
    FullGradientTrainer,
    LGMPreTrainer,
    MixtureGradientTrainer,
)
from .types import AnyHMoG

### Preamble ###

# Start logger
log = logging.getLogger(__name__)


### HMog Experiment ###


class HMoGModel(
    ClusteringModel,
    HasLogLikelihood,
    IsGenerative,
    HasSoftAssignments,
    CanComputePrototypes,
    ABC,
):
    """Model framework for HMoGs."""

    # Training configuration
    manifold: AnyHMoG
    pre: LGMPreTrainer
    lgm: FullGradientTrainer
    mix: MixtureGradientTrainer
    full: FullGradientTrainer

    lr_schedule: list[float]
    num_cycles: int

    lgm_noise_scale: float
    mix_init_scale: float

    # Analysis configuration
    analyses_config: ClusteringAnalysesConfig

    def __init__(
        self,
        data_dim: int,
        latent_dim: int,
        n_clusters: int,
        pre: LGMPreTrainer,
        lgm: FullGradientTrainer,
        mix: MixtureGradientTrainer,
        full: FullGradientTrainer,
        lr_scales: list[float],
        num_cycles: int,
        lgm_noise_scale: float,
        mix_init_scale: float,
        analyses: ClusteringAnalysesConfig,
        diagonal_latent: bool = True,
    ) -> None:
        super().__init__()

        self.manifold = differentiable_hmog(
            obs_dim=data_dim,
            obs_rep=Diagonal(),
            lat_dim=latent_dim,
            pst_rep=Diagonal() if diagonal_latent else PositiveDefinite(),
            n_components=n_clusters,
        )

        self.pre = pre
        self.lgm = lgm
        self.mix = mix
        self.full = full

        self.num_cycles = num_cycles
        self.lr_schedule = cycle_lr_schedule(lr_scales, num_cycles)

        self.lgm_noise_scale = lgm_noise_scale
        self.mix_init_scale = mix_init_scale

        self.analyses_config = analyses

    # Properties

    @property
    @override
    def n_epochs(self) -> int:
        """Calculate total number of epochs across all cycles."""
        return self.num_cycles * (
            self.lgm.n_epochs + self.mix.n_epochs + self.full.n_epochs
        )

    @property
    def latent_dim(self) -> int:
        return self.manifold.prr_man.obs_man.data_dim

    @property
    @override
    def n_clusters(self) -> int:
        return self.manifold.prr_man.n_categories

    @property
    @override
    def n_parameters(self) -> int:
        return self.manifold.dim

    # Methods

    @override
    def initialize_model(self, key: Array, data: Array) -> Array:
        """Initialize model parameters."""

        key_comp, key_int = jax.random.split(key, 2)

        obs_means = self.manifold.obs_man.average_sufficient_statistic(data)
        obs_means = self.manifold.obs_man.regularize_covariance(
            obs_means, self.lgm.obs_jitter_var, self.lgm.obs_min_var
        )
        obs_params = self.manifold.obs_man.to_natural(obs_means)

        mix_params = self.manifold.pst_man.initialize(
            key_comp, shape=self.mix_init_scale
        )

        int_noise = self.lgm_noise_scale * jax.random.normal(
            key_int, self.manifold.int_man.matrix_shape
        )
        int_params = self.manifold.int_man.rep.from_matrix(int_noise)

        return self.manifold.join_coords(obs_params, int_params, mix_params)

    @override
    def log_likelihood(self, params: Array, data: Array) -> float:
        return float(self.manifold.average_log_observable_density(params, data))

    @override
    def posterior_soft_assignments(self, params: Array, data: Array) -> Array:
        """Compute posterior responsibilities p(z|x) for all data."""
        return jax.lax.map(
            lambda x: self.manifold.posterior_soft_assignments(params, x),
            data,
            batch_size=2048,
        )

    @override
    def compute_cluster_prototypes(self, params: Array) -> list[Array]:
        """Compute model-derived prototypes for each cluster."""
        return get_component_prototypes(self.manifold, params)

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
            analyses.append(
                CoAssignmentHierarchyAnalysis(
                    filter_empty_clusters=cfg.co_assignment_hierarchy.filter_empty_clusters,
                    min_cluster_size=cfg.co_assignment_hierarchy.min_cluster_size,
                )
            )

        analyses.append(LoadingMatrixAnalysis())

        if cfg.kl_hierarchy.enabled:
            analyses.append(
                KLHierarchyAnalysis(
                    filter_empty_clusters=cfg.kl_hierarchy.filter_empty_clusters,
                    min_cluster_size=cfg.kl_hierarchy.min_cluster_size,
                )
            )

        # Merge analyses require ground truth labels
        if dataset.has_labels:
            if cfg.optimal_merge.enabled:
                analyses.append(
                    OptimalMergeAnalysis(
                        filter_empty_clusters=cfg.optimal_merge.filter_empty_clusters,
                        min_cluster_size=cfg.optimal_merge.min_cluster_size,
                    )
                )

            if cfg.co_assignment_merge.enabled:
                analyses.append(CoAssignmentMergeAnalysis())

            if cfg.kl_merge.enabled:
                analyses.append(KLMergeAnalysis())

        return analyses + list(dataset.get_dataset_analyses().values())

    @property
    @override
    def training_metric_keys(self) -> frozenset[str]:
        from .metrics import HMOG_PRE_TRAINING_METRIC_KEYS, HMOG_TRAINING_METRIC_KEYS
        from .trainers import ENTROPY_REG_METRIC_KEYS, PRECISION_REG_METRIC_KEYS

        return (
            HMOG_PRE_TRAINING_METRIC_KEYS
            | HMOG_TRAINING_METRIC_KEYS
            | L1_L2_METRIC_KEYS
            | PRECISION_REG_METRIC_KEYS
            | ENTROPY_REG_METRIC_KEYS
        )

    @override
    def generate(self, params: Array, key: Array, n_samples: int) -> Array:
        return self.manifold.observable_sample(key, params, n_samples)

    @override
    def cluster_assignments(self, params: Array, data: Array) -> Array:
        """Assign data points to clusters using the HMoG model."""
        return jax.lax.map(
            lambda x: self.manifold.posterior_hard_assignment(params, x),
            data,
            batch_size=2048,
        )

    @override
    def analyze(
        self,
        key: Array,
        handler: RunHandler,
        logger: Logger,
        dataset: ClusteringDataset,
    ) -> None:
        """Generate analysis artifacts from saved checkpoint results."""

        if handler.resolve_epoch is None:
            raise RuntimeError("No saved parameters found for analysis")
        epoch = handler.resolve_epoch

        if handler.recompute_artifacts:
            log.info("Recomputing artifacts from scratch.")
            # This shouldn't be necessary because key_model shouldn't be used, but just in case...
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
        """Train the HMoG model.

        Phase layout:
          1. Optional LGM pretraining (``pre.n_epochs`` epochs) — a single
             pass that seeds the latent space before cycling begins.
          2. ``num_cycles`` training cycles, each running three gradient
             sub-phases in order:
               - ``lgm`` — optimize LGM params with mixture frozen
               - ``mix`` — optimize mixture params with LGM fixed
               - ``full`` — end-to-end gradient descent
             Each cycle applies a scalar from ``lr_schedule`` (built from
             ``lr_scales``) to the base learning rates. A checkpoint is
             written at the end of every cycle so Optuna can prune between
             them.

        Resume semantics: on re-entry, the current epoch is mapped back to
        either a point inside pretraining (continue pretraining from there)
        or to a cycle index (resume cycling at the start of that cycle).
        The cycle orchestration is shared with ``MFAModel.train`` via
        ``run_cyclic_training`` in ``apps.interface.clustering.model``.
        """
        # Split PRNG key for different training phases
        init_key, pre_key, mix_reinit_key, cycle_key = jax.random.split(key, 4)

        params = self.prepare_model(init_key, handler, dataset.train_data)

        # Track total epochs
        epoch = handler.resolve_epoch or 0

        # Calculate training structure
        epochs_per_cycle = self.lgm.n_epochs + self.mix.n_epochs + self.full.n_epochs
        training_start_epoch = self.pre.n_epochs

        if epoch == 0:
            log.info("Starting training from scratch.")
        elif epoch < training_start_epoch:
            log.info("Continuing pretraining phase.")
        else:
            current_cycle = (epoch - training_start_epoch) // epochs_per_cycle
            log.info(
                f"Resuming from epoch {epoch}, cycle {current_cycle}/{self.num_cycles}"
            )

        if self.pre.n_epochs > epoch:
            obs_params, int_params, lat_params = self.manifold.split_coords(params)
            lat_obs_params, _, _ = self.manifold.pst_man.split_coords(lat_params)
            lgm = self.manifold.lwr_hrm
            lgm_params = lgm.join_coords(obs_params, int_params, lat_obs_params)

            log.info("Pretraining LGM parameters")
            log.info(f"Learning rate: {self.pre.lr:.2e}")
            lgm_params = self.pre.train(
                pre_key,
                handler,
                dataset,
                lgm,
                logger,
                epoch,
                lgm_params,
            )
            obs_params, int_params, lat_obs_params = lgm.split_coords(lgm_params)

            # Re-initialize the mixture-specific params (lat_int and cat) now that
            # the latent space has settled. The params from before pre-training are
            # meaningless in the trained latent space and cause analysis crashes.
            fresh_mix = self.manifold.pst_man.initialize(
                mix_reinit_key, shape=self.mix_init_scale
            )
            _, fresh_lat_int, fresh_cat = self.manifold.pst_man.split_coords(fresh_mix)
            lat_params = self.manifold.pst_man.join_coords(
                lat_obs_params, fresh_lat_int, fresh_cat
            )
            params = self.manifold.join_coords(obs_params, int_params, lat_params)
            epoch = self.pre.n_epochs
            self.process_checkpoint(key, handler, logger, dataset, self, epoch, params)
            log.info("Pretraining complete.")

        def run_cycle(
            cycle_k: Array,
            cycle_params: Array,
            cycle_idx: int,
            lr_scale: float,
            epoch_offset: int,
        ) -> tuple[Array, int]:
            key_lgm, key_mix, key_full = jax.random.split(cycle_k, 3)
            cur_epoch = epoch_offset
            if self.lgm.n_epochs > 0:
                log.info(
                    f"Cycle {cycle_idx + 1}/{self.num_cycles}: Training LGM parameters"
                )
                cycle_params = self.lgm.train(
                    key_lgm,
                    handler,
                    dataset,
                    self.manifold,
                    logger,
                    lr_scale,
                    cur_epoch,
                    cycle_params,
                )
                cur_epoch += self.lgm.n_epochs
            if self.mix.n_epochs > 0:
                log.info(
                    f"Cycle {cycle_idx + 1}/{self.num_cycles}: Training MoG parameters"
                )
                cycle_params = self.mix.train(
                    key_mix,
                    handler,
                    dataset,
                    self.manifold,
                    logger,
                    lr_scale,
                    cur_epoch,
                    cycle_params,
                )
                cur_epoch += self.mix.n_epochs
            if self.full.n_epochs > 0:
                log.info(
                    f"Cycle {cycle_idx + 1}/{self.num_cycles}: Training full model"
                )
                cycle_params = self.full.train(
                    key_full,
                    handler,
                    dataset,
                    self.manifold,
                    logger,
                    lr_scale,
                    cur_epoch,
                    cycle_params,
                )
                cur_epoch += self.full.n_epochs
            return cycle_params, cur_epoch

        run_cyclic_training(
            self,
            cycle_key,
            handler,
            logger,
            dataset,
            params,
            num_cycles=self.num_cycles,
            lr_scales=self.lr_schedule,
            epochs_per_cycle=epochs_per_cycle,
            cycle_start_epoch=training_start_epoch,
            current_epoch=epoch,
            run_cycle=run_cycle,
        )

    def get_cluster_prototypes(self, handler: RunHandler, epoch: int) -> list[Array]:
        """Get cluster prototypes by loading from ClusterStatistics artifact."""
        stats = handler.load_artifact(epoch, ClusterStatistics)
        return stats.prototypes

    def get_cluster_members(self, handler: RunHandler, epoch: int) -> list[Array]:
        """Get cluster members by loading from ClusterStatistics artifact."""
        stats = handler.load_artifact(epoch, ClusterStatistics)
        return stats.members

    def get_cluster_hierarchy(self, handler: RunHandler, epoch: int) -> Array:
        """Get co-assignment based hierarchy by loading from artifact."""
        hierarchy = handler.load_artifact(epoch, CoAssignmentHierarchy)
        return jnp.array(hierarchy.linkage_matrix)
