"""Base class for HMoG implementations."""

from __future__ import annotations

import logging
from abc import ABC
from typing import Any, override

import jax
import jax.numpy as jnp
import numpy as np
from goal.geometry import (
    Diagonal,
    Natural,
    Point,
)
from goal.models import (
    DifferentiableHMoG,
    differentiable_hmog,
)
from jax import Array

from apps.interface import (
    Analysis,
    ClusteringDataset,
    HierarchicalClusteringModel,
)
from apps.runtime import Logger, RunHandler

from .analysis.base import cluster_assignments as hmog_cluster_assignments
from .analysis.clusters import ClusterStatistics, ClusterStatisticsAnalysis
from .analysis.generative import GenerativeExamplesAnalysis
from .analysis.hierarchy import (
    CoAssignmentClusterHierarchy,
    CoAssignmentHierarchyAnalysis,
)
from .analysis.loadings import LoadingMatrixAnalysis
from .analysis.merge import (
    CoAssignmentMergeAnalysis,
    OptimalMergeAnalysis,
)
from .trainers import (
    FullGradientTrainer,
    LGMPreTrainer,
    MixtureGradientTrainer,
)

### Preamble ###

# Start logger
log = logging.getLogger(__name__)

# Helpers


def cycle_lr_schedule(keypoints: list[float], num_cycles: int) -> list[float]:
    """Return a list of `num_cycles` learning rate multipliers by interpolating keypoints."""
    n = len(keypoints)

    if n == 0:
        return [1.0] * num_cycles
    if n == 1:
        return [keypoints[0]] * num_cycles

    if num_cycles < n:
        log.warning(f"Too many keypoints ({n}) for {num_cycles} cycles. Subsampling.")
        indices = np.linspace(0, n - 1, num=num_cycles).round().astype(int)
        return [keypoints[i] for i in indices]

    x_keypoints = np.linspace(0, num_cycles - 1, num=n)
    x_full = np.arange(num_cycles)
    schedule = np.interp(x_full, x_keypoints, keypoints)
    return schedule.tolist()


### HMog Experiment ###


class HMoGModel(HierarchicalClusteringModel, ABC):
    """Model framework for HMoGs."""

    # Training configuration
    manifold: DifferentiableHMoG[Diagonal, Diagonal]
    pre: LGMPreTrainer
    lgm: FullGradientTrainer
    mix: MixtureGradientTrainer
    full: FullGradientTrainer

    lr_schedule: list[float]
    num_cycles: int

    lgm_noise_scale: float
    mix_noise_scale: float

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
        mix_noise_scale: float,
    ) -> None:
        super().__init__()

        self.manifold = differentiable_hmog(
            obs_dim=data_dim,
            obs_rep=Diagonal,
            lat_dim=latent_dim,
            pst_rep=Diagonal,
            n_components=n_clusters,
        )

        log.info(f"Created HMoG model with dimension {self.manifold.dim}.")

        self.pre = pre
        self.lgm = lgm
        self.mix = mix
        self.full = full

        self.num_cycles = num_cycles
        self.lr_schedule = cycle_lr_schedule(lr_scales, num_cycles)

        self.lgm_noise_scale = lgm_noise_scale
        self.mix_noise_scale = mix_noise_scale

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
        return self.manifold.lat_man.obs_man.data_dim

    @property
    @override
    def n_clusters(self) -> int:
        return self.manifold.lat_man.lat_man.dim + 1

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

        # with self.model.upr_hrm as uh:
        #     cat_params = uh.lat_man.initialize(key_cat, shape=self.mix_noise_scale)
        #     key_comps = jax.random.split(key_comp, self.n_clusters)
        #     anchor = uh.obs_man.initialize(key_comps[0], shape=self.mix_noise_scale)
        #
        #     component_list = [
        #         uh.obs_emb.sub_man.initialize(
        #             key_compi, shape=self.mix_noise_scale
        #         ).array
        #         for key_compi in key_comps[1:]
        #     ]
        #     components = jnp.stack(component_list)
        #     mix_params = uh.join_params(
        #         anchor, uh.int_man.point(components), cat_params
        #     )
        mix_params = self.manifold.upr_hrm.initialize(
            key_comp, shape=self.mix_noise_scale
        )

        int_noise = self.lgm_noise_scale * jax.random.normal(
            key_int, self.manifold.int_man.shape
        )
        int_params = self.manifold.int_man.point(
            self.manifold.int_man.rep.from_dense(int_noise)
        )

        return self.manifold.join_params(obs_params, int_params, mix_params).array

    def log_likelihood(
        self,
        params: Point[Natural, DifferentiableHMoG[Diagonal, Diagonal]],
        data: Array,
    ) -> Array:
        return self.manifold.average_log_observable_density(params, data)

    @override
    def get_analyses(
        self, dataset: ClusteringDataset
    ) -> list[Analysis[ClusteringDataset, DifferentiableHMoG[Diagonal, Diagonal], Any]]:
        analyses: list[
            Analysis[ClusteringDataset, DifferentiableHMoG[Diagonal, Diagonal], Any]
        ] = [
            ClusterStatisticsAnalysis(),
            # KLHierarchyAnalysis(),
            CoAssignmentHierarchyAnalysis(),
            GenerativeExamplesAnalysis(n_samples=1000),
            LoadingMatrixAnalysis(),
        ]

        if dataset.has_labels:
            analyses.extend(
                [
                    # KLMergeAnalysis(True, 0.0005),
                    CoAssignmentMergeAnalysis(True, 0.0005),
                    OptimalMergeAnalysis(True, 0.0005),
                ]
            )

        specialized_analyses = dataset.get_dataset_analyses()
        return analyses + list(specialized_analyses.values())

    @override
    def generate(
        self,
        params: Array,
        key: Array,
        n_samples: int,
    ) -> Array:
        return self.manifold.observable_sample(
            key, self.manifold.natural_point(params), n_samples
        )

    @override
    def cluster_assignments(self, params: Array, data: Array) -> Array:
        """Assign data points to clusters using the HMoG model."""
        return hmog_cluster_assignments(self.manifold, params, data)

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
                key_check, handler, logger, dataset, self.manifold, epoch, params_array
            )
        else:
            log.info("Loading existing artifacts.")
            self.process_checkpoint(key, handler, logger, dataset, self.manifold, epoch)

    @override
    def train(
        self,
        key: Array,
        handler: RunHandler,
        logger: Logger,
        dataset: ClusteringDataset,
    ) -> None:
        """Train HMoG model using alternating optimization."""
        # Split PRNG key for different training phases
        init_key, pre_key, *cycle_keys = jax.random.split(key, self.num_cycles + 2)

        params = self.manifold.natural_point(
            self.prepare_model(init_key, handler, dataset.train_data)
        )

        # Track total epochs
        epoch = handler.resolve_epoch or 0

        # Calculate training structure
        epochs_per_cycle = self.lgm.n_epochs + self.mix.n_epochs + self.full.n_epochs
        training_start_epoch = self.pre.n_epochs

        # Determine current cycle and remaining work
        if epoch == 0:
            log.info("Starting training from scratch.")
            current_cycle = 0

        elif epoch < training_start_epoch:
            current_cycle = 0
            log.info("Continuing pretraining phase.")
        else:
            current_cycle = (epoch - training_start_epoch) // epochs_per_cycle
            log.info(
                f"Resuming from epoch {epoch}, cycle {current_cycle}/{self.num_cycles}"
            )

        if self.pre.n_epochs > epoch:
            obs_params, int_params, lat_params = self.manifold.split_params(params)
            lat_obs_params, lat_int_params, cat_params = (
                self.manifold.upr_hrm.split_params(lat_params)
            )
            lgm = self.manifold.lwr_hrm
            lgm_params = lgm.join_params(obs_params, int_params, lat_obs_params)
            # Construct path to the pretrained file

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
            obs_params, int_params, lat_obs_params = lgm.split_params(lgm_params)
            lat_params = self.manifold.upr_hrm.join_params(
                lat_obs_params, lat_int_params, cat_params
            )
            params = self.manifold.join_params(obs_params, int_params, lat_params)
            epoch = self.pre.n_epochs

        # Cycle between Mixture and LGM training
        for cycle in range(current_cycle, self.num_cycles):
            current_lr_scale = self.lr_schedule[cycle]
            key_lgm, key_mix, key_full = jax.random.split(cycle_keys[cycle], 3)
            log.info("Starting training cycle %d", cycle + 1)
            log.info(f"Learning rate scale: {current_lr_scale:.3f}")

            # Train LGM (mixture params fixed)
            if self.lgm.n_epochs > 0:
                log.info(
                    f"Cycle {cycle + 1}/{self.num_cycles}: Training LGM parameters"
                )
                params = self.lgm.train(
                    key_lgm,
                    handler,
                    dataset,
                    self.manifold,
                    logger,
                    current_lr_scale,
                    epoch,
                    params,
                )
                epoch += self.lgm.n_epochs

            if self.mix.n_epochs > 0:
                log.info(
                    f"Cycle {cycle + 1}/{self.num_cycles}: Training MoG parameters"
                )
                params = self.mix.train(
                    key_mix,
                    handler,
                    dataset,
                    self.manifold,
                    logger,
                    current_lr_scale,
                    epoch,
                    params,
                )
                epoch += self.mix.n_epochs

            if self.full.n_epochs > 0:
                log.info(f"Cycle {cycle + 1}/{self.num_cycles}: Training full model")
                params = self.full.train(
                    key_full,
                    handler,
                    dataset,
                    self.manifold,
                    logger,
                    current_lr_scale,
                    epoch,
                    params,
                )
                epoch += self.full.n_epochs

            self.process_checkpoint(
                key, handler, logger, dataset, self.manifold, epoch, params.array
            )

            log.info(f"Completed cycle {cycle + 1}/{self.num_cycles}")

    @override
    def get_cluster_prototypes(self, handler: RunHandler, epoch: int) -> list[Array]:
        """Get cluster prototypes by loading from ClusterStatistics artifact."""
        # First try to load from file
        stats = handler.load_artifact(epoch, ClusterStatistics)
        return stats.prototypes

    @override
    def get_cluster_members(self, handler: RunHandler, epoch: int) -> list[Array]:
        """Get cluster members by loading from ClusterStatistics artifact."""
        stats = handler.load_artifact(epoch, ClusterStatistics)
        return stats.members

    @override
    def get_cluster_hierarchy(self, handler: RunHandler, epoch: int) -> Array:
        """Get co-assignment based hierarchy by loading from artifact."""
        hierarchy = handler.load_artifact(epoch, CoAssignmentClusterHierarchy)
        return jnp.array(hierarchy.linkage_matrix)
