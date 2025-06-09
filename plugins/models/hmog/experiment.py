"""Base class for HMoG implementations."""

from __future__ import annotations

import logging
from abc import ABC
from typing import override

import jax
import jax.numpy as jnp
import optax
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
    ClusteringDataset,
    HierarchicalClusteringExperiment,
)
from apps.runtime import JaxLogger, RunHandler

from .analysis.base import cluster_assignments as hmog_cluster_assignments
from .analysis.clusters import ClusterStatistics
from .analysis.hierarchy import CoAssignmentClusterHierarchy
from .analysis.logging import log_artifacts
from .trainers import (
    FullGradientTrainer,
    LGMPreTrainer,
    MixtureGradientTrainer,
)

### Preamble ###

# Start logger
log = logging.getLogger(__name__)


### HMog Experiment ###


class HMoGExperiment(HierarchicalClusteringExperiment, ABC):
    """Experiment framework for HMoGs."""

    # Training configuration
    model: DifferentiableHMoG[Diagonal, Diagonal]
    pre: LGMPreTrainer
    lgm: FullGradientTrainer
    mix: MixtureGradientTrainer
    full: FullGradientTrainer

    lr_scale_init: float
    lr_scale_final: float
    num_cycles: int

    lgm_noise_scale: float
    mix_noise_scale: float

    pretrain: bool

    def __init__(
        self,
        data_dim: int,
        latent_dim: int,
        n_clusters: int,
        pre: LGMPreTrainer,
        lgm: FullGradientTrainer,
        mix: MixtureGradientTrainer,
        full: FullGradientTrainer,
        lr_scale_init: float,
        lr_scale_final: float,
        num_cycles: int,
        lgm_noise_scale: float,
        mix_noise_scale: float,
        pretrain: bool,
    ) -> None:
        super().__init__()

        self.model = differentiable_hmog(
            obs_dim=data_dim,
            obs_rep=Diagonal,
            lat_dim=latent_dim,
            pst_rep=Diagonal,
            n_components=n_clusters,
        )

        self.pre = pre
        self.lgm = lgm
        self.mix = mix
        self.full = full

        log.info(f"Initialized HMoG model with dimension {self.model.dim}.")

        self.num_cycles = num_cycles
        self.lr_scale_init = lr_scale_init
        self.lr_scale_final = lr_scale_final

        self.lgm_noise_scale = lgm_noise_scale
        self.mix_noise_scale = mix_noise_scale

        self.pretrain = pretrain

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
        return self.model.lat_man.obs_man.data_dim

    @property
    @override
    def n_clusters(self) -> int:
        return self.model.lat_man.lat_man.dim + 1

    # Methods

    @override
    def initialize_model(self, key: Array, data: Array) -> Array:
        """Initialize model parameters."""

        keys = jax.random.split(key, 3)
        key_cat, key_comp, key_int = keys

        obs_means = self.model.obs_man.average_sufficient_statistic(data)
        obs_means = self.model.obs_man.regularize_covariance(
            obs_means, self.lgm.obs_jitter_var, self.lgm.obs_min_var
        )
        obs_params = self.model.obs_man.to_natural(obs_means)

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
        mix_params = self.model.upr_hrm.initialize(key_comp, shape=self.mix_noise_scale)

        int_noise = self.lgm_noise_scale * jax.random.normal(
            key_int, self.model.int_man.shape
        )
        int_params = self.model.int_man.point(
            self.model.int_man.rep.from_dense(int_noise)
        )

        return self.model.join_params(obs_params, int_params, mix_params).array

    def log_likelihood(
        self,
        params: Point[Natural, DifferentiableHMoG[Diagonal, Diagonal]],
        data: Array,
    ) -> Array:
        return self.model.average_log_observable_density(params, data)

    @override
    def generate(
        self,
        params: Array,
        key: Array,
        n_samples: int,
    ) -> Array:
        return self.model.observable_sample(
            key, self.model.natural_point(params), n_samples
        )

    @override
    def cluster_assignments(self, params: Array, data: Array) -> Array:
        """Assign data points to clusters using the HMoG model."""
        return hmog_cluster_assignments(self.model, params, data)

    @override
    def analyze(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        logger: JaxLogger,
    ) -> None:
        """Generate analysis artifacts from saved experiment results."""

        if handler.from_epoch is None:
            raise RuntimeError("No saved parameters found for analysis")
        epoch = handler.from_epoch

        if handler.from_scratch:
            log.info("Recomputing artifacts from scratch.")
            params_array = self.prepare_model(key, handler, dataset.train_data)
            params = self.model.natural_point(params_array)
            log_artifacts(handler, dataset, logger, self, self.model, epoch, params)
        else:
            log.info("Loading existing artifacts.")
            log_artifacts(handler, dataset, logger, self, self.model, epoch)

    @override
    def train(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        logger: JaxLogger,
    ) -> None:
        """Train HMoG model using alternating optimization."""
        # Split PRNG key for different training phases
        init_key, pre_key, *cycle_keys = jax.random.split(key, self.num_cycles + 2)

        params_array = self.prepare_model(init_key, handler, dataset.train_data)
        # Initialize model
        params = self.model.natural_point(params_array)

        # Track total epochs
        epoch = handler.from_epoch or 0

        if self.pre.n_epochs > epoch:
            obs_params, int_params, lat_params = self.model.split_params(params)
            lat_obs_params, lat_int_params, cat_params = (
                self.model.upr_hrm.split_params(lat_params)
            )
            lgm = self.model.lwr_hrm
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
            lat_params = self.model.upr_hrm.join_params(
                lat_obs_params, lat_int_params, cat_params
            )
            params = self.model.join_params(obs_params, int_params, lat_params)
            epoch = self.pre.n_epochs

        # cosine lr schedule
        multiplier_schedule = optax.cosine_onecycle_schedule(
            transition_steps=self.num_cycles,
            peak_value=1.0,  # Maximum multiplier value
            pct_start=0.2,  # 30% of steps for warmup
            div_factor=1 / self.lr_scale_init,
            final_div_factor=1 / self.lr_scale_final,
        )

        # Cycle between Mixture and LGM training
        for cycle in range(self.num_cycles):
            current_lr_scale = float(multiplier_schedule(cycle))
            key_lgm, key_mix, key_full = jax.random.split(cycle_keys[cycle], 3)
            log.info("Starting training cycle %d", cycle + 1)
            log.info(f"Learning rate scale: {current_lr_scale:.2e}")

            # Train LGM (mixture params fixed)
            if self.lgm.n_epochs > 0:
                log.info(
                    f"Cycle {cycle + 1}/{self.num_cycles}: Training LGM parameters"
                )
                params = self.lgm.train(
                    key_lgm,
                    handler,
                    dataset,
                    self.model,
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
                    self.model,
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
                    self.model,
                    logger,
                    current_lr_scale,
                    epoch,
                    params,
                )
                epoch += self.full.n_epochs

            log_artifacts(handler, dataset, logger, self, self.model, epoch, params)

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
