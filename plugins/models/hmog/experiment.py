"""Base class for HMoG implementations."""

from __future__ import annotations

import logging
from abc import ABC
from typing import override

import jax
import jax.numpy as jnp
from goal.geometry import (
    Diagonal,
    Natural,
    Point,
)
from goal.models import (
    SymmetricHMoG,
    symmetric_hmog,
)
from jax import Array

from apps.plugins import (
    ClusteringDataset,
    ClusteringModel,
)
from apps.runtime.handler import RunHandler
from apps.runtime.logger import JaxLogger

from .artifacts import AnalysisArgs, log_artifacts
from .trainers import (
    GradientTrainer,
    LGMTrainer,
)

### Preamble ###

# Start logger
log = logging.getLogger(__name__)


### HMog Experiment ###


class HMoGExperiment(ClusteringModel, ABC):
    """Experiment framework for HMoGs."""

    # Training configuration
    model: SymmetricHMoG[Diagonal, Diagonal]
    pre: LGMTrainer[Diagonal]
    trainer: GradientTrainer

    num_cycles: int

    analysis: AnalysisArgs

    def __init__(
        self,
        data_dim: int,
        latent_dim: int,
        n_clusters: int,
        pre: LGMTrainer[Diagonal],
        trainer: GradientTrainer,
        analysis: AnalysisArgs,
        num_cycles: int,
    ) -> None:
        super().__init__()

        self.model = symmetric_hmog(
            obs_dim=data_dim,
            obs_rep=Diagonal,
            lat_dim=latent_dim,
            lat_rep=Diagonal,
            n_components=n_clusters,
        )

        self.pre = pre
        self.trainer = trainer
        self.analysis = analysis

        log.info(f"Initialized HMoG model with dimension {self.model.dim}.")

        self.num_cycles = num_cycles
        self.pre = pre

    # Properties

    @property
    @override
    def n_epochs(self) -> int:
        """Calculate total number of epochs across all cycles."""
        return self.pre.n_epochs + self.num_cycles * self.trainer.n_epochs

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
        noise_scale = 0.01
        keys = jax.random.split(key, 3)
        key_cat, key_comp, key_int = keys

        obs_means = self.model.obs_man.average_sufficient_statistic(data)
        obs_means = self.model.obs_man.regularize_covariance(
            obs_means, self.lgm.jitter, self.lgm.min_var
        )
        obs_params = self.model.obs_man.to_natural(obs_means)

        with self.model.upr_hrm as uh:
            cat_params = uh.lat_man.initialize(key_cat, shape=noise_scale)
            key_comps = jax.random.split(key_comp, self.n_clusters)
            anchor = uh.obs_man.initialize(key_comps[0], shape=noise_scale)

            component_list = [
                uh.obs_emb.sub_man.initialize(key_compi, shape=noise_scale).array
                for key_compi in key_comps[1:]
            ]
            components = jnp.stack(component_list)
            mix_params = uh.join_params(
                anchor, uh.int_man.point(components), cat_params
            )

        int_noise = noise_scale * jax.random.normal(key_int, self.model.int_man.shape)
        lkl_params = self.model.lkl_man.join_params(
            obs_params,
            self.model.int_man.point(self.model.int_man.rep.from_dense(int_noise)),
        )

        return self.model.join_conjugated(lkl_params, mix_params).array

    def log_likelihood(
        self,
        params: Point[Natural, SymmetricHMoG[Diagonal, Diagonal]],
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
    def analyze(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        logger: JaxLogger,
    ) -> None:
        """Generate analysis artifacts from saved experiment results."""

        epoch = (
            self.analysis.epoch
            if self.analysis.epoch is not None
            else max(handler.get_available_epochs())
        )

        if self.analysis.from_scratch:
            log.info("Recomputing artifacts from scratch.")
            params = self.model.natural_point(handler.load_params(epoch))
            log_artifacts(handler, dataset, logger, self.model, epoch, params)
        else:
            log.info("Loading existing artifacts.")
            log_artifacts(handler, dataset, logger, self.model, epoch)

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

        # Initialize model
        params = self.model.natural_point(
            self.initialize_model(init_key, dataset.train_data)
        )

        # Track total epochs
        epoch = 0

        # Train LGM (mixture params fixed)
        if self.pre.n_epochs > 0:
            log.info("Pre-training LGM parameters")
            params = self.pre.train(
                pre_key,
                handler,
                dataset,
                self.model,
                logger,
                epoch,
                params,
            )
            epoch += self.pre.n_epochs

        # Cycle between Mixture and LGM training
        for cycle in range(self.num_cycles):
            key_lgm, key_mix, key_full = jax.random.split(cycle_keys[cycle], 3)

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
                    epoch,
                    params,
                )
                epoch += self.lgm.n_epochs

            if self.mix.n_epochs > 0:
                log.info(
                    f"Cycle {cycle + 1}/{self.num_cycles}: Training mixture parameters"
                )
                params = self.mix.train(
                    key_mix, handler, dataset, self.model, logger, epoch, params
                )
                epoch += self.mix.n_epochs

            if self.full.n_epochs > 0:
                log.info(f"Cycle {cycle + 1}/{self.num_cycles}: Training full model")
                params = self.full.train(
                    key_full, handler, dataset, self.model, logger, epoch, params
                )
                epoch += self.full.n_epochs

            log_artifacts(handler, dataset, logger, self.model, epoch, params)

            log.info(f"Completed cycle {cycle + 1}/{self.num_cycles}")


# class DifferentiableHMoGExperiment(ClusteringModel):
#     """Experiment for DifferentiableHMoG models."""
#
#     model: DifferentiableHMoG[Diagonal, Diagonal]
#     pre: LGMTrainer[Diagonal, Diagonal]
#     trainer: DifferentiableModelTrainer[Diagonal, Diagonal]
#     analysis: AnalysisArgs
#
#     def __init__(
#         self,
#         data_dim: int,
#         latent_dim: int,
#         n_clusters: int,
#         trainer: DifferentiableModelTrainer[Diagonal, Diagonal],
#         analysis: AnalysisArgs,
#     ) -> None:
#         self.model = differentiable_hmog(
#             obs_dim=data_dim,
#             obs_rep=Diagonal,
#             lat_dim=latent_dim,
#             pst_rep=Diagonal,
#             n_components=n_clusters,
#         )
#
#         self.trainer = trainer
#         self.analysis = analysis
#
#     @property
#     @override
#     def n_epochs(self) -> int:
#         return self.trainer.n_epochs
#
#     @property
#     @override
#     def n_clusters(self) -> int:
#         return self.model.upr_hrm.lat_man.dim + 1
#
#     @override
#     def initialize_model(self, key: Array, data: Array) -> Array:
#         """Initialize model parameters."""
#         noise_scale = 0.01
#         keys = jax.random.split(key, 3)
#         key_cat, key_comp, key_int = keys
#
#         obs_means = self.model.obs_man.average_sufficient_statistic(data)
#         obs_means = self.model.obs_man.regularize_covariance(
#             obs_means, self.trainer.obs_jitter, self.trainer.obs_min_var
#         )
#         obs_params = self.model.obs_man.to_natural(obs_means)
#
#         with self.model.upr_hrm as uh:
#             cat_params = uh.lat_man.initialize(key_cat, shape=noise_scale)
#             key_comps = jax.random.split(key_comp, self.n_clusters)
#
#             component_list = [
#                 uh.obs_man.initialize(key_compi, shape=noise_scale).array
#                 for key_compi in key_comps
#             ]
#             components = jnp.stack(component_list)
#             mix_params = uh.join_natural_mixture(
#                 uh.cmp_man.point(components), cat_params
#             )
#
#         int_noise = noise_scale * jax.random.normal(key_int, self.model.int_man.shape)
#         int_params: Point[Natural, LinearMap[Rectangular, Euclidean, Euclidean]] = (
#             self.model.lwr_hrm.int_man.point(
#                 self.model.int_man.rep.from_dense(int_noise)
#             )
#         )
#
#         params = self.model.join_params(
#             obs_params,
#             int_params,
#             mix_params,
#         )
#         return params.array
#
#     @override
#     def generate(self, params: Array, key: Array, n_samples: int) -> Array:
#         """Generate samples from the model."""
#         return self.model.observable_sample(
#             key, self.model.natural_point(params), n_samples
#         )
#
#     @override
#     def train(
#         self,
#         key: Array,
#         handler: RunHandler,
#         dataset: ClusteringDataset,
#         logger: JaxLogger,
#     ) -> None:
#         """Train the model using gradient-based optimization."""
#         # Initialize model
#         init_key, pre_key, train_key = jax.random.split(key, 3)
#         params = self.model.natural_point(
#             self.initialize_model(init_key, dataset.train_data)
#         )
#
#         # Track total epochs
#         epoch = 0
#
#         # Train LGM (mixture params fixed)
#         if self.pre.n_epochs > 0:
#             log.info("Pre-training LGM parameters")
#             params = self.pre.train(
#                 pre_key,
#                 handler,
#                 dataset,
#                 self.model,
#                 logger,
#                 epoch,
#                 params,
#             )
#             epoch += self.pre.n_epochs
#
#         # Train model with the trainer
#         params = self.trainer.train(
#             train_key,
#             handler,
#             dataset,
#             self.model,
#             logger,
#             0,  # epoch_offset
#             params,
#         )
#
#         # Log artifacts
#         log_artifacts(
#             handler, dataset, logger, self.model, self.trainer.n_epochs, params
#         )
#
#     @override
#     def analyze(
#         self,
#         key: Array,
#         handler: RunHandler,
#         dataset: ClusteringDataset,
#         logger: JaxLogger,
#     ) -> None:
#         """Generate analysis artifacts from saved experiment results."""
#         epoch = (
#             self.analysis.epoch
#             if self.analysis.epoch is not None
#             else max(handler.get_available_epochs())
#         )
#
#         if self.analysis.from_scratch:
#             params = self.model.natural_point(handler.load_params(epoch))
#             log_artifacts(handler, dataset, logger, self.model, epoch, params)
#         else:
#             log_artifacts(handler, dataset, logger, self.model, epoch)
