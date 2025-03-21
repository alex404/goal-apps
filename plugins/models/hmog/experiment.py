"""Base class for HMoG implementations."""

from __future__ import annotations

import logging
from abc import ABC
from typing import override

import jax
import jax.numpy as jnp
from goal.geometry import (
    Natural,
    Point,
    PositiveDefinite,
)
from goal.models import (
    DifferentiableHMoG,
    differentiable_hmog,
)
from jax import Array

from apps.plugins import (
    ClusteringDataset,
    ClusteringModel,
)
from apps.runtime.handler import RunHandler
from apps.runtime.logger import JaxLogger

from .artifacts import AnalysisArgs, log_artifacts
from .configs import (
    RepresentationType,
)
from .trainers import (
    FullModelTrainer,
    LGMTrainer,
    MixtureTrainer,
)

### Preamble ###

# Start logger
log = logging.getLogger(__name__)


### HMog Experiment ###


class HMoGExperiment[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    ClusteringModel, ABC
):
    """Experiment framework for HMoGs."""

    # Training configuration
    model: DifferentiableHMoG[ObsRep, LatRep]

    lgm: LGMTrainer[ObsRep, LatRep]
    mix: MixtureTrainer[ObsRep, LatRep]
    full: FullModelTrainer[ObsRep, LatRep]

    analysis: AnalysisArgs

    def __init__(
        self,
        data_dim: int,
        latent_dim: int,
        n_clusters: int,
        obs_rep: RepresentationType,
        lat_rep: RepresentationType,
        lgm: LGMTrainer[ObsRep, LatRep],
        mix: MixtureTrainer[ObsRep, LatRep],
        full: FullModelTrainer[ObsRep, LatRep],
        analysis: AnalysisArgs,
    ) -> None:
        obs_rep_type = obs_rep.value
        lat_rep_type = lat_rep.value

        self.model = differentiable_hmog(  # pyright: ignore[reportAttributeAccessIssue]
            obs_dim=data_dim,
            obs_rep=obs_rep_type,
            lat_dim=latent_dim,
            n_components=n_clusters,
            lat_rep=lat_rep_type,
        )

        self.lgm = lgm
        self.mix = mix
        self.full = full
        self.analysis = analysis

        log.info(f"Initialized HMoG model with dimension {self.model.dim}.")

    # Properties

    @property
    @override
    def n_epochs(self) -> int:
        return self.lgm.n_epochs + self.mix.n_epochs + self.full.n_epochs

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
            component_list = [
                uh.obs_man.initialize(key_compi, shape=noise_scale).array
                for key_compi in key_comps
            ]
            components = jnp.stack(component_list)
            mix_params = uh.join_natural_mixture(
                uh.cmp_man.natural_point(components), cat_params
            )

        int_noise = noise_scale * jax.random.normal(key_int, self.model.int_man.shape)
        lkl_params = self.model.lkl_man.join_params(
            obs_params,
            self.model.int_man.point(self.model.int_man.rep.from_dense(int_noise)),
        )
        return self.model.join_conjugated(lkl_params, mix_params).array

    def log_likelihood(
        self, params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]], data: Array
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


class ThreeStageHMoGExperiment(HMoGExperiment[PositiveDefinite, PositiveDefinite]):
    @override
    def train(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        logger: JaxLogger,
    ) -> None:
        keys = jax.random.split(key, 4)

        params0 = self.model.natural_point(
            self.initialize_model(keys[0], dataset.train_data)
        )
        epoch = 0

        params1 = self.lgm.train(
            keys[1], handler, dataset, self.model, logger, epoch, params0
        )

        epoch += self.lgm.n_epochs

        params2 = self.mix.train(
            keys[2], handler, dataset, self.model, logger, epoch, params1
        )

        epoch += self.mix.n_epochs
        log_artifacts(handler, dataset, logger, self.model, epoch, params2)

        params3 = self.full.train(
            keys[3], handler, dataset, self.model, logger, epoch, params2
        )

        epoch += self.full.n_epochs
        log_artifacts(handler, dataset, logger, self.model, epoch, params3)


class CyclicHMoGExperiment(HMoGExperiment[PositiveDefinite, PositiveDefinite]):
    """HMoG experiment using cyclical alternating optimization."""

    pre: LGMTrainer[PositiveDefinite, PositiveDefinite]
    num_cycles: int

    def __init__(
        self,
        data_dim: int,
        latent_dim: int,
        n_clusters: int,
        obs_rep: RepresentationType,
        lat_rep: RepresentationType,
        pre: LGMTrainer[PositiveDefinite, PositiveDefinite],
        lgm: LGMTrainer[PositiveDefinite, PositiveDefinite],
        mix: MixtureTrainer[PositiveDefinite, PositiveDefinite],
        full: FullModelTrainer[PositiveDefinite, PositiveDefinite],
        analysis: AnalysisArgs,
        num_cycles: int,
    ) -> None:
        super().__init__(
            data_dim=data_dim,
            latent_dim=latent_dim,
            n_clusters=n_clusters,
            obs_rep=obs_rep,
            lat_rep=lat_rep,
            lgm=lgm,
            mix=mix,
            full=full,
            analysis=analysis,
        )
        self.num_cycles = num_cycles
        self.pre = pre

    @property
    @override
    def n_epochs(self) -> int:
        """Calculate total number of epochs across all cycles."""
        return self.num_cycles * (
            self.lgm.n_epochs + self.mix.n_epochs + self.full.n_epochs
        )

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
