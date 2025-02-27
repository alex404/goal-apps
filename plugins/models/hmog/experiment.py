"""Base class for HMoG implementations."""

from __future__ import annotations

import logging
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
from .trainers import FullModelTrainer, LGMTrainer, MixtureTrainer

### Preamble ###

# Start logger
log = logging.getLogger(__name__)


### HMog Experiment ###


class HMoGExperiment[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    ClusteringModel
):
    """Experiment framework for HMoGs."""

    # Training configuration
    model: DifferentiableHMoG[ObsRep, LatRep]

    stage1: LGMTrainer[ObsRep, LatRep]
    stage2: MixtureTrainer[ObsRep, LatRep]
    stage3: FullModelTrainer[ObsRep, LatRep]

    analysis: AnalysisArgs

    def __init__(
        self,
        data_dim: int,
        latent_dim: int,
        n_clusters: int,
        obs_rep: RepresentationType,
        lat_rep: RepresentationType,
        stage1: LGMTrainer[ObsRep, LatRep],
        stage2: MixtureTrainer[ObsRep, LatRep],
        stage3: FullModelTrainer[ObsRep, LatRep],
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

        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3
        self.analysis = analysis

        log.info(f"Initialized HMoG model with dimension {self.model.dim}.")

    # Properties

    @property
    @override
    def n_epochs(self) -> int:
        return self.stage1.n_epochs + self.stage2.n_epochs + self.stage3.n_epochs

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
            obs_means, self.stage1.jitter, self.stage1.min_var
        )
        obs_params = self.model.obs_man.to_natural(obs_means)

        with self.model.upr_hrm as uh:
            cat_params = uh.lat_man.initialize(key_cat)
            key_comps = jax.random.split(key_comp, self.n_clusters)
            component_list = [
                uh.obs_man.initialize(key_compi).array for key_compi in key_comps
            ]
            components = jnp.stack(component_list)
            mix_params = uh.join_natural_mixture(
                uh.cmp_man.natural_point(components), cat_params
            )

        int_noise = 0.1 * jax.random.normal(key_int, self.model.int_man.shape)
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
    def cluster_assignments(self, params: Array, data: Array) -> Array:
        def data_point_cluster(x: Array) -> Array:
            with self.model as m:
                cat_pst = m.lat_man.prior(m.posterior_at(m.natural_point(params), x))
                with m.lat_man.lat_man as lm:
                    probs = lm.to_probs(lm.to_mean(cat_pst))
            return jnp.argmax(probs, axis=-1)

        return jax.vmap(data_point_cluster)(data)

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
        keys = jax.random.split(key, 4)

        params0 = self.model.natural_point(
            self.initialize_model(keys[0], dataset.train_data)
        )

        params1 = self.stage1.train(
            keys[1], handler, dataset, self.model, logger, params0
        )

        epoch = self.stage1.n_epochs

        params2 = self.stage2.train(
            keys[2], handler, dataset, self.model, logger, epoch, params1
        )

        epoch += self.stage2.n_epochs
        log_artifacts(handler, dataset, logger, self.model, epoch, params2)

        params3 = self.stage3.train(
            keys[3], handler, dataset, self.model, logger, epoch, params2
        )

        epoch += self.stage3.n_epochs
        log_artifacts(handler, dataset, logger, self.model, epoch, params3)
