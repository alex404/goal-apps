"""Base class for HMoG implementations."""

from __future__ import annotations

import logging
from typing import override

import jax
import jax.numpy as jnp
from goal.geometry import (
    Natural,
    Optimizer,
    OptState,
    Point,
    PositiveDefinite,
)
from goal.models import (
    DifferentiableHMoG,
    DifferentiableMixture,
    FullNormal,
    LinearGaussianModel,
    Normal,
    differentiable_hmog,
)
from jax import Array

from apps.plugins import (
    ClusteringDataset,
    ClusteringModel,
)
from apps.runtime.handler import RunHandler
from apps.runtime.logger import JaxLogger

from .artifacts import HMoGMetrics, log_artifacts
from .base import (
    RepresentationType,
    bound_hmog_mixture_probabilities,
    bound_mixture_probabilities,
    fori,
)

### Preamble ###

# Start logger
log = logging.getLogger(__name__)


### HMog Experiment ###


class HMoGExperiment[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    ClusteringModel
):
    """Experiment framework for HMoGs."""

    batch_size: int
    stage1_epochs: int
    stage2_epochs: int
    stage3_epochs: int
    stage2_learning_rate: float
    stage3_learning_rate: float
    obs_jitter: float
    obs_min_var: float
    from_scratch: bool
    analysis_epoch: int | None

    def __init__(
        self,
        data_dim: int,
        latent_dim: int,
        n_clusters: int,
        obs_rep: RepresentationType,
        lat_rep: RepresentationType,
        batch_size: int,
        stage1_epochs: int,
        stage2_epochs: int,
        stage3_epochs: int,
        stage2_learning_rate: float,
        stage3_learning_rate: float,
        obs_jitter: float,
        obs_min_var: float,
        from_scratch: bool,
        analysis_epoch: int,
    ) -> None:
        self.batch_size = batch_size
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.stage3_epochs = stage3_epochs
        self.stage2_learning_rate = stage2_learning_rate
        self.stage3_learning_rate = stage3_learning_rate
        self.obs_jitter = obs_jitter
        self.obs_min_var = obs_min_var
        self.from_scratch = from_scratch
        self.analysis_epoch = analysis_epoch

        obs_rep_type = obs_rep.value
        lat_rep_type = lat_rep.value

        self.model: DifferentiableHMoG[ObsRep, LatRep] = differentiable_hmog(  # pyright: ignore[reportAttributeAccessIssue]
            obs_dim=data_dim,
            obs_rep=obs_rep_type,
            lat_dim=latent_dim,
            n_components=n_clusters,
            lat_rep=lat_rep_type,
        )

        log.info(f"Initialized HMoG model with dimension {self.model.dim}.")

    """Base class for HMoG implementations."""

    # Properties

    @property
    @override
    def n_epochs(self) -> int:
        return self.stage1_epochs + self.stage2_epochs + self.stage3_epochs

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
            obs_means, self.obs_jitter, self.obs_min_var
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

    def log_epoch_metrics(
        self,
        logger: JaxLogger,
        epoch: int,
        hmog_params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
        train_sample: Array,
        test_sample: Array,
        log_freq: int = 1,
    ) -> None:
        """Log metrics for an epoch."""

        def compute_metrics():
            epoch_train_ll = self.model.average_log_observable_density(
                hmog_params, train_sample
            )
            epoch_test_ll = self.model.average_log_observable_density(
                hmog_params, test_sample
            )

            n_samps = train_sample.shape[0]
            epoch_train_bic = (
                self.model.dim * jnp.log(n_samps) / n_samps - 2 * epoch_train_ll
            )
            metrics = HMoGMetrics(
                train_ll=epoch_train_ll,
                test_ll=epoch_test_ll,
                train_average_bic=epoch_train_bic,
            )
            logger.log_metrics({k: metrics[k] for k in metrics}, epoch + 1)

        def no_op():
            return None

        jax.lax.cond(epoch % log_freq == 0, compute_metrics, no_op)

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
            self.analysis_epoch
            if self.analysis_epoch is not None
            else max(handler.get_available_epochs())
        )

        if self.from_scratch:
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
        init_key, fit_key = jax.random.split(key)
        params = self.initialize_model(init_key, dataset.train_data)
        self.fit(fit_key, handler, dataset, logger, self.model.natural_point(params))

    def fit(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        logger: JaxLogger,
        init_params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
    ) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]:
        """Three-stage minibatch training process."""

        train_sample = dataset.train_data
        test_sample = dataset.test_data

        lkl_params0, mix_params0 = self.model.split_conjugated(init_params)

        self.log_epoch_metrics(logger, -1, init_params, train_sample, test_sample)

        # Stage 1: Full-batch EM for LinearGaussianModel
        with self.model.lwr_hrm as lh:
            z = lh.lat_man.to_natural(lh.lat_man.standard_normal())
            lgm_params0 = lh.join_conjugated(lkl_params0, z)

            def stage1_step(
                epoch: int, lgm_params: Point[Natural, LinearGaussianModel[ObsRep]]
            ) -> Point[Natural, LinearGaussianModel[ObsRep]]:
                means = lh.expectation_step(lgm_params, train_sample)
                obs_means, int_means, lat_means = lh.split_params(means)
                obs_means = lh.obs_man.regularize_covariance(
                    obs_means, self.obs_jitter, self.obs_min_var
                )
                means = lh.join_params(obs_means, int_means, lat_means)
                params1 = lh.to_natural(means)
                lkl_params = lh.likelihood_function(params1)
                lgm_params = lh.join_conjugated(lkl_params, z)
                hmog_params = self.model.join_conjugated(lkl_params, mix_params0)
                self.log_epoch_metrics(
                    logger, epoch, hmog_params, train_sample, test_sample
                )
                return lgm_params

            lgm_params1 = fori(0, self.stage1_epochs, stage1_step, lgm_params0)
            lkl_params1 = lh.likelihood_function(lgm_params1)

        # Stage 2: Gradient descent for mixture components
        stage2_optimizer: Optimizer[
            Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]
        ] = Optimizer.adam(self.model.upr_hrm, learning_rate=self.stage2_learning_rate)
        stage2_opt_state = stage2_optimizer.init(mix_params0)

        def stage2_loss(
            params: Point[Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]],
            batch: Array,
        ) -> Array:
            hmog_params = self.model.join_conjugated(lkl_params1, params)
            return -self.model.average_log_observable_density(hmog_params, batch)

        def stage2_minibatch_step(
            carry: tuple[
                OptState,
                Point[Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]],
            ],
            batch: Array,
        ) -> tuple[
            tuple[
                OptState,
                Point[Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]],
            ],
            None,
        ]:
            opt_state, params = carry
            grad = self.model.upr_hrm.grad(lambda p: stage2_loss(p, batch), params)

            opt_state, params = stage2_optimizer.update(opt_state, grad, params)
            params = bound_mixture_probabilities(self.model.upr_hrm, params)
            return ((opt_state, params), None)

        def stage2_epoch(
            epoch: int,
            carry: tuple[
                OptState,
                Point[Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]],
                Array,
            ],
        ) -> tuple[
            OptState,
            Point[Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]],
            Array,
        ]:
            opt_state, params, key = carry

            # Shuffle data and truncate to fit batches evenly
            return_key, shuffle_key = jax.random.split(key)
            n_complete_batches = train_sample.shape[0] // self.batch_size
            n_samples_to_use = n_complete_batches * self.batch_size

            shuffled_indices = jax.random.permutation(
                shuffle_key, train_sample.shape[0]
            )[:n_samples_to_use]
            batched_data = train_sample[shuffled_indices].reshape(
                (n_complete_batches, self.batch_size, -1)
            )

            # Process batches
            (opt_state, params), _ = jax.lax.scan(
                stage2_minibatch_step,
                (opt_state, params),
                batched_data,
                None,
            )

            # Compute full dataset likelihood
            hmog_params = self.model.join_conjugated(lkl_params1, params)

            self.log_epoch_metrics(
                logger, epoch, hmog_params, train_sample, test_sample, log_freq=10
            )

            return (opt_state, params, return_key)

        (_, mix_params1, key) = fori(
            self.stage1_epochs,
            self.stage1_epochs + self.stage2_epochs,
            stage2_epoch,
            (stage2_opt_state, mix_params0, key),
        )

        # Stage 3: Similar structure to stage 2
        params1 = self.model.join_conjugated(lkl_params1, mix_params1)
        log_artifacts(
            handler,
            dataset,
            logger,
            self.model,
            self.stage1_epochs + self.stage2_epochs,
            params1,
        )

        stage3_optimizer: Optimizer[Natural, DifferentiableHMoG[ObsRep, LatRep]] = (
            Optimizer.adam(self.model.upr_hrm, learning_rate=self.stage3_learning_rate)
        )
        stage3_opt_state = stage3_optimizer.init(params1)

        def stage3_loss(
            params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
            batch: Array,
        ) -> Array:
            return -self.model.average_log_observable_density(params, batch)

        def stage3_minibatch_step(
            carry: tuple[OptState, Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]],
            batch: Array,
        ) -> tuple[
            tuple[OptState, Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]], None
        ]:
            opt_state, params = carry
            grad = self.model.grad(lambda p: stage3_loss(p, batch), params)
            opt_state, params = stage3_optimizer.update(opt_state, grad, params)
            params = bound_hmog_mixture_probabilities(self.model, params)
            return (opt_state, params), None

        def stage3_epoch(
            epoch: int,
            carry: tuple[
                OptState, Point[Natural, DifferentiableHMoG[ObsRep, LatRep]], Array
            ],
        ) -> tuple[OptState, Point[Natural, DifferentiableHMoG[ObsRep, LatRep]], Array]:
            opt_state, params, key = carry

            # Shuffle and batch data
            return_key, shuffle_key = jax.random.split(key)
            n_complete_batches = train_sample.shape[0] // self.batch_size
            n_samples_to_use = n_complete_batches * self.batch_size

            shuffled_indices = jax.random.permutation(
                shuffle_key, train_sample.shape[0]
            )[:n_samples_to_use]
            batched_data = train_sample[shuffled_indices].reshape(
                (n_complete_batches, self.batch_size, -1)
            )

            # Process batches
            (opt_state, params), _ = jax.lax.scan(
                stage3_minibatch_step,
                (opt_state, params),
                batched_data,
            )

            self.log_epoch_metrics(
                logger, epoch, params, train_sample, test_sample, log_freq=10
            )

            return opt_state, params, return_key

        (_, final_params, _) = fori(
            self.stage1_epochs + self.stage2_epochs,
            self.n_epochs,
            stage3_epoch,
            (stage3_opt_state, params1, key),
        )
        log_artifacts(handler, dataset, logger, self.model, self.n_epochs, final_params)
        return final_params
