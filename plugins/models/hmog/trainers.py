"""Trainers for HMoG model components."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, override

import jax
import jax.numpy as jnp
import optax
from goal.geometry import (
    Mean,
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
)
from jax import Array

from apps.configs import STATS_LEVEL
from apps.plugins import (
    ClusteringDataset,
)
from apps.runtime.handler import MetricDict, RunHandler
from apps.runtime.logger import JaxLogger

# Start logger
log = logging.getLogger(__name__)


### Helpers ###


def fori[X](lower: int, upper: int, body_fun: Callable[[Array, X], X], init: X) -> X:
    return jax.lax.fori_loop(lower, upper, body_fun, init)  # pyright: ignore[reportUnknownVariableType]


### Stabilizers ###


def bound_observable_variances[Rep: PositiveDefinite](
    model: LinearGaussianModel[Rep],
    means: Point[Mean, LinearGaussianModel[Rep]],
    min_var: float,
    jitter: float,
) -> Point[Mean, LinearGaussianModel[Rep]]:
    """Regularize observable covariance parameters in mean coordinates."""
    obs_means, int_means, lat_means = model.split_params(means)
    bounded_obs_means = model.obs_man.regularize_covariance(obs_means, jitter, min_var)
    return model.join_params(bounded_obs_means, int_means, lat_means)


def bound_mixture_parameters[Rep: PositiveDefinite](
    model: DifferentiableMixture[FullNormal, Normal[Rep]],
    params: Point[Natural, DifferentiableMixture[FullNormal, Normal[Rep]]],
    min_prob: float,
    min_var: float,
    jitter: float,
) -> Point[Natural, DifferentiableMixture[FullNormal, Normal[Rep]]]:
    """Bound mixture probabilities and latent variances."""
    lkl_params, cat_params = model.split_conjugated(params)
    lat_params, int_params = model.lkl_man.split_params(lkl_params)

    # Bound latent variances
    with model.obs_man as om:
        lat_means = om.to_mean(lat_params)
        bounded_lat_means = om.regularize_covariance(lat_means, jitter, min_var)
        bounded_lat_params = om.to_natural(bounded_lat_means)

    # Bound categorical probabilities
    with model.lat_man as lm:
        cat_means = lm.to_mean(cat_params)
        probs = lm.to_probs(cat_means)
        bounded_probs = jnp.clip(probs, min_prob, 1.0)
        bounded_probs = bounded_probs / jnp.sum(bounded_probs)
        bounded_cat_params = lm.to_natural(lm.from_probs(bounded_probs))

    bounded_lkl_params = model.lkl_man.join_params(bounded_lat_params, int_params)
    return model.join_conjugated(bounded_lkl_params, bounded_cat_params)


def bound_hmog_parameters[
    ObsRep: PositiveDefinite,
    LatRep: PositiveDefinite,
](
    model: DifferentiableHMoG[ObsRep, LatRep],
    params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
    min_prob: float,
    obs_min_var: float,
    obs_jitter: float,
    lat_min_var: float,
    lat_jitter: float,
) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]:
    """Apply all parameter bounds to HMoG."""
    # Split parameters
    lkl_params, mix_params = model.split_conjugated(params)

    # Bound mixture parameters
    bounded_mix_params = bound_mixture_parameters(
        model.upr_hrm, mix_params, min_prob, lat_min_var, lat_jitter
    )

    # Bound observable parameters
    obs_params, int_params = model.lkl_man.split_params(lkl_params)
    obs_means = model.obs_man.to_mean(obs_params)
    bounded_obs_means = model.obs_man.regularize_covariance(
        obs_means, obs_jitter, obs_min_var
    )
    bounded_obs_params = model.obs_man.to_natural(bounded_obs_means)
    bounded_lkl_params = model.lkl_man.join_params(bounded_obs_params, int_params)

    return model.join_conjugated(bounded_lkl_params, bounded_mix_params)


### Logging ###


def log_epoch_metrics[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    dataset: ClusteringDataset,
    hmog_model: DifferentiableHMoG[ObsRep, LatRep],
    logger: JaxLogger,
    params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
    epoch: Array,
    gradient_norms: Array | None = None,
    log_freq: int = 1,
) -> None:
    """Log metrics for an epoch."""
    train_data = dataset.train_data
    test_data = dataset.test_data

    def compute_metrics() -> None:
        # Compute core performance metrics
        epoch_train_ll = hmog_model.average_log_observable_density(params, train_data)
        epoch_test_ll = hmog_model.average_log_observable_density(params, test_data)

        n_samps = train_data.shape[0]
        epoch_negative_bic = -(
            hmog_model.dim * jnp.log(n_samps) / n_samps - 2 * epoch_train_ll
        )
        info = jnp.array(logging.INFO)

        # Build metrics dictionary with display names and levels
        metrics: MetricDict = {
            "Performance/Train Log-Likelihood": (
                info,
                epoch_train_ll,
            ),
            "Performance/Test Log-Likelihood": (
                info,
                epoch_test_ll,
            ),
            "Performance/Negative BIC": (
                info,
                epoch_negative_bic,
            ),
        }

        stats = jnp.array(STATS_LEVEL)

        def update_parameter_stats(params: Point[Natural, Any], name: str) -> None:
            array = params.array
            metrics.update(
                {
                    f"Params/{name} Min": (
                        stats,
                        jnp.min(array),
                    ),
                    f"Params/{name} Median": (
                        stats,
                        jnp.median(array),
                    ),
                    f"Params/{name} Max": (
                        stats,
                        jnp.max(array),
                    ),
                }
            )

        # Add parameter statistics at DEBUG level
        obs_params, lwr_int_params, upr_params = hmog_model.split_params(params)
        obs_loc_params, obs_prs_params = hmog_model.obs_man.split_location_precision(
            obs_params
        )
        lat_params, upr_int_params, cat_params = hmog_model.upr_hrm.split_params(
            upr_params
        )
        lat_loc_params, lat_prs_params = (
            hmog_model.upr_hrm.obs_man.split_location_precision(lat_params)
        )

        update_parameter_stats(obs_loc_params, "Obs Location")
        update_parameter_stats(obs_prs_params, "Obs Precision")
        update_parameter_stats(lwr_int_params, "Obs Interaction")
        update_parameter_stats(lat_loc_params, "Lat Location")
        update_parameter_stats(lat_prs_params, "Lat Precision")
        update_parameter_stats(upr_int_params, "Lat Interaction")
        update_parameter_stats(cat_params, "Categorical")

        if gradient_norms is not None:
            metrics.update(
                {
                    "Grad/Min Norm": (stats, jnp.min(gradient_norms)),
                    "Grad/Max Norm": (stats, jnp.max(gradient_norms)),
                    "Grad/Median Norm": (stats, jnp.median(gradient_norms)),
                }
            )

        logger.log_metrics(metrics, epoch + 1)

    def no_op() -> None:
        return None

    jax.lax.cond(epoch % log_freq == 0, compute_metrics, no_op)


### LGM Trainers ###


@dataclass(frozen=True)
class LGMTrainer[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](ABC):
    """Base trainer for LinearGaussianModel components."""

    n_epochs: int
    min_var: float
    jitter: float

    @abstractmethod
    def train(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: DifferentiableHMoG[ObsRep, LatRep],
        logger: JaxLogger,
        params0: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
    ) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]: ...


@dataclass(frozen=True)
class EMLGMTrainer[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    LGMTrainer[ObsRep, LatRep]
):
    """EM training for LinearGaussianModel."""

    @override
    def train(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: DifferentiableHMoG[ObsRep, LatRep],
        logger: JaxLogger,
        params0: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
    ) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]:
        # Standard normal latent for LGM
        train_data = dataset.train_data
        lkl_params0, mix_params0 = model.split_conjugated(params0)

        with model.lwr_hrm as lh:

            def step(
                epoch: Array, lgm_params: Point[Natural, LinearGaussianModel[ObsRep]]
            ) -> Point[Natural, LinearGaussianModel[ObsRep]]:
                lgm_means = lh.expectation_step(lgm_params, train_data)
                lgm_means = bound_observable_variances(
                    lh, lgm_means, self.min_var, self.jitter
                )
                new_lgm_params = lh.to_natural(lgm_means)
                lkl_params = lh.likelihood_function(new_lgm_params)
                lgm_params = lh.join_conjugated(lkl_params, z)

                # Create full HMOG params for evaluation
                hmog_params = model.join_conjugated(lkl_params, mix_params0)
                log_epoch_metrics(
                    dataset,
                    model,
                    logger,
                    hmog_params,
                    epoch,
                )

                logger.monitor_params(
                    {
                        "original": lgm_params.array,
                        "post_update": new_lgm_params.array,
                    },
                    handler,
                    context="stage1_epoch",
                )
                return lgm_params

            z = lh.lat_man.to_natural(lh.lat_man.standard_normal())
            lgm_params0 = lh.join_conjugated(lkl_params0, z)

            lgm_params1 = fori(0, self.n_epochs, step, lgm_params0)

            lkl_params1 = lh.likelihood_function(lgm_params1)
            return model.join_conjugated(lkl_params1, mix_params0)


### Mixture Trainers ###


@dataclass(frozen=True)
class MixtureTrainer[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](ABC):
    """Base trainer for mixture model components."""

    n_epochs: int
    min_prob: float
    min_var: float
    jitter: float

    @abstractmethod
    def train(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: DifferentiableHMoG[ObsRep, LatRep],
        logger: JaxLogger,
        epoch_offset: int,
        params1: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
    ) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]: ...


@dataclass(frozen=True)
class GradientMixtureTrainer[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    MixtureTrainer[ObsRep, LatRep]
):
    """Gradient-based training for mixture components."""

    lr_init: float
    lr_final_ratio: float
    batch_size: int

    @override
    def train(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: DifferentiableHMoG[ObsRep, LatRep],
        logger: JaxLogger,
        epoch_offset: int,
        params1: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
    ) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]:
        # Set up optimizer
        lkl_params1, mix_params0 = model.split_conjugated(params1)
        train_data = dataset.train_data
        n_complete_batches = train_data.shape[0] // self.batch_size

        if self.lr_final_ratio < 1.0:
            lr_schedule = optax.cosine_decay_schedule(
                init_value=self.lr_init,
                decay_steps=self.n_epochs,
                alpha=self.lr_final_ratio,
            )
        else:
            lr_schedule = self.lr_init

        optimizer: Optimizer[
            Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]
        ] = Optimizer.adamw(model.upr_hrm, learning_rate=lr_schedule)

        # Define minibatch step function
        def minibatch_step(
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
            Array,
        ]:
            # Define loss function
            def loss_fn(
                mix_params: Point[
                    Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]
                ],
            ) -> Array:
                params = model.join_conjugated(lkl_params1, mix_params)
                return -model.average_log_observable_density(params, batch)

            opt_state, mix_params = carry
            grad = model.upr_hrm.grad(loss_fn, mix_params)

            opt_state, new_mix_params = optimizer.update(opt_state, grad, mix_params)
            bound_mix_params = bound_mixture_parameters(
                model.upr_hrm,
                new_mix_params,
                self.min_prob,
                self.min_var,
                self.jitter / n_complete_batches,
            )

            logger.monitor_params(
                {
                    "original": mix_params.array,
                    "post_update": new_mix_params.array,
                    "post_bounds": bound_mix_params.array,
                    "batch": batch,
                    "grad": grad.array,
                },
                handler,
                context="stage2_step",
            )
            grad_norm = jnp.linalg.norm(grad.array)
            return ((opt_state, bound_mix_params), grad_norm)

        # Define epoch function
        def epoch_step(
            epoch: Array,
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
            opt_state, mix_params, epoch_key = carry

            # Shuffle data and truncate to fit batches evenly
            return_key, shuffle_key = jax.random.split(epoch_key)

            shuffled_indices = jax.random.permutation(shuffle_key, train_data.shape[0])
            batched_indices = shuffled_indices[: n_complete_batches * self.batch_size]
            leftover_indices = shuffled_indices[n_complete_batches * self.batch_size :]
            batched_data = train_data[batched_indices].reshape(
                (n_complete_batches, self.batch_size, -1)
            )
            leftover_data = train_data[leftover_indices]

            # Process batches
            (opt_state, new_mix_params), grad_norms = jax.lax.scan(
                minibatch_step,
                (opt_state, mix_params),
                batched_data,
            )
            (opt_state, new_mix_params), grad_norm = minibatch_step(
                (opt_state, new_mix_params), leftover_data
            )
            grad_norms = jnp.concatenate([grad_norms, jnp.array([grad_norm])])

            # Log metrics
            new_params = model.join_conjugated(lkl_params1, new_mix_params)
            log_epoch_metrics(
                dataset,
                model,
                logger,
                new_params,
                epoch_offset + epoch,
                gradient_norms=grad_norms,
                log_freq=10,
            )

            return (opt_state, new_mix_params, return_key)

        # Run training

        opt_state = optimizer.init(mix_params0)

        (_, mix_params1, _) = fori(
            0, self.n_epochs, epoch_step, (opt_state, mix_params0, key)
        )

        return model.join_conjugated(lkl_params1, mix_params1)


### Full Model Trainers ###


@dataclass(frozen=True)
class FullModelTrainer[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](ABC):
    """Base trainer for full HMoG model."""

    min_prob: float
    obs_min_var: float
    lat_min_var: float
    obs_jitter: float
    lat_jitter: float
    n_epochs: int

    @abstractmethod
    def train(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: DifferentiableHMoG[ObsRep, LatRep],
        logger: JaxLogger,
        epoch_offset: int,
        params2: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
    ) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]: ...


@dataclass(frozen=True)
class GradientFullModelTrainer[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    FullModelTrainer[ObsRep, LatRep]
):
    """Gradient-based training for full HMoG model."""

    lr_init: float
    lr_final_ratio: float
    grad_clip: float
    batch_size: int

    @override
    def train(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: DifferentiableHMoG[ObsRep, LatRep],
        logger: JaxLogger,
        epoch_offset: int,
        params2: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
    ) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]:
        # Set up optimizer with optional learning rate schedule
        train_data = dataset.train_data
        n_complete_batches = train_data.shape[0] // self.batch_size

        if self.lr_final_ratio < 1.0:
            lr_schedule = optax.cosine_decay_schedule(
                init_value=self.lr_init,
                decay_steps=self.n_epochs,
                alpha=self.lr_final_ratio,
            )
        else:
            lr_schedule = self.lr_init

        optimizer: Optimizer[Natural, DifferentiableHMoG[ObsRep, LatRep]] = (
            Optimizer.adamw(model, learning_rate=lr_schedule, grad_clip=self.grad_clip)
        )

        # Define minibatch step function
        def minibatch_step(
            carry: tuple[OptState, Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]],
            batch: Array,
        ) -> tuple[
            tuple[OptState, Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]], Array
        ]:
            # Define loss function
            def loss_fn(
                params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
            ) -> Array:
                return -model.average_log_observable_density(params, batch)

            opt_state, params = carry
            grad = model.grad(loss_fn, params)
            opt_state, new_params = optimizer.update(opt_state, grad, params)

            bound_params = bound_hmog_parameters(
                model,
                new_params,
                self.min_prob,
                self.obs_min_var,
                self.obs_jitter / n_complete_batches,
                self.lat_min_var,
                self.lat_jitter / n_complete_batches,
            )

            logger.monitor_params(
                {
                    "original": params.array,
                    "post_update": new_params.array,
                    "post_bounds": bound_params.array,
                    "batch": batch,
                    "grad": grad.array,
                },
                handler,
                context="stage3_step",
            )

            grad_norm = jnp.linalg.norm(grad.array)
            return (opt_state, bound_params), grad_norm

        # Define epoch function
        def epoch_fn(
            epoch: Array,
            carry: tuple[
                OptState, Point[Natural, DifferentiableHMoG[ObsRep, LatRep]], Array
            ],
        ) -> tuple[OptState, Point[Natural, DifferentiableHMoG[ObsRep, LatRep]], Array]:
            opt_state, params, epoch_key = carry

            # Shuffle data and batch
            return_key, shuffle_key = jax.random.split(epoch_key)

            shuffled_indices = jax.random.permutation(shuffle_key, train_data.shape[0])
            batched_indices = shuffled_indices[: n_complete_batches * self.batch_size]
            leftover_indices = shuffled_indices[n_complete_batches * self.batch_size :]
            batched_data = train_data[batched_indices].reshape(
                (n_complete_batches, self.batch_size, -1)
            )
            leftover_data = train_data[leftover_indices]

            # Process batches
            (opt_state, new_params), grad_norms = jax.lax.scan(
                minibatch_step,
                (opt_state, params),
                batched_data,
            )

            (opt_state, new_params), grad_norm = minibatch_step(
                (opt_state, new_params), leftover_data
            )

            grad_norms = jnp.concatenate([grad_norms, jnp.array([grad_norm])])

            # Log metrics
            log_epoch_metrics(
                dataset,
                model,
                logger,
                new_params,
                epoch_offset + epoch,
                gradient_norms=grad_norms,
                log_freq=10,
            )

            return opt_state, new_params, return_key

        # Run training

        opt_state = optimizer.init(params2)
        (_, params3, _) = fori(0, self.n_epochs, epoch_fn, (opt_state, params2, key))

        return params3
