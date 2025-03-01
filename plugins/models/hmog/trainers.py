"""Trainers for HMoG model components."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, override

import jax
import jax.numpy as jnp
import optax
from goal.geometry import (
    AffineMap,
    Manifold,
    Mean,
    Natural,
    Optimizer,
    OptState,
    Point,
    PositiveDefinite,
    Rectangular,
    Replicated,
)
from goal.models import (
    DifferentiableHMoG,
    DifferentiableMixture,
    Euclidean,
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


def prepare_batches(
    key: Array, train_data: Array, n_complete_batches: int, batch_size: int
) -> tuple[Array, Array | None]:
    # Calculate batch info
    n_complete_batched = n_complete_batches * batch_size

    # Shuffle the data
    shuffled_indices = jax.random.permutation(key, train_data.shape[0])

    # Split into complete batches and remainder
    batched_indices = shuffled_indices[:n_complete_batched]
    leftover_indices = shuffled_indices[n_complete_batched:]

    # Create batches from complete portion
    batched_data = train_data[batched_indices].reshape(
        (n_complete_batches, batch_size, -1)
    )

    # Create remainder batch if any data is left
    leftover_batch = None
    if leftover_indices.shape[0] > 0:
        leftover_batch = train_data[leftover_indices]

    return batched_data, leftover_batch


def get_learning_rate_schedule(
    init_value: float, decay_steps: int, final_ratio: float
) -> optax.ScalarOrSchedule:
    """Create a learning rate schedule with optional cosine decay."""
    if final_ratio < 1.0:
        return optax.cosine_decay_schedule(
            init_value=init_value,
            decay_steps=decay_steps,
            alpha=final_ratio,
        )
    return init_value


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


def bound_observable_precisions[Rep: PositiveDefinite](
    model: AffineMap[Rectangular, Euclidean, Euclidean, Normal[Rep]],
    params: Point[Natural, AffineMap[Rectangular, Euclidean, Euclidean, Normal[Rep]]],
    max_prs: float,
    jitter: float,
) -> Point[Natural, AffineMap[Rectangular, Euclidean, Euclidean, Normal[Rep]]]:
    """Bound observable precisions."""
    obs_params, int_params = model.split_params(params)
    obs_man = model.cod_sub.sup_man
    obs_means = obs_man.to_mean(obs_params)
    bounded_obs_means = obs_man.regularize_covariance(obs_means, jitter, max_prs)
    bounded_obs_params = obs_man.to_natural(bounded_obs_means)
    return model.join_params(bounded_obs_params, int_params)


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
    batch_grads: None
    | Point[Mean, Replicated[DifferentiableHMoG[ObsRep, LatRep]]] = None,
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
        epoch_scaled_bic = (
            -(hmog_model.dim * jnp.log(n_samps) / n_samps - 2 * epoch_train_ll) / 2
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
            "Performance/Scaled BIC": (
                info,
                epoch_scaled_bic,
            ),
        }

        stats = jnp.array(STATS_LEVEL)

        def update_parameter_stats[M: Manifold](
            name: str, params: Point[Natural, M]
        ) -> None:
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
        obs_loc_params, obs_prs_params = hmog_model.obs_man.split_params(obs_params)
        lat_params, upr_int_params, cat_params = hmog_model.upr_hrm.split_params(
            upr_params
        )
        lat_loc_params, lat_prs_params = hmog_model.upr_hrm.obs_man.split_params(
            lat_params
        )

        update_parameter_stats("Obs Location", obs_loc_params)
        update_parameter_stats("Obs Precision", obs_prs_params)
        update_parameter_stats("Obs Interaction", lwr_int_params)
        update_parameter_stats("Lat Location", lat_loc_params)
        update_parameter_stats("Lat Precision", lat_prs_params)
        update_parameter_stats("Lat Interaction", upr_int_params)
        update_parameter_stats("Categorical", cat_params)

        metrics.update(
            {
                "Params/Loading Sparsity": (
                    stats,
                    jnp.mean(jnp.abs(lwr_int_params.array) < 1e-6),
                )
            }
        )

        ### Grad Norms ###

        def update_grad_stats[M: Manifold](name: str, grad_norms: Array) -> None:
            metrics.update(
                {
                    f"Grad Norms/{name} Min": (
                        stats,
                        jnp.min(grad_norms),
                    ),
                    f"Grad Norms/{name} Median": (
                        stats,
                        jnp.median(grad_norms),
                    ),
                    f"Grad Norms/{name} Max": (
                        stats,
                        jnp.max(grad_norms),
                    ),
                }
            )

        def norm_grads(grad: Point[Mean, DifferentiableHMoG[ObsRep, LatRep]]) -> Array:
            obs_grad, lwr_int_grad, upr_grad = hmog_model.split_params(grad)
            obs_loc_grad, obs_prs_grad = hmog_model.obs_man.split_params(obs_grad)
            lat_grad, upr_int_grad, cat_grad = hmog_model.upr_hrm.split_params(upr_grad)
            lat_loc_grad, lat_prs_grad = hmog_model.upr_hrm.obs_man.split_params(
                lat_grad
            )
            return jnp.asarray(
                [
                    jnp.linalg.norm(grad.array)
                    for grad in [
                        obs_loc_grad,
                        obs_prs_grad,
                        lwr_int_grad,
                        lat_loc_grad,
                        lat_prs_grad,
                        upr_int_grad,
                        cat_grad,
                    ]
                ]
            )

        if batch_grads is not None:
            batch_man: Replicated[DifferentiableHMoG[ObsRep, LatRep]] = Replicated(
                hmog_model, batch_grads.shape[0]
            )
            grad_norms = batch_man.map(norm_grads, batch_grads).T

            update_grad_stats("Obs Location", grad_norms[0])
            update_grad_stats("Obs Precision", grad_norms[1])
            update_grad_stats("Obs Interaction", grad_norms[2])
            update_grad_stats("Lat Location", grad_norms[3])
            update_grad_stats("Lat Precision", grad_norms[4])
            update_grad_stats("Lat Interaction", grad_norms[5])
            update_grad_stats("Categorical", grad_norms[6])

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
        epoch_offset: int,
        params0: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
    ) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]: ...


@dataclass(frozen=True)
class EMLGMTrainer[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    LGMTrainer[ObsRep, LatRep]
):
    """EM training for LinearGaussianModel. Uses a standard normal prior to speed up training."""

    @override
    def train(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: DifferentiableHMoG[ObsRep, LatRep],
        logger: JaxLogger,
        epoch_offset: int,
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


type LikelihoodModel[ObsRep: PositiveDefinite] = AffineMap[
    Rectangular, Euclidean, Euclidean, Normal[ObsRep]
]


@dataclass(frozen=True)
class GradientLGMTrainer[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    LGMTrainer[ObsRep, LatRep]
):
    """Gradient-based training for LinearGaussianModel Uses the given prior."""

    lr_init: float
    lr_final_ratio: float
    batch_size: int
    l1_reg: float
    l2_reg: float
    re_reg: float
    grad_clip: float

    @override
    def train(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: DifferentiableHMoG[ObsRep, LatRep],
        logger: JaxLogger,
        epoch_offset: int,
        params0: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
    ) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]:
        """Train LinearGaussianModel using gradient descent."""
        train_data = dataset.train_data
        lkl_params0, mix_params0 = model.split_conjugated(params0)

        n_complete_batches = train_data.shape[0] // self.batch_size

        # Initialize standard normal latent
        lr_schedule = get_learning_rate_schedule(
            self.lr_init, self.n_epochs, self.lr_final_ratio
        )

        # Configure optimizer
        optimizer: Optimizer[Natural, LikelihoodModel[ObsRep]] = Optimizer.adamw(
            model.lkl_man, learning_rate=lr_schedule, weight_decay=self.l2_reg
        )

        if self.grad_clip > 0.0:
            optimizer = optimizer.with_grad_clip(self.grad_clip)

        opt_state = optimizer.init(lkl_params0)

        # Define minibatch training step
        def minibatch_step(
            carry: tuple[OptState, Point[Natural, LikelihoodModel[ObsRep]]],
            batch: Array,
        ) -> tuple[tuple[OptState, Point[Natural, LikelihoodModel[ObsRep]]], Array]:
            # Define loss function for each minibatch
            def loss_fn(
                lkl_params: Point[Natural, LikelihoodModel[ObsRep]],
            ) -> Array:
                params = model.join_conjugated(lkl_params, mix_params0)
                ce_loss = -model.average_log_observable_density(params, batch)
                _, int_params = model.lkl_man.split_params(lkl_params)
                l1_loss = self.l1_reg * jnp.sum(jnp.abs(int_params.array))

                with model.lwr_hrm as lh:
                    z = lh.lat_man.to_natural(lh.lat_man.standard_normal())
                    lgm_params = lh.join_conjugated(lkl_params, z)
                    lgm_means = lh.expectation_step(lgm_params, batch)
                    lgm_lat_means = lh.split_params(lgm_means)[2]
                    re_loss = self.re_reg * lh.lat_man.relative_entropy(
                        lgm_lat_means, z
                    )

                return ce_loss + l1_loss + re_loss

            opt_state, lkl_params = carry
            grad = model.lkl_man.grad(loss_fn, lkl_params)

            opt_state, new_lkl_params = optimizer.update(opt_state, grad, lkl_params)

            # Extract likelihood parameters and rejoin with constant latent z

            bounded_lkl_params = bound_observable_precisions(
                model.lkl_man, new_lkl_params, self.min_var, self.jitter
            )

            # Monitor parameters for debugging
            logger.monitor_params(
                {
                    "original": lkl_params.array,
                    "post_update": new_lkl_params.array,
                    "post_bounds": bounded_lkl_params.array,
                    "batch": batch,
                    "grad": grad.array,
                },
                handler,
                context="lgm_grad_step",
            )

            return (opt_state, bounded_lkl_params), grad.array

        # Define epoch function
        def epoch_step(
            epoch: Array,
            carry: tuple[OptState, Point[Natural, LikelihoodModel[ObsRep]], Array],
        ) -> tuple[OptState, Point[Natural, LikelihoodModel[ObsRep]], Array]:
            opt_state, lkl_params, epoch_key = carry

            # Shuffle data and batch
            return_key, shuffle_key = jax.random.split(epoch_key)

            batched_data, _ = prepare_batches(
                shuffle_key,
                train_data,
                n_complete_batches,
                self.batch_size,
            )

            # Process main batches
            (opt_state, new_lkl_params), grads_array = jax.lax.scan(
                minibatch_step,
                (opt_state, lkl_params),
                batched_data,
            )

            batch_man: Replicated[LikelihoodModel[ObsRep]] = Replicated(
                model.lkl_man, grads_array.shape[0]
            )

            lkl_batch_grads: Point[Mean, ...] = batch_man.point(grads_array)

            def pad_grad(
                lkl_grad: Point[Mean, LikelihoodModel[ObsRep]],
            ) -> Point[Mean, DifferentiableHMoG[ObsRep, LatRep]]:
                # zeros like mix
                mix_grad = model.upr_hrm.mean_point(jnp.zeros_like(mix_params0.array))
                obs_means, int_means = model.lkl_man.split_params(lkl_grad)
                return model.join_params(obs_means, int_means, mix_grad)

            batch_grads = batch_man.man_map(pad_grad, lkl_batch_grads)

            # Extract likelihood for full model evaluation
            params = model.join_conjugated(lkl_params, mix_params0)

            # Log metrics
            log_epoch_metrics(
                dataset,
                model,
                logger,
                params,
                epoch + epoch_offset,
                batch_grads,
                10,
            )

            return opt_state, new_lkl_params, return_key

        # Run training loop
        (_, lkl_params1, _) = fori(
            0, self.n_epochs, epoch_step, (opt_state, lkl_params0, key)
        )

        # Return only the likelihood parameters
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
    l2_reg: float
    grad_clip: float

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

        lr_schedule = get_learning_rate_schedule(
            self.lr_init, self.n_epochs, self.lr_final_ratio
        )

        optimizer: Optimizer[
            Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]
        ] = Optimizer.adamw(
            model.upr_hrm, learning_rate=lr_schedule, weight_decay=self.l2_reg
        )

        if self.grad_clip > 0.0:
            optimizer = optimizer.with_grad_clip(self.grad_clip)

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
            return ((opt_state, bound_mix_params), grad.array)

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

            batched_data, _ = prepare_batches(
                shuffle_key,
                train_data,
                n_complete_batches,
                self.batch_size,
            )

            # Process batches
            (opt_state, new_mix_params), grads_array = jax.lax.scan(
                minibatch_step,
                (opt_state, mix_params),
                batched_data,
            )

            batch_man: Replicated[DifferentiableMixture[FullNormal, Normal[LatRep]]] = (
                Replicated(model.upr_hrm, grads_array.shape[0])
            )

            mix_batch_grads: Point[Mean, ...] = batch_man.point(grads_array)

            # Create padded HMOG gradients
            def pad_grad(
                mix_grad: Point[
                    Mean, DifferentiableMixture[FullNormal, Normal[LatRep]]
                ],
            ) -> Point[Mean, DifferentiableHMoG[ObsRep, LatRep]]:
                # Create zero LGM gradient
                lkl_grad: Point[Mean, ...] = model.lkl_man.point(
                    jnp.zeros_like(lkl_params1.array)
                )
                obs_grad, int_grad = model.lkl_man.split_params(lkl_grad)
                return model.join_params(obs_grad, int_grad, mix_grad)

            batch_grads = batch_man.man_map(pad_grad, mix_batch_grads)

            # Log metrics
            new_params = model.join_conjugated(lkl_params1, new_mix_params)
            log_epoch_metrics(
                dataset,
                model,
                logger,
                new_params,
                epoch_offset + epoch,
                batch_grads,
                10,
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
    batch_size: int
    l1_reg: float
    l2_reg: float
    re_reg: float
    grad_clip: float

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

        lr_schedule = get_learning_rate_schedule(
            self.lr_init, self.n_epochs, self.lr_final_ratio
        )

        optimizer: Optimizer[Natural, DifferentiableHMoG[ObsRep, LatRep]] = (
            Optimizer.adamw(model, learning_rate=lr_schedule, weight_decay=self.l2_reg)
        )
        if self.grad_clip > 0.0:
            optimizer = optimizer.with_grad_clip(self.grad_clip)

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
                ce_loss = -model.average_log_observable_density(params, batch)
                obs_params, int_params, _ = model.split_params(params)

                l1_loss = self.l1_reg * jnp.sum(jnp.abs(int_params.array))
                with model.lwr_hrm as lh:
                    z = lh.lat_man.to_natural(lh.lat_man.standard_normal())
                    lkl_params = lh.lkl_man.join_params(obs_params, int_params)
                    lgm_params = lh.join_conjugated(lkl_params, z)
                    lgm_means = lh.expectation_step(lgm_params, batch)
                    lgm_lat_means = lh.split_params(lgm_means)[2]
                    re_loss = self.re_reg * lh.lat_man.relative_entropy(
                        lgm_lat_means, z
                    )

                return ce_loss + l1_loss + re_loss

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

            return (opt_state, bound_params), grad.array

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

            batched_data, leftover_data = prepare_batches(
                shuffle_key,
                train_data,
                n_complete_batches,
                self.batch_size,
            )

            # Process batches
            (opt_state, new_params), grads_array = jax.lax.scan(
                minibatch_step,
                (opt_state, params),
                batched_data,
            )

            if leftover_data is not None:
                (opt_state, new_params), grad_array = minibatch_step(
                    (opt_state, new_params), leftover_data
                )

                grads_array = jnp.concatenate([grads_array, jnp.array([grad_array])])

            batch_man: Replicated[DifferentiableHMoG[ObsRep, LatRep]] = Replicated(
                model, grads_array.shape[0]
            )

            batch_grads: Point[Mean, Replicated[DifferentiableHMoG[ObsRep, LatRep]]] = (
                batch_man.point(grads_array)
            )

            # Log metrics
            log_epoch_metrics(
                dataset,
                model,
                logger,
                new_params,
                epoch_offset + epoch,
                batch_grads,
                log_freq=10,
            )

            return opt_state, new_params, return_key

        # Run training

        opt_state = optimizer.init(params2)
        (_, params3, _) = fori(0, self.n_epochs, epoch_fn, (opt_state, params2, key))

        return params3
