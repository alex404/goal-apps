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
    Null,
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


### Gradient Trainer ###


@dataclass(frozen=True)
class GradientTrainer[
    ObsRep: PositiveDefinite,
    LatRep: PositiveDefinite,
    Model: Manifold,
    Masked: Manifold,
](ABC):
    """Base trainer for gradient-based training of HMoG models."""

    n_epochs: int
    lr_init: float
    lr_final_ratio: float
    batch_size: int
    l2_reg: float
    grad_clip: float

    @abstractmethod
    def model(self, hmog_model: DifferentiableHMoG[ObsRep, LatRep]) -> Model: ...

    @abstractmethod
    def masked_model(
        self, hmog_model: DifferentiableHMoG[ObsRep, LatRep]
    ) -> Masked: ...

    @abstractmethod
    def bound_parameters(
        self,
        model: Model,
        params: Point[Natural, Model],
    ) -> Point[Natural, Model]: ...

    @abstractmethod
    def make_loss_fn(
        self,
        hmog_model: DifferentiableHMoG[ObsRep, LatRep],
        masked_params: Point[Natural, Masked],
        batch: Array,
    ) -> Callable[[Point[Natural, Model]], Array]: ...

    @abstractmethod
    def make_pad_grad(
        self,
        hmog_model: DifferentiableHMoG[ObsRep, LatRep],
    ) -> Callable[
        [Point[Mean, Model]], Point[Mean, DifferentiableHMoG[ObsRep, LatRep]]
    ]: ...

    @abstractmethod
    def to_hmog_params(
        self,
        hmog_model: DifferentiableHMoG[ObsRep, LatRep],
        params: Point[Natural, Model],
        masked_params: Point[Natural, Masked],
    ) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]: ...

    @abstractmethod
    def from_hmog_params(
        self,
        hmog_model: DifferentiableHMoG[ObsRep, LatRep],
        params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
    ) -> tuple[Point[Natural, Model], Point[Natural, Masked]]: ...

    def make_minibatch_step(
        self,
        handler: RunHandler,
        hmog_model: DifferentiableHMoG[ObsRep, LatRep],
        logger: JaxLogger,
        optimizer: Optimizer[Natural, Model],
        masked_params: Point[Natural, Masked],
    ) -> Callable[
        [tuple[OptState, Point[Natural, Model]], Array],
        tuple[tuple[OptState, Point[Natural, Model]], Array],
    ]:
        model = self.model(hmog_model)

        def minibatch_step(
            carry: tuple[OptState, Point[Natural, Model]],
            batch: Array,
        ) -> tuple[tuple[OptState, Point[Natural, Model]], Array]:
            opt_state, params = carry
            grad = model.grad(
                self.make_loss_fn(hmog_model, masked_params, batch), params
            )

            opt_state, new_params = optimizer.update(opt_state, grad, params)
            bound_params = self.bound_parameters(model, new_params)

            # Monitor parameters for debugging
            logger.monitor_params(
                {
                    "original": params.array,
                    "post_update": new_params.array,
                    "post_bounds": bound_params.array,
                    "batch": batch,
                    "grad": grad.array,
                },
                handler,
            )

            return (opt_state, bound_params), grad.array

        return minibatch_step

    # Define epoch function
    def make_epoch_step(
        self,
        handler: RunHandler,
        dataset: ClusteringDataset,
        hmog_model: DifferentiableHMoG[ObsRep, LatRep],
        logger: JaxLogger,
        optimizer: Optimizer[Natural, Model],
        masked_params: Point[Natural, Masked],
        epoch_offset: int,
        n_batches: int,
    ) -> Callable[
        [Array, tuple[OptState, Point[Natural, Model], Array]],
        tuple[OptState, Point[Natural, Model], Array],
    ]:
        def epoch_step(
            epoch: Array,
            carry: tuple[OptState, Point[Natural, Model], Array],
        ) -> tuple[OptState, Point[Natural, Model], Array]:
            opt_state, params, epoch_key = carry

            model = self.model(hmog_model)

            # Shuffle data and batch
            return_key, shuffle_key = jax.random.split(epoch_key)

            minibatch_step = self.make_minibatch_step(
                handler, hmog_model, logger, optimizer, masked_params
            )

            shuffled_indices = jax.random.permutation(
                shuffle_key, dataset.train_data.shape[0]
            )
            batched_indices = shuffled_indices[: n_batches * self.batch_size]
            batched_data = dataset.train_data[batched_indices].reshape(
                (n_batches, self.batch_size, -1)
            )

            # Process main batches
            (opt_state, new_params), grads_array = jax.lax.scan(
                minibatch_step,
                (opt_state, params),
                batched_data,
            )

            batch_man: Replicated[Model] = Replicated(model, grads_array.shape[0])
            batch_grads: Point[Mean, ...] = batch_man.point(grads_array)
            pad_grad = self.make_pad_grad(hmog_model)
            batch_grads = batch_man.man_map(pad_grad, batch_grads)

            # Extract likelihood for full model evaluation
            hmog_params = self.to_hmog_params(hmog_model, new_params, masked_params)

            # Log metrics
            log_epoch_metrics(
                dataset,
                hmog_model,
                logger,
                hmog_params,
                epoch + epoch_offset,
                batch_grads,
                10,
            )

            return opt_state, new_params, return_key

        return epoch_step

    def train(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        hmog_model: DifferentiableHMoG[ObsRep, LatRep],
        logger: JaxLogger,
        epoch_offset: int,
        hmog_params0: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
    ) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]:
        """Train LinearGaussianModel using gradient descent."""

        model = self.model(hmog_model)
        train_data = dataset.train_data
        params0, masked_params0 = self.from_hmog_params(hmog_model, hmog_params0)

        n_batches = train_data.shape[0] // self.batch_size

        lr_schedule = self.lr_init

        # Initialize standard normal latent
        if self.lr_final_ratio < 1.0:
            lr_schedule = optax.cosine_decay_schedule(
                init_value=self.lr_init,
                decay_steps=self.n_epochs,
                alpha=self.lr_final_ratio,
            )
        # Configure optimizer
        optimizer: Optimizer[Natural, Model] = Optimizer.adamw(
            model, learning_rate=lr_schedule, weight_decay=self.l2_reg
        )

        if self.grad_clip > 0.0:
            optimizer = optimizer.with_grad_clip(self.grad_clip)

        opt_state = optimizer.init(params0)

        epoch_step = self.make_epoch_step(
            handler,
            dataset,
            hmog_model,
            logger,
            optimizer,
            masked_params0,
            epoch_offset,
            n_batches,
        )

        # Run training loop
        (_, params1, _) = fori(0, self.n_epochs, epoch_step, (opt_state, params0, key))

        # Return only the likelihood parameters
        return self.to_hmog_params(hmog_model, params1, masked_params0)


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
        hmog_model: DifferentiableHMoG[ObsRep, LatRep],
        logger: JaxLogger,
        epoch_offset: int,
        hmog_params0: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
    ) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]: ...


type LikelihoodModel[ObsRep: PositiveDefinite] = AffineMap[
    Rectangular, Euclidean, Euclidean, Normal[ObsRep]
]


@dataclass(frozen=True)
class GradientLGMTrainer[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    GradientTrainer[
        ObsRep,
        LatRep,
        LikelihoodModel[ObsRep],  # Model being trained (just the likelihood part)
        DifferentiableMixture[FullNormal, Normal[LatRep]],  # Masked part (mixture)
    ],
    LGMTrainer[ObsRep, LatRep],
):
    """Gradient-based training for LinearGaussianModel."""

    min_var: float
    jitter: float
    l1_reg: float
    re_reg: float

    @override
    def model(
        self, hmog_model: DifferentiableHMoG[ObsRep, LatRep]
    ) -> LikelihoodModel[ObsRep]:
        return hmog_model.lkl_man

    @override
    def masked_model(
        self, hmog_model: DifferentiableHMoG[ObsRep, LatRep]
    ) -> DifferentiableMixture[FullNormal, Normal[LatRep]]:
        return hmog_model.upr_hrm

    @override
    def bound_parameters(
        self,
        model: LikelihoodModel[ObsRep],
        params: Point[Natural, LikelihoodModel[ObsRep]],
    ) -> Point[Natural, LikelihoodModel[ObsRep]]:
        """Bound observable precisions."""
        obs_params, int_params = model.split_params(params)
        obs_man = model.cod_sub.sup_man
        obs_means = obs_man.to_mean(obs_params)
        bounded_obs_means = obs_man.regularize_covariance(
            obs_means, self.jitter, self.min_var
        )
        bounded_obs_params = obs_man.to_natural(bounded_obs_means)
        return model.join_params(bounded_obs_params, int_params)

    @override
    def make_loss_fn(
        self,
        hmog_model: DifferentiableHMoG[ObsRep, LatRep],
        masked_params: Point[
            Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]
        ],
        batch: Array,
    ) -> Callable[[Point[Natural, LikelihoodModel[ObsRep]]], Array]:
        def _loss_fn(lkl_params: Point[Natural, LikelihoodModel[ObsRep]]) -> Array:
            # Join likelihood and mixture to evaluate full model
            params = hmog_model.join_conjugated(lkl_params, masked_params)

            # Cross-entropy loss
            ce_loss = -hmog_model.average_log_observable_density(params, batch)

            # L1 regularization on interaction matrix
            _, int_params = hmog_model.lkl_man.split_params(lkl_params)
            l1_loss = self.l1_reg * jnp.sum(jnp.abs(int_params.array))

            # Relative entropy regularization
            with hmog_model.lwr_hrm as lh:
                z = lh.lat_man.to_natural(lh.lat_man.standard_normal())
                lgm_params = lh.join_conjugated(lkl_params, z)
                lgm_means = lh.expectation_step(lgm_params, batch)
                lgm_lat_means = lh.split_params(lgm_means)[2]
                re_loss = self.re_reg * lh.lat_man.relative_entropy(lgm_lat_means, z)

            return ce_loss + l1_loss + re_loss

        return _loss_fn

    @override
    def make_pad_grad(
        self,
        hmog_model: DifferentiableHMoG[ObsRep, LatRep],
    ) -> Callable[
        [Point[Mean, LikelihoodModel[ObsRep]]],
        Point[Mean, DifferentiableHMoG[ObsRep, LatRep]],
    ]:
        def pad_grad(
            grad: Point[Mean, LikelihoodModel[ObsRep]],
        ) -> Point[Mean, DifferentiableHMoG[ObsRep, LatRep]]:
            """Pad LGM gradient with zeros for mixture component."""
            # Create zero gradient for mixture component
            mix_grad: Point[Mean, ...] = self.masked_model(hmog_model).point(
                jnp.zeros(self.masked_model(hmog_model).dim)
            )

            obs_grad, int_grad = hmog_model.lkl_man.split_params(grad)
            return hmog_model.join_params(obs_grad, int_grad, mix_grad)

        return pad_grad

    @override
    def to_hmog_params(
        self,
        hmog_model: DifferentiableHMoG[ObsRep, LatRep],
        params: Point[Natural, LikelihoodModel[ObsRep]],
        masked_params: Point[
            Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]
        ],
    ) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]:
        """Convert trained LGM parameters and fixed mixture parameters to full HMoG parameters."""
        return hmog_model.join_conjugated(params, masked_params)

    @override
    def from_hmog_params(
        self,
        hmog_model: DifferentiableHMoG[ObsRep, LatRep],
        params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
    ) -> tuple[
        Point[Natural, LikelihoodModel[ObsRep]],
        Point[Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]],
    ]:
        """Extract LGM parameters and mixture parameters from full HMoG parameters."""
        lkl_params, mix_params = hmog_model.split_conjugated(params)

        return lkl_params, mix_params


@dataclass(frozen=True)
class GradientLGMPretrainer[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    GradientLGMTrainer[
        ObsRep,
        LatRep,
    ],
    LGMTrainer[ObsRep, LatRep],
):
    @override
    def make_loss_fn(
        self,
        hmog_model: DifferentiableHMoG[ObsRep, LatRep],
        masked_params: Point[
            Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]
        ],
        batch: Array,
    ) -> Callable[[Point[Natural, LikelihoodModel[ObsRep]]], Array]:
        def _loss_fn(lkl_params: Point[Natural, LikelihoodModel[ObsRep]]) -> Array:
            # Join likelihood and mixture to evaluate full model
            lgm = hmog_model.lwr_hrm
            z = lgm.lat_man.to_natural(lgm.lat_man.standard_normal())
            params = lgm.join_conjugated(lkl_params, z)

            # Cross-entropy loss
            ce_loss = -lgm.average_log_observable_density(params, batch)

            # L1 regularization on interaction matrix
            _, int_params = lgm.lkl_man.split_params(lkl_params)
            l1_loss = self.l1_reg * jnp.sum(jnp.abs(int_params.array))

            return ce_loss + l1_loss

        return _loss_fn


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
        hmog_model: DifferentiableHMoG[ObsRep, LatRep],
        logger: JaxLogger,
        epoch_offset: int,
        hmog_params0: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
    ) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]: ...


@dataclass(frozen=True)
class GradientMixtureTrainer[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    GradientTrainer[
        ObsRep,
        LatRep,
        DifferentiableMixture[FullNormal, Normal[LatRep]],
        LikelihoodModel[ObsRep],
    ],
    MixtureTrainer[ObsRep, LatRep],
):
    """Gradient-based training for mixture components."""

    min_prob: float
    min_var: float
    jitter: float

    @override
    def model(
        self, hmog_model: DifferentiableHMoG[ObsRep, LatRep]
    ) -> DifferentiableMixture[FullNormal, Normal[LatRep]]:
        return hmog_model.upr_hrm

    @override
    def masked_model(
        self, hmog_model: DifferentiableHMoG[ObsRep, LatRep]
    ) -> LikelihoodModel[ObsRep]:
        return hmog_model.lkl_man

    @override
    def bound_parameters(
        self,
        model: DifferentiableMixture[FullNormal, Normal[LatRep]],
        params: Point[Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]],
    ) -> Point[Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]]:
        return bound_mixture_parameters(
            model, params, self.min_prob, self.min_var, self.jitter
        )

    @override
    def make_loss_fn(
        self,
        hmog_model: DifferentiableHMoG[ObsRep, LatRep],
        masked_params: Point[Natural, LikelihoodModel[ObsRep]],
        batch: Array,
    ) -> Callable[
        [Point[Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]]], Array
    ]:
        def _loss_fn(
            mix_params: Point[
                Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]
            ],
        ) -> Array:
            # Join likelihood and mixture to evaluate full model
            params = hmog_model.join_conjugated(masked_params, mix_params)

            # Cross-entropy loss (negative log-likelihood)
            return -hmog_model.average_log_observable_density(params, batch)

        return _loss_fn

    @override
    def make_pad_grad(
        self,
        hmog_model: DifferentiableHMoG[ObsRep, LatRep],
    ) -> Callable[
        [Point[Mean, DifferentiableMixture[FullNormal, Normal[LatRep]]]],
        Point[Mean, DifferentiableHMoG[ObsRep, LatRep]],
    ]:
        masked_model = self.masked_model(hmog_model)

        def pad_grad(
            grad: Point[Mean, DifferentiableMixture[FullNormal, Normal[LatRep]]],
        ) -> Point[Mean, DifferentiableHMoG[ObsRep, LatRep]]:
            """Pad mixture gradient with zeros for LGM component."""
            # Create zero gradient for LGM component
            lkl_grad: Point[Mean, ...] = masked_model.point(jnp.zeros(masked_model.dim))

            # Extract components
            obs_grad, int_grad = masked_model.split_params(lkl_grad)

            # Join into full HMoG gradient
            return hmog_model.join_params(obs_grad, int_grad, grad)

        return pad_grad

    @override
    def to_hmog_params(
        self,
        hmog_model: DifferentiableHMoG[ObsRep, LatRep],
        params: Point[Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]],
        masked_params: Point[Natural, LikelihoodModel[ObsRep]],
    ) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]:
        """Convert trained mixture parameters and fixed LGM parameters to full HMoG parameters."""
        return hmog_model.join_conjugated(masked_params, params)

    @override
    def from_hmog_params(
        self,
        hmog_model: DifferentiableHMoG[ObsRep, LatRep],
        params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
    ) -> tuple[
        Point[Natural, DifferentiableMixture[FullNormal, Normal[LatRep]]],
        Point[Natural, LikelihoodModel[ObsRep]],
    ]:
        """Extract mixture parameters and LGM parameters from full HMoG parameters."""
        lkl_params, mix_params = hmog_model.split_conjugated(params)

        return mix_params, lkl_params


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
        hmog_model: DifferentiableHMoG[ObsRep, LatRep],
        logger: JaxLogger,
        epoch_offset: int,
        hmog_params0: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
    ) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]: ...


@dataclass(frozen=True)
class GradientFullModelTrainer[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
    GradientTrainer[
        ObsRep,
        LatRep,
        DifferentiableHMoG[ObsRep, LatRep],
        Null,
    ],
    FullModelTrainer[ObsRep, LatRep],
):
    """Gradient-based training for full HMoG model."""

    min_prob: float
    obs_min_var: float
    lat_min_var: float
    obs_jitter: float
    lat_jitter: float
    l1_reg: float
    re_reg: float

    @override
    def model(
        self, hmog_model: DifferentiableHMoG[ObsRep, LatRep]
    ) -> DifferentiableHMoG[ObsRep, LatRep]:
        return hmog_model

    @override
    def masked_model(self, hmog_model: DifferentiableHMoG[ObsRep, LatRep]) -> Null:
        return Null()

    @override
    def bound_parameters(
        self,
        model: DifferentiableHMoG[ObsRep, LatRep],
        params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
    ) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]:
        """Apply all parameter bounds to HMoG."""
        # Split parameters
        lkl_params, mix_params = model.split_conjugated(params)

        # Bound mixture parameters
        bounded_mix_params = bound_mixture_parameters(
            model.upr_hrm, mix_params, self.min_prob, self.lat_min_var, self.lat_jitter
        )

        # Bound observable parameters
        obs_params, int_params = model.lkl_man.split_params(lkl_params)
        obs_means = model.obs_man.to_mean(obs_params)
        bounded_obs_means = model.obs_man.regularize_covariance(
            obs_means, self.obs_jitter, self.obs_min_var
        )
        bounded_obs_params = model.obs_man.to_natural(bounded_obs_means)
        bounded_lkl_params = model.lkl_man.join_params(bounded_obs_params, int_params)

        return model.join_conjugated(bounded_lkl_params, bounded_mix_params)

    @override
    def make_loss_fn(
        self,
        hmog_model: DifferentiableHMoG[ObsRep, LatRep],
        masked_params: Point[Natural, Null],
        batch: Array,
    ) -> Callable[[Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]], Array]:
        def _loss_fn(
            params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
        ) -> Array:
            # Cross-entropy loss
            ce_loss = -hmog_model.average_log_observable_density(params, batch)

            # L1 regularization on interaction matrix
            obs_params, int_params, _ = hmog_model.split_params(params)
            l1_loss = self.l1_reg * jnp.sum(jnp.abs(int_params.array))

            # Relative entropy regularization
            with hmog_model.lwr_hrm as lh:
                z = lh.lat_man.to_natural(lh.lat_man.standard_normal())
                lkl_params = lh.lkl_man.join_params(obs_params, int_params)
                lgm_params = lh.join_conjugated(lkl_params, z)
                lgm_means = lh.expectation_step(lgm_params, batch)
                lgm_lat_means = lh.split_params(lgm_means)[2]
                re_loss = self.re_reg * lh.lat_man.relative_entropy(lgm_lat_means, z)

            return ce_loss + l1_loss + re_loss

        return _loss_fn

    @override
    def make_pad_grad(
        self,
        hmog_model: DifferentiableHMoG[ObsRep, LatRep],
    ) -> Callable[
        [Point[Mean, DifferentiableHMoG[ObsRep, LatRep]]],
        Point[Mean, DifferentiableHMoG[ObsRep, LatRep]],
    ]:
        """Identity function since we're already working with the full HMoG gradient."""

        def pad_grad(
            grad: Point[Mean, DifferentiableHMoG[ObsRep, LatRep]],
        ) -> Point[Mean, DifferentiableHMoG[ObsRep, LatRep]]:
            """Identity function since we're already working with the full HMoG gradient."""
            return grad

        return pad_grad

    @override
    def to_hmog_params(
        self,
        hmog_model: DifferentiableHMoG[ObsRep, LatRep],
        params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
        masked_params: Point[Natural, Null],
    ) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]:
        """Identity function since we're already working with full HMoG parameters."""
        return params

    @override
    def from_hmog_params(
        self,
        hmog_model: DifferentiableHMoG[ObsRep, LatRep],
        params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
    ) -> tuple[
        Point[Natural, DifferentiableHMoG[ObsRep, LatRep]], Point[Natural, Null]
    ]:
        """Return the full parameters without splitting."""
        return params, Null().point(jnp.asarray([]))


# @dataclass(frozen=True)
# class EMLGMTrainer[ObsRep: PositiveDefinite, LatRep: PositiveDefinite](
#     LGMTrainer[ObsRep, LatRep]
# ):
#     """EM training for LinearGaussianModel. Uses a standard normal prior to speed up training."""
#
#     @override
#     def train(
#         self,
#         key: Array,
#         handler: RunHandler,
#         dataset: ClusteringDataset,
#         model: DifferentiableHMoG[ObsRep, LatRep],
#         logger: JaxLogger,
#         epoch_offset: int,
#         params0: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
#     ) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]:
#         # Standard normal latent for LGM
#         train_data = dataset.train_data
#         lkl_params0, mix_params0 = model.split_conjugated(params0)
#
#         with model.lwr_hrm as lh:
#
#             def step(
#                 epoch: Array, lgm_params: Point[Natural, LinearGaussianModel[ObsRep]]
#             ) -> Point[Natural, LinearGaussianModel[ObsRep]]:
#                 lgm_means = lh.expectation_step(lgm_params, train_data)
# def bound_observable_variances[Rep: PositiveDefinite](
#     model: LinearGaussianModel[Rep],
#     means: Point[Mean, LinearGaussianModel[Rep]],
#     min_var: float,
#     jitter: float,
# ) -> Point[Mean, LinearGaussianModel[Rep]]:
#     """Regularize observable covariance parameters in mean coordinates."""
#     obs_means, int_means, lat_means = model.split_params(means)
#     bounded_obs_means = model.obs_man.regularize_covariance(obs_means, jitter, min_var)
#     return model.join_params(bounded_obs_means, int_means, lat_means)
#


#                 lgm_means = bound_observable_variances(
#                     lh, lgm_means, self.min_var, self.jitter
#                 )
#                 new_lgm_params = lh.to_natural(lgm_means)
#                 lkl_params = lh.likelihood_function(new_lgm_params)
#                 lgm_params = lh.join_conjugated(lkl_params, z)
#
#                 # Create full HMOG params for evaluation
#                 hmog_params = model.join_conjugated(lkl_params, mix_params0)
#                 log_epoch_metrics(
#                     dataset,
#                     model,
#                     logger,
#                     hmog_params,
#                     epoch,
#                 )
#
#                 logger.monitor_params(
#                     {
#                         "original": lgm_params.array,
#                         "post_update": new_lgm_params.array,
#                     },
#                     handler,
#                     context="stage1_epoch",
#                 )
#                 return lgm_params
#
#             z = lh.lat_man.to_natural(lh.lat_man.standard_normal())
#             lgm_params0 = lh.join_conjugated(lkl_params0, z)
#
#             lgm_params1 = fori(0, self.n_epochs, step, lgm_params0)
#
#             lkl_params1 = lh.likelihood_function(lgm_params1)
#             return model.join_conjugated(lkl_params1, mix_params0)
