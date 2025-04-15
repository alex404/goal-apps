"""Trainers for HMoG model components."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import jax
import jax.numpy as jnp
import optax
from goal.geometry import (
    AffineMap,
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
    AnalyticLinearGaussianModel,
    DifferentiableMixture,
    Euclidean,
    FullNormal,
    Normal,
    differentiable_hmog,
)
from jax import Array

from apps.plugins import (
    ClusteringDataset,
)
from apps.runtime.handler import RunHandler
from apps.runtime.logger import JaxLogger

from .base import HMoG, fori, log_epoch_metrics, relative_entropy_regularization_full

### Constants ###


class MaskingStrategy(Enum):
    """Enum defining which parameters to update during training."""

    LGM = auto()  # Only update LGM parameters (obs_params and int_params)
    MIXTURE = auto()  # Only update mixture parameters (lat_params)
    FULL = auto()  # Update all parameters


# Start logger
log = logging.getLogger(__name__)


### Helpers ###


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


### Symmetric Gradient Trainer ###


@dataclass(frozen=True)
class GradientTrainer:
    """Base trainer for gradient-based training of HMoG models."""

    # Training hyperparameters
    n_epochs: int
    lr_init: float
    lr_final_ratio: float
    batch_size: int
    grad_clip: float

    # Regularization parameters
    l1_reg: float
    l2_reg: float
    re_reg: float

    # Parameter bounds
    min_prob: float
    obs_min_var: float
    lat_min_var: float
    obs_jitter: float
    lat_jitter: float

    # Strategy
    mask_type: MaskingStrategy

    def bound_parameters(
        self, model: HMoG, params: Point[Natural, HMoG]
    ) -> Point[Natural, HMoG]:
        """Apply bounds to parameters for numerical stability."""
        # Split parameters
        obs_params, int_params, lat_params = model.split_params(params)

        # Bound observable parameters
        with model.obs_man as om:
            obs_means = om.to_mean(obs_params)
            bounded_obs_means = om.regularize_covariance(
                obs_means, self.obs_jitter, self.obs_min_var
            )
            bounded_obs_params = om.to_natural(bounded_obs_means)

        # Additional bounds could be added here for other parameters

        # Rejoin parameters
        return model.join_params(bounded_obs_params, int_params, lat_params)

    def make_loss_fn(
        self, model: HMoG, batch: Array
    ) -> Callable[[Point[Natural, HMoG]], Array]:
        """Create a universal loss function for any HMoG model."""

        def loss_fn(params: Point[Natural, HMoG]) -> Array:
            # Core negative log-likelihood
            ce_loss = -model.average_log_observable_density(params, batch)

            # Extract components for regularization
            obs_params, int_params, _ = model.split_params(params)

            # L1 regularization on interaction matrix
            l1_loss = self.l1_reg * jnp.sum(jnp.abs(int_params.array))

            # L2 regularization on all parameters
            l2_loss = self.l2_reg * jnp.sum(jnp.square(params.array))

            # Relative entropy regularization
            lkl_params = model.lkl_man.join_params(obs_params, int_params)
            re_loss = (
                self.re_reg
                * relative_entropy_regularization_full(
                    model.lwr_hrm, batch, lkl_params
                )[1]
            )

            return ce_loss + l1_loss + l2_loss + re_loss

        return loss_fn

    def create_gradient_mask(
        self, model: HMoG
    ) -> Callable[[Point[Mean, HMoG]], Point[Mean, HMoG]]:
        """Create a function that masks gradients for specific training regimes."""
        if self.mask_type == MaskingStrategy.LGM:
            # Only update LGM parameters (obs_params and int_params)
            def mask_fn(grad: Point[Mean, HMoG]) -> Point[Mean, HMoG]:
                obs_grad, int_grad, lat_grad = model.split_params(grad)
                zero_lat_grad = model.lat_man.mean_point(jnp.zeros_like(lat_grad.array))
                return model.join_params(obs_grad, int_grad, zero_lat_grad)

        elif self.mask_type == MaskingStrategy.MIXTURE:
            # Only update mixture parameters (lat_params)
            def mask_fn(grad: Point[Mean, HMoG]) -> Point[Mean, HMoG]:
                obs_grad, int_grad, lat_grad = model.split_params(grad)
                zero_obs_grad = model.obs_man.point(jnp.zeros_like(obs_grad.array))
                zero_int_grad = model.int_man.point(jnp.zeros_like(int_grad.array))
                return model.join_params(zero_obs_grad, zero_int_grad, lat_grad)

        else:
            # No masking - update all parameters
            def mask_fn(grad: Point[Mean, HMoG]) -> Point[Mean, HMoG]:
                return grad

        return mask_fn

    def make_minibatch_step(
        self,
        handler: RunHandler,
        model: HMoG,
        logger: JaxLogger,
        optimizer: Optimizer[Natural, HMoG],
        mask_type: str,
    ) -> Callable[
        [tuple[OptState, Point[Natural, HMoG]], Array],
        tuple[tuple[OptState, Point[Natural, HMoG]], Array],
    ]:
        """Create step function for processing a single minibatch."""
        # Create gradient mask function based on training phase
        mask_gradient = self.create_gradient_mask(model)

        def minibatch_step(
            carry: tuple[OptState, Point[Natural, HMoG]],
            batch: Array,
        ) -> tuple[tuple[OptState, Point[Natural, HMoG]], Array]:
            opt_state, params = carry

            # Compute full gradient
            grad = model.grad(self.make_loss_fn(model, batch), params)

            # Apply gradient mask
            masked_grad = mask_gradient(grad)

            # Update parameters
            opt_state, new_params = optimizer.update(opt_state, masked_grad, params)

            # Apply parameter bounds
            bound_params = self.bound_parameters(model, new_params)

            # Monitor parameters for debugging
            logger.monitor_params(
                {
                    "original": params.array,
                    "post_update": new_params.array,
                    "post_bounds": bound_params.array,
                    "batch": batch,
                    "grad": grad.array,
                    "masked_grad": masked_grad.array,
                },
                handler,
                context=mask_type,
            )

            return (opt_state, bound_params), masked_grad.array

        return minibatch_step

    def train(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: HMoG,
        logger: JaxLogger,
        epoch_offset: int,
        params0: Point[Natural, HMoG],
        mask_type: str,
    ) -> Point[Natural, HMoG]:
        """Train the model with the specified gradient masking strategy."""
        n_epochs = self.n_epochs

        train_data = dataset.train_data
        n_batches = train_data.shape[0] // self.batch_size

        # Configure learning rate
        if self.lr_final_ratio < 1.0:
            lr_schedule = optax.cosine_decay_schedule(
                init_value=self.lr_init,
                decay_steps=n_epochs,
                alpha=self.lr_final_ratio,
            )
        else:
            lr_schedule = self.lr_init

        # Create optimizer
        optim = optax.adamw(learning_rate=lr_schedule)
        optimizer: Optimizer[Natural, HMoG] = Optimizer(optim, model)

        if self.grad_clip > 0.0:
            optimizer = optimizer.with_grad_clip(self.grad_clip)

        # Initialize optimizer state
        opt_state = optimizer.init(params0)

        # Log training phase
        log.info(f"Training {mask_type} parameters for {n_epochs} epochs")

        # Create epoch step function
        def epoch_step(
            epoch: Array,
            carry: tuple[OptState, Point[Natural, HMoG], Array],
        ) -> tuple[OptState, Point[Natural, HMoG], Array]:
            opt_state, params, epoch_key = carry

            # Split key for shuffling
            shuffle_key, next_key = jax.random.split(epoch_key)

            # Create minibatch step with the appropriate mask
            minibatch_step = self.make_minibatch_step(
                handler, model, logger, optimizer, mask_type
            )

            # Shuffle and batch data
            shuffled_indices = jax.random.permutation(shuffle_key, train_data.shape[0])
            batched_indices = shuffled_indices[: n_batches * self.batch_size]
            batched_data = train_data[batched_indices].reshape(
                (n_batches, self.batch_size, -1)
            )

            # Process all batches
            (opt_state, new_params), grads_array = jax.lax.scan(
                minibatch_step, (opt_state, params), batched_data
            )

            # Create batch gradients for logging
            batch_man = Replicated(model, grads_array.shape[0])
            batch_grads = batch_man.point(grads_array)

            # Log metrics
            log_epoch_metrics(
                dataset,
                model,
                logger,
                new_params,
                epoch + epoch_offset,
                batch_grads,
                log_freq=10,
            )

            return opt_state, new_params, next_key

        # Run training loop
        (_, params_final, _) = fori(0, n_epochs, epoch_step, (opt_state, params0, key))

        return params_final


type LinearModel[ObsRep: PositiveDefinite] = AffineMap[
    Rectangular, Euclidean, Euclidean, Normal[ObsRep]
]


@dataclass(frozen=True)
class LGMTrainer[ObsRep: PositiveDefinite]:
    """Standalone trainer for AnalyticLinearGaussianModel.

    This trainer provides a simplified way to pre-train an LGM component
    before integrating it into a more complex HMoG model.
    """

    # Training hyperparameters
    n_epochs: int
    batch_size: int
    lr_init: float
    lr_final_ratio: float = 0.1
    grad_clip: float = 10.0

    # Regularization parameters
    l1_reg: float = 0.0
    l2_reg: float = 1e-4

    # Parameter bounds
    min_var: float = 1e-6
    jitter: float = 0.0

    def bound_parameters(
        self,
        lgm: AnalyticLinearGaussianModel[ObsRep],
        lkl_params: Point[Natural, LinearModel[ObsRep]],
    ) -> Point[Natural, LinearModel[ObsRep]]:
        """Apply bounds to LGM parameters for numerical stability."""
        obs_params, int_params = lgm.lkl_man.split_params(lkl_params)

        # Bound observable variances
        with lgm.obs_man as om:
            obs_means = om.to_mean(obs_params)
            bounded_obs_means = om.regularize_covariance(
                obs_means, self.jitter, self.min_var
            )
            bounded_obs_params = om.to_natural(bounded_obs_means)

        # Rejoin parameters
        return lgm.lkl_man.join_params(bounded_obs_params, int_params)

    def make_loss_fn(
        self, lgm: AnalyticLinearGaussianModel[ObsRep], batch: Array
    ) -> Callable[[Point[Natural, AnalyticLinearGaussianModel[ObsRep]]], Array]:
        """Create a loss function for LGM training."""

        def loss_fn(
            params: Point[Natural, AnalyticLinearGaussianModel[ObsRep]],
        ) -> Array:
            # Core negative log-likelihood
            ce_loss = -lgm.average_log_observable_density(params, batch)

            # Split parameters for regularization
            _, int_params, _ = lgm.split_params(params)

            # L1 regularization on interaction matrix
            l1_loss = self.l1_reg * jnp.sum(jnp.abs(int_params.array))

            # L2 regularization on all parameters
            l2_loss = self.l2_reg * jnp.sum(jnp.square(params.array))

            return ce_loss + l1_loss + l2_loss

        return loss_fn

    def train(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        lgm: AnalyticLinearGaussianModel[ObsRep],
        logger: JaxLogger,
        lkl_params: Point[Natural, LinearModel[ObsRep]],
    ) -> Point[Natural, LinearModel[ObsRep]]:
        # Initialize parameters if not provided
        # Combine components
        ana_hmog = differentiable_hmog(
            lgm.obs_dim, lgm.obs_rep, lgm.lat_dim, PositiveDefinite, 10
        )
        mix_params = ana_hmog.upr_hrm.zeros()
        mix_grad = ana_hmog.upr_hrm.zeros()

        z = lgm.lat_man.to_natural(lgm.lat_man.standard_normal())

        train_data = dataset.train_data
        n_batches = train_data.shape[0] // self.batch_size

        # Configure learning rate
        if self.lr_final_ratio < 1.0:
            lr_schedule = optax.cosine_decay_schedule(
                init_value=self.lr_init,
                decay_steps=self.n_epochs,
                alpha=self.lr_final_ratio,
            )
        else:
            lr_schedule = self.lr_init

        # Create optimizer
        optim = optax.adamw(learning_rate=lr_schedule)
        optimizer: Optimizer[Natural, LinearModel[ObsRep]] = Optimizer(
            optim, lgm.lkl_man
        )

        if self.grad_clip > 0.0:
            optimizer = optimizer.with_grad_clip(self.grad_clip)

        # Initialize optimizer state
        opt_state = optimizer.init(lkl_params)

        log.info(f"Pre-training LGM for {self.n_epochs} epochs")

        # Define minibatch step
        def minibatch_step(
            carry: tuple[
                OptState,
                Point[
                    Natural,
                    LinearModel[ObsRep],
                ],
            ],
            batch: Array,
        ) -> tuple[
            tuple[
                OptState,
                Point[
                    Natural,
                    LinearModel[ObsRep],
                ],
            ],
            Array,
        ]:
            opt_state, lkl_params = carry

            params = lgm.join_conjugated(lkl_params, z)
            # Compute gradient
            grad = lgm.grad(self.make_loss_fn(lgm, batch), params)
            obs_grad, int_grad, _ = lgm.split_params(grad)
            lkl_grad = lgm.lkl_man.join_params(obs_grad, int_grad)

            # Update parameters
            opt_state, new_params = optimizer.update(opt_state, lkl_grad, lkl_params)

            # Apply parameter bounds
            bound_params = self.bound_parameters(lgm, new_params)

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
                context="lgm_pretrain",
            )

            return (opt_state, bound_params), grad.array

        # Define epoch step function
        def epoch_step(
            epoch: Array,
            carry: tuple[
                OptState,
                Point[Natural, LinearModel[ObsRep]],
                Array,
            ],
        ) -> tuple[
            OptState,
            Point[Natural, LinearModel[ObsRep]],
            Array,
        ]:
            opt_state, params, epoch_key = carry

            # Split key for shuffling
            shuffle_key, next_key = jax.random.split(epoch_key)

            # Shuffle and batch data
            shuffled_indices = jax.random.permutation(shuffle_key, train_data.shape[0])
            batched_indices = shuffled_indices[: n_batches * self.batch_size]
            batched_data = train_data[batched_indices].reshape(
                (n_batches, self.batch_size, -1)
            )

            # Process all batches
            (opt_state, new_params), lkl_grads_array = jax.lax.scan(
                minibatch_step, (opt_state, params), batched_data
            )

            # Function to convert LGM grad to padded HMoG grad
            def pad_lgm_grad(lkl_grad: Array) -> Array:
                return jnp.concatenate([lkl_grad, mix_grad.array])

            # Map this function over all batch gradients
            hmog_grads_array = jax.vmap(pad_lgm_grad)(lkl_grads_array)
            lgm_params = lgm.join_conjugated(new_params, z)
            (obs_params, int_params, lat_params) = lgm.split_params(lgm_params)
            _, lat_int_params, lat_lat_params = ana_hmog.upr_hrm.split_params(
                mix_params
            )
            new_mix_params = ana_hmog.upr_hrm.join_params(
                lat_params, lat_int_params, lat_lat_params
            )
            hmog_params = ana_hmog.join_params(obs_params, int_params, new_mix_params)

            # Create batch gradients for logging
            batch_man = Replicated(ana_hmog, hmog_grads_array.shape[0])
            batch_grads = batch_man.point(hmog_grads_array)

            # Log metrics
            log_epoch_metrics(
                dataset,
                ana_hmog,
                logger,
                hmog_params,
                epoch,
                batch_grads,
                log_freq=10,
            )

            return opt_state, new_params, next_key

        # Run training loop
        (_, lkl_params_final, _) = fori(
            0, self.n_epochs, epoch_step, (opt_state, lkl_params, key)
        )

        return lkl_params_final
