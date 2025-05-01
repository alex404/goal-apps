"""Trainers for HMoG model components."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

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
    Euclidean,
    Normal,
    differentiable_hmog,
)
from jax import Array

from apps.plugins import (
    ClusteringDataset,
)
from apps.runtime.handler import RunHandler
from apps.runtime.logger import JaxLogger

from .base import HMoG, fori, log_epoch_metrics

### Constants ###


class MaskingStrategy(Enum):
    """Enum defining which parameters to update during training."""

    LGM = auto()  # Only update LGM parameters (obs_params and int_params)
    MIXTURE = auto()  # Only update mixture parameters (lat_params)
    FULL = auto()  # Update all parameters


# Start logger
log = logging.getLogger(__name__)


### Helpers ###


### Symmetric Gradient Trainer ###


@dataclass(frozen=True)
class GradientTrainer:
    """Base trainer for gradient-based training of HMoG models."""

    # Training hyperparameters
    n_epochs: int
    lr_init: float
    lr_final_ratio: float
    batch_size: None | int
    batch_steps: int
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

    def bound_means(self, model: HMoG, means: Point[Mean, HMoG]) -> Point[Mean, HMoG]:
        """Apply bounds to posterior statistics for numerical stability."""
        # Split posterior statistics
        obs_means, int_means, lat_means = model.split_params(means)

        # For observable parameters, bound the variances
        bounded_obs_means = model.lwr_hrm.obs_man.regularize_covariance(
            obs_means, self.obs_jitter, self.obs_min_var
        )

        # Bound latent parameters
        with model.upr_hrm as uh:
            # Split latent parameters
            comp_meanss, prob_means = uh.split_mean_mixture(lat_means)
            lat_obs_means, _, _ = model.upr_hrm.split_params(lat_means)

            def whiten_and_bound(
                comp_means: Point[Mean, Normal[Any]],
            ) -> Point[Mean, Normal[Any]]:
                whitened_comp_means = uh.obs_man.whiten(comp_means, lat_obs_means)
                return uh.obs_man.regularize_covariance(
                    whitened_comp_means, self.obs_jitter, self.obs_min_var
                )

            bounded_comp_meanss = uh.cmp_man.man_map(whiten_and_bound, comp_meanss)

            probs = uh.lat_man.to_probs(prob_means)
            bounded_probs0 = jnp.clip(probs, self.min_prob, 1.0 - self.min_prob)
            bounded_probs = bounded_probs0 / jnp.sum(bounded_probs0)
            bounded_prob_means = uh.lat_man.from_probs(bounded_probs)

            bounded_lat_means = uh.join_mean_mixture(
                bounded_comp_meanss, bounded_prob_means
            )

        # Rejoin all parameters
        return model.join_params(bounded_obs_means, int_means, bounded_lat_means)

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
            # lkl_params = model.lkl_man.join_params(obs_params, int_params)
            # re_loss = (
            #     self.re_reg
            #     * relative_entropy_regularization_full(
            #         model.lwr_hrm, batch, lkl_params
            #     )[1]
            # )

            return ce_loss + l1_loss + l2_loss  # + re_loss

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

    def make_batch_step(
        self,
        handler: RunHandler,
        model: HMoG,
        logger: JaxLogger,
        optimizer: Optimizer[Natural, HMoG],
    ) -> Callable[
        [tuple[OptState, Point[Natural, HMoG]], Array],
        tuple[tuple[OptState, Point[Natural, HMoG]], Array],
    ]:
        """Create step function for processing a single batch.

        Works for both standard SGD and EM-style updates.
        """
        # Create gradient mask function
        mask_gradient = self.create_gradient_mask(model)

        # define a function that extracts model.upr_hrm.obs_man parameters from a mean hmog point
        def debug_lat_obs_means(
            means: Point[Mean, HMoG],
            pst: bool,
        ) -> None:
            _, _, lat_means = model.split_params(means)
            obs_lat_means, _, _ = model.upr_hrm.split_params(lat_means)
            olm_mean, olm_cov = model.upr_hrm.obs_man.split_mean_covariance(
                obs_lat_means
            )
            if pst:
                jax.debug.print("Posterior obs mean: {}", olm_mean.array)
                jax.debug.print("Posterior obs cov: {}", olm_cov.array)
            else:
                jax.debug.print("Prior obs means: {}", obs_lat_means.array)

        def batch_step(
            carry: tuple[OptState, Point[Natural, HMoG]],
            batch: Array,
        ) -> tuple[tuple[OptState, Point[Natural, HMoG]], Array]:
            opt_state, params = carry

            # Compute posterior statistics once for this batch
            posterior_stats = model.mean_posterior_statistics(params, batch)

            # Apply bounds to posterior statistics
            bounded_posterior = self.bound_means(model, posterior_stats)
            # debug_lat_obs_means(bounded_posterior, True)

            # Define the inner step function for scan
            def inner_step(
                carry: tuple[OptState, Point[Natural, HMoG]],
                _: None,  # Dummy input since we don't need per-step inputs
            ) -> tuple[tuple[OptState, Point[Natural, HMoG]], Array]:
                current_opt_state, current_params = carry

                # Compute gradient as difference between posterior and current prior
                prior_stats = model.to_mean(current_params)
                grad = prior_stats - bounded_posterior

                # Apply gradient mask
                masked_grad = mask_gradient(grad)

                # Update parameters
                current_opt_state, new_params = optimizer.update(
                    current_opt_state, masked_grad, current_params
                )

                # Monitor parameters for debugging
                logger.monitor_params(
                    {
                        "original": current_params.array,
                        "post_update": new_params.array,
                        "batch": batch,
                        "grad": grad.array,
                        "masked_grad": masked_grad.array,
                    },
                    handler,
                    context=f"{self.mask_type}",
                )

                return (current_opt_state, new_params), masked_grad.array

            # Run inner steps using jax.lax.scan
            (final_opt_state, final_params), all_grads = jax.lax.scan(
                inner_step,
                (opt_state, params),
                None,  # No inputs needed
                length=self.batch_steps,
            )

            return (final_opt_state, final_params), all_grads

        return batch_step

    def train(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: HMoG,
        logger: JaxLogger,
        epoch_offset: int,
        params0: Point[Natural, HMoG],
    ) -> Point[Natural, HMoG]:
        """Train the model with the specified gradient masking strategy."""
        n_epochs = self.n_epochs

        train_data = dataset.train_data
        if self.batch_size is None:
            batch_size = train_data.shape[0]
            n_batches = 1
        else:
            n_batches = train_data.shape[0] // self.batch_size
            batch_size = self.batch_size

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
        log.info(f"Training {self.mask_type} parameters for {n_epochs} epochs")

        # Create epoch step function
        def epoch_step(
            epoch: Array,
            carry: tuple[OptState, Point[Natural, HMoG], Array],
        ) -> tuple[OptState, Point[Natural, HMoG], Array]:
            opt_state, params, epoch_key = carry

            # Split key for shuffling
            shuffle_key, next_key = jax.random.split(epoch_key)

            # Create minibatch step with the appropriate mask
            batch_step = self.make_batch_step(handler, model, logger, optimizer)

            # Shuffle and batch data
            shuffled_indices = jax.random.permutation(shuffle_key, train_data.shape[0])
            batched_indices = shuffled_indices[: n_batches * batch_size]
            batched_data = train_data[batched_indices].reshape(
                (n_batches, batch_size, -1)
            )

            # Process all batches
            (opt_state, new_params), gradss_array = jax.lax.scan(
                batch_step, (opt_state, params), batched_data
            )
            # gradss_array shape: (n_batches, n_steps, param_dim)
            grads_array = gradss_array.reshape(-1, *gradss_array.shape[2:])

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
class LGMPretrainer[ObsRep: PositiveDefinite]:
    """Standalone trainer for AnalyticLinearGaussianModel.

    This trainer provides a simplified way to pre-train an LGM component
    before integrating it into a more complex HMoG model.
    """

    # Training hyperparameters
    n_epochs: int
    batch_size: int
    lr_init: float
    lr_final_ratio: float
    grad_clip: float

    # Regularization parameters
    l1_reg: float
    l2_reg: float

    # Parameter bounds
    min_var: float
    jitter: float

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
        hmog = differentiable_hmog(
            lgm.obs_dim, lgm.obs_rep, lgm.lat_dim, PositiveDefinite, 10
        )
        mix_params = hmog.upr_hrm.zeros()
        mix_grad = hmog.upr_hrm.zeros()

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

            lgm_params = lgm.join_conjugated(lkl_params, z)
            # Compute gradient
            lkl_grad = lgm.grad(self.make_loss_fn(lgm, batch), lgm_params)
            obs_grad, int_grad, _ = lgm.split_params(lkl_grad)
            lkl_grad = lgm.lkl_man.join_params(obs_grad, int_grad)

            # Update parameters
            opt_state, new_lkl_params = optimizer.update(
                opt_state, lkl_grad, lkl_params
            )

            # Apply parameter bounds
            bound_lkl_params = self.bound_parameters(lgm, new_lkl_params)

            # Monitor parameters for debugging
            logger.monitor_params(
                {
                    "original": lkl_params.array,
                    "post_update": new_lkl_params.array,
                    "post_bounds": bound_lkl_params.array,
                    "batch": batch,
                    "grad": lkl_grad.array,
                },
                handler,
                context="lgm_pretrain",
            )

            return (opt_state, bound_lkl_params), lkl_grad.array

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
            _, lat_int_params, lat_lat_params = hmog.upr_hrm.split_params(mix_params)
            new_mix_params = hmog.upr_hrm.join_params(
                lat_params, lat_int_params, lat_lat_params
            )
            hmog_params = hmog.join_params(obs_params, int_params, new_mix_params)

            # Create batch gradients for logging
            batch_man = Replicated(hmog, hmog_grads_array.shape[0])
            batch_grads = batch_man.point(hmog_grads_array)

            # Log metrics
            log_epoch_metrics(
                dataset,
                hmog,
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
