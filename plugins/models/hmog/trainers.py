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
    Mean,
    Natural,
    Optimizer,
    OptState,
    Point,
    Replicated,
)
from goal.models import (
    DiagonalNormal,
    Normal,
)
from jax import Array

from apps.plugins import (
    ClusteringDataset,
)
from apps.runtime.handler import RunHandler
from apps.runtime.logger import JaxLogger

from .analysis import log_epoch_metrics, pre_log_epoch_metrics
from .base import LGM, HMoG, fori

### Constants ###


class MaskingStrategy(Enum):
    """Enum defining which parameters to update during training."""

    LGM = auto()  # Only update LGM parameters (obs_params and int_params)
    MIXTURE = auto()  # Only update mixture parameters (lat_params)
    FULL = auto()  # Update all parameters


# Start logger
log = logging.getLogger(__name__)


### Symmetric Gradient Trainer ###


@dataclass(frozen=True)
class GradientTrainer:
    """Base trainer for gradient-based training of HMoG models."""

    # Training hyperparameters
    lr: float
    n_epochs: int
    batch_size: None | int
    batch_steps: int
    grad_clip: float

    # Regularization parameters
    l1_reg: float
    l2_reg: float

    # Parameter bounds
    min_prob: float
    obs_min_var: float
    lat_min_var: float
    obs_jitter_var: float
    lat_jitter_var: float
    eigen_floor_prs: float

    # Strategy
    mask_type: MaskingStrategy

    def bound_means(self, model: HMoG, means: Point[Mean, HMoG]) -> Point[Mean, HMoG]:
        """Apply bounds to posterior statistics for numerical stability."""
        # Split posterior statistics
        obs_means, int_means, lat_means = model.split_params(means)

        # For observable parameters, bound the variances
        bounded_obs_means = model.lwr_hrm.obs_man.regularize_covariance(
            obs_means, self.obs_jitter_var, self.obs_min_var
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
                    whitened_comp_means, self.lat_jitter_var, self.lat_min_var
                )

            bounded_comp_meanss = uh.cmp_man.man_map(whiten_and_bound, comp_meanss)

            probs = uh.lat_man.to_probs(prob_means)
            bounded_probs0 = jnp.clip(probs, self.min_prob, 1.0)
            bounded_probs = bounded_probs0 / jnp.sum(bounded_probs0)
            bounded_prob_means = uh.lat_man.from_probs(bounded_probs)

            bounded_lat_means0 = uh.join_mean_mixture(
                bounded_comp_meanss, bounded_prob_means
            )

            _, bounded_lat_int, bounded_lat_cat = uh.split_params(bounded_lat_means0)

            z = uh.obs_man.standard_normal()

            bounded_lat_means = uh.join_params(z, bounded_lat_int, bounded_lat_cat)

        # Rejoin all parameters
        return model.join_params(bounded_obs_means, int_means, bounded_lat_means)

    def ensure_positive_definite_components(
        self,
        model: HMoG,
        params: Point[Natural, HMoG],
        eigen_floor_prs: float,
    ) -> Point[Natural, HMoG]:
        """
        Ensure the precision matrices of the latent components are positive definite.
        """
        obs_params, int_params, mix_params = model.split_params(params)
        lkl_params = model.lwr_hrm.lkl_man.join_params(obs_params, int_params)
        rho = model.lwr_hrm.conjugation_parameters(lkl_params)
        cmp_params, prob_params = model.upr_hrm.split_natural_mixture(mix_params)

        def ensure_positive_definite(
            dia_params: Point[Natural, DiagonalNormal],
        ) -> Point[Natural, DiagonalNormal]:
            with model.con_upr_hrm as cuh:
                con_cmp_params = cuh.obs_emb.translate(rho, dia_params)

                prs_params = cuh.obs_man.split_location_precision(con_cmp_params)[1]
                prs_dense = cuh.obs_man.cov_man.to_dense(prs_params)
                eigenvalues, eigenvectors = jnp.linalg.eigh(prs_dense)

            min_eig = jnp.min(eigenvalues)

            with model.upr_hrm as uh:
                diag_shift = jnp.maximum(eigen_floor_prs - min_eig, 0)
                shift_array = jnp.full(uh.obs_man.cov_man.dim, diag_shift)
                shift_point = uh.obs_man.cov_man.natural_point(shift_array)

                loc_params, prs_params = uh.obs_man.split_location_precision(dia_params)
                shifted_dia_params = prs_params + shift_point
                return uh.obs_man.join_location_precision(
                    loc_params, shifted_dia_params
                )

        # Apply the function to all components
        pd_cmp_params = model.upr_hrm.cmp_man.man_map(
            ensure_positive_definite, cmp_params
        )
        pd_mix_params = model.upr_hrm.join_natural_mixture(pd_cmp_params, prob_params)
        return model.join_params(obs_params, int_params, pd_mix_params)

    def make_regularizer(self, model: HMoG) -> Callable[[Point[Natural, HMoG]], Array]:
        """Create a universal loss function for any HMoG model."""

        def loss_fn(params: Point[Natural, HMoG]) -> Array:
            # Core negative log-likelihood

            # Extract components for regularization
            _, int_params, _ = model.split_params(params)

            # L1 regularization on interaction matrix
            l1_loss = self.l1_reg * jnp.sum(jnp.abs(int_params.array))

            # L2 regularization on all parameters
            l2_loss = self.l2_reg * jnp.sum(jnp.square(params.array))

            return l1_loss + l2_loss

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

        def batch_step(
            carry: tuple[OptState, Point[Natural, HMoG]],
            batch: Array,
        ) -> tuple[tuple[OptState, Point[Natural, HMoG]], Array]:
            opt_state, params = carry

            # Compute posterior statistics once for this batch
            posterior_stats = model.mean_posterior_statistics(params, batch)

            # Apply bounds to posterior statistics
            bounded_posterior_stats = self.bound_means(model, posterior_stats)
            # debug_lat_obs_means(bounded_posterior, True)

            # Define the inner step function for scan
            def inner_step(
                carry: tuple[OptState, Point[Natural, HMoG]],
                _: None,  # Dummy input since we don't need per-step inputs
            ) -> tuple[tuple[OptState, Point[Natural, HMoG]], Array]:
                current_opt_state, current_params = carry

                # Compute gradient as difference between posterior and current prior
                prior_stats = model.to_mean(current_params)
                grad = prior_stats - bounded_posterior_stats
                reg_fn = self.make_regularizer(model)
                reg_grad = jax.grad(reg_fn)(current_params)
                grad = grad + reg_grad

                # Apply gradient mask
                masked_grad = mask_gradient(grad)

                # Update parameters
                new_opt_state, new_params = optimizer.update(
                    current_opt_state, masked_grad, current_params
                )

                # Monitor parameters for debugging
                logger.monitor_params(
                    {
                        "original_params": current_params.array,
                        "updated_params": new_params.array,
                        "batch": batch,
                        "posterior_stats": posterior_stats.array,
                        "bounded_posterior_stats": bounded_posterior_stats.array,
                        "prior_stats": prior_stats.array,
                        "grad": grad.array,
                        "masked_grad": masked_grad.array,
                    },
                    handler,
                    context=f"{self.mask_type}",
                )

                return (new_opt_state, new_params), masked_grad.array

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
        learning_rate_scale: float,
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

        # Create optimizer
        optim = optax.adamw(learning_rate=self.lr * learning_rate_scale)
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

            if self.eigen_floor_prs > 0.0:
                new_params = self.ensure_positive_definite_components(
                    model, new_params, eigen_floor_prs=self.eigen_floor_prs
                )

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


@dataclass(frozen=True)
class PreTrainer:
    """Base trainer for gradient-based training of HMoG models."""

    # Training hyperparameters
    lr: float
    n_epochs: int
    batch_size: None | int
    batch_steps: int
    grad_clip: float

    # Regularization parameters
    l1_reg: float
    l2_reg: float

    # Parameter bounds
    min_var: float
    jitter_var: float

    def bound_means(self, model: LGM, means: Point[Mean, LGM]) -> Point[Mean, LGM]:
        """Apply bounds to posterior statistics for numerical stability."""
        # Split posterior statistics
        obs_means, int_means, lat_means = model.split_params(means)

        # For observable parameters, bound the variances
        bounded_obs_means = model.obs_man.regularize_covariance(
            obs_means, self.jitter_var, self.min_var
        )
        z = model.lat_man.standard_normal()

        # Rejoin all parameters
        return model.join_params(bounded_obs_means, int_means, z)

    def make_regularizer(self, model: LGM) -> Callable[[Point[Natural, LGM]], Array]:
        def loss_fn(params: Point[Natural, LGM]) -> Array:
            # Core negative log-likelihood

            # Extract components for regularization
            _, int_params, _ = model.split_params(params)

            # L1 regularization on interaction matrix
            l1_loss = self.l1_reg * jnp.sum(jnp.abs(int_params.array))

            # L2 regularization on all parameters
            l2_loss = self.l2_reg * jnp.sum(jnp.square(params.array))

            return l1_loss + l2_loss

        return loss_fn

    def make_batch_step(
        self,
        handler: RunHandler,
        model: LGM,
        logger: JaxLogger,
        optimizer: Optimizer[Natural, LGM],
    ) -> Callable[
        [tuple[OptState, Point[Natural, LGM]], Array],
        tuple[tuple[OptState, Point[Natural, LGM]], Array],
    ]:
        def batch_step(
            carry: tuple[OptState, Point[Natural, LGM]],
            batch: Array,
        ) -> tuple[tuple[OptState, Point[Natural, LGM]], Array]:
            opt_state, params = carry

            # Compute posterior statistics once for this batch
            posterior_stats = model.mean_posterior_statistics(params, batch)

            # Apply bounds to posterior statistics
            bounded_posterior_stats = self.bound_means(model, posterior_stats)
            # debug_lat_obs_means(bounded_posterior, True)

            # Define the inner step function for scan
            def inner_step(
                carry: tuple[OptState, Point[Natural, LGM]],
                _: None,  # Dummy input since we don't need per-step inputs
            ) -> tuple[tuple[OptState, Point[Natural, LGM]], Array]:
                current_opt_state, current_params = carry

                # Compute gradient as difference between posterior and current prior
                prior_stats = model.to_mean(current_params)
                grad = prior_stats - bounded_posterior_stats
                reg_fn = self.make_regularizer(model)
                reg_grad = jax.grad(reg_fn)(current_params)
                grad = grad + reg_grad

                # Update parameters
                new_opt_state, new_params = optimizer.update(
                    current_opt_state, grad, current_params
                )

                # Monitor parameters for debugging
                logger.monitor_params(
                    {
                        "original_params": current_params.array,
                        "updated_params": new_params.array,
                        "batch": batch,
                        "posterior_stats": posterior_stats.array,
                        "bounded_posterior_stats": bounded_posterior_stats.array,
                        "prior_stats": prior_stats.array,
                        "grad": grad.array,
                    },
                    handler,
                    context="PRE",
                )

                return (new_opt_state, new_params), grad.array

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
        model: LGM,
        logger: JaxLogger,
        epoch_offset: int,
        params0: Point[Natural, LGM],
    ) -> Point[Natural, LGM]:
        """Train the model with the specified gradient masking strategy."""
        n_epochs = self.n_epochs

        train_data = dataset.train_data
        if self.batch_size is None:
            batch_size = train_data.shape[0]
            n_batches = 1
        else:
            n_batches = train_data.shape[0] // self.batch_size
            batch_size = self.batch_size

        # Create optimizer
        optim = optax.adamw(learning_rate=self.lr)
        optimizer: Optimizer[Natural, LGM] = Optimizer(optim, model)

        if self.grad_clip > 0.0:
            optimizer = optimizer.with_grad_clip(self.grad_clip)

        # Initialize optimizer state
        opt_state = optimizer.init(params0)

        # Log training phase
        log.info(f"Training PRE parameters for {n_epochs} epochs")

        # Create epoch step function
        def epoch_step(
            epoch: Array,
            carry: tuple[OptState, Point[Natural, LGM], Array],
        ) -> tuple[OptState, Point[Natural, LGM], Array]:
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
            pre_log_epoch_metrics(
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
