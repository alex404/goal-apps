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
from goal.geometry.manifold.util import batched_mean
from goal.models import (
    FullNormal,
    Normal,
)
from jax import Array

from apps.configs import STATS_NUM
from apps.plugins import (
    ClusteringDataset,
)
from apps.runtime.handler import MetricDict, RunHandler
from apps.runtime.logger import JaxLogger

from .analysis.logging import log_epoch_metrics, pre_log_epoch_metrics
from .base import LGM, HMoG, Mixture, fori

### Constants ###


class MaskingStrategy(Enum):
    """Enum defining which parameters to update during training."""

    LGM = auto()  # Only update LGM parameters (obs_params and int_params)
    MIXTURE = auto()  # Only update mixture parameters (lat_params)
    FULL = auto()  # Update all parameters


# Start logger
log = logging.getLogger(__name__)

STATS_LEVEL = jnp.array(STATS_NUM)
INFO_LEVEL = jnp.array(logging.INFO)

### Helpers ###


def precision_regularizer(
    model: HMoG,
    rho: Point[Natural, Mixture],
    mix_params: Point[Natural, Mixture],
    upr_prs_reg: float,
    lwr_prs_reg: float,
) -> tuple[Array, MetricDict]:
    """Compute regularization based on log-determinant of precision matrices.

    This regularization encourages components to have well-conditioned
    precision matrices by penalizing small determinants. This improves
    numerical stability and can prevent degenerate solutions.

    Args:
        model: HMoG model
        params: Model parameters

    Returns:
        Regularization term (negative sum of log-determinants)
    """
    con_mix_params = model.pst_lat_emb.translate(rho, mix_params)
    cmp_params, _ = model.con_upr_hrm.split_natural_mixture(con_mix_params)

    def compute_trace(
        nor_params: Point[Natural, FullNormal],
    ) -> Array:
        with model.con_upr_hrm as cuh:
            prs_params = cuh.obs_man.split_location_precision(nor_params)[1]
            prs_dense = cuh.obs_man.cov_man.to_dense(prs_params)
            return jnp.trace(prs_dense)

    traces = model.con_upr_hrm.cmp_man.map(compute_trace, cmp_params)
    trace_sum = jnp.sum(traces)
    trace_reg = upr_prs_reg * trace_sum

    def compute_logdet(
        nor_params: Point[Natural, FullNormal],
    ) -> Array:
        with model.con_upr_hrm as cuh:
            prs_params = cuh.obs_man.split_location_precision(nor_params)[1]
            prs_dense = cuh.obs_man.cov_man.to_dense(prs_params)
            return -jnp.linalg.slogdet(prs_dense)[1]

    logdets = model.con_upr_hrm.cmp_man.map(compute_logdet, cmp_params)
    logdet_sum = jnp.sum(logdets)
    logdet_reg = lwr_prs_reg * logdet_sum

    # Create metrics dictionary
    metrics: MetricDict = {
        "Regularization/Precision Trace Sum": (STATS_LEVEL, trace_sum),
        "Regularization/Precision LogDet Sum": (STATS_LEVEL, logdet_sum),
        "Regularization/Trace Penalty": (STATS_LEVEL, trace_reg),
        "Regularization/LogDet Penalty": (STATS_LEVEL, logdet_reg),
    }

    return trace_reg + logdet_reg, metrics


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
    upr_prs_reg: float
    lwr_prs_reg: float

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

    def make_regularizer(
        self, model: HMoG
    ) -> Callable[[Point[Natural, HMoG]], tuple[Point[Mean, HMoG], MetricDict]]:
        """Create a unified regularizer that returns loss, metrics, and gradient."""

        def loss_with_metrics(params: Point[Natural, HMoG]) -> tuple[Array, MetricDict]:
            # Extract components for regularization
            obs_params, int_params, mix_params = model.split_params(params)
            lkl_params = model.lkl_man.join_params(obs_params, int_params)
            rho = model.conjugation_parameters(lkl_params)

            # L1 regularization
            l1_norm = jnp.sum(jnp.abs(int_params.array))
            l1_loss = self.l1_reg * l1_norm

            # L2 regularization
            l2_norm = jnp.sum(jnp.square(params.array))
            l2_loss = self.l2_reg * l2_norm

            # Component precision regularization
            prs_loss, prs_metrics = precision_regularizer(
                model, rho, mix_params, self.upr_prs_reg, self.lwr_prs_reg
            )

            # Combine into total loss and metrics
            total_loss = l1_loss + l2_loss + prs_loss

            metrics = {
                "Regularization/L1 Norm": (STATS_LEVEL, l1_norm),
                "Regularization/L2 Norm": (STATS_LEVEL, l2_norm),
                "Regularization/L1 Penalty": (STATS_LEVEL, l1_loss),
                "Regularization/L2 Penalty": (STATS_LEVEL, l2_loss),
            }
            metrics.update(prs_metrics)

            return total_loss, metrics

        # Create a function that computes loss, metrics, and gradients in one pass
        return jax.grad(loss_with_metrics, has_aux=True)

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
                zero_obs_grad = model.obs_man.zeros()
                zero_int_grad = model.int_man.zeros()
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
                reg_grad = reg_fn(current_params)[0]
                grad = grad + reg_grad

                # Apply gradient mask
                masked_grad = mask_gradient(grad)

                # Update parameters
                new_opt_state, new_params = optimizer.update(
                    current_opt_state, masked_grad, current_params
                )

                # new_params = self.reset_non_pd(model, new_params)

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

            reg_fn = self.make_regularizer(model)
            metrics = reg_fn(new_params)[1]

            # Log metrics
            log_epoch_metrics(
                dataset,
                model,
                logger,
                new_params,
                epoch + epoch_offset,
                metrics,
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

    def make_regularizer(
        self, model: LGM
    ) -> Callable[[Point[Natural, LGM]], tuple[Point[Mean, LGM], MetricDict]]:
        """Create a unified regularizer that returns loss, metrics, and gradient."""

        def loss_with_metrics(params: Point[Natural, LGM]) -> tuple[Array, MetricDict]:
            # Extract components for regularization
            _, int_params, _ = model.split_params(params)

            # L1 regularization
            l1_norm = jnp.sum(jnp.abs(int_params.array))
            l1_loss = self.l1_reg * l1_norm

            # L2 regularization
            l2_norm = jnp.sum(jnp.square(params.array))
            l2_loss = self.l2_reg * l2_norm

            # Combine into total loss and metrics
            total_loss = l1_loss + l2_loss

            metrics = {
                "Regularization/L1 Norm": (STATS_LEVEL, l1_norm),
                "Regularization/L2 Norm": (STATS_LEVEL, l2_norm),
                "Regularization/L1 Penalty": (STATS_LEVEL, l1_loss),
                "Regularization/L2 Penalty": (STATS_LEVEL, l2_loss),
            }

            return total_loss, metrics

        # Create a function that computes loss, metrics, and gradients in one pass
        return jax.grad(loss_with_metrics, has_aux=True)

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
                reg_grad = reg_fn(current_params)[0]
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

            reg_fn = self.make_regularizer(model)
            metrics = reg_fn(new_params)[1]

            # Log metrics
            pre_log_epoch_metrics(
                dataset,
                model,
                logger,
                new_params,
                epoch + epoch_offset,
                metrics,
                batch_grads,
                log_freq=10,
            )

            return opt_state, new_params, next_key

        # Run training loop
        (_, params_final, _) = fori(0, n_epochs, epoch_step, (opt_state, params0, key))

        return params_final


@dataclass(frozen=True)
class FixedObservableTrainer:
    """Trainer that holds observable parameters fixed and only updates mixture parameters.

    For high-dimensional data, this caches the mapping from observation to latent space,
    avoiding repeated expensive computations.
    """

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
    lat_min_var: float
    lat_jitter_var: float
    upr_prs_reg: float
    lwr_prs_reg: float

    def precompute_observable_mappings(
        self, model: HMoG, params: Point[Natural, HMoG], data: Array
    ) -> tuple[Array, Point[Natural, Mixture]]:
        log.info("Precomputing latent statistics for fixed observable training")

        # Get the posterior function (the affine map)
        obs_params, int_params, _ = model.split_params(params)
        lkl_params = model.lkl_man.join_params(obs_params, int_params)
        rho = model.conjugation_parameters(lkl_params)

        def project_fn(x: Array) -> Array:
            x_means = model.obs_man.loc_man.mean_point(x)
            return model.int_man.transpose_apply(int_params, x_means).array

        return jax.vmap(project_fn)(data), rho

    def mean_posterior_statistics(
        self,
        model: HMoG,
        mix_params: Point[Natural, Mixture],
        latent_locations: Array,
    ) -> Point[Mean, Mixture]:
        # Get sufficient statistics of observation

        def posterior_statistics(
            latent_array: Array,
        ) -> Array:
            # Get posterior statistics for this observation
            latent_location = model.upr_hrm.obs_man.loc_man.natural_point(latent_array)
            mix_obs, mix_int, mix_lat = model.upr_hrm.split_params(mix_params)
            posterior_params = model.int_lat_emb.translate(mix_params, latent_location)
            posterior_means = model.upr_hrm.to_mean(posterior_params)
            return posterior_means.array

        return model.upr_hrm.mean_point(
            batched_mean(posterior_statistics, latent_locations, 256)
        )

    def prior_statistics(
        self,
        model: HMoG,
        mix_params: Point[Natural, Mixture],
        rho: Point[Natural, Mixture],
    ) -> Point[Mean, Mixture]:
        # Get sufficient statistics of observation

        def log_partition_function(
            params: Point[Natural, Mixture],
        ) -> Array:
            # Split parameters
            con_params = model.pst_lat_emb.translate(rho, params)
            return model.con_upr_hrm.log_partition_function(con_params)

        return model.upr_hrm.grad(log_partition_function, mix_params)

    def bound_mixture_means(
        self, model: HMoG, mix_means: Point[Mean, Mixture]
    ) -> Point[Mean, Mixture]:
        """Apply bounds to mixture components for numerical stability."""
        with model.upr_hrm as uh:
            # Bound component means
            cmp_meanss, cat_means = uh.split_mean_mixture(mix_means)
            lat_obs_means, _, _ = model.upr_hrm.split_params(mix_means)

            def whiten_and_bound(
                comp_means: Point[Mean, Normal[Any]],
            ) -> Point[Mean, Normal[Any]]:
                whitened_comp_means = uh.obs_man.whiten(comp_means, lat_obs_means)
                return uh.obs_man.regularize_covariance(
                    whitened_comp_means, self.lat_jitter_var, self.lat_min_var
                )

            bounded_cmp_meanss = uh.cmp_man.man_map(whiten_and_bound, cmp_meanss)

            # Bound probabilities
            probs = uh.lat_man.to_probs(cat_means)
            bounded_probs = jnp.clip(probs, self.min_prob, 1.0)
            bounded_probs = bounded_probs / jnp.sum(bounded_probs)
            bounded_prob_means = uh.lat_man.from_probs(bounded_probs)

            return uh.join_mean_mixture(bounded_cmp_meanss, bounded_prob_means)

    def make_regularizer(
        self, model: HMoG, rho: Point[Natural, Mixture]
    ) -> Callable[[Point[Natural, Mixture]], tuple[Point[Mean, Mixture], MetricDict]]:
        """Create a unified regularizer that returns loss, metrics, and gradient."""

        def loss_with_metrics(
            mix_params: Point[Natural, Mixture],
        ) -> tuple[Array, MetricDict]:
            # Extract components for regularization

            # L2 regularization
            l2_norm = jnp.sum(jnp.square(mix_params.array))
            l2_loss = self.l2_reg * l2_norm

            # Component precision regularization
            prs_loss, prs_metrics = precision_regularizer(
                model, rho, mix_params, self.upr_prs_reg, self.lwr_prs_reg
            )

            # Combine into total loss and metrics
            total_loss = l2_loss + prs_loss

            metrics = {
                "Regularization/L2 Norm": (STATS_LEVEL, l2_norm),
                "Regularization/L2 Penalty": (STATS_LEVEL, l2_loss),
            }
            metrics.update(prs_metrics)

            return total_loss, metrics

        # Create a function that computes loss, metrics, and gradients in one pass
        return jax.grad(loss_with_metrics, has_aux=True)

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
        """Train only the mixture parameters with fixed observable mapping.

        Args:
            key: Random key
            handler: Run handler for saving artifacts
            dataset: Dataset for training
            model: HMoG model
            logger: Logger
            learning_rate_scale: Scale factor for learning rate
            epoch_offset: Offset for epoch numbering
            params0: Initial parameters

        Returns:
            Updated parameters
        """
        n_epochs = self.n_epochs
        train_data = dataset.train_data

        # Extract fixed observable parameters
        obs_params0, int_params0, mix_params0 = model.split_params(params0)

        # Precompute all latent statistics using the fixed observable mapping
        latent_locations, rho = self.precompute_observable_mappings(
            model, params0, train_data
        )

        # Setup optimizer for mixture parameters
        optim = optax.adam(learning_rate=self.lr * learning_rate_scale)
        optimizer = Optimizer(optim, model.upr_hrm)
        if self.grad_clip > 0.0:
            optimizer = optimizer.with_grad_clip(self.grad_clip)

        # Initialize optimizer state
        opt_state = optimizer.init(mix_params0)

        # Determine batch sizes
        if self.batch_size is None:
            batch_size = latent_locations.shape[0]
            n_batches = 1
        else:
            batch_size = self.batch_size
            n_batches = latent_locations.shape[0] // batch_size

        # Log training start
        log.info(
            f"Training mixture with fixed observable mapping for {n_epochs} epochs"
        )

        # Create regularizer
        reg_fn = self.make_regularizer(model, rho)

        # Define batch step function
        def batch_step(
            carry: tuple[OptState, Point[Natural, Mixture]],
            batch_locations: Array,
        ) -> tuple[tuple[OptState, Point[Natural, Mixture]], Array]:
            opt_state, params = carry

            posterior_stats = self.mean_posterior_statistics(
                model, params, batch_locations
            )

            bounded_posterior_stats = self.bound_mixture_means(model, posterior_stats)

            # Define the inner step function
            def inner_step(
                carry: tuple[OptState, Point[Natural, Mixture]],
                _: None,
            ) -> tuple[tuple[OptState, Point[Natural, Mixture]], Array]:
                current_opt_state, current_params = carry

                # Current parameters as means
                prior_stats = self.prior_statistics(model, current_params, rho)

                # Compute gradient as difference between posterior and current prior
                grad = prior_stats - bounded_posterior_stats

                # Add regularization
                reg_grad = reg_fn(current_params)[0]
                grad = grad + reg_grad

                # Update parameters
                new_opt_state, new_params = optimizer.update(
                    current_opt_state, grad, current_params
                )
                full_params = model.join_params(
                    obs_params0, int_params0, current_params
                )
                new_full_params = model.join_params(
                    obs_params0, int_params0, new_params
                )
                full_grad = model.join_params(
                    model.lwr_hrm.obs_man.zeros(), model.lwr_hrm.int_man.zeros(), grad
                )

                # Monitor parameters for debugging
                logger.monitor_params(
                    {
                        "original_params": full_params.array,
                        "updated_params": new_full_params.array,
                        "mixture_stats": posterior_stats.array,
                        "bounded_mixture_stats": bounded_posterior_stats.array,
                        "prior_stats": prior_stats.array,
                        "grad": full_grad.array,
                    },
                    handler,
                    context="FIXED_OBSERVABLE",
                )

                return (new_opt_state, new_params), full_grad.array

            # Run inner steps
            (final_opt_state, final_params), all_grads = jax.lax.scan(
                inner_step,
                (opt_state, params),
                None,
                length=self.batch_steps,
            )

            return (final_opt_state, final_params), all_grads

        # Create epoch step function
        def epoch_step(
            epoch: Array,
            carry: tuple[OptState, Point[Natural, Mixture], Array],
        ) -> tuple[OptState, Point[Natural, Mixture], Array]:
            opt_state, mix_params, epoch_key = carry

            # Split key for shuffling
            shuffle_key, next_key = jax.random.split(epoch_key)

            # Shuffle and batch data
            shuffled_indices = jax.random.permutation(
                shuffle_key, latent_locations.shape[0]
            )
            batched_indices = shuffled_indices[: n_batches * batch_size]
            batched_locations = latent_locations[batched_indices].reshape(
                (n_batches, batch_size, -1)
            )

            # Process all batches
            (opt_state, new_mix_params), gradss_array = jax.lax.scan(
                batch_step, (opt_state, mix_params), batched_locations
            )
            # gradss_array shape: (n_batches, n_steps, param_dim)
            grads_array = gradss_array.reshape(-1, *gradss_array.shape[2:])

            # Create batch gradients for logging
            batch_man = Replicated(model, grads_array.shape[0])
            batch_grads = batch_man.point(grads_array)

            # Reconstruct full model parameters for evaluation
            full_params = model.join_params(obs_params0, int_params0, new_mix_params)

            reg_fn = self.make_regularizer(model, rho)
            metrics = reg_fn(new_mix_params)[1]

            # Log metrics
            log_epoch_metrics(
                dataset,
                model,
                logger,
                full_params,
                epoch + epoch_offset,
                metrics,
                batch_grads,
                log_freq=10,
            )

            return opt_state, new_mix_params, next_key

        # Run training loop
        (_, mix_params_final, _) = fori(
            0, n_epochs, epoch_step, (opt_state, mix_params0, key)
        )

        # Reconstruct full parameters with updated mixture
        params_final = model.join_params(obs_params0, int_params0, mix_params_final)

        return params_final
