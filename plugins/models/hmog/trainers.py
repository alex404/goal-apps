"""Trainers forAnyHMoG model components."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

import jax
import jax.numpy as jnp
import optax
from goal.geometry.manifold.util import batched_mean
from jax import Array

from apps.interface import (
    ClusteringDataset,
)
from apps.runtime import STATS_NUM, Logger, MetricDict, RunHandler

from .metrics import log_epoch_metrics, pre_log_epoch_metrics
from .types import AnyHMoG, AnyLGM

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
    model: AnyHMoG,
    rho: Array,
    mix_params: Array,
    upr_prs_reg: float,
    lwr_prs_reg: float,
) -> tuple[Array, MetricDict]:
    """Compute regularization based on precision matrix eigenvalues.

    This regularizer controls the condition number and scale of the latent
    precision matrices by penalizing extreme eigenvalues:

    R(Lambda) = upr_prs_reg * tr(Lambda) - lwr_prs_reg * log|Lambda|

    where:
    - tr(Lambda) = sum of eigenvalues (penalizes large eigenvalues)
    - log|Lambda| = sum of log eigenvalues (penalizes small eigenvalues)

    Mathematical insight: For a precision matrix with eigenvalues lambda_i,
    the gradient of this regularizer is:

        dR/dlambda_i = upr_prs_reg - lwr_prs_reg/lambda_i

    Setting to zero gives the optimal eigenvalue: lambda* = lwr_prs_reg/upr_prs_reg

    When upr_prs_reg = lwr_prs_reg:
    - All eigenvalues are pushed toward 1
    - Creates an isotropic latent space with uniform scaling
    - This is the natural scale in information geometry
    - Empirically gives best model performance in most cases

    When upr_prs_reg != lwr_prs_reg:
    - Eigenvalues pushed toward lwr_prs_reg/upr_prs_reg
    - Use upr_prs_reg > lwr_prs_reg to encourage smaller eigenvalues (more regularization)
    - Use upr_prs_reg < lwr_prs_reg to allow larger eigenvalues (less regularization)
    - May be useful for specific domain knowledge or computational constraints

    Args:
        model:AnyHMoG model
        rho: Conjugation parameters
        mix_params: Mixture parameters in natural coordinates
        upr_prs_reg: Weight for trace penalty (controls upper bound on eigenvalues)
        lwr_prs_reg: Weight for log-det penalty (controls lower bound on eigenvalues)

    Returns:
        Tuple of (regularization_loss, metrics_dict)
    """
    con_mix_params = model.pst_prr_emb.translate(rho, mix_params)
    cmp_params, _ = model.prr_man.split_natural_mixture(con_mix_params)

    def compute_trace(
        nor_params: Array,
    ) -> Array:
        with model.prr_man as cuh:
            prs_params = cuh.obs_man.split_location_precision(nor_params)[1]
            prs_dense = cuh.obs_man.cov_man.to_matrix(prs_params)
            return jnp.trace(prs_dense)

    traces = model.prr_man.cmp_man.map(compute_trace, cmp_params)
    trace_sum = jnp.sum(traces)
    trace_reg = upr_prs_reg * trace_sum

    def compute_logdet(
        nor_params: Array,
    ) -> Array:
        with model.prr_man as cuh:
            prs_params = cuh.obs_man.split_location_precision(nor_params)[1]
            prs_dense = cuh.obs_man.cov_man.to_matrix(prs_params)
            return -jnp.linalg.slogdet(prs_dense)[1]

    logdets = model.prr_man.cmp_man.map(compute_logdet, cmp_params)
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


def mixture_entropy_regularizer(
    model: AnyHMoG,
    rho: Array,
    mix_params: Array,
    entropy_reg: float,
) -> tuple[Array, MetricDict]:
    """Entropy regularization on mixture weights.

    Adds entropy_reg * neg_entropy(π) to the loss, where neg_entropy = Σ πᵢ log πᵢ
    is the Categorical.negative_entropy() from the Analytic base class (≤ 0).
    Its gradient pushes the mixing distribution toward uniformity (maximum entropy).
    Default entropy_reg=0.0 disables the penalty entirely.
    """
    con_mix_params = model.pst_prr_emb.translate(rho, mix_params)
    _, cat_nat_params = model.prr_man.split_natural_mixture(con_mix_params)

    # natural → mean params (required by negative_entropy, which is defined on Analytic)
    cat_means = model.prr_man.lat_man.to_mean(cat_nat_params)
    neg_entropy = model.prr_man.lat_man.negative_entropy(cat_means)

    # neg_entropy ≤ 0; minimizing entropy_reg * neg_entropy pushes toward uniform
    entropy_loss = entropy_reg * neg_entropy

    metrics: MetricDict = {
        "Regularization/Mixture Entropy": (STATS_LEVEL, -neg_entropy),
        "Regularization/Entropy Penalty": (STATS_LEVEL, entropy_loss),
    }
    return entropy_loss, metrics


### Symmetric Gradient Trainer ###


@dataclass(frozen=True)
class FullGradientTrainer:
    """Base trainer for gradient-based training ofAnyHMoG models."""

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

    # Numerical stability
    mixture_entropy_reg: float

    epoch_reset: bool = True

    def bound_means(
        self,
        model: AnyHMoG,
        means: Array,
    ) -> Array:
        """Apply bounds to posterior statistics for numerical stability."""
        obs_means, int_means, lat_means = model.split_coords(means)

        bounded_obs_means = model.lwr_hrm.obs_man.regularize_covariance(
            obs_means, self.obs_jitter_var, self.obs_min_var
        )

        with model.pst_man as uh:
            comp_means, prob_means = uh.split_mean_mixture(lat_means)
            probs = uh.lat_man.to_probs(prob_means)
            bounded_probs = jnp.clip(probs, self.min_prob, 1.0)
            bounded_probs = bounded_probs / jnp.sum(bounded_probs)
            bounded_prob_means = uh.lat_man.from_probs(bounded_probs)
            bounded_lat_means = uh.join_mean_mixture(comp_means, bounded_prob_means)

        return model.whiten_prior(
            model.join_coords(bounded_obs_means, int_means, bounded_lat_means)
        )

    def make_regularizer(
        self, model: AnyHMoG
    ) -> Callable[
        [Array],
        tuple[Array, MetricDict],
    ]:
        """Create a unified regularizer that returns loss, metrics, and gradient."""

        def loss_with_metrics(
            params: Array,
        ) -> tuple[Array, MetricDict]:
            # Extract components for regularization
            obs_params, int_params, mix_params = model.split_coords(params)
            lkl_params = model.lkl_fun_man.join_coords(obs_params, int_params)
            rho = model.conjugation_parameters(lkl_params)

            # L1 regularization
            l1_norm = jnp.sum(jnp.abs(int_params))
            l1_loss = self.l1_reg * l1_norm

            # L2 regularization
            l2_norm = jnp.sum(jnp.square(params))
            l2_loss = self.l2_reg * l2_norm

            # Component precision regularization
            prs_loss, prs_metrics = precision_regularizer(
                model, rho, mix_params, self.upr_prs_reg, self.lwr_prs_reg
            )

            # Entropy penalty on mixture weights
            ent_loss, ent_metrics = mixture_entropy_regularizer(
                model, rho, mix_params, self.mixture_entropy_reg
            )

            # Combine into total loss and metrics
            total_loss = l1_loss + l2_loss + prs_loss + ent_loss

            metrics = {
                "Regularization/L1 Norm": (STATS_LEVEL, l1_norm),
                "Regularization/L2 Norm": (STATS_LEVEL, l2_norm),
                "Regularization/L1 Penalty": (STATS_LEVEL, l1_loss),
                "Regularization/L2 Penalty": (STATS_LEVEL, l2_loss),
            }
            metrics.update(prs_metrics)
            metrics.update(ent_metrics)

            return total_loss, metrics

        # Create a function that computes loss, metrics, and gradients in one pass
        return jax.grad(loss_with_metrics, has_aux=True)

    def create_gradient_mask(self, model: AnyHMoG) -> Callable[[Array], Array]:
        """Create a function that masks gradients for specific training regimes."""
        if self.mask_type == MaskingStrategy.LGM:
            # Only update LGM parameters (obs_params and int_params)
            def mask_fn(
                grad: Array,
            ) -> Array:
                obs_grad, int_grad, lat_grad = model.split_coords(grad)
                zero_lat_grad = jnp.zeros_like(lat_grad)
                return model.join_coords(obs_grad, int_grad, zero_lat_grad)

        elif self.mask_type == MaskingStrategy.MIXTURE:
            # Only update mixture parameters (lat_params)
            def mask_fn(
                grad: Array,
            ) -> Array:
                _, _, lat_grad = model.split_coords(grad)
                zero_obs_grad = model.obs_man.zeros()
                zero_int_grad = model.int_man.zeros()
                return model.join_coords(zero_obs_grad, zero_int_grad, lat_grad)

        else:
            # No masking - update all parameters
            def mask_fn(
                grad: Array,
            ) -> Array:
                return grad

        return mask_fn

    def make_batch_step(
        self,
        handler: RunHandler,
        model: AnyHMoG,
        logger: Logger,
        optimizer: optax.GradientTransformation,
    ) -> Callable[
        [tuple[optax.OptState, Array], Array],
        tuple[tuple[optax.OptState, Array], Array],
    ]:
        """Create step function for processing a single batch.

        Works for both standard SGD and EM-style updates.
        """
        # Create gradient mask function
        mask_gradient = self.create_gradient_mask(model)

        def batch_step(
            carry: tuple[optax.OptState, Array],
            batch: Array,
        ) -> tuple[tuple[optax.OptState, Array], Array]:
            opt_state, params = carry

            # Compute posterior statistics once for this batch
            posterior_stats = model.mean_posterior_statistics(params, batch)

            # Apply bounds to posterior statistics
            bounded_posterior_stats = self.bound_means(model, posterior_stats)
            # debug_lat_obs_means(bounded_posterior, True)

            # Define the inner step function for scan
            def inner_step(
                carry: tuple[optax.OptState, Array],
                _: None,  # Dummy input since we don't need per-step inputs
            ) -> tuple[tuple[optax.OptState, Array], Array]:
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
                updates, new_opt_state = optimizer.update(
                    masked_grad, current_opt_state, current_params
                )
                new_params = optax.apply_updates(current_params, updates)

                # new_params = self.reset_non_pd(model, new_params)

                # Monitor parameters for debugging
                logger.monitor_params(
                    {
                        "original_params": current_params,
                        "updated_params": new_params,
                        "batch": batch,
                        "posterior_stats": posterior_stats,
                        "bounded_posterior_stats": bounded_posterior_stats,
                        "prior_stats": prior_stats,
                        "grad": grad,
                        "masked_grad": masked_grad,
                    },
                    handler,
                    context=f"{self.mask_type}",
                )

                return (new_opt_state, new_params), masked_grad

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
        model: AnyHMoG,
        logger: Logger,
        learning_rate_scale: float,
        epoch_offset: int,
        params0: Array,
    ) -> Array:
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
        if self.grad_clip > 0.0:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.grad_clip),
                optax.adamw(learning_rate=self.lr * learning_rate_scale),
            )
        else:
            optimizer = optax.adamw(learning_rate=self.lr * learning_rate_scale)

        # Initialize optimizer state
        opt_state = optimizer.init(params0)

        # Log training phase
        log.info(f"Training {self.mask_type} parameters for {n_epochs} epochs")

        # Create epoch step function
        def epoch_step(
            epoch: Array,
            carry: tuple[optax.OptState, Array, Array],
        ) -> tuple[optax.OptState, Array, Array]:
            opt_state, params, epoch_key = carry

            if self.epoch_reset:
                opt_state = optimizer.init(params)

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
            batch_grads = gradss_array.reshape(-1, *gradss_array.shape[2:])

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
        (_, params_final, _) = jax.lax.fori_loop(
            0, n_epochs, epoch_step, (opt_state, params0, key)
        )

        return params_final


@dataclass(frozen=True)
class LGMPreTrainer:
    """Base trainer for gradient-based training ofAnyHMoG models."""

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

    epoch_reset: bool = True

    def bound_means(self, model: AnyLGM, means: Array) -> Array:
        """Apply bounds to posterior statistics for numerical stability."""
        obs_means, int_means, lat_means = model.split_coords(means)
        bounded_obs_means = model.obs_man.regularize_covariance(
            obs_means, self.jitter_var, self.min_var
        )
        return model.join_coords(bounded_obs_means, int_means, lat_means)

    def make_regularizer(
        self, model: AnyLGM
    ) -> Callable[[Array], tuple[Array, MetricDict]]:
        """Create a unified regularizer that returns loss, metrics, and gradient."""

        def loss_with_metrics(params: Array) -> tuple[Array, MetricDict]:
            # Extract components for regularization
            _, int_params, _ = model.split_coords(params)

            # L1 regularization
            l1_norm = jnp.sum(jnp.abs(int_params))
            l1_loss = self.l1_reg * l1_norm

            # L2 regularization
            l2_norm = jnp.sum(jnp.square(params))
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
        model: AnyLGM,
        logger: Logger,
        optimizer: optax.GradientTransformation,
    ) -> Callable[
        [tuple[optax.OptState, Array], Array],
        tuple[tuple[optax.OptState, Array], Array],
    ]:
        def batch_step(
            carry: tuple[optax.OptState, Array],
            batch: Array,
        ) -> tuple[tuple[optax.OptState, Array], Array]:
            opt_state, params = carry

            # Compute posterior statistics once for this batch
            posterior_stats = model.mean_posterior_statistics(params, batch)

            # Apply bounds to posterior statistics
            bounded_posterior_stats = self.bound_means(model, posterior_stats)
            # debug_lat_obs_means(bounded_posterior, True)

            # Define the inner step function for scan
            def inner_step(
                carry: tuple[optax.OptState, Array],
                _: None,  # Dummy input since we don't need per-step inputs
            ) -> tuple[tuple[optax.OptState, Array], Array]:
                current_opt_state, current_params = carry

                # Compute gradient as difference between posterior and current prior
                prior_stats = model.to_mean(current_params)
                grad = prior_stats - bounded_posterior_stats
                reg_fn = self.make_regularizer(model)
                reg_grad = reg_fn(current_params)[0]
                grad = grad + reg_grad

                # Update parameters
                updates, new_opt_state = optimizer.update(
                    grad, current_opt_state, current_params
                )
                new_params = optax.apply_updates(current_params, updates)

                # Monitor parameters for debugging
                logger.monitor_params(
                    {
                        "original_params": current_params,
                        "updated_params": new_params,
                        "batch": batch,
                        "posterior_stats": posterior_stats,
                        "bounded_posterior_stats": bounded_posterior_stats,
                        "prior_stats": prior_stats,
                        "grad": grad,
                    },
                    handler,
                    context="PRE",
                )

                return (new_opt_state, new_params), grad

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
        model: AnyLGM,
        logger: Logger,
        epoch_offset: int,
        params0: Array,
    ) -> Array:
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
        if self.grad_clip > 0.0:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.grad_clip),
                optax.adamw(learning_rate=self.lr),
            )
        else:
            optimizer = optax.adamw(learning_rate=self.lr)

        # Initialize optimizer state
        opt_state = optimizer.init(params0)

        # Log training phase
        log.info(f"Training PRE parameters for {n_epochs} epochs")

        # Create epoch step function
        def epoch_step(
            epoch: Array,
            carry: tuple[optax.OptState, Array, Array],
        ) -> tuple[optax.OptState, Array, Array]:
            opt_state, params, epoch_key = carry

            if self.epoch_reset:
                opt_state = optimizer.init(params)

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
            batch_grads = gradss_array.reshape(-1, *gradss_array.shape[2:])

            opt_state = optimizer.init(new_params)

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
        (_, params_final, _) = jax.lax.fori_loop(
            0, n_epochs, epoch_step, (opt_state, params0, key)
        )

        return params_final


@dataclass(frozen=True)
class MixtureGradientTrainer:
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
    l2_reg: float

    # Parameter bounds
    min_prob: float
    lat_min_var: float
    lat_jitter_var: float
    upr_prs_reg: float
    lwr_prs_reg: float
    mixture_entropy_reg: float

    epoch_reset: bool = True

    def precompute_observable_mappings(
        self,
        model: AnyHMoG,
        params: Array,
        data: Array,
    ) -> tuple[Array, Array]:
        log.info("Precomputing latent statistics for fixed observable training")

        # Get the posterior function (the affine map)
        obs_params, int_params, _ = model.split_coords(params)
        lkl_params = model.lkl_fun_man.join_coords(obs_params, int_params)
        rho = model.conjugation_parameters(lkl_params)

        def project_fn(x_means: Array) -> Array:
            return model.int_man.transpose_apply(int_params, x_means)

        return jax.vmap(project_fn)(data), rho

    def mean_posterior_statistics(
        self,
        model: AnyHMoG,
        mix_params: Array,
        latent_locations: Array,
    ) -> Array:
        # Get sufficient statistics of observation

        def posterior_statistics(
            latent_loc: Array,
        ) -> Array:
            # Get posterior statistics for this observation
            posterior_params = model.int_man.dom_emb.translate(mix_params, latent_loc)
            return model.pst_man.to_mean(posterior_params)

        return batched_mean(posterior_statistics, latent_locations, 256)

    def prior_statistics(
        self,
        model: AnyHMoG,
        mix_params: Array,
        rho: Array,
    ) -> Array:
        # Get sufficient statistics of observation

        def log_partition_function(
            params: Array,
        ) -> Array:
            # Split parameters
            con_params = model.pst_prr_emb.translate(rho, params)
            return model.prr_man.log_partition_function(con_params)

        return jax.grad(log_partition_function)(mix_params)

    def bound_mixture_means(self, model: AnyHMoG, mix_means: Array) -> Array:
        """Apply bounds to mixture weights for numerical stability."""
        with model.pst_man as uh:
            comp_meanss, cat_means = uh.split_mean_mixture(mix_means)
            probs = uh.lat_man.to_probs(cat_means)
            bounded_probs = jnp.clip(probs, self.min_prob, 1.0)
            bounded_probs = bounded_probs / jnp.sum(bounded_probs)
            bounded_prob_means = uh.lat_man.from_probs(bounded_probs)
            return uh.join_mean_mixture(comp_meanss, bounded_prob_means)

    def make_regularizer(
        self, model: AnyHMoG, rho: Array
    ) -> Callable[[Array], tuple[Array, MetricDict]]:
        """Create a unified regularizer that returns loss, metrics, and gradient."""

        def loss_with_metrics(
            mix_params: Array,
        ) -> tuple[Array, MetricDict]:
            # Extract components for regularization

            # L2 regularization
            l2_norm = jnp.sum(jnp.square(mix_params))
            l2_loss = self.l2_reg * l2_norm

            # Component precision regularization
            prs_loss, prs_metrics = precision_regularizer(
                model, rho, mix_params, self.upr_prs_reg, self.lwr_prs_reg
            )

            # Entropy penalty on mixture weights
            ent_loss, ent_metrics = mixture_entropy_regularizer(
                model, rho, mix_params, self.mixture_entropy_reg
            )

            # Combine into total loss and metrics
            total_loss = l2_loss + prs_loss + ent_loss

            metrics = {
                "Regularization/L2 Norm": (STATS_LEVEL, l2_norm),
                "Regularization/L2 Penalty": (STATS_LEVEL, l2_loss),
            }
            metrics.update(prs_metrics)
            metrics.update(ent_metrics)

            return total_loss, metrics

        # Create a function that computes loss, metrics, and gradients in one pass
        return jax.grad(loss_with_metrics, has_aux=True)

    def train(
        self,
        key: Array,
        handler: RunHandler,
        dataset: ClusteringDataset,
        model: AnyHMoG,
        logger: Logger,
        learning_rate_scale: float,
        epoch_offset: int,
        params0: Array,
    ) -> Array:
        """Train only the mixture parameters with fixed observable mapping.

        Args:
            key: Random key
            handler: Run handler for saving artifacts
            dataset: Dataset for training
            model:AnyHMoG model
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
        obs_params0, int_params0, mix_params0 = model.split_coords(params0)

        # Precompute all latent statistics using the fixed observable mapping
        latent_locations, rho = self.precompute_observable_mappings(
            model, params0, train_data
        )

        # Setup optimizer for mixture parameters
        if self.grad_clip > 0.0:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.grad_clip),
                optax.adam(learning_rate=self.lr * learning_rate_scale),
            )
        else:
            optimizer = optax.adam(learning_rate=self.lr * learning_rate_scale)

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
            carry: tuple[optax.OptState, Array],
            batch_locations: Array,
        ) -> tuple[tuple[optax.OptState, Array], Array]:
            opt_state, params = carry

            posterior_stats = self.mean_posterior_statistics(
                model, params, batch_locations
            )

            bounded_posterior_stats = self.bound_mixture_means(model, posterior_stats)

            # Define the inner step function
            def inner_step(
                carry: tuple[optax.OptState, Array],
                _: None,
            ) -> tuple[tuple[optax.OptState, Array], Array]:
                current_opt_state, current_params = carry

                # Current parameters as means
                prior_stats = self.prior_statistics(model, current_params, rho)

                # Compute gradient as difference between posterior and current prior
                grad = prior_stats - bounded_posterior_stats

                # Add regularization
                reg_grad = reg_fn(current_params)[0]
                grad = grad + reg_grad

                # Update parameters
                updates, new_opt_state = optimizer.update(
                    grad, current_opt_state, current_params
                )
                new_params = optax.apply_updates(current_params, updates)
                full_params = model.join_coords(
                    obs_params0, int_params0, current_params
                )
                new_full_params = model.join_coords(
                    obs_params0, int_params0, new_params
                )
                full_grad = model.join_coords(
                    model.lwr_hrm.obs_man.zeros(), model.lwr_hrm.int_man.zeros(), grad
                )

                # Monitor parameters for debugging
                logger.monitor_params(
                    {
                        "original_params": full_params,
                        "updated_params": new_full_params,
                        "mixture_stats": posterior_stats,
                        "bounded_mixture_stats": bounded_posterior_stats,
                        "prior_stats": prior_stats,
                        "grad": full_grad,
                    },
                    handler,
                    context="FIXED_OBSERVABLE",
                )

                return (new_opt_state, new_params), full_grad

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
            carry: tuple[optax.OptState, Array, Array],
        ) -> tuple[optax.OptState, Array, Array]:
            opt_state, mix_params, epoch_key = carry

            if self.epoch_reset:
                opt_state = optimizer.init(mix_params)

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
            batch_grads = gradss_array.reshape(-1, *gradss_array.shape[2:])

            # Reconstruct full model parameters for evaluation
            full_params = model.join_coords(obs_params0, int_params0, new_mix_params)

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
        (_, mix_params_final, _) = jax.lax.fori_loop(
            0, n_epochs, epoch_step, (opt_state, mix_params0, key)
        )

        # Reconstruct full parameters with updated mixture
        return model.join_coords(obs_params0, int_params0, mix_params_final)
