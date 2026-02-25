"""Training algorithms for MFA model using HMOG-style gradient computation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from jax import Array

from apps.interface import ClusteringDataset
from apps.runtime import STATS_NUM, Logger, MetricDict

from goal.models import MixtureOfFactorAnalyzers

from .metrics import log_epoch_metrics
from .types import MFA

log = logging.getLogger(__name__)

STATS_LEVEL = jnp.array(STATS_NUM)


def mixture_entropy_regularizer(
    mfa: MFA,
    rho: Array,
    lat_params: Array,
    entropy_reg: float,
) -> tuple[Array, MetricDict]:
    """Entropy regularization on mixture weights.

    Adds entropy_reg * neg_entropy(π) to the loss, pushing the mixing
    distribution toward uniformity (maximum entropy).
    Default entropy_reg=0.0 disables the penalty entirely.
    """
    con_lat_params = mfa.pst_prr_emb.translate(rho, lat_params)
    _, cat_nat_params = mfa.prr_man.split_natural_mixture(con_lat_params)

    cat_means = mfa.prr_man.lat_man.to_mean(cat_nat_params)
    neg_entropy = mfa.prr_man.lat_man.negative_entropy(cat_means)

    entropy_loss = entropy_reg * neg_entropy

    metrics: MetricDict = {
        "Regularization/Mixture Entropy": (STATS_LEVEL, -neg_entropy),
        "Regularization/Entropy Penalty": (STATS_LEVEL, entropy_loss),
    }
    return entropy_loss, metrics


def precision_regularizer(
    mfa: MFA,
    rho: Array,
    lat_params: Array,
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

    Args:
        mfa: MFA model
        rho: Conjugation parameters
        lat_params: Latent mixture parameters in natural coordinates
        upr_prs_reg: Weight for trace penalty (controls upper bound on eigenvalues)
        lwr_prs_reg: Weight for log-det penalty (controls lower bound on eigenvalues)

    Returns:
        Tuple of (regularization_loss, metrics_dict)
    """
    # Apply conjugation to get prior-space parameters (matches HMOG pattern)
    con_lat_params = mfa.pst_prr_emb.translate(rho, lat_params)
    prr_man = mfa.prr_man  # CompleteMixture[Normal] - prior manifold

    # Split into component params and categorical
    comp_params, _ = prr_man.split_natural_mixture(con_lat_params)

    def compute_trace(nor_params: Array) -> Array:
        _, prs = prr_man.obs_man.split_location_precision(nor_params)
        prs_dense = prr_man.obs_man.cov_man.to_matrix(prs)
        return jnp.trace(prs_dense)

    traces = prr_man.cmp_man.map(compute_trace, comp_params)
    trace_sum = jnp.sum(traces)
    trace_reg = upr_prs_reg * trace_sum

    def compute_logdet(nor_params: Array) -> Array:
        _, prs = prr_man.obs_man.split_location_precision(nor_params)
        prs_dense = prr_man.obs_man.cov_man.to_matrix(prs)
        return -jnp.linalg.slogdet(prs_dense)[1]

    logdets = prr_man.cmp_man.map(compute_logdet, comp_params)
    logdet_sum = jnp.sum(logdets)
    logdet_reg = lwr_prs_reg * logdet_sum

    metrics: MetricDict = {
        "Regularization/Precision Trace Sum": (STATS_LEVEL, trace_sum),
        "Regularization/Precision LogDet Sum": (STATS_LEVEL, -logdet_sum),
        "Regularization/Trace Penalty": (STATS_LEVEL, trace_reg),
        "Regularization/LogDet Penalty": (STATS_LEVEL, logdet_reg),
    }
    return trace_reg + logdet_reg, metrics


@dataclass(frozen=True)
class GradientTrainer:
    """Gradient descent trainer for MFA using HMOG-style gradient computation.

    Computes gradients manually as (prior_stats - bounded_posterior_stats),
    which allows bounding statistics in mean coordinate space.
    """

    # Training hyperparameters
    lr: float
    """Learning rate for optimizer."""

    n_epochs: int
    """Number of training epochs."""

    batch_size: int | None
    """Batch size (None = full batch)."""

    batch_steps: int
    """Number of gradient steps per batch."""

    grad_clip: float
    """Maximum gradient norm for clipping."""

    # Regularization parameters
    l1_reg: float
    """L1 regularization on interaction parameters."""

    l2_reg: float
    """L2 regularization on all parameters."""

    upr_prs_reg: float
    """Upper bound regularization on precision eigenvalues."""

    lwr_prs_reg: float
    """Lower bound regularization on precision eigenvalues."""

    mixture_entropy_reg: float
    """Entropy regularization on mixture weights (pushes toward uniformity)."""

    log_freq: int
    """Log metrics every log_freq epochs."""

    # Parameter bounds (applied in mean space)
    min_prob: float
    """Minimum cluster probability."""

    obs_min_var: float
    """Minimum observable variance."""

    lat_min_var: float
    """Minimum latent variance."""

    obs_jitter_var: float
    """Jitter for observable variance."""

    lat_jitter_var: float
    """Jitter for latent variance."""

    obs_max_var: float
    """Maximum observable variance (0.0 = disabled). Matches reference sqrt_D ≤ 1.0 when set to 1.0."""

    epoch_reset: bool
    """Reset optimizer state at the start of each epoch."""

    use_adamw: bool
    """Use AdamW (with weight decay) instead of plain Adam."""

    def bound_means(self, mfa: MFA, means: Array) -> Array:
        """Apply bounds to posterior statistics: bound observable covariance, whiten, and clip mixture weights."""
        obs_means, int_means, lat_means = mfa.split_coords(means)

        bounded_obs_means = mfa.bas_hrm.obs_man.regularize_covariance(
            obs_means, self.obs_jitter_var, self.obs_min_var
        )

        if self.obs_max_var > 0.0:
            obs_man = mfa.bas_hrm.obs_man
            obs_mean_vec, obs_cov = obs_man.split_mean_covariance(bounded_obs_means)
            clipped_cov = obs_man.cov_man.map_diagonal(
                obs_cov, lambda x: jnp.minimum(x, self.obs_max_var)
            )
            bounded_obs_means = obs_man.join_mean_covariance(obs_mean_vec, clipped_cov)

        bounded_means = mfa.join_coords(bounded_obs_means, int_means, lat_means)

        if isinstance(mfa, MixtureOfFactorAnalyzers):
            bounded_means = mfa.whiten_prior(bounded_means)

        obs_w, int_w, lat_w = mfa.split_coords(bounded_means)

        with mfa.pst_man as pm:
            comp_w, cat_w = pm.split_mean_mixture(lat_w)
            probs = pm.lat_man.to_probs(cat_w)
            bounded_probs = jnp.clip(probs, self.min_prob, 1.0)
            bounded_probs = bounded_probs / jnp.sum(bounded_probs)
            bounded_cat_w = pm.lat_man.from_probs(bounded_probs)
            bounded_lat_w = pm.join_mean_mixture(comp_w, bounded_cat_w)

        return mfa.join_coords(obs_w, int_w, bounded_lat_w)

    def make_regularizer(
        self, mfa: MFA
    ) -> Callable[[Array], tuple[Array, tuple[Array, MetricDict]]]:
        """Create a unified regularizer that returns loss, metrics, and gradient."""

        def loss_with_metrics(params: Array) -> tuple[Array, MetricDict]:
            # Extract components for regularization (matches HMOG pattern)
            obs_params, int_params, lat_params = mfa.split_coords(params)
            lkl_params = mfa.lkl_fun_man.join_coords(obs_params, int_params)
            rho = mfa.conjugation_parameters(lkl_params)

            # L1 regularization on interactions
            l1_norm = jnp.sum(jnp.abs(int_params))
            l1_loss = self.l1_reg * l1_norm

            # L2 regularization on all params
            l2_norm = jnp.sum(jnp.square(params))
            l2_loss = self.l2_reg * l2_norm

            # Component precision regularization
            prs_loss, prs_metrics = precision_regularizer(
                mfa, rho, lat_params, self.upr_prs_reg, self.lwr_prs_reg
            )

            # Entropy regularization on mixture weights
            ent_loss, ent_metrics = mixture_entropy_regularizer(
                mfa, rho, lat_params, self.mixture_entropy_reg
            )

            # Combine into total loss and metrics
            total_loss = l1_loss + l2_loss + prs_loss + ent_loss

            metrics: MetricDict = {
                "Regularization/L1 Norm": (STATS_LEVEL, l1_norm),
                "Regularization/L2 Norm": (STATS_LEVEL, l2_norm),
                "Regularization/L1 Penalty": (STATS_LEVEL, l1_loss),
                "Regularization/L2 Penalty": (STATS_LEVEL, l2_loss),
            }
            metrics.update(prs_metrics)
            metrics.update(ent_metrics)

            return total_loss, metrics

        return jax.grad(loss_with_metrics, has_aux=True)

    def train(
        self,
        mfa: MFA,
        dataset: ClusteringDataset,
        logger: Logger,
        epoch_offset: int,
        params0: Array,
        key: Array | None = None,
    ) -> Array:
        """Execute training loop with HMOG-style gradient computation.

        The gradient is computed as:
            grad = prior_stats - bounded_posterior_stats + reg_grad

        where statistics are in mean coordinates, allowing stable bounding.
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        data = dataset.train_data

        # Setup batching
        n_samples = data.shape[0]
        if self.batch_size is None:
            batch_size = n_samples
            n_batches = 1
        else:
            batch_size = self.batch_size
            n_batches = n_samples // batch_size

        # Create optimizer
        base_optimizer = (
            optax.adamw(learning_rate=self.lr)
            if self.use_adamw
            else optax.adam(self.lr)
        )
        if self.grad_clip > 0.0:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.grad_clip),
                base_optimizer,
            )
        else:
            optimizer = base_optimizer
        opt_state = optimizer.init(params0)

        # Create regularizer
        reg_fn = self.make_regularizer(mfa)

        log.info(f"Training MFA for {self.n_epochs} epochs (lr={self.lr})")

        def batch_step(
            carry: tuple[optax.OptState, Array],
            batch: Array,
        ) -> tuple[tuple[optax.OptState, Array], Array]:
            opt_state, params = carry

            # Compute posterior statistics (mean coordinates)
            posterior_stats = mfa.mean_posterior_statistics(params, batch)

            bounded_posterior_stats = self.bound_means(mfa, posterior_stats)

            def inner_step(
                carry: tuple[optax.OptState, Array],
                _: None,
            ) -> tuple[tuple[optax.OptState, Array], Array]:
                current_opt_state, current_params = carry

                # Prior statistics (mean coordinates)
                prior_stats = mfa.to_mean(current_params)

                # Gradient = prior - bounded_posterior
                grad = prior_stats - bounded_posterior_stats

                # Add regularization gradient
                reg_grad, _aux = reg_fn(current_params)
                del _aux  # Unused auxiliary output
                grad = grad + reg_grad

                # Update parameters
                updates, new_opt_state = optimizer.update(
                    grad, current_opt_state, current_params
                )
                new_params = optax.apply_updates(current_params, updates)

                return (new_opt_state, new_params), grad

            # Run inner steps
            (final_opt_state, final_params), all_grads = jax.lax.scan(
                inner_step,
                (opt_state, params),
                None,
                length=self.batch_steps,
            )

            return (final_opt_state, final_params), all_grads

        def epoch_step(
            epoch: Array,
            carry: tuple[optax.OptState, Array, Array],
        ) -> tuple[optax.OptState, Array, Array]:
            opt_state, params, epoch_key = carry

            if self.epoch_reset:
                opt_state = optimizer.init(params)

            shuffle_key, next_key = jax.random.split(epoch_key)

            # Shuffle and batch data
            shuffled_indices = jax.random.permutation(shuffle_key, n_samples)
            batched_indices = shuffled_indices[: n_batches * batch_size]
            batched_data = data[batched_indices].reshape((n_batches, batch_size, -1))

            # Process all batches
            (opt_state, new_params), gradss = jax.lax.scan(
                batch_step, (opt_state, params), batched_data
            )

            _, reg_metrics = reg_fn(new_params)

            # Log metrics using the new metrics module
            # gradss shape is (n_batches, batch_steps, param_dim)
            # Flatten to (n_batches * batch_steps, param_dim) for gradient norm computation
            flat_grads = gradss.reshape(-1, gradss.shape[-1])

            log_epoch_metrics(
                dataset,
                mfa,
                logger,
                new_params,
                epoch + epoch_offset,
                reg_metrics,
                flat_grads,
                self.log_freq,
            )

            return opt_state, new_params, next_key

        # Run training loop
        _, final_params, _ = jax.lax.fori_loop(
            0, self.n_epochs, epoch_step, (opt_state, params0, key)
        )

        return final_params
