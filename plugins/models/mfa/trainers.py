"""Training algorithms for MFA model using HMOG-style gradient computation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from goal.geometry import Optimizer, OptState
from goal.models import Normal
from goal.models.graphical.mixture import MixtureOfConjugated
from jax import Array

from apps.runtime import STATS_NUM, Logger, MetricDict

log = logging.getLogger(__name__)

# Type alias for MFA model
type MFA = MixtureOfConjugated[Normal, Normal]

STATS_LEVEL = jnp.array(STATS_NUM)
INFO_LEVEL = jnp.array(logging.INFO)


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
    lat_man = mfa.lat_man  # CompleteMixture[Normal]

    # Split into component params and categorical
    comp_params, _ = lat_man.split_natural_mixture(con_lat_params)

    def compute_trace(nor_params: Array) -> Array:
        _, prs = lat_man.obs_man.split_location_precision(nor_params)
        prs_dense = lat_man.obs_man.cov_man.to_matrix(prs)
        return jnp.trace(prs_dense)

    traces = lat_man.cmp_man.map(compute_trace, comp_params)
    trace_sum = jnp.sum(traces)
    trace_reg = upr_prs_reg * trace_sum

    def compute_logdet(nor_params: Array) -> Array:
        _, prs = lat_man.obs_man.split_location_precision(nor_params)
        prs_dense = lat_man.obs_man.cov_man.to_matrix(prs)
        return -jnp.linalg.slogdet(prs_dense)[1]

    logdets = lat_man.cmp_man.map(compute_logdet, comp_params)
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

    batch_size: int | None = None
    """Batch size (None = full batch)."""

    batch_steps: int = 1
    """Number of gradient steps per batch."""

    grad_clip: float = 1.0
    """Maximum gradient norm for clipping."""

    # Regularization parameters
    l1_reg: float = 0.0
    """L1 regularization on interaction parameters."""

    l2_reg: float = 0.0
    """L2 regularization on all parameters."""

    upr_prs_reg: float = 1e-3
    """Upper bound regularization on precision eigenvalues."""

    lwr_prs_reg: float = 1e-3
    """Lower bound regularization on precision eigenvalues."""

    # Parameter bounds (applied in mean space)
    min_prob: float = 1e-4
    """Minimum cluster probability."""

    obs_min_var: float = 1e-5
    """Minimum observable variance."""

    lat_min_var: float = 1e-6
    """Minimum latent variance (currently unused - latents reset to standard normal)."""

    obs_jitter_var: float = 0.0
    """Jitter for observable variance."""

    lat_jitter_var: float = 0.0
    """Jitter for latent variance (currently unused - latents reset to standard normal)."""

    def bound_means(self, mfa: MFA, means: Array) -> Array:
        """Apply bounds to posterior/prior statistics in mean coordinates.

        Bounds observable covariances, clips mixture weights, and resets all
        latent components to standard normal N(0, I).

        This matches the gans-n-gmms approach where the latent prior is fixed
        at standard normal for ALL mixture components. All component variation
        must be expressed through the loading matrices and observation parameters.
        """
        obs_means, int_means, lat_means = mfa.split_coords(means)

        # Bound observable covariance
        bounded_obs_means = mfa.hrm.obs_man.regularize_covariance(
            obs_means, self.obs_jitter_var, self.obs_min_var
        )

        # Bound latent mixture parameters
        lat_man = mfa.lat_man  # CompleteMixture[Normal]

        # Split mixture into component means and categorical means
        _, cat_means = lat_man.split_mean_mixture(lat_means)

        # Reset ALL component latents to standard normal N(0, I)
        # This enforces identifiability by fixing the latent prior
        standard = lat_man.obs_man.standard_normal()
        reset_comp_meanss = jnp.tile(standard, lat_man.n_categories)

        # Bound categorical probabilities
        probs = lat_man.lat_man.to_probs(cat_means)
        bounded_probs = jnp.clip(probs, self.min_prob, 1.0)
        bounded_probs = bounded_probs / jnp.sum(bounded_probs)
        bounded_cat_means = lat_man.lat_man.from_probs(bounded_probs)

        # Rejoin mixture with bounded components
        bounded_lat_means = lat_man.join_mean_mixture(
            reset_comp_meanss, bounded_cat_means
        )

        return mfa.join_coords(bounded_obs_means, int_means, bounded_lat_means)

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

            # Combine into total loss and metrics
            total_loss = l1_loss + l2_loss + prs_loss

            metrics: MetricDict = {
                "Regularization/L1 Norm": (STATS_LEVEL, l1_norm),
                "Regularization/L2 Norm": (STATS_LEVEL, l2_norm),
                "Regularization/L1 Penalty": (STATS_LEVEL, l1_loss),
                "Regularization/L2 Penalty": (STATS_LEVEL, l2_loss),
            }
            metrics.update(prs_metrics)

            return total_loss, metrics

        return jax.grad(loss_with_metrics, has_aux=True)

    def train(
        self,
        mfa: MFA,
        data: Array,
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

        # Setup batching
        n_samples = data.shape[0]
        if self.batch_size is None:
            batch_size = n_samples
            n_batches = 1
        else:
            batch_size = self.batch_size
            n_batches = n_samples // batch_size

        # Create optimizer
        optim = optax.adamw(learning_rate=self.lr)
        optimizer: Optimizer[MFA] = Optimizer(optim, mfa)
        if self.grad_clip > 0.0:
            optimizer = optimizer.with_grad_clip(self.grad_clip)
        opt_state = optimizer.init(params0)

        # Create regularizer
        reg_fn = self.make_regularizer(mfa)

        log.info(f"Training MFA for {self.n_epochs} epochs (lr={self.lr})")

        def batch_step(
            carry: tuple[OptState, Array],
            batch: Array,
        ) -> tuple[tuple[OptState, Array], Array]:
            opt_state, params = carry

            # Compute posterior statistics (mean coordinates)
            posterior_stats = mfa.mean_posterior_statistics(params, batch)

            # Bound posterior statistics in mean space
            bounded_posterior_stats = self.bound_means(mfa, posterior_stats)

            def inner_step(
                carry: tuple[OptState, Array],
                _: None,
            ) -> tuple[tuple[OptState, Array], Array]:
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
                new_opt_state, new_params = optimizer.update(
                    current_opt_state, grad, current_params
                )

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
            carry: tuple[OptState, Array, Array],
        ) -> tuple[OptState, Array, Array]:
            opt_state, params, epoch_key = carry

            shuffle_key, next_key = jax.random.split(epoch_key)

            # Shuffle and batch data
            shuffled_indices = jax.random.permutation(shuffle_key, n_samples)
            batched_indices = shuffled_indices[: n_batches * batch_size]
            batched_data = data[batched_indices].reshape((n_batches, batch_size, -1))

            # Process all batches
            (opt_state, new_params), gradss = jax.lax.scan(
                batch_step, (opt_state, params), batched_data
            )

            # Compute metrics for logging
            train_ll = mfa.average_log_observable_density(new_params, data)
            grad_norms = jnp.sqrt(jnp.sum(gradss**2, axis=-1))
            avg_grad_norm = jnp.mean(grad_norms)

            metrics: MetricDict = {
                "train/log_likelihood": (INFO_LEVEL, train_ll),
                "train/grad_norm": (INFO_LEVEL, avg_grad_norm),
            }

            logger.log_metrics(metrics, epoch + epoch_offset + 1)

            return opt_state, new_params, next_key

        # Run training loop
        _, final_params, _ = jax.lax.fori_loop(
            0, self.n_epochs, epoch_step, (opt_state, params0, key)
        )

        return final_params
