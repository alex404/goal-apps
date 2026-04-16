"""Training algorithms for MFA model using HMOG-style gradient computation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from goal.models import MixtureOfFactorAnalyzers
from jax import Array

from apps.interface import ClusteringDataset
from apps.runtime import STATS_LEVEL, Logger, MetricDict, l1_l2_regularizer

from .metrics import log_epoch_metrics
from .types import MFA

log = logging.getLogger(__name__)

ENTROPY_REG_METRIC_KEYS: frozenset[str] = frozenset({
    "Regularization/Mixture Entropy",
    "Regularization/Entropy Penalty",
})


def entropy_regularizer(
    mfa: MFA,
    rho: Array,
    lat_params: Array,
    ent_reg: float,
) -> tuple[Array, MetricDict]:
    """Entropy regularization on mixture weights.

    Adds ent_reg * neg_entropy(π) to the loss, pushing the mixing
    distribution toward uniformity (maximum entropy).

    Uses dual_potential: φ(η) = η·μ(η) - ψ(η) = -H(π) for Categorical,
    computed via logsumexp-based log_partition and softmax-based to_mean.
    Unlike direct p*log(p), never evaluates log(0) for dying components.
    """
    con_lat_params = mfa.pst_prr_emb.translate(rho, lat_params)
    _, cat_nat_params = mfa.prr_man.split_natural_mixture(con_lat_params)

    neg_entropy = mfa.prr_man.lat_man.dual_potential(cat_nat_params)

    entropy_loss = ent_reg * neg_entropy

    metrics: MetricDict = {
        "Regularization/Mixture Entropy": (STATS_LEVEL, -neg_entropy),
        "Regularization/Entropy Penalty": (STATS_LEVEL, entropy_loss),
    }
    return entropy_loss, metrics



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
    """L2 gradient penalty on all parameters. Stabilizes precision reg backward pass through Adam's moment estimates."""

    ent_reg: float
    """Entropy regularization on mixture weights (pushes toward uniformity)."""

    weight_decay: float
    """AdamW decoupled weight decay (0 = plain Adam)."""

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

    def bound_means(self, mfa: MFA, means: Array) -> Array:
        """Apply bounds to posterior statistics: regularize covariances, clip probabilities, then whiten."""
        obs_means, int_means, lat_means = mfa.split_coords(means)

        # Regularize observable covariance
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

        # Regularize per-component latent covariance
        pm = mfa.pst_man
        comp_w, cat_w = pm.split_mean_mixture(lat_means)

        comp_w = pm.cmp_man.map(
            lambda c: pm.obs_man.regularize_covariance(
                c, self.lat_jitter_var, self.lat_min_var
            ),
            comp_w,
            flatten=True,
        )

        # Clip mixture probabilities
        probs = pm.lat_man.to_probs(cat_w)
        bounded_probs = jnp.clip(probs, self.min_prob, 1.0)
        bounded_probs = bounded_probs / jnp.sum(bounded_probs)
        bounded_cat_w = pm.lat_man.from_probs(bounded_probs)
        bounded_lat_w = pm.join_mean_mixture(comp_w, bounded_cat_w)

        # Rejoin and whiten last
        bounded_means = mfa.join_coords(bounded_obs_means, int_means, bounded_lat_w)

        if isinstance(mfa, MixtureOfFactorAnalyzers):
            bounded_means = mfa.whiten_prior(bounded_means)

        return bounded_means

    def make_regularizer(
        self, mfa: MFA
    ) -> Callable[[Array], tuple[Array, MetricDict]]:
        """Create a unified regularizer that returns loss, metrics, and gradient."""

        def loss_with_metrics(params: Array) -> tuple[Array, MetricDict]:
            obs_params, int_params, lat_params = mfa.split_coords(params)

            total_loss, metrics = l1_l2_regularizer(
                params, int_params, self.l1_reg, self.l2_reg
            )

            # Gating on Python-level constants (frozen dataclass fields) so JAX eliminates
            # dead code at trace time.
            if self.ent_reg > 0:
                lkl_params = mfa.lkl_fun_man.join_coords(obs_params, int_params)
                rho = mfa.conjugation_parameters(lkl_params)

                ent_loss, ent_metrics = entropy_regularizer(
                    mfa, rho, lat_params, self.ent_reg
                )
                total_loss = total_loss + ent_loss
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
            optax.adamw(learning_rate=self.lr, weight_decay=self.weight_decay)
            if self.weight_decay > 0
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
                new_params: Array = optax.apply_updates(current_params, updates)  # pyright: ignore[reportAssignmentType]

                # Return per-component gradient norms (4 scalars) instead of full grad
                # to avoid accumulating (batch_steps, param_dim) arrays in scan.
                obs_g, int_g, lat_g = mfa.split_coords(grad)
                obs_loc_g, obs_prs_g = mfa.bas_hrm.obs_man.split_coords(obs_g)
                grad_norms = jnp.array([
                    jnp.linalg.norm(obs_loc_g),
                    jnp.linalg.norm(obs_prs_g),
                    jnp.linalg.norm(int_g),
                    jnp.linalg.norm(lat_g),
                ])

                return (new_opt_state, new_params), grad_norms

            # Run inner steps; each step returns (4,) grad norms, not full grad
            (final_opt_state, final_params), all_norm_arrs = jax.lax.scan(
                inner_step,
                (opt_state, params),
                None,
                length=self.batch_steps,
            )

            return (final_opt_state, final_params), all_norm_arrs

        def epoch_step(
            epoch: Array,
            carry: tuple[optax.OptState, Array, Array],
        ) -> tuple[optax.OptState, Array, Array]:
            opt_state, params, epoch_key = carry

            if self.epoch_reset:
                opt_state = optimizer.init(params)

            # Shuffle and batch data
            if n_batches > 1:
                shuffle_key, next_key = jax.random.split(epoch_key)
                shuffled_indices = jax.random.permutation(shuffle_key, n_samples)
                batched_indices = shuffled_indices[: n_batches * batch_size]
                batched_data = data[batched_indices].reshape((n_batches, batch_size, -1))
            else:
                next_key = epoch_key
                batched_data = data[None]  # shape (1, n_samples, data_dim)

            # Process all batches
            (opt_state, new_params), gradss = jax.lax.scan(
                batch_step, (opt_state, params), batched_data
            )

            _, reg_metrics = reg_fn(new_params)

            # gradss shape is (n_batches, batch_steps, 4) — pre-computed grad norms
            # Flatten to (n_batches * batch_steps, 4) for gradient norm logging
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
