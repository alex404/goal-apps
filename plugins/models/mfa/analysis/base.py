"""Helper functions for MFA analysis."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from ..base import MFA


def get_responsibilities(
    mfa: MFA,
    params: Array,
    data: Array,
) -> Array:
    """Compute posterior responsibilities p(z|x) for all data.

    This function computes the posterior probability that each data point
    belongs to each mixture component.

    Args:
        mfa: MixtureOfConjugated model instance
        params: Model parameters in natural coordinates
        data: Data array of shape (n_samples, data_dim)

    Returns:
        Array of shape (n_samples, n_clusters) with posterior probabilities,
        where responsibilities[i, k] = p(z=k | x=i, params)
    """

    def get_responsibilities_one(x: Array) -> Array:
        """Get responsibilities for one sample."""
        # Compute posterior latent parameters
        posterior_params = mfa.posterior_at(params, x)

        # Split CompleteMixture coordinates to extract categorical parameters
        # CompleteMixture structure: (observable_params, interaction_params, categorical_params)
        _, _, cat_params = mfa.lat_man.split_coords(posterior_params)

        # Convert categorical natural parameters to probabilities
        # cat_params is in natural coordinates for Categorical (n_categories - 1,)
        # Add reference category (0.0) and apply softmax
        cat_natural = jnp.concatenate([cat_params, jnp.array([0.0])])
        probs = jax.nn.softmax(cat_natural)

        return probs

    return jax.vmap(get_responsibilities_one)(data)
