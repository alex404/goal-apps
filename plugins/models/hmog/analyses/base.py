"""HMoG-specific analysis utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from goal.models import FullNormal, NormalAnalyticLGM
from jax import Array

from ..types import AnyHMoG


### HMoG-specific Analysis ###


def cluster_assignments(model: AnyHMoG, params: Array, data: Array) -> Array:
    """Assign data points to clusters using the model.

    Args:
        model: DifferentiableHMoG model
        params: Model parameters
        data: Data points to assign

    Returns:
        Array of cluster assignments
    """
    return jax.lax.map(
        lambda x: model.posterior_hard_assignment(params, x), data, batch_size=2048
    )


def symmetric_kl_matrix(
    model: AnyHMoG,
    params: Array,
) -> Array:
    mix_params = model.prior(params)
    with model.prr_man as ch:
        comp_lats, _ = ch.split_natural_mixture(mix_params)

        # Convert flat components to 2D for indexing inside vmap
        comp_lats_2d = ch.cmp_man.to_2d(comp_lats)

        def kl_div_between_components(i: Array, j: Array) -> Array:
            comp_i = comp_lats_2d[i]
            comp_i_mean = ch.obs_man.to_mean(comp_i)
            comp_j = comp_lats_2d[j]
            return ch.obs_man.relative_entropy(comp_i_mean, comp_j)

        idxs = jnp.arange(ch.n_categories)

    def kl_div_from_one_to_all(i: Array) -> Array:
        return jax.vmap(kl_div_between_components, in_axes=(None, 0))(i, idxs)

    kl_matrix = jax.lax.map(kl_div_from_one_to_all, idxs)
    return (kl_matrix + kl_matrix.T) / 2


def get_component_prototypes(
    model: AnyHMoG,
    params: Array,
) -> list[Array]:
    # Split into likelihood and mixture parameters
    lkl_params, mix_params = model.split_conjugated(params)

    # Extract components from mixture
    comp_lats, _ = model.prr_man.split_natural_mixture(mix_params)

    # For each component, compute the observable distribution and get its mean
    prototypes: list[Array] = []

    ana_lgm = NormalAnalyticLGM(
        obs_dim=model.lwr_hrm.obs_dim,  # Original latent becomes observable
        obs_rep=model.lwr_hrm.obs_rep,
        lat_dim=model.lwr_hrm.lat_dim,  # Original observable becomes latent
    )

    for i in range(model.prr_man.cmp_man.n_reps):
        # Get latent params for this component (comp_lats is flat)
        comp_lat_params = model.prr_man.cmp_man.get_replicate(comp_lats, i)
        lwr_hrm_params = ana_lgm.join_conjugated(lkl_params, comp_lat_params)
        lwr_hrm_means = ana_lgm.to_mean(lwr_hrm_params)
        lwr_hrm_obs = ana_lgm.split_coords(lwr_hrm_means)[0]
        obs_means = ana_lgm.obs_man.split_mean_second_moment(lwr_hrm_obs)[0]

        prototypes.append(obs_means)

    return prototypes


def cluster_probabilities(
    model: AnyHMoG, params: Array, data: Array
) -> Array:
    """Get cluster probability distributions for each data point.

    Args:
        model: DifferentiableHMoG model
        params: Model parameters
        data: Data points to get probabilities for

    Returns:
        Array of shape (n_samples, n_clusters) with probability distributions
    """
    return jax.lax.map(
        lambda x: model.posterior_soft_assignments(params, x), data, batch_size=2048
    )


def analyze_component(nor_man: FullNormal, nrm_params: Array) -> Array:
    nrm_means = nor_man.to_mean(nrm_params)
    loc, prs = nor_man.split_location_precision(nrm_params)
    mean, cov = nor_man.split_mean_covariance(nrm_means)
    dns_prs = nor_man.cov_man.to_matrix(prs)
    dns_cov = nor_man.cov_man.to_matrix(cov)
    loc_nrm = jnp.linalg.norm(loc)
    mean_nrm = jnp.linalg.norm(mean)
    prs_cond = jnp.linalg.cond(dns_prs)
    cov_cond = jnp.linalg.cond(dns_cov)
    prs_ldet = jnp.linalg.slogdet(dns_prs)[1]
    cov_ldet = jnp.linalg.slogdet(dns_cov)[1]
    return jnp.asarray(
        [
            loc_nrm,
            mean_nrm,
            prs_cond,
            cov_cond,
            prs_ldet,
            cov_ldet,
        ]
    )
