"""Base class for HMoG implementations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import jax
import jax.numpy as jnp
from goal.geometry import (
    Diagonal,
    Mean,
    Natural,
    Point,
    PositiveDefinite,
    Scale,
)
from goal.models import (
    DifferentiableHMoG,
    DifferentiableMixture,
    FullNormal,
    LinearGaussianModel,
    Normal,
)
from hydra.core.config_store import ConfigStore
from jax import Array
from omegaconf import MISSING

from apps.configs import ClusteringModelConfig

### Preamble ###

# Start logger
log = logging.getLogger(__name__)


# Specify config
class RepresentationType(Enum):
    scale = Scale
    diagonal = Diagonal
    positive_definite = PositiveDefinite


@dataclass
class HMoGConfig(ClusteringModelConfig):
    """Configuration for Hierarchical Mixture of Gaussians model.

    Model Architecture:
        latent_dim: Dimension of latent space [default: 10]
        n_clusters: Number of mixture components [default: 10]
        data_dim: Dimension of input data [set by dataset]
        obs_rep: Representation type for observations. Options: scale, diagonal, positive_definite [default: diagonal]
        lat_rep: Representation type for latents. Options: scale, diagonal, positive_definite [default: diagonal]

    Training Parameters:
        batch_size: Batch size for stage 3 [default: 256]
        stage1_epochs: Number of epochs for EM initialization [default: 100]
        stage2_epochs: Number of epochs for mixture component training [default: 100]
        stage3_epochs: Number of epochs for full model training [default: 100]
        stage2_learning_rate: Learning rate for stage 2 [default: 0.001]
        stage3_learning_rate: Learning rate for stage 3 [default: 0.0003]
    """

    _target_: str = "plugins.models.hmog.experiment.HMoGExperiment"
    data_dim: int = MISSING
    latent_dim: int = 10
    n_clusters: int = 10
    obs_rep: RepresentationType = RepresentationType.diagonal
    lat_rep: RepresentationType = RepresentationType.diagonal
    batch_size: int = 256
    stage1_epochs: int = 100
    stage2_epochs: int = 100
    stage3_epochs: int = 100
    stage2_learning_rate: float = 1e-3
    stage3_learning_rate: float = 3e-4
    min_prob: float = 1e-3
    obs_jitter: float = 1e-5
    obs_min_var: float = 1e-4
    lat_jitter: float = 1e-7
    lat_min_var: float = 1e-6
    from_scratch: bool = False
    analysis_epoch: int | None = None


# Register config
cs = ConfigStore.instance()
cs.store(group="model", name="hmog", node=HMoGConfig)

### Metrics ###


### Helper Functions ###


def fori[X](lower: int, upper: int, body_fun: Callable[[Array, X], X], init: X) -> X:
    return jax.lax.fori_loop(lower, upper, body_fun, init)  # pyright: ignore[reportUnknownVariableType]


### Stabilizers ###


def bound_observable_variances[Rep: PositiveDefinite](
    model: LinearGaussianModel[Rep],
    means: Point[Mean, LinearGaussianModel[Rep]],
    min_var: float,
    jitter: float,
) -> Point[Mean, LinearGaussianModel[Rep]]:
    """Regularize observable covariance parameters in mean coordinates."""
    obs_means, int_means, lat_means = model.split_params(means)
    bounded_obs_means = model.obs_man.regularize_covariance(obs_means, jitter, min_var)
    return model.join_params(bounded_obs_means, int_means, lat_means)


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


def bound_hmog_parameters[
    ObsRep: PositiveDefinite,
    LatRep: PositiveDefinite,
](
    model: DifferentiableHMoG[ObsRep, LatRep],
    params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
    min_prob: float,
    obs_min_var: float,
    obs_jitter: float,
    lat_min_var: float,
    lat_jitter: float,
) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]:
    """Apply all parameter bounds to HMoG."""
    # Split parameters
    lkl_params, mix_params = model.split_conjugated(params)

    # Bound mixture parameters
    bounded_mix_params = bound_mixture_parameters(
        model.upr_hrm, mix_params, min_prob, lat_min_var, lat_jitter
    )

    # Bound observable parameters
    obs_params, int_params = model.lkl_man.split_params(lkl_params)
    obs_means = model.obs_man.to_mean(obs_params)
    bounded_obs_means = model.obs_man.regularize_covariance(
        obs_means, obs_jitter, obs_min_var
    )
    bounded_obs_params = model.obs_man.to_natural(bounded_obs_means)
    bounded_lkl_params = model.lkl_man.join_params(bounded_obs_params, int_params)

    return model.join_conjugated(bounded_lkl_params, bounded_mix_params)
