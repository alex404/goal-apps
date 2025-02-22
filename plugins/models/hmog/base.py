"""Base class for HMoG implementations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypedDict

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
    obs_jitter: float = 1e-8
    obs_min_var: float = 1e-7
    from_scratch: bool = False
    analysis_epoch: int | None = None


# Register config
cs = ConfigStore.instance()
cs.store(group="model", name="hmog", node=HMoGConfig)

### Metrics ###


class HMoGMetrics(TypedDict):
    train_ll: Array
    test_ll: Array
    negative_bic: Array


### Helper Functions ###


def fori[X](lower: int, upper: int, body_fun: Callable[[int, X], X], init: X) -> X:
    return jax.lax.fori_loop(lower, upper, body_fun, init)  # pyright: ignore[reportUnknownVariableType]


### Stabilizers ###


def bound_lgm_means[Rep: PositiveDefinite](
    model: LinearGaussianModel[Rep],
    means: Point[Mean, LinearGaussianModel[Rep]],
    min_var: float,
    jitter: float,
) -> Point[Mean, LinearGaussianModel[Rep]]:
    """Regularize covariance parameters in LinearGaussianModel mean coordinates."""
    obs_means, int_means, lat_means = model.split_params(means)

    # Regularize observation covariance
    bounded_obs_means = model.obs_man.regularize_covariance(obs_means, jitter, min_var)

    # Rejoin parameters
    return model.join_params(bounded_obs_means, int_means, lat_means)


def bound_mixture_probabilities[Rep: PositiveDefinite](
    model: DifferentiableMixture[FullNormal, Normal[Rep]],
    params: Point[Natural, DifferentiableMixture[FullNormal, Normal[Rep]]],
    min_prob: float,
) -> Point[Natural, DifferentiableMixture[FullNormal, Normal[Rep]]]:
    """Bound mixture probabilities away from 0."""
    comps, cat_params = model.split_natural_mixture(params)

    with model.lat_man as lm:
        cat_means = lm.to_mean(cat_params)
        probs = lm.to_probs(cat_means)
        bounded_probs = jnp.clip(probs, min_prob, 1.0)
        bounded_probs = bounded_probs / jnp.sum(bounded_probs)
        bounded_cat_params = lm.to_natural(lm.from_probs(bounded_probs))

    return model.join_natural_mixture(comps, bounded_cat_params)


def bound_hmog_parameters[
    ObsRep: PositiveDefinite,
    LatRep: PositiveDefinite,
](
    model: DifferentiableHMoG[ObsRep, LatRep],
    params: Point[Natural, DifferentiableHMoG[ObsRep, LatRep]],
    min_prob: float,
    min_var: float,
    jitter: float,
) -> Point[Natural, DifferentiableHMoG[ObsRep, LatRep]]:
    """Bound both mixture probabilities and observable covariance parameters in a single operation."""
    # Split once
    lkl_params, mix_params = model.split_conjugated(params)

    # Bound mixture probabilities
    bounded_mix_params = bound_mixture_probabilities(
        model.upr_hrm, mix_params, min_prob
    )
    params = model.join_conjugated(lkl_params, bounded_mix_params)

    # Bound observable parameters
    obs_params, int_params, mix_params = model.split_params(params)

    # Convert to mean coordinates for bounding
    obs_means = model.obs_man.to_mean(obs_params)

    # Apply regularization in mean coordinates
    bounded_obs_means = model.obs_man.regularize_covariance(obs_means, jitter, min_var)

    # Convert back to natural coordinates
    bounded_obs_params = model.obs_man.to_natural(bounded_obs_means)

    # Join only once at the end
    return model.join_params(bounded_obs_params, int_params, mix_params)
