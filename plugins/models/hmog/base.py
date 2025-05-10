"""Configuration for HMoG implementations."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable

import jax
from goal.geometry import (
    Diagonal,
    PositiveDefinite,
    Scale,
)
from goal.models import (
    AnalyticMixture,
    DifferentiableHMoG,
    DifferentiableLinearGaussianModel,
    Normal,
)
from jax import Array

# Start logger
log = logging.getLogger(__name__)


### Covariance Reps ###


class RepresentationType(Enum):
    scale = Scale
    diagonal = Diagonal
    positive_definite = PositiveDefinite


### HMoG Protocol ###

type HMoG = DifferentiableHMoG[Any, Any]
type LGM = DifferentiableLinearGaussianModel[Any, Any]
type Mixture = AnalyticMixture[Normal[Any]]


### Helpers ###


def fori[X](lower: int, upper: int, body_fun: Callable[[Array, X], X], init: X) -> X:
    return jax.lax.fori_loop(lower, upper, body_fun, init)  # pyright: ignore[reportUnknownVariableType]
