"""Configuration for HMoG implementations."""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Enum

import jax
from goal.geometry import (
    Diagonal,
    PositiveDefinite,
    Scale,
)
from jax import Array

# Start logger
log = logging.getLogger(__name__)


### Covariance Reps ###


class RepresentationType(Enum):
    scale = Scale
    diagonal = Diagonal
    positive_definite = PositiveDefinite


### Helpers ###


def fori[X](lower: int, upper: int, body_fun: Callable[[Array, X], X], init: X) -> X:
    return jax.lax.fori_loop(lower, upper, body_fun, init)
