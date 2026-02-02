"""Shared type definitions for MFA plugin."""

from goal.models import DiagonalNormal, FullNormal
from goal.models.graphical.mixture import (
    CompleteMixtureOfConjugated,
    CompleteMixtureOfSymmetric,
)

type MFA = (
    CompleteMixtureOfSymmetric[DiagonalNormal, FullNormal]
    | CompleteMixtureOfConjugated[DiagonalNormal, DiagonalNormal, FullNormal]
)
