"""Shared type definitions for MFA plugin."""

from goal.models import DiagonalNormal, FullNormal, MixtureOfFactorAnalyzers
from goal.models.graphical.mixture import CompleteMixtureOfConjugated

type MFA = (
    MixtureOfFactorAnalyzers
    | CompleteMixtureOfConjugated[DiagonalNormal, DiagonalNormal, FullNormal]
)
