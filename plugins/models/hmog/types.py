"""Shared type definitions for HMoG plugin."""

from goal.geometry import Diagonal, PositiveDefinite
from goal.models import DifferentiableHMoG, NormalLGM

type DiagonalHMoG = DifferentiableHMoG[Diagonal, Diagonal]
type FullLatentHMoG = DifferentiableHMoG[Diagonal, PositiveDefinite]
type AnyHMoG = DiagonalHMoG | FullLatentHMoG

type AnyLGM = NormalLGM[Diagonal, Diagonal] | NormalLGM[Diagonal, PositiveDefinite]
