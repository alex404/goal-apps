"""Shared type definitions for HMoG plugin."""

from goal.geometry import Diagonal, PositiveDefinite
from goal.models import DifferentiableHMoG
from goal.models.harmonium.lgm import NormalAnalyticLGM

type DiagonalHMoG = DifferentiableHMoG[Diagonal, Diagonal]
type FullLatentHMoG = DifferentiableHMoG[Diagonal, PositiveDefinite]
type AnyHMoG = DiagonalHMoG | FullLatentHMoG

type AnyLGM = NormalAnalyticLGM[Diagonal]
