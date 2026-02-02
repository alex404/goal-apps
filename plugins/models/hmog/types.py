"""Shared type definitions for HMoG plugin."""

from goal.geometry import Diagonal
from goal.models import DifferentiableHMoG, NormalLGM

type DiagonalHMoG = DifferentiableHMoG[Diagonal, Diagonal]
type DiagonalLGM = NormalLGM[Diagonal, Diagonal]
