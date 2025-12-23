"""MFA (Mixture of Factor Analyzers) model plugin for goal-apps."""

from .configs import MFAConfig
from .model import MFAModel

__all__ = ["MFAConfig", "MFAModel"]
