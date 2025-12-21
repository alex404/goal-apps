"""Shared utilities for GOAL examples."""

from __future__ import annotations

# Re-export run configs from their canonical locations
from ..interface import RunConfig
from ..interface.clustering.config import ClusteringRunConfig

__all__ = [
    "ClusteringRunConfig",
    "RunConfig",
]
