"""Shared utilities for GOAL examples."""

from __future__ import annotations

# Re-export run configs from their canonical locations
from ..interface.clustering.config import ClusteringRunConfig, RunConfig

__all__ = [
    "RunConfig",
    "ClusteringRunConfig",
]
