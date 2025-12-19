"""Training and evaluation metrics analysis for MFA model.

This module now uses the generic clustering metrics from apps.interface.clustering,
demonstrating how model-specific analyses can be built on top of reusable
framework components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from apps.interface.clustering.analyses import (
    ClusteringMetrics,
    ClusteringMetricsAnalysis,
)

if TYPE_CHECKING:
    from ..model import MFAModel

# Re-export the artifact for backwards compatibility
TrainingMetrics = ClusteringMetrics

# MFA uses the generic clustering metrics analysis directly
# No customization needed - just specify the model type
# Using string annotation to avoid circular import
TrainingMetricsAnalysis = ClusteringMetricsAnalysis["MFAModel"]  # type: ignore[valid-type]
