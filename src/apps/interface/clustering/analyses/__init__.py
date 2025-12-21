"""Reusable analysis implementations for clustering models."""

from .clusters import ClusterStatistics, ClusterStatisticsAnalysis
from .hierarchy import (
    ClusterHierarchy,
    CoAssignmentHierarchy,
    CoAssignmentHierarchyAnalysis,
    build_hierarchy_from_distance,
    compute_co_assignment_matrix,
    plot_hierarchy_dendrogram,
)
from .merge import (
    CoAssignmentMergeAnalysis,
    CoAssignmentMergeResults,
    MergeAnalysis,
    MergeResults,
    OptimalMergeAnalysis,
    OptimalMergeResults,
    compute_merge_metrics,
    compute_optimal_mapping,
    get_valid_clusters,
    hierarchy_to_mapping,
    plot_merge_results,
)

__all__ = [
    "ClusterStatistics",
    "ClusterStatisticsAnalysis",
    # Hierarchy
    "ClusterHierarchy",
    "CoAssignmentHierarchy",
    "CoAssignmentHierarchyAnalysis",
    "build_hierarchy_from_distance",
    "compute_co_assignment_matrix",
    "plot_hierarchy_dendrogram",
    # Merge
    "CoAssignmentMergeAnalysis",
    "CoAssignmentMergeResults",
    "MergeAnalysis",
    "MergeResults",
    "OptimalMergeAnalysis",
    "OptimalMergeResults",
    "compute_merge_metrics",
    "compute_optimal_mapping",
    "get_valid_clusters",
    "hierarchy_to_mapping",
    "plot_merge_results",
]
