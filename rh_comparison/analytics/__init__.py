# rh_comparison/analytics package — QT1–QT4 analytical functions (Phase 2 & 3)

from rh_comparison.analytics.structural_diff import compute_structural_diff
from rh_comparison.analytics.solution_diff import (
    compute_solution_diff,
    get_variable_family_details,
)
from rh_comparison.analytics.backward_compat import assess_backward_compat
from rh_comparison.analytics.attribution_analysis import compute_attribution_analysis

__all__ = [
    "compute_structural_diff",
    "compute_solution_diff",
    "get_variable_family_details",
    "assess_backward_compat",
    "compute_attribution_analysis",
]
