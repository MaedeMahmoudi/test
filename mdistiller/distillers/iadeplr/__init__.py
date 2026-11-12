"""Interaction-Aware Dynamic Edge Pruning with Low-Rank curvature."""

from .pruning import (
    IADEPLR,
    compute_edge_gradients,
    compute_interaction_matrix,
    iadeplr_pruning_decision,
    nystrom_hessian_approx,
    apply_inverse_nystrom,
    diagnostic_experiment,
)

__all__ = [
    "IADEPLR", "compute_edge_gradients", "compute_interaction_matrix",
    "iadeplr_pruning_decision", "nystrom_hessian_approx",
    "apply_inverse_nystrom", "diagnostic_experiment",
]
