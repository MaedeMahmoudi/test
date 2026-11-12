# IADEPLR implementation notes

## Objective

IADEPLR treats the VRM ISV relation graph as a dynamic mini-batch graph. For
edge loss `ell_e(theta)`, let `g_e = grad ell_e`. With validation objective
`J(theta)` and a small perturbation coefficient `eta`, the second-order
simultaneous-deletion interaction is approximated by

`I_ij ~= eta^2 g_i^T H_J g_j`.

The implementation never constructs `H_J`. A randomized low-rank basis is
obtained from Hessian-vector products and the reduced Hessian is

a `U diag(S) U^T` approximation. Edge gradients are projected into `U`, so
all pairwise interactions are computed as `eta^2 Z diag(S) Z^T`.

## Large-batch approximation

A 64x64 VRM graph has 4032 directed off-diagonal edges. Computing a full
parameter-space gradient for every edge is prohibitively expensive. The
implementation therefore selects the highest current edge-loss candidates
(`MAX_CANDIDATES`) and computes exact gradients only for them. The final
pruning ranking is still based on the exact gradient norms of those candidates.
Set `MAX_CANDIDATES <= 0` for exhaustive mode on small graphs/diagnostics.

## Important experimental controls

For a paper, report at least:

1. VRM baseline.
2. VRM + random edge pruning at the same keep ratio.
3. VRM + gradient-only edge pruning (no interaction constraint).
4. VRM + IADEPLR (full method).
5. Sensitivity to rank, pruning fraction, update interval and interaction threshold.
6. Correlation between predicted `I_ij` and finite-difference interaction on a
   small held-out set of edge pairs.
7. Wall-clock overhead and peak GPU memory.

The online mask is recomputed for the current mini-batch, so it is a dynamic
relation-graph intervention rather than a permanent dataset-level adjacency
matrix.
