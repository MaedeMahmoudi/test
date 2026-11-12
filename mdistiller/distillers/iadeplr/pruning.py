"""IADEPLR: Interaction-Aware Dynamic Edge Pruning with Low-Rank curvature.

The implementation follows the second-order interaction model

    I_ij ~= eta^2 g_i^T H_J g_j,

where g_i is the gradient of the i-th relation edge and H_J is the Hessian
of a validation objective.  The Hessian is never materialised.  Instead,
a randomized low-rank basis is built from Hessian-vector products (HVPs),
and edge gradients are projected into that basis.

The code deliberately exposes the approximations used for large batches:
`max_candidates` limits exact edge-gradient evaluation to a shortlist, while
`rank` controls the curvature approximation.  Setting max_candidates <= 0
switches to all-edge gradient evaluation (appropriate only for diagnostics or
small graphs).
"""

from __future__ import division

import math
import time
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


def _trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def _flatten_tensors(tensors):
    chunks = []
    for t in tensors:
        if t is not None:
            chunks.append(t.contiguous().view(-1))
    if not chunks:
        return None
    return torch.cat(chunks)


def _split_vector(vec, params):
    out = []
    offset = 0
    for p in params:
        n = p.numel()
        out.append(vec[offset:offset + n].view_as(p))
        offset += n
    if offset != vec.numel():
        raise ValueError("Vector length does not match model parameters")
    return out


def _safe_grad(loss, params, create_graph=False, retain_graph=True):
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )
    return [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]


def _hvp_from_grads(first_grads, vector, params, retain_graph=True):
    dot = None
    chunks = _split_vector(vector, params)
    for g, q in zip(first_grads, chunks):
        term = (g * q).sum()
        dot = term if dot is None else dot + term
    hv = torch.autograd.grad(
        dot,
        params,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    return [torch.zeros_like(p) if h is None else h for p, h in zip(params, hv)]


def _edge_indices(edge_losses: Tensor, candidates=None):
    b0, b1 = edge_losses.shape[:2]
    if candidates is None:
        return [(i, j) for i in range(b0) for j in range(b1) if i != j]
    return [(int(i), int(j)) for i, j in candidates if int(i) != int(j)]


def compute_edge_gradients(edge_losses: Tensor, model, candidates=None,
                           create_graph: bool = False):
    """Compute exact gradients for selected relation edges.

    Args:
        edge_losses: differentiable [B,B] tensor, one scalar per edge.
        model: student network whose parameters define g_e.
        candidates: iterable of (i,j). None means all off-diagonal edges.
        create_graph: whether edge gradients should retain a higher-order graph.

    Returns:
        ``indices`` and a list of flattened gradient vectors.  A list is used
        intentionally: materialising an E x P matrix is prohibitive for modern
        CNNs.  Callers can project each vector immediately into a low-rank basis.
    """
    if edge_losses.dim() != 2:
        raise ValueError("edge_losses must have shape [B, B]")
    params = _trainable_params(model)
    indices = _edge_indices(edge_losses, candidates)
    gradients = []
    for i, j in indices:
        g = _safe_grad(
            edge_losses[i, j], params,
            create_graph=create_graph,
            retain_graph=True,
        )
        gradients.append(_flatten_tensors(g))
    return indices, gradients


def nystrom_hessian_approx(loss: Tensor, model, rank: int = 8,
                           damping: float = 1e-3,
                           generator: Optional[torch.Generator] = None):
    """Build a randomized low-rank Hessian approximation using HVPs.

    The implementation uses a randomized low-rank Nyström-style range finder.
    For random probes q_k, Y=[Hq_1,...,Hq_r], and U=orth(Y), the reduced
    curvature matrix is B=U^T H U. We use its symmetric eigendecomposition
    B=V diag(S) V^T and return U_bar=UV so that H ~= U_bar diag(S) U_bar^T.
    This is the Hessian-only, HVP form of a randomized Nyström approximation;
    no dense Hessian or parameter-by-rank random matrix is materialised.

    ``damping`` is added to eigenvalues only when constructing the inverse;
    the returned curvature eigenvalues are the undamped estimate.
    """
    if rank < 1:
        raise ValueError("rank must be >= 1")
    params = _trainable_params(model)
    if not params:
        raise ValueError("model has no trainable parameters")

    # First derivative creates the graph needed by all HVPs.
    first = _safe_grad(loss, params, create_graph=True, retain_graph=True)
    n_params = sum(p.numel() for p in params)
    device = params[0].device
    dtype = params[0].dtype
    rank = min(rank, n_params)

    ys = []
    for k in range(rank):
        if generator is None:
            q = torch.randn(n_params, device=device, dtype=dtype)
        else:
            q = torch.randn(n_params, generator=generator, device=device, dtype=dtype)
        q = q / (q.norm() + 1e-12)
        hv = _hvp_from_grads(first, q, params, retain_graph=True)
        ys.append(_flatten_tensors(hv).detach())

    Y = torch.stack(ys, dim=1)  # [P, r]
    # QR is considerably more stable than normalising individual HVPs.
    U, _ = torch.linalg.qr(Y, mode="reduced")

    # H times each basis vector; the first derivative graph is still alive.
    hu_cols = []
    for k in range(U.shape[1]):
        hv = _hvp_from_grads(first, U[:, k], params, retain_graph=True)
        hu_cols.append(_flatten_tensors(hv).detach())
    HU = torch.stack(hu_cols, dim=1)
    B = 0.5 * (U.t().mm(HU) + HU.t().mm(U))
    evals, evecs = torch.linalg.eigh(B)
    order = torch.argsort(evals.abs(), descending=True)
    evals = evals[order]
    evecs = evecs[:, order]
    Ubar = U.mm(evecs)

    # Numerical rank can collapse for nearly flat directions.
    keep = evals.abs() > max(damping * 1e-3, 1e-10)
    if keep.sum().item() == 0:
        keep[0] = True
    Ubar = Ubar[:, keep]
    evals = evals[keep]
    return Ubar.detach(), evals.detach()


def apply_inverse_nystrom(vector: Tensor, basis: Tensor, spectrum: Tensor,
                           damping: float = 1e-3):
    """Apply a damped inverse of a low-rank Hessian model to a vector.

    The complement of the low-rank subspace is treated as having curvature
    ``damping``. This is a stable Woodbury-style approximation:

        (lambda I + U S U^T)^-1 v.
    """
    if vector.dim() != 1:
        vector = vector.view(-1)
    lam = float(max(damping, 1e-12))
    proj = basis.t().mv(vector)
    correction = basis.mv((1.0 / (lam + spectrum) - 1.0 / lam) * proj)
    return vector / lam + correction


def compute_interaction_matrix(edge_gradients: Sequence[Tensor], basis: Tensor,
                               spectrum: Tensor, eta: float = 1.0):
    """Compute C_ij ~= eta^2 g_i^T H_hat g_j without forming H_hat or G."""
    if len(edge_gradients) == 0:
        return torch.empty(0, 0, device=basis.device)
    projected = []
    for g in edge_gradients:
        projected.append(basis.t().mv(g.to(basis.device)))
    Z = torch.stack(projected, dim=0)  # [E,r]
    weighted = Z * spectrum.view(1, -1)
    C = weighted.mm(Z.t())
    C = 0.5 * (C + C.t())
    return (float(eta) ** 2) * C


def iadeplr_pruning_decision(indices: Sequence[Tuple[int, int]],
                             gradient_norms: Tensor,
                             interaction: Tensor,
                             prune_fraction: float = 0.25,
                             interaction_threshold: float = 0.0,
                             min_keep: int = 1):
    """Greedily select low-saliency edges while blocking harmful pairs.

    Edges are considered in ascending single-edge gradient norm. An edge is
    accepted for deletion only if its interaction with every already deleted
    edge is <= ``interaction_threshold``. The diagonal is ignored.
    """
    n = len(indices)
    if n == 0:
        return [], torch.empty(0, dtype=torch.bool, device=gradient_norms.device)
    requested = int(math.floor(n * max(0.0, min(1.0, prune_fraction))))
    requested = min(requested, max(0, n - min_keep))
    order = torch.argsort(gradient_norms.detach()).tolist()
    selected = []
    selected_set = set()
    for k in order:
        if len(selected) >= requested:
            break
        ok = True
        for j in selected:
            if float(interaction[k, j].detach().cpu()) > interaction_threshold:
                ok = False
                break
        if ok:
            selected.append(k)
            selected_set.add(k)
    mask = torch.ones(n, dtype=torch.bool, device=gradient_norms.device)
    if selected:
        mask[torch.tensor(selected, device=mask.device)] = False
    return [indices[k] for k in selected], mask


class IADEPLR(object):
    """Online IADEPLR controller for a VRM mini-batch relation graph."""

    def __init__(self, model, rank=8, eta=1.0, prune_fraction=0.25,
                 interaction_threshold=0.0, max_candidates=64,
                 damping=1e-3, validation_batches=1, seed=17):
        self.model = model
        self.rank = int(rank)
        self.eta = float(eta)
        self.prune_fraction = float(prune_fraction)
        self.interaction_threshold = float(interaction_threshold)
        self.max_candidates = int(max_candidates)
        self.damping = float(damping)
        self.validation_batches = int(max(1, validation_batches))
        self.seed = int(seed)
        self.last_report = {}

    def _candidate_edges(self, edge_losses):
        b = edge_losses.size(0)
        all_edges = [(i, j) for i in range(b) for j in range(b) if i != j]
        if self.max_candidates <= 0 or self.max_candidates >= len(all_edges):
            return all_edges
        # Cheap shortlist. Exact gradient norms are computed only for these.
        flat = edge_losses.detach().clone()
        flat.fill_diagonal_(float("-inf"))
        k = min(self.max_candidates, flat.numel() - b)
        vals = flat.view(-1).topk(k=k, largest=True).indices.tolist()
        return [(v // b, v % b) for v in vals]

    def step(self, edge_losses: Tensor, validation_loss: Tensor,
             validation_objective=None):
        """Compute a new edge mask.

        ``validation_loss`` must be differentiable w.r.t. the student model.
        ``validation_objective`` is accepted for future extensions and logging.
        """
        t0 = time.time()
        candidates = self._candidate_edges(edge_losses)
        indices, gradients = compute_edge_gradients(
            edge_losses, self.model, candidates=candidates, create_graph=False
        )
        norms = torch.stack([g.norm() for g in gradients]).detach()

        # PyTorch 1.9 does not consistently expose device-aware Generator; use
        # a deterministic local seed without requiring a second RNG object.
        if edge_losses.is_cuda:
            torch.cuda.manual_seed(self.seed)
        else:
            torch.manual_seed(self.seed)
        gen = None
        basis, spectrum = nystrom_hessian_approx(
            validation_loss, self.model, rank=self.rank,
            damping=self.damping, generator=gen
        )
        C = compute_interaction_matrix(gradients, basis, spectrum, eta=self.eta)
        deleted, local_keep = iadeplr_pruning_decision(
            indices, norms, C,
            prune_fraction=self.prune_fraction,
            interaction_threshold=self.interaction_threshold,
            min_keep=1,
        )

        full_mask = torch.ones_like(edge_losses, dtype=edge_losses.dtype)
        for i, j in deleted:
            full_mask[i, j] = 0.0
        # Do not prune diagonal/self relations; they are not graph edges.
        full_mask.fill_diagonal_(0.0 if False else 1.0)

        self.last_report = {
            "num_edges": edge_losses.numel() - edge_losses.size(0),
            "num_candidates": len(indices),
            "num_pruned": len(deleted),
            "prune_fraction_actual": len(deleted) / max(1, len(indices)),
            "interaction_max": float(C.max().detach().cpu()) if C.numel() else 0.0,
            "interaction_min": float(C.min().detach().cpu()) if C.numel() else 0.0,
            "grad_norm_mean": float(norms.mean().cpu()),
            "elapsed_sec": time.time() - t0,
            "deleted_edges": deleted,
        }
        return full_mask.detach(), self.last_report


def diagnostic_experiment(student, train_batch, validation_batch,
                          edge_loss_fn, rank=4, max_candidates=16,
                          prune_fraction=0.25, interaction_threshold=0.0):
    """Small real-batch diagnostic used to validate the IADEPLR pipeline.

    This function intentionally does not update model parameters. It returns
    the mask and diagnostic statistics so a script can compare loss/accuracy
    before and after pruning.
    """
    student.eval()
    train_input, train_target = train_batch
    val_input, val_target = validation_batch
    train_input = train_input.cuda(non_blocking=True)
    train_target = train_target.cuda(non_blocking=True)
    val_input = val_input.cuda(non_blocking=True)
    val_target = val_target.cuda(non_blocking=True)
    logits_w = student(train_input)[0]
    logits_s = student(train_input)[0]
    edge_losses = edge_loss_fn(logits_w, logits_s)
    val_logits = student(val_input)[0]
    val_loss = F.cross_entropy(val_logits, val_target)
    controller = IADEPLR(student, rank=rank, max_candidates=max_candidates,
                         prune_fraction=prune_fraction,
                         interaction_threshold=interaction_threshold)
    mask, report = controller.step(edge_losses, val_loss)
    return mask, report
