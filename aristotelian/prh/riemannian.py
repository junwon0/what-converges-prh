"""Locally estimated Riemannian geometry for PRH representation analysis.

The routines in this module estimate a data-driven local metric from an
L2-normalized representation point cloud.  For each sample, a neighborhood is
mapped to the tangent space of the unit sphere with the spherical logarithm
map, a low-rank local PCA is fit, and inverse local covariance defines an
anisotropic quadratic form.  The implementation never materializes a D x D
metric tensor: each local PCA basis is constructed only while the distances
incident to that sample are evaluated and is then discarded.

Two distance products are exposed:

* ``local_riemannian_knn_log_distances`` returns normalized log distances only
  on an already fixed evaluation kNN support.  This is intended for
  distance-aware mKNN and keeps support completely unchanged.
* ``local_riemannian_geodesic_distance_matrix`` weights a symmetric local kNN
  graph with the estimated Riemannian edge lengths and computes all-pairs
  shortest paths.  This is intended for global distance-aware MST evaluation,
  where direct extrapolation of a local metric to long edges would be
  inappropriate.

The neighborhood size used to estimate the metric is deliberately separate
from the evaluation k used by mKNN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components, shortest_path
except ImportError:  # pragma: no cover
    csr_matrix = None
    connected_components = None
    shortest_path = None


@dataclass(frozen=True)
class LocalRiemannianGraph:
    """Sparse locally estimated Riemannian graph for one representation layer."""

    n: int
    edge_codes: np.ndarray
    edge_lengths: np.ndarray
    scale: float


def _validate_riemannian_parameters(
    *,
    n: int,
    metric_k: int,
    tangent_dim: int,
    regularization: float,
) -> None:
    if n <= 1:
        return
    if not (1 <= int(metric_k) < n):
        raise ValueError(
            f"riemannian_k must satisfy 1 <= k < n={n}, got {metric_k}"
        )
    if not (1 <= int(tangent_dim) <= int(metric_k)):
        raise ValueError(
            "riemannian_dim must satisfy 1 <= dim <= riemannian_k; "
            f"got dim={tangent_dim}, k={metric_k}"
        )
    if float(regularization) < 0.0:
        raise ValueError(
            f"riemannian_reg must be non-negative, got {regularization}"
        )


def _spherical_log_map(
    base: torch.Tensor,
    targets: torch.Tensor,
    *,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Map unit-sphere targets to the tangent space at ``base``.

    ``base`` has shape ``(D,)`` and ``targets`` has shape ``(m, D)``.  Inputs
    are assumed to be L2-normalized.  The returned rows have norm equal to the
    spherical angle up to floating-point error.
    """
    if targets.numel() == 0:
        return targets.clone()

    cos_theta = (targets @ base).clamp(min=-1.0 + eps, max=1.0)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta.square(), min=0.0))

    # theta / sin(theta) -> 1 as theta -> 0.  Avoid a 0/0 numerical branch.
    small = theta.abs() < 1e-4
    safe_sin = sin_theta.clamp_min(eps)
    factor = theta / safe_sin
    if small.any():
        theta2 = theta[small].square()
        factor = factor.clone()
        factor[small] = 1.0 + theta2 / 6.0

    tangent = targets - cos_theta.unsqueeze(1) * base.unsqueeze(0)
    return factor.unsqueeze(1) * tangent



def _spherical_log_map_batched(
    base: torch.Tensor,
    targets: torch.Tensor,
    *,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Batched spherical log map: base (B,D), targets (B,m,D)."""
    cos_theta = torch.einsum("bd,bmd->bm", base, targets).clamp(
        min=-1.0 + eps, max=1.0
    )
    theta = torch.acos(cos_theta)
    sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta.square(), min=0.0))
    small = theta.abs() < 1e-4
    factor = theta / sin_theta.clamp_min(eps)
    if small.any():
        factor = torch.where(small, 1.0 + theta.square() / 6.0, factor)
    tangent = targets - cos_theta.unsqueeze(-1) * base.unsqueeze(1)
    return factor.unsqueeze(-1) * tangent

def _ambient_knn_indices(x: torch.Tensor, *, k: int) -> torch.Tensor:
    """Ambient nearest neighbors for an L2-normalized layer.

    Cosine order and Euclidean chord-distance order are identical on the unit
    sphere, so this is consistent with the existing PRH kNN construction.
    """
    n = x.shape[0]
    sim = x @ x.T
    sim = sim.clone()
    sim.fill_diagonal_(float("-inf"))
    return torch.topk(sim, k=int(k), dim=1, largest=True, sorted=True).indices


def _undirected_edge_codes_from_knn(knn_idx: torch.Tensor) -> torch.Tensor:
    """Return sorted unique ``u * n + v`` codes from a directed kNN support."""
    n, k = knn_idx.shape
    rows = torch.arange(n, device=knn_idx.device).view(-1, 1).expand(n, k)
    lo = torch.minimum(rows, knn_idx)
    hi = torch.maximum(rows, knn_idx)
    codes = (lo * n + hi).reshape(-1)
    return torch.unique(codes, sorted=True)


def _local_metric_squared_lengths(
    x: torch.Tensor,
    metric_knn: torch.Tensor,
    edge_codes: torch.Tensor,
    *,
    tangent_dim: int,
    regularization: float,
    eps: float,
    batch_size: int = 32,
) -> torch.Tensor:
    """Evaluate symmetric local-Riemannian squared lengths on selected edges.

    Expensive local PCA work is batched over samples.  The ambient tangent
    bases exist only for one small batch and are discarded immediately, so the
    method remains memory-bounded even when D is several thousand.
    """
    n = x.shape[0]
    m = edge_codes.numel()
    if m == 0:
        return torch.empty((0,), device=x.device, dtype=x.dtype)

    u = torch.div(edge_codes, n, rounding_mode="floor")
    v = edge_codes.remainder(n)
    endpoint_sq = torch.empty((m, 2), device=x.device, dtype=x.dtype)

    incident_edges: list[list[int]] = [[] for _ in range(n)]
    incident_side: list[list[int]] = [[] for _ in range(n)]
    u_cpu = u.detach().cpu().tolist()
    v_cpu = v.detach().cpu().tolist()
    for edge_idx, (uu, vv) in enumerate(zip(u_cpu, v_cpu)):
        incident_edges[uu].append(edge_idx)
        incident_side[uu].append(0)
        incident_edges[vv].append(edge_idx)
        incident_side[vv].append(1)

    h = metric_knn.shape[1]
    d = int(tangent_dim)
    lambda_floor = torch.finfo(x.dtype).eps
    batch_size = max(1, int(batch_size))

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        idx = torch.arange(start, stop, device=x.device)
        base = x[idx]
        neigh = metric_knn[idx]
        V = _spherical_log_map_batched(base, x[neigh], eps=eps)  # (B,h,D)

        gram = torch.bmm(V, V.transpose(1, 2)) / float(h)
        evals, U = torch.linalg.eigh(gram)
        evals = evals[:, -d:]
        U = U[:, :, -d:]
        safe_evals = evals.clamp_min(lambda_floor)

        # Q_b = V_b^T U_b / sqrt(h * lambda_b), shape (B,D,d).
        Q = torch.bmm(V.transpose(1, 2), U)
        Q = Q / torch.sqrt(float(h) * safe_evals).unsqueeze(1)
        mean_eval = safe_evals.mean(dim=1, keepdim=True).clamp_min(lambda_floor)
        denom = safe_evals + float(regularization) * mean_eval

        for local_idx, i in enumerate(range(start, stop)):
            edge_ids_py = incident_edges[i]
            if not edge_ids_py:
                continue
            edge_ids = torch.tensor(edge_ids_py, device=x.device, dtype=torch.long)
            sides = torch.tensor(incident_side[i], device=x.device, dtype=torch.long)
            uu = u[edge_ids]
            vv = v[edge_ids]
            other = torch.where(uu == i, vv, uu)
            W = _spherical_log_map(x[i], x[other], eps=eps)
            proj = W @ Q[local_idx]
            sq = (proj.square() / denom[local_idx].unsqueeze(0)).sum(dim=1)
            endpoint_sq[edge_ids, sides] = sq

    return 0.5 * (endpoint_sq[:, 0] + endpoint_sq[:, 1])


def build_local_riemannian_graph(
    layer: torch.Tensor,
    *,
    metric_k: int = 30,
    tangent_dim: int = 10,
    regularization: float = 1e-3,
    q: float = 0.9,
    eps: float = 1e-12,
) -> LocalRiemannianGraph:
    """Estimate local metric tensors implicitly and return a weighted kNN graph.

    The layer is L2-normalized internally.  Edge lengths are symmetric averages
    of the two endpoint-local quadratic forms.  ``scale`` is the q-quantile of
    positive graph-edge lengths and is provided so local kNN values can use the
    same per-layer normalization independently of the evaluation k.
    """
    if not (0.0 < float(q) <= 1.0):
        raise ValueError(f"q must be in (0, 1], got {q}")

    if layer.ndim != 2:
        raise ValueError(f"layer must have shape (n, D), got {tuple(layer.shape)}")

    x = F.normalize(layer.detach().float(), dim=1)
    n = x.shape[0]
    _validate_riemannian_parameters(
        n=n,
        metric_k=metric_k,
        tangent_dim=tangent_dim,
        regularization=regularization,
    )

    if n <= 1:
        return LocalRiemannianGraph(
            n=n,
            edge_codes=np.empty((0,), dtype=np.int64),
            edge_lengths=np.empty((0,), dtype=np.float64),
            scale=1.0,
        )

    metric_knn = _ambient_knn_indices(x, k=int(metric_k))
    edge_codes = _undirected_edge_codes_from_knn(metric_knn)
    edge_sq = _local_metric_squared_lengths(
        x,
        metric_knn,
        edge_codes,
        tangent_dim=int(tangent_dim),
        regularization=float(regularization),
        eps=float(eps),
    )
    edge_lengths = torch.sqrt(edge_sq.clamp_min(float(eps)))

    positive = edge_lengths[edge_lengths > 0]
    if positive.numel() == 0:
        scale = 1.0
    else:
        scale = max(float(torch.quantile(positive, float(q)).item()), float(eps))

    return LocalRiemannianGraph(
        n=n,
        edge_codes=edge_codes.detach().cpu().numpy().astype(np.int64, copy=False),
        edge_lengths=edge_lengths.detach().cpu().numpy().astype(np.float64, copy=False),
        scale=scale,
    )


def local_riemannian_knn_log_distances(
    layer: torch.Tensor,
    eval_knn: torch.Tensor,
    *,
    metric_k: int = 30,
    tangent_dim: int = 10,
    regularization: float = 1e-3,
    q: float = 0.9,
    eps: float = 1e-12,
    normalization: str = "quantile",
) -> torch.Tensor:
    """Normalized log local-Riemannian distances on a fixed evaluation support.

    ``metric_k`` must be at least the evaluation k.  The returned tensor has
    shape ``(n, k_eval)`` and can be cached across all permutations.
    """
    n, k_eval = eval_knn.shape
    if int(metric_k) < int(k_eval):
        raise ValueError(
            "riemannian_k must be >= the evaluation k so every fixed mKNN "
            f"support edge has a locally estimated graph length; got "
            f"riemannian_k={metric_k}, eval_k={k_eval}"
        )

    graph = build_local_riemannian_graph(
        layer,
        metric_k=metric_k,
        tangent_dim=tangent_dim,
        regularization=regularization,
        q=q,
        eps=eps,
    )

    if n <= 1:
        return torch.zeros_like(eval_knn, dtype=layer.dtype)

    rows = torch.arange(n, device=eval_knn.device).view(-1, 1).expand_as(eval_knn)
    lo = torch.minimum(rows, eval_knn)
    hi = torch.maximum(rows, eval_knn)
    directed_codes = (lo * n + hi).reshape(-1).detach().cpu().numpy()

    # edge_codes is sorted by torch.unique(sorted=True), enabling O(log m)
    # vectorized lookup without a Python dictionary.
    positions = np.searchsorted(graph.edge_codes, directed_codes)
    safe_positions = np.clip(positions, 0, max(graph.edge_codes.size - 1, 0))
    valid = (positions < graph.edge_codes.size)
    if graph.edge_codes.size:
        valid &= graph.edge_codes[safe_positions] == directed_codes
    if not bool(np.all(valid)):
        raise RuntimeError(
            "Fixed evaluation kNN support contains edges absent from the "
            "Riemannian metric graph. Increase --riemannian-k."
        )

    raw_values = graph.edge_lengths[positions]
    if normalization == "quantile":
        positive = graph.edge_lengths[graph.edge_lengths > 0]
        scale = (
            max(float(np.quantile(positive, float(q))), float(eps))
            if positive.size
            else 1.0
        )
    elif normalization == "mean":
        positive = graph.edge_lengths[graph.edge_lengths > 0]
        scale = (
            max(float(np.mean(positive)), float(eps))
            if positive.size
            else 1.0
        )
    elif normalization == "none":
        scale = 1.0
    else:
        raise ValueError(
            "distance_normalization must be 'quantile', 'mean', or 'none'; "
            f"got {normalization!r}"
        )

    values = raw_values / scale
    values = np.maximum(values, float(eps)).reshape(n, k_eval)
    return torch.from_numpy(np.log(values)).to(
        device=eval_knn.device,
        dtype=layer.dtype,
    )


def local_riemannian_geodesic_distance_matrix(
    layer: torch.Tensor,
    *,
    metric_k: int = 30,
    tangent_dim: int = 10,
    regularization: float = 1e-3,
    q: float = 0.9,
    eps: float = 1e-12,
    normalize: bool = True,
) -> np.ndarray:
    """All-pairs geodesics on a locally estimated Riemannian kNN graph.

    This is the global counterpart used for distance-aware MST values.  The
    support MST itself is still constructed independently from the configured
    ambient topology distance; only numerical values are replaced.
    """
    if csr_matrix is None or connected_components is None or shortest_path is None:
        raise ImportError("scipy is required for local-Riemannian geodesics")

    graph_cache = build_local_riemannian_graph(
        layer,
        metric_k=metric_k,
        tangent_dim=tangent_dim,
        regularization=regularization,
        q=q,
        eps=eps,
    )
    n = graph_cache.n
    if n <= 1:
        return np.zeros((n, n), dtype=np.float64)

    u = graph_cache.edge_codes // n
    v = graph_cache.edge_codes % n
    w = np.maximum(graph_cache.edge_lengths, float(eps))

    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    vals = np.concatenate([w, w])
    graph = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)

    n_components, _ = connected_components(graph, directed=False, return_labels=True)
    if n_components != 1:
        raise ValueError(
            f"local-Riemannian kNN graph is disconnected ({n_components} components) "
            f"for k={metric_k}. Increase --riemannian-k or report a sensitivity sweep."
        )

    distances = shortest_path(graph, directed=False, unweighted=False, method="D")
    distances = np.asarray(distances, dtype=np.float64)
    if not np.isfinite(distances).all():
        raise RuntimeError("Local-Riemannian shortest paths produced NaN or Inf")

    np.fill_diagonal(distances, 0.0)
    positive = distances[distances > 0]
    if normalize and positive.size:
        scale = max(float(np.quantile(positive, float(q))), float(eps))
        distances = distances / scale
    np.fill_diagonal(distances, 0.0)
    return distances
