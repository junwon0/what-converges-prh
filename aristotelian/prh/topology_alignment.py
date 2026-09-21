from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Sequence, Tuple

import numpy as np
import torch

from ..metrics.aggregation import agg_max, compute_null_summary, gated_rescaled
from ..metrics.topology import (
    _diagram_distance,
    _distance_to_similarity,
    _pairwise_distances,
    _quantile_normalize,
    _ripser_diagrams,
)
from .geodesic import graph_geodesic_from_distances
from .riemannian import local_riemannian_geodesic_distance_matrix

try:
    from scipy.sparse.csgraph import minimum_spanning_tree
except ImportError:  # pragma: no cover
    minimum_spanning_tree = None


TOPOLOGY_METRICS = {
    "mst_overlap",
    "mst_min_overlap",
    "mst_overlap_dist",
    "mst_overlap_sim",
    # "rtd",
    "rtd_lite",
    "rtd_lite_tai",
    "pd0_bottleneck",
    "pd1_bottleneck",
    "pd01_bottleneck",
    "pd0_wasserstein",
    "pd1_wasserstein",
    "pd01_wasserstein",
}

def _parse_topology_metric(metric: str) -> tuple[str, float | None]:
    """Parse topology metrics with tau in their name.

    Example:
        mst_overlap_dist_tau0p1 -> ("mst_overlap_dist", 0.1)
    """
    dist_prefix = "mst_overlap_dist_tau"
    sim_prefix = "mst_overlap_sim_tau"

    if metric.startswith(dist_prefix):
        tau_str = metric[len(dist_prefix):].replace("p", ".").replace("m", "-")
        return "mst_overlap_dist", float(tau_str)

    if metric.startswith(sim_prefix):
        tau_str = metric[len(sim_prefix):].replace("p", ".").replace("m", "-")
        return "mst_overlap_sim", float(tau_str)

    if metric == "mst_overlap_dist":
        return "mst_overlap_dist", 0.01
    if metric == "mst_overlap_sim":
        return "mst_overlap_sim", 0.01

    return metric, None


@dataclass
class RTDLiteLayerCache:
    # Distances used to define topology/support (MST, RTD-Lite, etc.).
    distances: list[np.ndarray]
    mst_weights: list[float]
    mst_edges: list[np.ndarray]
    # Numerical values used only by distance-aware MST overlap.  Keeping this
    # separate lets us hold MST support fixed while replacing ambient values by
    # graph-geodesic values.
    value_distances: list[np.ndarray]


@dataclass
class MSTOverlapLayerCache:
    """Lean cache for correspondence-aware distance-weighted MST overlap."""

    n: int
    mst_edges: list[np.ndarray]
    mst_edge_values: list[np.ndarray]


@dataclass
class PDLayerCache:
    diagrams: list[list[np.ndarray]]


def _mst_total_weight_numpy(D: np.ndarray) -> float:
    """Fast MST total weight using scipy when available."""
    D = np.asarray(D, dtype=np.float64)

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be square, got {D.shape}")

    if not np.isfinite(D).all():
        raise ValueError("Distance matrix contains NaN or Inf")

    n = D.shape[0]
    if n <= 1:
        return 0.0

    if minimum_spanning_tree is not None:
        return float(minimum_spanning_tree(D).sum())

    # Fallback dense Prim implementation in NumPy.
    visited = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf, dtype=np.float64)
    parent = np.full(n, -1, dtype=np.int64)

    v = 0
    total = 0.0

    for _ in range(n - 1):
        visited[v] = True

        better = (~visited) & (D[v] < min_dist)
        min_dist[better] = D[v, better]
        parent[better] = v

        masked = min_dist.copy()
        masked[visited] = np.inf
        v = int(np.argmin(masked))

        if parent[v] < 0:
            raise RuntimeError("Failed to construct MST.")

        total += float(D[v, parent[v]])

    return total

def _mst_edge_codes_numpy(D: np.ndarray) -> np.ndarray:
    """Return MST edge identities as sorted integer codes ``u * n + v``.

    This dense Prim implementation mirrors ``_mst_total_weight_numpy`` and is
    deterministic under ties. The returned edge labels are sample-index labels,
    which makes them suitable for correspondence-aware overlap scores.
    """
    D = np.asarray(D, dtype=np.float64)

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be square, got {D.shape}")

    if not np.isfinite(D).all():
        raise ValueError("Distance matrix contains NaN or Inf")

    n = D.shape[0]
    if n <= 1:
        return np.empty((0,), dtype=np.int64)

    visited = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf, dtype=np.float64)
    parent = np.full(n, -1, dtype=np.int64)

    v = 0
    edges: list[int] = []

    for _ in range(n - 1):
        visited[v] = True

        better = (~visited) & (D[v] < min_dist)
        min_dist[better] = D[v, better]
        parent[better] = v

        masked = min_dist.copy()
        masked[visited] = np.inf
        v = int(np.argmin(masked))

        if parent[v] < 0:
            raise RuntimeError("Failed to construct MST.")

        u = min(v, int(parent[v]))
        w = max(v, int(parent[v]))
        edges.append(u * n + w)

    return np.asarray(edges, dtype=np.int64)


def _edge_overlap_fraction(edges_a: np.ndarray, edges_b: np.ndarray, n: int) -> float:
    """Compute ``|edges_a ∩ edges_b| / (n - 1)`` for encoded MST edges."""
    if n <= 1:
        return 1.0
    return float(np.intersect1d(edges_a, edges_b, assume_unique=False).size / (n - 1))


def _permute_edge_codes(edge_codes: np.ndarray, perm: np.ndarray, n: int) -> np.ndarray:
    """Relabel cached MST edges after applying ``D_perm = D[perm][:, perm]``.

    If an original edge is ``(u, v)``, its coordinates in the permuted matrix are
    ``(inv_perm[u], inv_perm[v])``.
    """
    if n <= 1:
        return np.empty((0,), dtype=np.int64)

    inv_perm = np.empty(n, dtype=np.int64)
    inv_perm[perm] = np.arange(n, dtype=np.int64)

    u = edge_codes // n
    v = edge_codes % n
    pu = inv_perm[u]
    pv = inv_perm[v]
    lo = np.minimum(pu, pv)
    hi = np.maximum(pu, pv)
    return (lo * n + hi).astype(np.int64, copy=False)


def _prepare_rtd_lite_cache(
    layers: Sequence[torch.Tensor],
    *,
    dist_metric: str = "euclidean",
    normalize: bool = True,
    q: float = 0.9,
    distance_value_geometry: str = "ambient",
    geodesic_k: int = 20,
    riemannian_k: int = 30,
    riemannian_dim: int = 10,
    riemannian_reg: float = 1e-3,
) -> RTDLiteLayerCache:
    """Build topology support and optional distance-value caches per layer.

    ``dist_metric`` always defines the MST/topological support.  For
    For non-ambient value geometries, only the values used by
    ``mst_overlap_dist`` are replaced; MST edge identities remain those induced
    by ``dist_metric``. ``geodesic`` uses ambient-weighted graph geodesics, while
    ``riemannian`` uses shortest paths on a locally covariance-adapted graph.
    """
    distances: list[np.ndarray] = []
    mst_weights: list[float] = []
    mst_edges: list[np.ndarray] = []
    value_distances: list[np.ndarray] = []

    for layer in layers:
        with torch.no_grad():
            D_raw = _pairwise_distances(
                layer.detach().float().cpu(), metric=dist_metric
            )

            D_support = _quantile_normalize(D_raw, q=q) if normalize else D_raw
            D_np = D_support.numpy().astype(np.float64, copy=False)

            if distance_value_geometry == "ambient":
                D_value_np = D_np
            elif distance_value_geometry == "geodesic":
                G_np = graph_geodesic_from_distances(
                    D_raw.numpy().astype(np.float64, copy=False),
                    k=geodesic_k,
                )
                G = torch.from_numpy(G_np)
                if normalize:
                    G = _quantile_normalize(G, q=q)
                D_value_np = G.numpy().astype(np.float64, copy=False)
            elif distance_value_geometry == "riemannian":
                D_value_np = local_riemannian_geodesic_distance_matrix(
                    layer.detach().float().cpu(),
                    metric_k=riemannian_k,
                    tangent_dim=riemannian_dim,
                    regularization=riemannian_reg,
                    q=q,
                    normalize=normalize,
                ).astype(np.float64, copy=False)
            else:
                raise ValueError(
                    "distance_value_geometry must be 'ambient', 'geodesic', or "
                    f"'riemannian'; got {distance_value_geometry!r}"
                )

        distances.append(D_np)
        mst_weights.append(_mst_total_weight_numpy(D_np))
        mst_edges.append(_mst_edge_codes_numpy(D_np))
        value_distances.append(D_value_np)

    return RTDLiteLayerCache(
        distances=distances,
        mst_weights=mst_weights,
        mst_edges=mst_edges,
        value_distances=value_distances,
    )


def _rtd_lite_similarity_from_cached(
    D_x: np.ndarray,
    mst_x: float,
    D_y: np.ndarray,
    mst_y: float,
) -> float:
    D_min = np.minimum(D_x, D_y)
    mst_min = _mst_total_weight_numpy(D_min)

    dist = 0.5 * ((mst_x - mst_min) + (mst_y - mst_min))
    return _distance_to_similarity(dist)

def _iqr(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    q75, q25 = np.percentile(values, [75.0, 25.0])
    return float(q75 - q25)


def _rtd_lite_distance_from_cached(
    cache_x,
    cache_y,
    i: int,
    j: int,
    *,
    perm: np.ndarray | None = None,
) -> float:
    Dx = cache_x.distances[i]
    Dy = cache_y.distances[j]

    if perm is not None:
        Dy = Dy[np.ix_(perm, perm)]

    sx = float(cache_x.mst_weights[i])
    sy = float(cache_y.mst_weights[j])

    smin = float(_mst_total_weight_numpy(np.minimum(Dx, Dy)))

    d = 0.5 * ((sx - smin) + (sy - smin))
    return float(max(d, 0.0))


def _rtd_lite_distance_matrix_from_cached(
    cache_x,
    cache_y,
    *,
    perm: np.ndarray | None = None,
) -> np.ndarray:
    n_x = len(cache_x.distances)
    n_y = len(cache_y.distances)

    D = np.empty((n_x, n_y), dtype=float)

    for i in range(n_x):
        for j in range(n_y):
            D[i, j] = _rtd_lite_distance_from_cached(
                cache_x,
                cache_y,
                i,
                j,
                perm=perm,
            )

    return D


def _rtd_lite_tai_result(
    cache_x,
    cache_y,
    *,
    num_permutations: int,
    seed: int = 0,
    eps: float = 1e-12,
) -> dict:
    D_obs_matrix = _rtd_lite_distance_matrix_from_cached(cache_x, cache_y)

    best_flat = int(np.argmin(D_obs_matrix))
    best_i, best_j = np.unravel_index(best_flat, D_obs_matrix.shape)
    d_obs = float(D_obs_matrix[best_i, best_j])

    if num_permutations <= 0:
        return {
            "raw_score": float("nan"),
            "topo_effect": float("nan"),
            "raw_distance": d_obs,
            "best_indices": (int(best_i), int(best_j)),
            "p_value": float("nan"),
            "tau_alpha": float("nan"),
            "tail_strength": float("nan"),
            "g_score": float("nan"),
            "mu0": float("nan"),
            "sd0": float("nan"),
            "null_distance_median": float("nan"),
            "null_distance_iqr": float("nan"),
            "null_distance_mean": float("nan"),
            "null_distance_std": float("nan"),
            "num_permutations": int(num_permutations),
        }

    rng = np.random.default_rng(seed)
    n = cache_x.distances[0].shape[0]

    null_distances = np.empty(num_permutations, dtype=float)

    for b in range(num_permutations):
        perm = rng.permutation(n)
        D_perm_matrix = _rtd_lite_distance_matrix_from_cached(
            cache_x,
            cache_y,
            perm=perm,
        )
        null_distances[b] = float(np.min(D_perm_matrix))

    null_median = float(np.median(null_distances))
    null_iqr = float(_iqr(null_distances))
    null_mean = float(np.mean(null_distances))
    null_std = float(np.std(null_distances, ddof=1)) if num_permutations > 1 else float("nan")

    topo_effect = (null_median - d_obs) / (null_iqr + eps)

    p_value = float(
        (1.0 + np.sum(null_distances <= d_obs))
        / (float(num_permutations) + 1.0)
    )

    return {
        "raw_score": float(topo_effect),
        "topo_effect": float(topo_effect),
        "raw_distance": d_obs,
        "best_indices": (int(best_i), int(best_j)),
        "p_value": p_value,
        "tau_alpha": float("nan"),
        "tail_strength": float(topo_effect),
        "g_score": float(topo_effect),
        "mu0": null_mean,
        "sd0": null_std,
        "null_distance_median": null_median,
        "null_distance_iqr": null_iqr,
        "null_distance_mean": null_mean,
        "null_distance_std": null_std,
        "num_permutations": int(num_permutations),
    }


def _rtd_lite_similarity_matrix(
    x_cache: RTDLiteLayerCache,
    y_cache: RTDLiteLayerCache,
    *,
    y_perm: np.ndarray | None = None,
) -> torch.Tensor:
    n_x = len(x_cache.distances)
    n_y = len(y_cache.distances)

    S = torch.empty((n_x, n_y), dtype=torch.float32)

    for i in range(n_x):
        D_x = x_cache.distances[i]
        mst_x = x_cache.mst_weights[i]

        for j in range(n_y):
            D_y = y_cache.distances[j]
            mst_y = y_cache.mst_weights[j]

            if y_perm is not None:
                # Permuting rows of Y corresponds to permuting both axes
                # of its pairwise distance matrix.
                D_y_eff = D_y[np.ix_(y_perm, y_perm)]
            else:
                D_y_eff = D_y

            S[i, j] = _rtd_lite_similarity_from_cached(
                D_x,
                mst_x,
                D_y_eff,
                mst_y,
            )

    return S

def _mst_overlap_similarity_from_cached(
    edges_x: np.ndarray,
    edges_y: np.ndarray,
    *,
    n: int,
) -> float:
    """Direct MST support overlap: ``|T_x ∩ T_y| / (n - 1)``."""
    return _edge_overlap_fraction(edges_x, edges_y, n)


def _mst_min_overlap_similarity_from_cached(
    D_x: np.ndarray,
    edges_x: np.ndarray,
    D_y: np.ndarray,
    edges_y: np.ndarray,
) -> float:
    """Min-graph support overlap used as an RTD-Lite diagnostic."""
    n = D_x.shape[0]
    if n <= 1:
        return 1.0

    edges_min = _mst_edge_codes_numpy(np.minimum(D_x, D_y))
    overlap_x = _edge_overlap_fraction(edges_min, edges_x, n)
    overlap_y = _edge_overlap_fraction(edges_min, edges_y, n)
    return 0.5 * (overlap_x + overlap_y)


def _mst_overlap_similarity_matrix(
    x_cache: RTDLiteLayerCache,
    y_cache: RTDLiteLayerCache,
    *,
    y_perm: np.ndarray | None = None,
) -> torch.Tensor:
    n_x = len(x_cache.mst_edges)
    n_y = len(y_cache.mst_edges)
    n = x_cache.distances[0].shape[0]

    S = torch.empty((n_x, n_y), dtype=torch.float32)

    for i in range(n_x):
        edges_x = x_cache.mst_edges[i]
        for j in range(n_y):
            edges_y = y_cache.mst_edges[j]
            if y_perm is not None:
                edges_y = _permute_edge_codes(edges_y, y_perm, n)

            S[i, j] = _mst_overlap_similarity_from_cached(
                edges_x,
                edges_y,
                n=n,
            )

    return S


def _mst_min_overlap_similarity_matrix(
    x_cache: RTDLiteLayerCache,
    y_cache: RTDLiteLayerCache,
    *,
    y_perm: np.ndarray | None = None,
) -> torch.Tensor:
    n_x = len(x_cache.distances)
    n_y = len(y_cache.distances)
    n = x_cache.distances[0].shape[0]

    S = torch.empty((n_x, n_y), dtype=torch.float32)

    for i in range(n_x):
        D_x = x_cache.distances[i]
        edges_x = x_cache.mst_edges[i]

        for j in range(n_y):
            D_y = y_cache.distances[j]
            edges_y = y_cache.mst_edges[j]

            if y_perm is not None:
                D_y = D_y[np.ix_(y_perm, y_perm)]
                edges_y = _permute_edge_codes(edges_y, y_perm, n)

            S[i, j] = _mst_min_overlap_similarity_from_cached(
                D_x,
                edges_x,
                D_y,
                edges_y,
            )

    return S

def _mst_distance_overlap_similarity_from_cached(
    D_x: np.ndarray,
    edges_x: np.ndarray,
    D_y: np.ndarray,
    edges_y: np.ndarray,
    *,
    n: int,
    distance_tau: float,
    y_perm: np.ndarray | None = None,
    eps: float = 1e-12,
) -> float:
    """Distance-aware direct MST support overlap.

    If y_perm is given, compare X against the permuted Y distance matrix
    without explicitly materializing D_y[np.ix_(y_perm, y_perm)].
    """
    if n <= 1:
        return 1.0

    if distance_tau <= 0:
        raise ValueError(f"distance_tau must be positive, got {distance_tau}")

    if y_perm is not None:
        edges_y_eff = _permute_edge_codes(edges_y, y_perm, n)
    else:
        edges_y_eff = edges_y

    common = np.intersect1d(edges_x, edges_y_eff, assume_unique=True)

    if common.size == 0:
        return 0.0

    u = common // n
    v = common % n

    dx = np.maximum(D_x[u, v], eps)

    if y_perm is None:
        dy = np.maximum(D_y[u, v], eps)
    else:
        # D_y_perm[u, v] = D_y[y_perm[u], y_perm[v]]
        dy = np.maximum(D_y[y_perm[u], y_perm[v]], eps)

    log_diff = np.abs(np.log(dx) - np.log(dy))
    weights = np.exp(-log_diff / float(distance_tau))

    return float(weights.sum() / float(n - 1))

def _mst_distance_overlap_similarity_matrix(
    x_cache: RTDLiteLayerCache,
    y_cache: RTDLiteLayerCache,
    *,
    distance_tau: float,
    y_perm: np.ndarray | None = None,
) -> torch.Tensor:
    n_x = len(x_cache.distances)
    n_y = len(y_cache.distances)
    n = x_cache.distances[0].shape[0]

    S = torch.empty((n_x, n_y), dtype=torch.float32)

    for i in range(n_x):
        D_x = x_cache.value_distances[i]
        edges_x = x_cache.mst_edges[i]

        for j in range(n_y):
            D_y = y_cache.value_distances[j]
            edges_y = y_cache.mst_edges[j]

            S[i, j] = _mst_distance_overlap_similarity_from_cached(
                D_x,
                edges_x,
                D_y,
                edges_y,
                n=n,
                distance_tau=distance_tau,
                y_perm=y_perm,
            )

    return S




def _normalize_distance_array(
    D: np.ndarray,
    *,
    mode: str = "quantile",
    q: float = 0.9,
    eps: float = 1e-12,
) -> np.ndarray:
    """Normalize numerical distance values without changing support."""
    D = np.asarray(D, dtype=np.float64)
    if mode == "none" or D.shape[0] <= 1:
        out = D.copy()
        np.fill_diagonal(out, 0.0)
        return out

    values = D[D > 0]
    if values.size == 0:
        out = D.copy()
        np.fill_diagonal(out, 0.0)
        return out
    if mode == "quantile":
        if not (0.0 < float(q) <= 1.0):
            raise ValueError(f"distance_normalization_q must be in (0, 1], got {q}")
        scale = float(np.quantile(values, float(q)))
    elif mode == "mean":
        scale = float(np.mean(values))
    else:
        raise ValueError(
            "distance_normalization must be 'quantile', 'mean', or 'none'; "
            f"got {mode!r}"
        )
    scale = max(scale, float(eps))
    out = D / scale
    np.fill_diagonal(out, 0.0)
    return out


def prepare_mst_overlap_cache(
    layers: Sequence[torch.Tensor],
    *,
    dist_metric: str = "euclidean",
    normalize: bool = True,
    q: float = 0.9,
    distance_normalization: str = "quantile",
    distance_value_geometry: str = "ambient",
    geodesic_k: int = 20,
    riemannian_k: int = 30,
    riemannian_dim: int = 10,
    riemannian_reg: float = 1e-3,
) -> MSTOverlapLayerCache:
    """Build a memory-light MST support/value cache once per model.

    Only ``n-1`` support edge identities and their distance values are retained
    per layer.  This is sufficient even under label permutations because a
    permuted Y-MST edge keeps the value of its corresponding original Y edge.
    Expensive geodesic/Riemannian geometry is therefore computed once per
    model/layer rather than once per cross-model pair.
    """
    edges_all: list[np.ndarray] = []
    values_all: list[np.ndarray] = []
    n_ref: int | None = None

    for layer in layers:
        with torch.no_grad():
            D_raw = _pairwise_distances(layer.detach().float().cpu(), metric=dist_metric)
            D_support = _quantile_normalize(D_raw, q=q) if normalize else D_raw
            D_support_np = D_support.numpy().astype(np.float64, copy=False)
            n = D_support_np.shape[0]
            if n_ref is None:
                n_ref = n
            elif n != n_ref:
                raise ValueError("All layers must have the same number of samples")

            edges = _mst_edge_codes_numpy(D_support_np)

            if distance_value_geometry == "ambient":
                D_value_raw = D_raw.numpy().astype(np.float64, copy=False)
            elif distance_value_geometry == "geodesic":
                D_value_raw = graph_geodesic_from_distances(
                    D_raw.numpy().astype(np.float64, copy=False),
                    k=geodesic_k,
                )
            elif distance_value_geometry == "riemannian":
                D_value_raw = local_riemannian_geodesic_distance_matrix(
                    layer.detach().float().cpu(),
                    metric_k=riemannian_k,
                    tangent_dim=riemannian_dim,
                    regularization=riemannian_reg,
                    q=q,
                    normalize=False,
                )
            else:
                raise ValueError(
                    "distance_value_geometry must be 'ambient', 'geodesic', or "
                    f"'riemannian'; got {distance_value_geometry!r}"
                )

            D_value_np = (
                _normalize_distance_array(
                    D_value_raw,
                    mode=distance_normalization,
                    q=q,
                )
                if normalize
                else np.asarray(D_value_raw, dtype=np.float64)
            )

        u = edges // n
        v = edges % n
        values = np.asarray(D_value_np[u, v], dtype=np.float64)
        edges_all.append(edges)
        values_all.append(values)

    return MSTOverlapLayerCache(
        n=0 if n_ref is None else int(n_ref),
        mst_edges=edges_all,
        mst_edge_values=values_all,
    )



def prepare_mst_similarity_cache(
    layers: Sequence[torch.Tensor],
    *,
    dist_metric: str = "euclidean",
    normalize_support: bool = True,
    q: float = 0.9,
) -> MSTOverlapLayerCache:
    """Fixed MST support with cosine-similarity values on the selected edges."""
    edges_all: list[np.ndarray] = []
    values_all: list[np.ndarray] = []
    n_ref: int | None = None

    for layer in layers:
        with torch.no_grad():
            layer_cpu = layer.detach().float().cpu()
            D_raw = _pairwise_distances(layer_cpu, metric=dist_metric)
            D_support = _quantile_normalize(D_raw, q=q) if normalize_support else D_raw
            D_support_np = D_support.numpy().astype(np.float64, copy=False)
            n = D_support_np.shape[0]
            if n_ref is None:
                n_ref = n
            elif n != n_ref:
                raise ValueError("All layers must have the same number of samples")

            edges = _mst_edge_codes_numpy(D_support_np)

            x = torch.nn.functional.normalize(layer_cpu, dim=1)
            S = (x @ x.T).numpy().astype(np.float64, copy=False)

        u = edges // n
        v = edges % n
        edges_all.append(edges)
        values_all.append(np.asarray(S[u, v], dtype=np.float64))

    return MSTOverlapLayerCache(
        n=0 if n_ref is None else int(n_ref),
        mst_edges=edges_all,
        mst_edge_values=values_all,
    )


def _mst_edge_similarity_overlap_from_cached(
    edges_x: np.ndarray,
    values_x: np.ndarray,
    edges_y: np.ndarray,
    values_y: np.ndarray,
    *,
    n: int,
    similarity_tau: float,
    y_perm: np.ndarray | None = None,
) -> float:
    if n <= 1:
        return 1.0
    if similarity_tau <= 0:
        raise ValueError(f"similarity_tau must be positive, got {similarity_tau}")

    edges_y_eff = _permute_edge_codes(edges_y, y_perm, n) if y_perm is not None else edges_y
    order_x = np.argsort(edges_x)
    sx, vx = edges_x[order_x], values_x[order_x]
    order_y = np.argsort(edges_y_eff)
    sy, vy = edges_y_eff[order_y], values_y[order_y]

    common = np.intersect1d(sx, sy, assume_unique=True)
    if common.size == 0:
        return 0.0

    ix = np.searchsorted(sx, common)
    iy = np.searchsorted(sy, common)
    weights = np.exp(-np.abs(vx[ix] - vy[iy]) / float(similarity_tau))
    return float(weights.sum() / float(n - 1))


def _mst_edge_value_overlap_from_cached(
    edges_x: np.ndarray,
    values_x: np.ndarray,
    edges_y: np.ndarray,
    values_y: np.ndarray,
    *,
    n: int,
    distance_tau: float,
    y_perm: np.ndarray | None = None,
    eps: float = 1e-12,
) -> float:
    if n <= 1:
        return 1.0
    if distance_tau <= 0:
        raise ValueError(f"distance_tau must be positive, got {distance_tau}")

    edges_y_eff = _permute_edge_codes(edges_y, y_perm, n) if y_perm is not None else edges_y

    order_x = np.argsort(edges_x)
    sx = edges_x[order_x]
    vx = values_x[order_x]
    order_y = np.argsort(edges_y_eff)
    sy = edges_y_eff[order_y]
    vy = values_y[order_y]

    common = np.intersect1d(sx, sy, assume_unique=True)
    if common.size == 0:
        return 0.0

    ix = np.searchsorted(sx, common)
    iy = np.searchsorted(sy, common)
    dx = np.maximum(vx[ix], eps)
    dy = np.maximum(vy[iy], eps)
    weights = np.exp(-np.abs(np.log(dx) - np.log(dy)) / float(distance_tau))
    return float(weights.sum() / float(n - 1))


def _mst_edge_value_similarity_matrix(
    x_cache: MSTOverlapLayerCache,
    y_cache: MSTOverlapLayerCache,
    *,
    distance_tau: float,
    y_perm: np.ndarray | None = None,
) -> torch.Tensor:
    if x_cache.n != y_cache.n:
        raise ValueError("X and Y MST caches must have the same sample count")
    S = torch.empty((len(x_cache.mst_edges), len(y_cache.mst_edges)), dtype=torch.float32)
    for i, (ex, vx) in enumerate(zip(x_cache.mst_edges, x_cache.mst_edge_values)):
        for j, (ey, vy) in enumerate(zip(y_cache.mst_edges, y_cache.mst_edge_values)):
            S[i, j] = _mst_edge_value_overlap_from_cached(
                ex, vx, ey, vy,
                n=x_cache.n,
                distance_tau=distance_tau,
                y_perm=y_perm,
            )
    return S


def compute_alignment_gated_mst_distance_from_caches(
    x_cache: MSTOverlapLayerCache,
    y_cache: MSTOverlapLayerCache,
    *,
    distance_tau: float = 0.01,
    num_permutations: int = 0,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Dict[str, float | Tuple[int, int]]:
    """Distance-aware MST overlap from lean model-level caches."""
    S = _mst_edge_value_similarity_matrix(
        x_cache, y_cache, distance_tau=distance_tau
    )
    agg = agg_max(S, return_indices=True)
    T_obs = float(agg.value)
    best_indices = (
        (int(agg.indices["i"]), int(agg.indices["j"])) if agg.indices else (0, 0)
    )

    if num_permutations <= 0:
        return {
            "raw_score": T_obs,
            "best_indices": best_indices,
            "p_value": float("nan"),
            "tau_alpha": float("nan"),
            "tail_strength": float("nan"),
            "g_score": T_obs,
            "mu0": float("nan"),
            "sd0": float("nan"),
            "distance_tau": float(distance_tau),
            "metric_base": "mst_overlap_dist",
        }

    rng = np.random.default_rng(seed)
    null_samples = []
    for _ in range(num_permutations):
        perm = rng.permutation(x_cache.n)
        S_perm = _mst_edge_value_similarity_matrix(
            x_cache, y_cache, distance_tau=distance_tau, y_perm=perm
        )
        null_samples.append(float(agg_max(S_perm).value))

    summary = compute_null_summary(
        torch.tensor(null_samples, dtype=torch.float32), T_obs=T_obs, alpha=alpha
    )
    return {
        "raw_score": T_obs,
        "best_indices": best_indices,
        "p_value": summary["p_value"],
        "tau_alpha": summary["tau_alpha"],
        "tail_strength": summary["tail_strength"],
        "g_score": gated_rescaled(T_obs, tau_alpha=summary["tau_alpha"], s_max=1.0),
        "mu0": summary["mu0"],
        "sd0": summary["sd0"],
        "distance_tau": float(distance_tau),
        "metric_base": "mst_overlap_dist",
    }


def _mst_edge_similarity_similarity_matrix(
    x_cache: MSTOverlapLayerCache,
    y_cache: MSTOverlapLayerCache,
    *,
    similarity_tau: float,
    y_perm: np.ndarray | None = None,
) -> torch.Tensor:
    if x_cache.n != y_cache.n:
        raise ValueError("X and Y MST caches must have the same sample count")
    S = torch.empty((len(x_cache.mst_edges), len(y_cache.mst_edges)), dtype=torch.float32)
    for i, (ex, vx) in enumerate(zip(x_cache.mst_edges, x_cache.mst_edge_values)):
        for j, (ey, vy) in enumerate(zip(y_cache.mst_edges, y_cache.mst_edge_values)):
            S[i, j] = _mst_edge_similarity_overlap_from_cached(
                ex,
                vx,
                ey,
                vy,
                n=x_cache.n,
                similarity_tau=similarity_tau,
                y_perm=y_perm,
            )
    return S


def compute_alignment_gated_mst_similarity_from_caches(
    x_cache: MSTOverlapLayerCache,
    y_cache: MSTOverlapLayerCache,
    *,
    similarity_tau: float = 0.01,
    num_permutations: int = 0,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Dict[str, float | Tuple[int, int]]:
    """Cosine-similarity-aware MST overlap on fixed spanning support."""
    S = _mst_edge_similarity_similarity_matrix(
        x_cache, y_cache, similarity_tau=similarity_tau
    )
    agg = agg_max(S, return_indices=True)
    T_obs = float(agg.value)
    best_indices = (
        (int(agg.indices["i"]), int(agg.indices["j"])) if agg.indices else (0, 0)
    )

    if num_permutations <= 0:
        return {
            "raw_score": T_obs,
            "best_indices": best_indices,
            "p_value": float("nan"),
            "tau_alpha": float("nan"),
            "tail_strength": float("nan"),
            "g_score": T_obs,
            "mu0": float("nan"),
            "sd0": float("nan"),
            "similarity_tau": float(similarity_tau),
            "metric_base": "mst_overlap_sim",
        }

    rng = np.random.default_rng(seed)
    null_samples = []
    for _ in range(num_permutations):
        perm = rng.permutation(x_cache.n)
        S_perm = _mst_edge_similarity_similarity_matrix(
            x_cache, y_cache, similarity_tau=similarity_tau, y_perm=perm
        )
        null_samples.append(float(agg_max(S_perm).value))

    summary = compute_null_summary(
        torch.tensor(null_samples, dtype=torch.float32), T_obs=T_obs, alpha=alpha
    )
    return {
        "raw_score": T_obs,
        "best_indices": best_indices,
        "p_value": summary["p_value"],
        "tau_alpha": summary["tau_alpha"],
        "tail_strength": summary["tail_strength"],
        "g_score": gated_rescaled(T_obs, tau_alpha=summary["tau_alpha"], s_max=1.0),
        "mu0": summary["mu0"],
        "sd0": summary["sd0"],
        "similarity_tau": float(similarity_tau),
        "metric_base": "mst_overlap_sim",
    }


def _pd_metric_spec(metric: str) -> tuple[tuple[int, ...], Literal["bottleneck", "wasserstein"]]:
    if metric == "pd0_bottleneck":
        return (0,), "bottleneck"
    if metric == "pd1_bottleneck":
        return (1,), "bottleneck"
    if metric == "pd01_bottleneck":
        return (0, 1), "bottleneck"

    if metric == "pd0_wasserstein":
        return (0,), "wasserstein"
    if metric == "pd1_wasserstein":
        return (1,), "wasserstein"
    if metric == "pd01_wasserstein":
        return (0, 1), "wasserstein"

    raise ValueError(f"Unsupported PD topology metric: {metric}")


def _prepare_pd_cache(
    layers: Sequence[torch.Tensor],
    *,
    maxdim: int,
    dist_metric: Literal["euclidean", "cosine"] = "euclidean",
    normalize: bool = True,
    q: float = 0.9,
) -> PDLayerCache:
    diagrams: list[list[np.ndarray]] = []

    for layer in layers:
        dgms = _ripser_diagrams(
            layer,
            maxdim=maxdim,
            dist_metric=dist_metric,
            normalize=normalize,
            q=q,
        )
        diagrams.append(dgms)

    return PDLayerCache(diagrams=diagrams)


def _pd_similarity_from_cached(
    dgms_x: list[np.ndarray],
    dgms_y: list[np.ndarray],
    *,
    homology_dims: tuple[int, ...],
    metric: Literal["bottleneck", "wasserstein"],
) -> float:
    weight = 1.0 / len(homology_dims)
    total = 0.0

    for dim in homology_dims:
        dgm_x = dgms_x[dim] if dim < len(dgms_x) else np.empty((0, 2), dtype=np.float64)
        dgm_y = dgms_y[dim] if dim < len(dgms_y) else np.empty((0, 2), dtype=np.float64)

        total += weight * _diagram_distance(dgm_x, dgm_y, metric=metric)

    return _distance_to_similarity(total)


def _pd_similarity_matrix(
    x_cache: PDLayerCache,
    y_cache: PDLayerCache,
    *,
    homology_dims: tuple[int, ...],
    metric: Literal["bottleneck", "wasserstein"],
) -> torch.Tensor:
    n_x = len(x_cache.diagrams)
    n_y = len(y_cache.diagrams)

    S = torch.empty((n_x, n_y), dtype=torch.float32)

    for i in range(n_x):
        for j in range(n_y):
            S[i, j] = _pd_similarity_from_cached(
                x_cache.diagrams[i],
                y_cache.diagrams[j],
                homology_dims=homology_dims,
                metric=metric,
            )

    return S


def _uncalibrated_result(
    S: torch.Tensor,
) -> Dict[str, float | Tuple[int, int]]:
    agg = agg_max(S, return_indices=True)
    T_obs = float(agg.value)

    best_indices = (
        (int(agg.indices["i"]), int(agg.indices["j"]))
        if agg.indices
        else (0, 0)
    )

    return {
        "raw_score": T_obs,
        "best_indices": best_indices,
        "p_value": float("nan"),
        "tau_alpha": float("nan"),
        "tail_strength": float("nan"),
        "g_score": T_obs,
        "mu0": float("nan"),
        "sd0": float("nan"),
    }


def compute_alignment_gated_topology_cached(
    x_layers: Sequence[torch.Tensor],
    y_layers: Sequence[torch.Tensor],
    *,
    metric: str,
    num_permutations: int = 0,
    alpha: float = 0.05,
    seed: int | None = None,
    dist_metric: str = "euclidean",
    normalize_distances: bool = True,
    q: float = 0.9,
    distance_value_geometry: str = "ambient",
    geodesic_k: int = 20,
    riemannian_k: int = 30,
    riemannian_dim: int = 10,
    riemannian_reg: float = 1e-3,
) -> Dict[str, float | Tuple[int, int]]:
    """Cached topology-aware alignment.

    RTD-Lite:
        - caches pairwise distance matrices per layer
        - caches MST(D_layer) per layer
        - optionally supports a small number of row permutations

    PD metrics:
        - caches persistence diagrams per layer
        - disables row-permutation calibration because PDs are row-order invariant
    """
    metric_for_compute, distance_tau = _parse_topology_metric(metric)

    if metric_for_compute not in TOPOLOGY_METRICS:
        raise ValueError(f"Unsupported topology metric: {metric}")
    
    if len(x_layers) == 0 or len(y_layers) == 0:
        raise ValueError("x_layers and y_layers must be nonempty")

    n = x_layers[0].shape[0]
    if any(layer.shape[0] != n for layer in x_layers):
        raise ValueError("All x_layers must have the same number of samples")
    if any(layer.shape[0] != n for layer in y_layers):
        raise ValueError("All y_layers must have the same number of samples as x_layers")

    if metric_for_compute  == "rtd_lite":
        x_cache = _prepare_rtd_lite_cache(
            x_layers,
            dist_metric=dist_metric,
            normalize=normalize_distances,
            q=q,
        )
        y_cache = _prepare_rtd_lite_cache(
            y_layers,
            dist_metric=dist_metric,
            normalize=normalize_distances,
            q=q,
        )

        S = _rtd_lite_similarity_matrix(x_cache, y_cache)
        agg = agg_max(S, return_indices=True)
        T_obs = float(agg.value)

        best_indices = (
            (int(agg.indices["i"]), int(agg.indices["j"]))
            if agg.indices
            else (0, 0)
        )

        if num_permutations <= 0:
            return {
                "raw_score": T_obs,
                "best_indices": best_indices,
                "p_value": float("nan"),
                "tau_alpha": float("nan"),
                "tail_strength": float("nan"),
                "g_score": T_obs,
                "mu0": float("nan"),
                "sd0": float("nan"),
            }

        rng = np.random.default_rng(seed)
        null_samples = []

        for _ in range(num_permutations):
            perm = rng.permutation(n)
            S_perm = _rtd_lite_similarity_matrix(x_cache, y_cache, y_perm=perm)
            null_samples.append(float(agg_max(S_perm).value))

        null_samples_t = torch.tensor(null_samples, dtype=torch.float32)
        summary = compute_null_summary(null_samples_t, T_obs=T_obs, alpha=alpha)

        g_score = gated_rescaled(
            T_obs,
            tau_alpha=summary["tau_alpha"],
            s_max=1.0,
        )

        return {
            "raw_score": T_obs,
            "best_indices": best_indices,
            "p_value": summary["p_value"],
            "tau_alpha": summary["tau_alpha"],
            "tail_strength": summary["tail_strength"],
            "g_score": g_score,
            "mu0": summary["mu0"],
            "sd0": summary["sd0"],
        }

    if metric_for_compute == "mst_overlap_sim":
        if distance_tau is None:
            distance_tau = 0.01
        x_cache = prepare_mst_similarity_cache(
            x_layers,
            dist_metric=dist_metric,
            normalize_support=normalize_distances,
            q=q,
        )
        y_cache = prepare_mst_similarity_cache(
            y_layers,
            dist_metric=dist_metric,
            normalize_support=normalize_distances,
            q=q,
        )
        return compute_alignment_gated_mst_similarity_from_caches(
            x_cache,
            y_cache,
            similarity_tau=distance_tau,
            num_permutations=num_permutations,
            alpha=alpha,
            seed=seed,
        )

    if metric_for_compute in {"mst_overlap", "mst_overlap_dist", "mst_min_overlap"}:
        value_geometry = (
            distance_value_geometry
            if metric_for_compute == "mst_overlap_dist"
            else "ambient"
        )
        x_cache = _prepare_rtd_lite_cache(
            x_layers,
            dist_metric=dist_metric,
            normalize=normalize_distances,
            q=q,
            distance_value_geometry=value_geometry,
            geodesic_k=geodesic_k,
            riemannian_k=riemannian_k,
            riemannian_dim=riemannian_dim,
            riemannian_reg=riemannian_reg,
        )
        y_cache = _prepare_rtd_lite_cache(
            y_layers,
            dist_metric=dist_metric,
            normalize=normalize_distances,
            q=q,
            distance_value_geometry=value_geometry,
            geodesic_k=geodesic_k,
            riemannian_k=riemannian_k,
            riemannian_dim=riemannian_dim,
            riemannian_reg=riemannian_reg,
        )

        if metric_for_compute == "mst_overlap":
            def matrix_fn(x_cache, y_cache, *, y_perm=None):
                return _mst_overlap_similarity_matrix(
                    x_cache,
                    y_cache,
                    y_perm=y_perm,
                )
        elif metric_for_compute == "mst_overlap_dist":
            if distance_tau is None:
                distance_tau = 0.01
            def matrix_fn(x_cache, y_cache, *, y_perm=None):
                return _mst_distance_overlap_similarity_matrix(
                    x_cache,
                    y_cache,
                    distance_tau=distance_tau,
                    y_perm=y_perm,
                )
        elif metric_for_compute == "mst_min_overlap":
            def matrix_fn(x_cache, y_cache, *, y_perm=None):
                return _mst_min_overlap_similarity_matrix(
                    x_cache,
                    y_cache,
                    y_perm=y_perm,
                )

        else:
            raise ValueError(f"Unsupported MST-style metric: {metric}")
        S = matrix_fn(x_cache, y_cache)
        agg = agg_max(S, return_indices=True)
        T_obs = float(agg.value)

        best_indices = (
            (int(agg.indices["i"]), int(agg.indices["j"]))
            if agg.indices
            else (0, 0)
        )

        if num_permutations <= 0:
            out = {
                "raw_score": T_obs,
                "best_indices": best_indices,
                "p_value": float("nan"),
                "tau_alpha": float("nan"),
                "tail_strength": float("nan"),
                "g_score": T_obs,
                "mu0": float("nan"),
                "sd0": float("nan"),
                "metric_base": metric_for_compute,
            }
            if distance_tau is not None:
                out["distance_tau"] = float(distance_tau)
            if metric_for_compute == "mst_overlap_dist":
                out["distance_value_geometry"] = distance_value_geometry
                out["geodesic_k"] = int(geodesic_k)
                out["riemannian_k"] = int(riemannian_k)
                out["riemannian_dim"] = int(riemannian_dim)
                out["riemannian_reg"] = float(riemannian_reg)
            return out

        rng = np.random.default_rng(seed)
        null_samples = []

        for _ in range(num_permutations):
            perm = rng.permutation(n)
            S_perm = matrix_fn(x_cache, y_cache, y_perm=perm)
            null_samples.append(float(agg_max(S_perm).value))

        null_samples_t = torch.tensor(null_samples, dtype=torch.float32)
        summary = compute_null_summary(null_samples_t, T_obs=T_obs, alpha=alpha)

        g_score = gated_rescaled(
            T_obs,
            tau_alpha=summary["tau_alpha"],
            s_max=1.0,
        )

        out = {
            "raw_score": T_obs,
            "best_indices": best_indices,
            "p_value": summary["p_value"],
            "tau_alpha": summary["tau_alpha"],
            "tail_strength": summary["tail_strength"],
            "g_score": g_score,
            "mu0": summary["mu0"],
            "sd0": summary["sd0"],
            "metric_base": metric_for_compute,
        }
        if distance_tau is not None:
            out["distance_tau"] = float(distance_tau)
        if metric_for_compute == "mst_overlap_dist":
            out["distance_value_geometry"] = distance_value_geometry
            out["geodesic_k"] = int(geodesic_k)
            out["riemannian_k"] = int(riemannian_k)
            out["riemannian_dim"] = int(riemannian_dim)
            out["riemannian_reg"] = float(riemannian_reg)
        return out
    
    if metric_for_compute  == "rtd_lite_tai":
        cache_x = _prepare_rtd_lite_cache(
            x_layers,
            dist_metric=dist_metric,
            normalize=normalize_distances,
            q=q,
        )
        cache_y = _prepare_rtd_lite_cache(
            y_layers,
            dist_metric=dist_metric,
            normalize=normalize_distances,
            q=q,
        )

        return _rtd_lite_tai_result(
            cache_x,
            cache_y,
            num_permutations=num_permutations,
            seed=seed,
        )

    # PD metrics: cache diagrams and do not run row-permutation calibration.
    homology_dims, pd_metric = _pd_metric_spec(metric)
    maxdim = max(homology_dims)

    x_cache = _prepare_pd_cache(
        x_layers,
        maxdim=maxdim,
        dist_metric=dist_metric,
        normalize=normalize_distances,
        q=q,
    )
    y_cache = _prepare_pd_cache(
        y_layers,
        maxdim=maxdim,
        dist_metric=dist_metric,
        normalize=normalize_distances,
        q=q,
    )

    S = _pd_similarity_matrix(
        x_cache,
        y_cache,
        homology_dims=homology_dims,
        metric=pd_metric,
    )

    return _uncalibrated_result(S)