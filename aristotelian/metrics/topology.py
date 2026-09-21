# aristotelian/metrics/topology.py

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import torch

from .base import BaseMetric, MetricConfig
from .registry import register_metric
from .utils import EPS


# ---------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------

DistanceMetric = Literal["euclidean", "cosine"]
PDDistanceMetric = Literal["bottleneck", "wasserstein"]


def _pairwise_distances(
    X: torch.Tensor,
    metric: str = "euclidean",
    eps: float = 1e-12,
) -> torch.Tensor:
    X = X.float()

    if metric == "euclidean":
        return torch.cdist(X, X, p=2)

    if metric == "cosine":
        Xn = X / (X.norm(dim=1, keepdim=True).clamp_min(eps))
        sim = Xn @ Xn.T
        sim = sim.clamp(-1.0, 1.0)
        D = 1.0 - sim
        D.fill_diagonal_(0.0)
        return D.clamp_min(0.0)

    if metric == "normalized_euclidean":
        Xn = X / (X.norm(dim=1, keepdim=True).clamp_min(eps))
        D = torch.cdist(Xn, Xn, p=2)
        D.fill_diagonal_(0.0)
        return D

    if metric == "angular":
        Xn = X / (X.norm(dim=1, keepdim=True).clamp_min(eps))
        sim = (Xn @ Xn.T).clamp(-1.0, 1.0)
        D = torch.acos(sim) / torch.pi
        D.fill_diagonal_(0.0)
        return D

    raise ValueError(f"Unknown distance metric: {metric}")


def _quantile_normalize(
    D: torch.Tensor,
    *,
    q: float = 0.9,
) -> torch.Tensor:
    """Normalize a distance matrix by a robust distance scale."""
    if not (0.0 < q <= 1.0):
        raise ValueError(f"q must be in (0, 1], got {q}")

    vals = D[D > 0]
    if vals.numel() == 0:
        return D

    scale = torch.quantile(vals, q).clamp_min(EPS)
    return D / scale


def _distance_to_similarity(distance: float) -> float:
    """Convert a nonnegative distance to a bounded similarity score.

    This is a monotone decreasing transform:
        distance = 0      -> similarity = 1
        distance -> inf   -> similarity -> 0

    It is used to align distance-based topology metrics with the
    Aristotelian convention that larger scores indicate stronger alignment.
    """
    distance = max(float(distance), 0.0)
    # return 1.0 / (1.0 + distance)
    return - distance


def _validate_same_number_of_samples(X: torch.Tensor, Y: torch.Tensor) -> None:
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(
            f"X and Y must be 2D tensors, got {tuple(X.shape)} and {tuple(Y.shape)}"
        )

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of samples, "
            f"got {X.shape[0]} and {Y.shape[0]}"
        )


def _get_config_attr(config: MetricConfig, name: str, default):
    """Read optional topology config fields without requiring MetricConfig edits."""
    return getattr(config, name, default)

def _prim_mst_edge_codes(D: torch.Tensor) -> torch.Tensor:
    """Dense Prim MST edge set for a complete weighted graph.

    Edges are encoded as ``u * n + v`` with ``u < v`` so that two MSTs can be
    compared by edge identity. The implementation mirrors
    ``_prim_mst_total_weight`` to keep tie-breaking deterministic.
    """
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be a square distance matrix, got {tuple(D.shape)}")

    if not torch.isfinite(D).all():
        raise ValueError("D contains NaN or Inf")

    n = D.shape[0]
    if n <= 1:
        return torch.empty((0,), dtype=torch.long, device=D.device)

    inf = torch.tensor(float("inf"), dtype=D.dtype, device=D.device)

    visited = torch.zeros(n, dtype=torch.bool, device=D.device)
    min_dist = torch.full((n,), inf, dtype=D.dtype, device=D.device)
    parent = torch.full((n,), -1, dtype=torch.long, device=D.device)

    v = torch.tensor(0, dtype=torch.long, device=D.device)
    edge_codes: list[torch.Tensor] = []

    for _ in range(n - 1):
        visited[v] = True

        better = (~visited) & (D[v] < min_dist)
        min_dist[better] = D[v, better]
        parent[better] = v

        masked = min_dist.clone()
        masked[visited] = inf
        v = torch.argmin(masked)

        if parent[v] < 0:
            raise RuntimeError("Failed to construct MST. Check distance matrix.")

        u = torch.minimum(v, parent[v])
        w = torch.maximum(v, parent[v])
        edge_codes.append(u * n + w)

    return torch.stack(edge_codes).long()


def mst_overlap_from_distances(
    D1: torch.Tensor,
    D2: torch.Tensor,
    *,
    normalize: bool = True,
    q: float = 0.9,
) -> float:
    """Compute direct MST edge overlap from two pairwise distance matrices.

    Returns ``|MST(D1) ∩ MST(D2)| / (n - 1)``. This is a support-level
    similarity: it compares which sample pairs are selected by the MST, not the
    selected edge weights.
    """
    if D1.shape != D2.shape:
        raise ValueError(f"D1 and D2 must have same shape, got {D1.shape}, {D2.shape}")

    if D1.ndim != 2 or D1.shape[0] != D1.shape[1]:
        raise ValueError(f"D1 and D2 must be square matrices, got {tuple(D1.shape)}")

    D1 = D1.float()
    D2 = D2.float()

    if not torch.isfinite(D1).all() or not torch.isfinite(D2).all():
        raise ValueError("D1 or D2 contains NaN or Inf")

    if normalize:
        D1 = _quantile_normalize(D1, q=q)
        D2 = _quantile_normalize(D2, q=q)

    n = D1.shape[0]
    if n <= 1:
        return 1.0

    e1 = _prim_mst_edge_codes(D1)
    e2 = _prim_mst_edge_codes(D2)
    overlap = torch.isin(e1, e2).float().sum()
    return float((overlap / float(n - 1)).detach().cpu().item())


def mst_min_overlap_from_distances(
    D1: torch.Tensor,
    D2: torch.Tensor,
    *,
    normalize: bool = True,
    q: float = 0.9,
) -> float:
    """Compute RTD-Lite min-graph MST support overlap.

    Let ``C = min(D1, D2)``. Returns
    ``0.5 * (|MST(C) ∩ MST(D1)| + |MST(C) ∩ MST(D2)|) / (n - 1)``.
    This is best used as a diagnostic for how much the RTD-Lite auxiliary MST
    is supported by the two original MSTs.
    """
    if D1.shape != D2.shape:
        raise ValueError(f"D1 and D2 must have same shape, got {D1.shape}, {D2.shape}")

    if D1.ndim != 2 or D1.shape[0] != D1.shape[1]:
        raise ValueError(f"D1 and D2 must be square matrices, got {tuple(D1.shape)}")

    D1 = D1.float()
    D2 = D2.float()

    if not torch.isfinite(D1).all() or not torch.isfinite(D2).all():
        raise ValueError("D1 or D2 contains NaN or Inf")

    if normalize:
        D1 = _quantile_normalize(D1, q=q)
        D2 = _quantile_normalize(D2, q=q)

    n = D1.shape[0]
    if n <= 1:
        return 1.0

    Dmin = torch.minimum(D1, D2)
    e1 = _prim_mst_edge_codes(D1)
    e2 = _prim_mst_edge_codes(D2)
    emin = _prim_mst_edge_codes(Dmin)

    overlap_1 = torch.isin(emin, e1).float().sum()
    overlap_2 = torch.isin(emin, e2).float().sum()
    score = 0.5 * (overlap_1 + overlap_2) / float(n - 1)
    return float(score.detach().cpu().item())


def mst_overlap(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    dist_metric: DistanceMetric = "euclidean",
    normalize: bool = True,
    q: float = 0.9,
) -> float:
    """Compute direct MST edge overlap between two representation matrices."""
    _validate_same_number_of_samples(X, Y)
    D1 = _pairwise_distances(X, metric=dist_metric)
    D2 = _pairwise_distances(Y, metric=dist_metric)
    return mst_overlap_from_distances(D1, D2, normalize=normalize, q=q)


def mst_min_overlap(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    dist_metric: DistanceMetric = "euclidean",
    normalize: bool = True,
    q: float = 0.9,
) -> float:
    """Compute min-graph MST support overlap between two representations."""
    _validate_same_number_of_samples(X, Y)
    D1 = _pairwise_distances(X, metric=dist_metric)
    D2 = _pairwise_distances(Y, metric=dist_metric)
    return mst_min_overlap_from_distances(D1, D2, normalize=normalize, q=q)

# ---------------------------------------------------------------------
# RTD-Lite implementation
# ---------------------------------------------------------------------

def _prim_mst_total_weight(D: torch.Tensor) -> torch.Tensor:
    """Dense Prim MST total weight for a complete weighted graph.

    Args:
        D: Square distance matrix of shape (n, n).

    Returns:
        Total weight of the minimum spanning tree.
    """
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be a square distance matrix, got {tuple(D.shape)}")

    if not torch.isfinite(D).all():
        raise ValueError("D contains NaN or Inf")

    n = D.shape[0]
    if n <= 1:
        return torch.zeros((), dtype=D.dtype, device=D.device)

    inf = torch.tensor(float("inf"), dtype=D.dtype, device=D.device)

    visited = torch.zeros(n, dtype=torch.bool, device=D.device)
    min_dist = torch.full((n,), inf, dtype=D.dtype, device=D.device)
    parent = torch.full((n,), -1, dtype=torch.long, device=D.device)

    v = torch.tensor(0, dtype=torch.long, device=D.device)
    total = torch.zeros((), dtype=D.dtype, device=D.device)

    for _ in range(n - 1):
        visited[v] = True

        better = (~visited) & (D[v] < min_dist)
        min_dist[better] = D[v, better]
        parent[better] = v

        masked = min_dist.clone()
        masked[visited] = inf
        v = torch.argmin(masked)

        if parent[v] < 0:
            raise RuntimeError("Failed to construct MST. Check distance matrix.")

        total = total + D[v, parent[v]]

    return total


def rtd_lite_distance_from_distances(
    D1: torch.Tensor,
    D2: torch.Tensor,
    *,
    normalize: bool = True,
    q: float = 0.9,
) -> float:
    """Compute RTD-Lite distance from two pairwise distance matrices.

    RTD-Lite summary:
        0.5 * [(MST(D1) - MST(min(D1, D2)))
             + (MST(D2) - MST(min(D1, D2)))]
    """
    if D1.shape != D2.shape:
        raise ValueError(f"D1 and D2 must have same shape, got {D1.shape}, {D2.shape}")

    if D1.ndim != 2 or D1.shape[0] != D1.shape[1]:
        raise ValueError(f"D1 and D2 must be square matrices, got {tuple(D1.shape)}")

    D1 = D1.float()
    D2 = D2.float()

    if not torch.isfinite(D1).all() or not torch.isfinite(D2).all():
        raise ValueError("D1 or D2 contains NaN or Inf")

    if normalize:
        D1 = _quantile_normalize(D1, q=q)
        D2 = _quantile_normalize(D2, q=q)

    Dmin = torch.minimum(D1, D2)

    s_min = _prim_mst_total_weight(Dmin)
    s1 = _prim_mst_total_weight(D1)
    s2 = _prim_mst_total_weight(D2)

    dist = 0.5 * ((s1 - s_min) + (s2 - s_min))
    return float(dist.detach().cpu().item())


def rtd_lite_distance(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    dist_metric: DistanceMetric = "euclidean",
    normalize: bool = True,
    q: float = 0.9,
) -> float:
    """Compute RTD-Lite distance between two representation matrices."""
    _validate_same_number_of_samples(X, Y)

    D1 = _pairwise_distances(X, metric=dist_metric)
    D2 = _pairwise_distances(Y, metric=dist_metric)

    return rtd_lite_distance_from_distances(D1, D2, normalize=normalize, q=q)


@register_metric
class RTDLiteSimilarity(BaseMetric):
    """RTD-Lite converted to similarity.

    This is a correspondence-aware topology metric.
    Larger score means more similar connectivity structure.
    """

    name = "rtd_lite"
    min_score = 0.0
    max_score = 1.0
    supports_calibration = True
    supports_caching = False

    def _compute_raw(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        config: MetricConfig,
    ) -> float:
        dist_metric = _get_config_attr(config, "topology_distance_metric", "euclidean")
        normalize = _get_config_attr(config, "topology_normalize", True)
        q = _get_config_attr(config, "topology_quantile", 0.9)

        dist = rtd_lite_distance(
            X,
            Y,
            dist_metric=dist_metric,
            normalize=normalize,
            q=q,
        )
        return _distance_to_similarity(dist)


# ---------------------------------------------------------------------
# Persistence diagram utilities
# ---------------------------------------------------------------------

def _ripser_diagrams(
    X: torch.Tensor,
    *,
    maxdim: int = 1,
    dist_metric: DistanceMetric = "euclidean",
    normalize: bool = True,
    q: float = 0.9,
) -> list[np.ndarray]:
    """Compute Vietoris-Rips persistence diagrams up to maxdim.

    Returns:
        List of diagrams [H0, H1, ..., Hmaxdim].
        Each diagram has shape (n_features, 2).
    """
    try:
        from ripser import ripser
    except ImportError as exc:
        raise ImportError(
            "PD metrics require ripser and persim. Install with: pip install ripser persim"
        ) from exc

    if maxdim < 0:
        raise ValueError(f"maxdim must be nonnegative, got {maxdim}")

    X_cpu = X.detach().float().cpu()

    D = _pairwise_distances(X_cpu, metric=dist_metric)

    if not torch.isfinite(D).all():
        raise ValueError("Distance matrix contains NaN or Inf")

    if normalize:
        D = _quantile_normalize(D, q=q)

    D_np = D.numpy()

    out = ripser(D_np, distance_matrix=True, maxdim=maxdim)
    dgms = out["dgms"]

    cleaned: list[np.ndarray] = []
    for dim in range(maxdim + 1):
        if dim >= len(dgms):
            cleaned.append(np.empty((0, 2), dtype=np.float64))
            continue

        dgm = np.asarray(dgms[dim], dtype=np.float64)

        if dgm.size == 0:
            cleaned.append(np.empty((0, 2), dtype=np.float64))
            continue

        # Remove infinite deaths. This matters especially for H0.
        dgm = dgm[np.isfinite(dgm[:, 1])]
        cleaned.append(dgm.astype(np.float64, copy=False))

    return cleaned


def _ripser_diagram(
    X: torch.Tensor,
    *,
    homology_dim: int = 1,
    maxdim: int | None = None,
    dist_metric: DistanceMetric = "euclidean",
    normalize: bool = True,
    q: float = 0.9,
) -> np.ndarray:
    """Compute a single homology dimension persistence diagram."""
    if homology_dim < 0:
        raise ValueError(f"homology_dim must be nonnegative, got {homology_dim}")

    if maxdim is None:
        maxdim = homology_dim
    elif maxdim < homology_dim:
        raise ValueError(
            f"maxdim must be >= homology_dim, got maxdim={maxdim}, "
            f"homology_dim={homology_dim}"
        )

    dgms = _ripser_diagrams(
        X,
        maxdim=maxdim,
        dist_metric=dist_metric,
        normalize=normalize,
        q=q,
    )

    if homology_dim >= len(dgms):
        return np.empty((0, 2), dtype=np.float64)

    return dgms[homology_dim]


def _distance_to_empty_diagram(
    dgm: np.ndarray,
    *,
    metric: PDDistanceMetric,
) -> float:
    """Distance from a diagram to the empty diagram.

    For bottleneck, this is max persistence / 2.
    For Wasserstein with p=1 and diagonal cost, this is sum persistence / 2.

    This fallback avoids possible library edge cases with empty diagrams.
    """
    if dgm.size == 0:
        return 0.0

    lifetimes = np.maximum(dgm[:, 1] - dgm[:, 0], 0.0)
    diag_costs = 0.5 * lifetimes

    if metric == "bottleneck":
        return float(np.max(diag_costs)) if diag_costs.size else 0.0

    if metric == "wasserstein":
        return float(np.sum(diag_costs))

    raise ValueError(f"Unknown PD distance metric: {metric}")


def _diagram_distance(
    dgm_x: np.ndarray,
    dgm_y: np.ndarray,
    *,
    metric: PDDistanceMetric = "bottleneck",
    wasserstein_matching: bool = False,
) -> float:
    """Compute bottleneck or Wasserstein distance between two diagrams."""
    if dgm_x.size == 0 and dgm_y.size == 0:
        return 0.0

    if dgm_x.size == 0:
        return _distance_to_empty_diagram(dgm_y, metric=metric)

    if dgm_y.size == 0:
        return _distance_to_empty_diagram(dgm_x, metric=metric)

    try:
        from persim import bottleneck, wasserstein
    except ImportError as exc:
        raise ImportError(
            "PD metrics require persim. Install with: pip install ripser persim"
        ) from exc

    if metric == "bottleneck":
        return float(bottleneck(dgm_x, dgm_y))

    if metric == "wasserstein":
        return float(wasserstein(dgm_x, dgm_y, matching=wasserstein_matching))

    raise ValueError(f"Unknown PD distance metric: {metric}")


def persistence_diagram_distance(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    homology_dim: int = 1,
    metric: PDDistanceMetric = "bottleneck",
    dist_metric: DistanceMetric = "euclidean",
    normalize: bool = True,
    q: float = 0.9,
    wasserstein_matching: bool = False,
) -> float:
    """Compute a single-dimension persistence diagram distance."""
    _validate_same_number_of_samples(X, Y)

    maxdim = homology_dim

    dgm_x = _ripser_diagram(
        X,
        homology_dim=homology_dim,
        maxdim=maxdim,
        dist_metric=dist_metric,
        normalize=normalize,
        q=q,
    )
    dgm_y = _ripser_diagram(
        Y,
        homology_dim=homology_dim,
        maxdim=maxdim,
        dist_metric=dist_metric,
        normalize=normalize,
        q=q,
    )

    return _diagram_distance(
        dgm_x,
        dgm_y,
        metric=metric,
        wasserstein_matching=wasserstein_matching,
    )


def persistence_diagram_distance_multi(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    homology_dims: Sequence[int] = (0, 1),
    metric: PDDistanceMetric = "bottleneck",
    weights: Sequence[float] | None = None,
    dist_metric: DistanceMetric = "euclidean",
    normalize: bool = True,
    q: float = 0.9,
    wasserstein_matching: bool = False,
) -> float:
    """Compute weighted multi-dimensional persistence diagram distance.

    Example:
        homology_dims=(0, 1), weights=(0.5, 0.5)
    """
    _validate_same_number_of_samples(X, Y)

    homology_dims = tuple(int(dim) for dim in homology_dims)
    if any(dim < 0 for dim in homology_dims):
        raise ValueError(f"homology_dims must be nonnegative, got {homology_dims}")

    if len(homology_dims) == 0:
        raise ValueError("homology_dims must not be empty")

    if weights is None:
        weights = tuple(1.0 / len(homology_dims) for _ in homology_dims)
    else:
        weights = tuple(float(w) for w in weights)

    if len(weights) != len(homology_dims):
        raise ValueError(
            f"weights and homology_dims must have same length, "
            f"got {len(weights)} and {len(homology_dims)}"
        )

    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError(f"weights must have positive sum, got {weights}")

    # Normalize weights to avoid accidental scale changes.
    weights = tuple(w / weight_sum for w in weights)

    maxdim = max(homology_dims)

    dgms_x = _ripser_diagrams(
        X,
        maxdim=maxdim,
        dist_metric=dist_metric,
        normalize=normalize,
        q=q,
    )
    dgms_y = _ripser_diagrams(
        Y,
        maxdim=maxdim,
        dist_metric=dist_metric,
        normalize=normalize,
        q=q,
    )

    total = 0.0
    for dim, weight in zip(homology_dims, weights):
        dgm_x = dgms_x[dim] if dim < len(dgms_x) else np.empty((0, 2), dtype=np.float64)
        dgm_y = dgms_y[dim] if dim < len(dgms_y) else np.empty((0, 2), dtype=np.float64)

        dist = _diagram_distance(
            dgm_x,
            dgm_y,
            metric=metric,
            wasserstein_matching=wasserstein_matching,
        )
        total += weight * dist

    return float(total)


def _pd_similarity(
    X: torch.Tensor,
    Y: torch.Tensor,
    config: MetricConfig,
    *,
    homology_dims: Sequence[int],
    metric: PDDistanceMetric,
    weights: Sequence[float] | None = None,
) -> float:
    dist_metric = _get_config_attr(config, "topology_distance_metric", "euclidean")
    normalize = _get_config_attr(config, "topology_normalize", True)
    q = _get_config_attr(config, "topology_quantile", 0.9)

    if len(tuple(homology_dims)) == 1:
        dist = persistence_diagram_distance(
            X,
            Y,
            homology_dim=tuple(homology_dims)[0],
            metric=metric,
            dist_metric=dist_metric,
            normalize=normalize,
            q=q,
        )
    else:
        dist = persistence_diagram_distance_multi(
            X,
            Y,
            homology_dims=homology_dims,
            metric=metric,
            weights=weights,
            dist_metric=dist_metric,
            normalize=normalize,
            q=q,
        )

    return _distance_to_similarity(dist)


@register_metric
class MSTOverlapSimilarity(BaseMetric):
    """Direct MST edge overlap similarity.

    Larger score means that the two representation spaces select more of the
    same sample-pair edges in their 0-dimensional connectivity skeletons.
    """

    name = "mst_overlap"
    min_score = 0.0
    max_score = 1.0
    supports_calibration = True
    supports_caching = False

    def _compute_raw(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        config: MetricConfig,
    ) -> float:
        dist_metric = _get_config_attr(config, "topology_distance_metric", "euclidean")
        normalize = _get_config_attr(config, "topology_normalize", True)
        q = _get_config_attr(config, "topology_quantile", 0.9)

        return mst_overlap(
            X,
            Y,
            dist_metric=dist_metric,
            normalize=normalize,
            q=q,
        )


@register_metric
class MSTMinOverlapSimilarity(BaseMetric):
    """RTD-Lite min-graph MST support overlap similarity."""

    name = "mst_min_overlap"
    min_score = 0.0
    max_score = 1.0
    supports_calibration = True
    supports_caching = False

    def _compute_raw(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        config: MetricConfig,
    ) -> float:
        dist_metric = _get_config_attr(config, "topology_distance_metric", "euclidean")
        normalize = _get_config_attr(config, "topology_normalize", True)
        q = _get_config_attr(config, "topology_quantile", 0.9)

        return mst_min_overlap(
            X,
            Y,
            dist_metric=dist_metric,
            normalize=normalize,
            q=q,
        )

# ---------------------------------------------------------------------
# Persistence diagram metric classes
# ---------------------------------------------------------------------

@register_metric
class PD0BottleneckSimilarity(BaseMetric):
    """H0 bottleneck similarity between persistence diagrams.

    This compares clustering / merge structure.
    This metric is row-permutation invariant, so standard row-permutation
    calibration should not be interpreted as a correspondence-aware test.
    """

    name = "pd0_bottleneck"
    min_score = 0.0
    max_score = 1.0
    supports_calibration = False
    supports_caching = False

    def _compute_raw(self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig) -> float:
        return _pd_similarity(X, Y, config, homology_dims=(0,), metric="bottleneck")


@register_metric
class PD1BottleneckSimilarity(BaseMetric):
    """H1 bottleneck similarity between persistence diagrams."""

    name = "pd1_bottleneck"
    min_score = 0.0
    max_score = 1.0
    supports_calibration = False
    supports_caching = False

    def _compute_raw(self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig) -> float:
        return _pd_similarity(X, Y, config, homology_dims=(1,), metric="bottleneck")


@register_metric
class PD01BottleneckSimilarity(BaseMetric):
    """Combined H0/H1 bottleneck similarity between persistence diagrams."""

    name = "pd01_bottleneck"
    min_score = 0.0
    max_score = 1.0
    supports_calibration = False
    supports_caching = False

    def _compute_raw(self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig) -> float:
        return _pd_similarity(
            X,
            Y,
            config,
            homology_dims=(0, 1),
            metric="bottleneck",
            weights=(0.5, 0.5),
        )


@register_metric
class PD0WassersteinSimilarity(BaseMetric):
    """H0 Wasserstein similarity between persistence diagrams."""

    name = "pd0_wasserstein"
    min_score = 0.0
    max_score = 1.0
    supports_calibration = False
    supports_caching = False

    def _compute_raw(self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig) -> float:
        return _pd_similarity(X, Y, config, homology_dims=(0,), metric="wasserstein")


@register_metric
class PD1WassersteinSimilarity(BaseMetric):
    """H1 Wasserstein similarity between persistence diagrams."""

    name = "pd1_wasserstein"
    min_score = 0.0
    max_score = 1.0
    supports_calibration = False
    supports_caching = False

    def _compute_raw(self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig) -> float:
        return _pd_similarity(X, Y, config, homology_dims=(1,), metric="wasserstein")


@register_metric
class PD01WassersteinSimilarity(BaseMetric):
    """Combined H0/H1 Wasserstein similarity between persistence diagrams."""

    name = "pd01_wasserstein"
    min_score = 0.0
    max_score = 1.0
    supports_calibration = False
    supports_caching = False

    def _compute_raw(self, X: torch.Tensor, Y: torch.Tensor, config: MetricConfig) -> float:
        return _pd_similarity(
            X,
            Y,
            config,
            homology_dims=(0, 1),
            metric="wasserstein",
            weights=(0.5, 0.5),
        )


# ---------------------------------------------------------------------
# Optional original RTD wrapper
# ---------------------------------------------------------------------

@register_metric
class RTDSimilarity(BaseMetric):
    """Original RTD wrapper converted to similarity.

    Requires external RTD dependencies. This metric can be much heavier
    than RTD-Lite and is not recommended as a default full-suite metric.

        Install the optional RTD dependencies separately before using this legacy metric.
    """

    name = "rtd"
    min_score = 0.0
    max_score = 1.0
    supports_calibration = True
    supports_caching = False

    def _compute_raw(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        config: MetricConfig,
    ) -> float:
        _validate_same_number_of_samples(X, Y)

        try:
            import rtd
        except ImportError as exc:
            raise ImportError(
                "Original RTD requires the rtd package and ripserplusplus. "
                "Install them separately before using metric='rtd'."
            ) from exc

        X_np = X.detach().float().cpu().numpy()
        Y_np = Y.detach().float().cpu().numpy()

        trials = int(_get_config_attr(config, "rtd_trials", 3))
        batch = int(_get_config_attr(config, "rtd_batch", min(256, X_np.shape[0])))

        # Try the common RTD API first.
        if hasattr(rtd, "rtd"):
            try:
                dist = rtd.rtd(
                    X_np,
                    Y_np,
                    pdist_device="cpu",
                    trials=trials,
                    batch=batch,
                )
            except TypeError:
                dist = rtd.rtd(X_np, Y_np)
        elif hasattr(rtd, "calc_embed_dist"):
            dist = rtd.calc_embed_dist(X_np, Y_np)
        else:
            raise AttributeError(
                "Could not find a supported RTD entry point. "
                "Expected rtd.rtd or rtd.calc_embed_dist."
            )

        return _distance_to_similarity(float(dist))


__all__ = [
    "mst_overlap",
    "mst_overlap_from_distances",
    "mst_min_overlap",
    "mst_min_overlap_from_distances",
    "rtd_lite_distance",
    "rtd_lite_distance_from_distances",
    "persistence_diagram_distance",
    "persistence_diagram_distance_multi",
]