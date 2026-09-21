"""Layer-wise alignment engine for the paper-relevant mKNN variants."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch

from .. import compute_nearest_neighbors
from ..metrics.aggregation import agg_max, compute_null_summary, gated_rescaled
from ..metrics.utils import knn_indicator
from ..prh.layers import _as_layers, _normalize_layers
from ..prh.geodesic import graph_geodesic_from_distances
from ..prh.riemannian import local_riemannian_knn_log_distances

def _build_knn_cache(
    feats: torch.Tensor | Sequence[torch.Tensor],
    *,
    topk: int,
    normalize: bool,
    return_indices: bool,
) -> Tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor] | None]:
    """Build normalized layer-wise kNN masks and, optionally, neighbor indices."""
    layers = _as_layers(feats)
    if normalize:
        layers = _normalize_layers(layers)

    indices: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for layer in layers:
        knn_idx = compute_nearest_neighbors(layer, topk=topk)
        if return_indices:
            indices.append(knn_idx)
        masks.append(knn_indicator(knn_idx, layer.shape[0]))

    return layers, masks, indices if return_indices else None


def build_knn_cache(
    feats: torch.Tensor | Sequence[torch.Tensor],
    *,
    topk: int,
    normalize: bool = True,
) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Build the layer-wise kNN cache used for relational-structure mKNN."""
    layers, masks, _ = _build_knn_cache(
        feats, topk=topk, normalize=normalize, return_indices=False
    )
    return layers, masks


def build_knn_cache_with_indices(
    feats: torch.Tensor | Sequence[torch.Tensor],
    *,
    topk: int,
    normalize: bool = True,
) -> Tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Build layer, neighbor-index, and kNN-mask caches for paper experiments."""
    layers, masks, indices = _build_knn_cache(
        feats, topk=topk, normalize=normalize, return_indices=True
    )
    return layers, indices or [], masks


def build_gram_cache(
    feats: torch.Tensor | Sequence[torch.Tensor],
    *,
    normalize: bool = True,
    kernel: str = "linear",
    rbf_sigma: float = 1.0,
) -> list[torch.Tensor]:
    """Build layer-wise Gram matrices.

    Kept as a lightweight compatibility helper for the shared experiment
    runner; the paper-specific run scripts do not require CKA.
    """
    layers = _as_layers(feats)
    if normalize:
        layers = _normalize_layers(layers)

    grams: list[torch.Tensor] = []
    for layer in layers:
        if kernel == "linear":
            K = layer @ layer.T
        elif kernel == "rbf":
            dists = torch.cdist(layer, layer, p=2)
            K = torch.exp(-(dists**2) / (2.0 * float(rbf_sigma) ** 2))
        else:
            raise ValueError(f"Unsupported kernel: {kernel}")
        grams.append(K)
    return grams


def _knn_overlap_from_indices(
    knn_a: torch.Tensor, knn_b: torch.Tensor, *, topk: int
) -> torch.Tensor:
    matches = (knn_a.unsqueeze(2) == knn_b.unsqueeze(1)).sum(dim=(1, 2)).float()
    return matches.mean() / float(topk)

def _normalize_distance_matrix(
    D: torch.Tensor,
    *,
    mode: str = "quantile",
    q: float = 0.9,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Normalize one pairwise distance matrix by a representation-level scale.

    ``mode="quantile"`` reproduces the current method when ``q=0.9``.
    ``mode="mean"`` uses the off-diagonal mean, while ``mode="none"`` leaves
    distances unchanged. This changes numerical values only; support is fixed.
    """
    if mode not in {"quantile", "mean", "none"}:
        raise ValueError(
            "distance_normalization must be 'quantile', 'mean', or 'none'; "
            f"got {mode!r}"
        )
    if mode == "none" or D.shape[0] <= 1:
        out = D.clone()
        out.fill_diagonal_(0.0)
        return out

    n = D.shape[0]
    offdiag = ~torch.eye(n, device=D.device, dtype=torch.bool)
    values = D[offdiag]
    if mode == "quantile":
        if not (0.0 < float(q) <= 1.0):
            raise ValueError(f"distance_normalization_q must be in (0, 1], got {q}")
        scale = torch.quantile(values, float(q))
    else:
        scale = values.mean()

    out = D / scale.clamp_min(eps)
    out.fill_diagonal_(0.0)
    return out

def _knn_distance_matrix_from_layer(
    layer: torch.Tensor,
    *,
    q: float = 0.9,
    eps: float = 1e-12,
    distance_normalization: str = "quantile",
    distance_value_geometry: str = "ambient",
    geodesic_k: int = 20,
) -> torch.Tensor:
    """Distance values used by distance-aware kNN overlap.

    The kNN *support* is built elsewhere and is unchanged by this function.
    ``distance_value_geometry="ambient"`` reproduces the original behavior:
    normalized Euclidean chord distances on L2-normalized PRH features.

    ``distance_value_geometry="geodesic"`` keeps that support fixed but
    replaces the numerical distance values with Isomap-style graph-geodesic
    distances computed on a symmetric ``geodesic_k``-NN graph.  The geodesic
    graph itself uses the same ambient Euclidean chord distance.
    """
    sim = layer @ layer.T
    D_ambient = torch.sqrt(torch.clamp(2.0 - 2.0 * sim, min=0.0))
    D_ambient.fill_diagonal_(0.0)

    if distance_value_geometry == "ambient":
        D = D_ambient
    elif distance_value_geometry == "geodesic":
        G_np = graph_geodesic_from_distances(
            D_ambient.detach().float().cpu().numpy(),
            k=geodesic_k,
            eps=eps,
        )
        D = torch.from_numpy(G_np).to(device=layer.device, dtype=layer.dtype)
    else:
        raise ValueError(
            "distance_value_geometry must be 'ambient' or 'geodesic', "
            f"got {distance_value_geometry!r}"
        )

    return _normalize_distance_matrix(
        D,
        mode=distance_normalization,
        q=q,
        eps=eps,
    )

def _knn_log_distance_values_from_layer(
    layer: torch.Tensor,
    knn_idx: torch.Tensor,
    *,
    q: float = 0.9,
    eps: float = 1e-12,
    distance_normalization: str = "quantile",
    distance_value_geometry: str = "ambient",
    geodesic_k: int = 20,
    riemannian_k: int = 30,
    riemannian_dim: int = 10,
    riemannian_reg: float = 1e-3,
) -> torch.Tensor:
    """Log distance values only on the fixed kNN support.

    Ambient and graph-geodesic modes briefly construct a dense n x n matrix and
    immediately gather the fixed support.  The local-Riemannian mode is more
    memory efficient: it estimates a low-rank local metric one sample at a time
    and returns only the n x k support values.
    """
    if distance_value_geometry == "riemannian":
        return local_riemannian_knn_log_distances(
            layer,
            knn_idx,
            metric_k=riemannian_k,
            tangent_dim=riemannian_dim,
            regularization=riemannian_reg,
            q=q,
            eps=eps,
            normalization=distance_normalization,
        )

    D = _knn_distance_matrix_from_layer(
        layer,
        q=q,
        eps=eps,
        distance_normalization=distance_normalization,
        distance_value_geometry=distance_value_geometry,
        geodesic_k=geodesic_k,
    )
    n = knn_idx.shape[0]
    rows = torch.arange(n, device=knn_idx.device).view(-1, 1)
    return torch.log(D[rows, knn_idx].clamp_min(eps))

def build_knn_log_distance_cache(
    layers: Sequence[torch.Tensor],
    knn_indices: Sequence[torch.Tensor],
    *,
    distance_value_geometry: str = "ambient",
    geodesic_k: int = 20,
    riemannian_k: int = 30,
    riemannian_dim: int = 10,
    riemannian_reg: float = 1e-3,
    q: float = 0.9,
    distance_normalization: str = "quantile",
    eps: float = 1e-12,
) -> list[torch.Tensor]:
    """Build n x k log-distance caches once per layer for distance-aware mKNN."""
    if len(layers) != len(knn_indices):
        raise ValueError("layers and knn_indices must have matching lengths")
    return [
        _knn_log_distance_values_from_layer(
            layer,
            knn_idx,
            q=q,
            eps=eps,
            distance_normalization=distance_normalization,
            distance_value_geometry=distance_value_geometry,
            geodesic_k=geodesic_k,
            riemannian_k=riemannian_k,
            riemannian_dim=riemannian_dim,
            riemannian_reg=riemannian_reg,
        )
        for layer, knn_idx in zip(layers, knn_indices)
    ]

def _knn_distance_overlap_from_log_values(
    knn_a: torch.Tensor,
    knn_b: torch.Tensor,
    log_dist_a: torch.Tensor,
    log_dist_b: torch.Tensor,
    *,
    topk: int,
    tau: float = 0.01,
) -> torch.Tensor:
    """Distance-aware kNN overlap using only matched neighbor pairs.

    This is algebraically identical to forming the full (n, k, k) distance
    weight tensor and then masking it, but log/abs/exp are evaluated only for
    common neighbors.
    """
    matches = knn_a.unsqueeze(2).eq(knn_b.unsqueeze(1))
    rows, pos_a, pos_b = matches.nonzero(as_tuple=True)

    if rows.numel() == 0:
        return log_dist_a.new_zeros(())

    log_diff = (
        log_dist_a[rows, pos_a] - log_dist_b[rows, pos_b]
    ).abs()
    weights = torch.exp(-log_diff / float(tau))
    return weights.sum() / float(knn_a.shape[0] * topk)

def build_knn_similarity_cache(
    layers: Sequence[torch.Tensor],
    knn_indices: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """Cosine-similarity values on an already fixed kNN support."""
    if len(layers) != len(knn_indices):
        raise ValueError("layers and knn_indices must have matching lengths")

    out: list[torch.Tensor] = []
    for layer, knn_idx in zip(layers, knn_indices):
        sim = layer @ layer.T
        n = knn_idx.shape[0]
        rows = torch.arange(n, device=knn_idx.device).view(-1, 1)
        out.append(sim[rows, knn_idx])
    return out

def _knn_similarity_overlap_from_values(
    knn_a: torch.Tensor,
    knn_b: torch.Tensor,
    sim_a: torch.Tensor,
    sim_b: torch.Tensor,
    *,
    topk: int,
    tau: float = 0.01,
) -> torch.Tensor:
    """Similarity-aware kNN overlap on fixed shared-neighbor support."""
    matches = knn_a.unsqueeze(2).eq(knn_b.unsqueeze(1))
    rows, pos_a, pos_b = matches.nonzero(as_tuple=True)
    if rows.numel() == 0:
        return sim_a.new_zeros(())

    sim_diff = (sim_a[rows, pos_a] - sim_b[rows, pos_b]).abs()
    weights = torch.exp(-sim_diff / float(tau))
    return weights.sum() / float(knn_a.shape[0] * topk)

def _build_gated_summary(
    null_samples: Sequence[float],
    *,
    T_obs: float,
    best_indices: Tuple[int, int],
    alpha: float,
) -> Dict[str, float | Tuple[int, int]]:
    summary = compute_null_summary(null_samples, T_obs=T_obs, alpha=alpha)
    g_score = gated_rescaled(T_obs, tau_alpha=summary["tau_alpha"], s_max=1.0)
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

def compute_alignment_gated_knn_cached(
    x_knn: Sequence[torch.Tensor],
    y_knn: Sequence[torch.Tensor],
    *,
    topk: int = 10,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Dict[str, float | Tuple[int, int]]:
    if not x_knn or not y_knn:
        raise ValueError("x_knn and y_knn must be non-empty")
    device = x_knn[0].device
    n = x_knn[0].shape[0]

    S = torch.empty((len(x_knn), len(y_knn)), device=device)
    for i, knn_A in enumerate(x_knn):
        for j, knn_B in enumerate(y_knn):
            S[i, j] = _knn_overlap_from_indices(knn_A, knn_B, topk=topk)

    agg = agg_max(S, return_indices=True)
    T_obs = float(agg.value)
    best_indices = (
        (int(agg.indices["i"]), int(agg.indices["j"])) if agg.indices else (0, 0)
    )

    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    null_samples = []
    for _ in range(num_permutations):
        perm = torch.randperm(n, generator=rng, device=device)
        perm_inv = torch.empty_like(perm)
        perm_inv[perm] = torch.arange(n, device=device)
        S_perm = torch.empty_like(S)
        for j, knn_B in enumerate(y_knn):
            knn_B_perm = perm_inv[knn_B[perm]]
            for i, knn_A in enumerate(x_knn):
                S_perm[i, j] = _knn_overlap_from_indices(knn_A, knn_B_perm, topk=topk)
        null_samples.append(float(agg_max(S_perm).value))

    return _build_gated_summary(
        null_samples, T_obs=T_obs, best_indices=best_indices, alpha=alpha
    )

def compute_alignment_gated_knn_distance_values_cached(
    x_knn: Sequence[torch.Tensor],
    y_knn: Sequence[torch.Tensor],
    x_log_dists: Sequence[torch.Tensor],
    y_log_dists: Sequence[torch.Tensor],
    *,
    topk: int = 10,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
    distance_tau: float = 0.01,
) -> Dict[str, float | Tuple[int, int]]:
    """Calibrate distance-aware kNN from already cached n x k log values.

    This separates expensive geometry estimation from permutation scoring.  In
    particular, locally estimated Riemannian distances can be built once per
    model/layer and reused across every cross-model pair and every permutation.
    """
    if not x_knn or not y_knn:
        raise ValueError("x_knn and y_knn must be non-empty")
    if len(x_knn) != len(x_log_dists) or len(y_knn) != len(y_log_dists):
        raise ValueError("kNN and log-distance cache lists must have matching lengths")

    device = x_knn[0].device
    n = x_knn[0].shape[0]

    S = torch.empty((len(x_knn), len(y_knn)), device=device)
    for i, (knn_A, log_dist_A) in enumerate(zip(x_knn, x_log_dists)):
        for j, (knn_B, log_dist_B) in enumerate(zip(y_knn, y_log_dists)):
            S[i, j] = _knn_distance_overlap_from_log_values(
                knn_A,
                knn_B,
                log_dist_A,
                log_dist_B,
                topk=topk,
                tau=distance_tau,
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
        }

    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    null_samples = []
    rows = torch.arange(n, device=device)
    for _ in range(num_permutations):
        perm = torch.randperm(n, generator=rng, device=device)

        # Shared across every layer pair in this permutation.
        perm_inv = torch.empty_like(perm)
        perm_inv[perm] = rows

        # Each Y support/value cache is relabeled once per permutation rather
        # than once per X layer.
        y_perm_cache = []
        for knn_B, log_dist_B in zip(y_knn, y_log_dists):
            knn_B_orig = knn_B[perm]
            knn_B_perm = perm_inv[knn_B_orig]
            log_dist_B_perm = log_dist_B[perm]
            y_perm_cache.append((knn_B_perm, log_dist_B_perm))

        S_perm = torch.empty_like(S)
        for i, (knn_A, log_dist_A) in enumerate(zip(x_knn, x_log_dists)):
            for j, (knn_B_perm, log_dist_B_perm) in enumerate(y_perm_cache):
                S_perm[i, j] = _knn_distance_overlap_from_log_values(
                    knn_A,
                    knn_B_perm,
                    log_dist_A,
                    log_dist_B_perm,
                    topk=topk,
                    tau=distance_tau,
                )

        null_samples.append(float(agg_max(S_perm).value))

    return _build_gated_summary(
        null_samples, T_obs=T_obs, best_indices=best_indices, alpha=alpha
    )

def compute_alignment_gated_knn_similarity_values_cached(
    x_knn: Sequence[torch.Tensor],
    y_knn: Sequence[torch.Tensor],
    x_sims: Sequence[torch.Tensor],
    y_sims: Sequence[torch.Tensor],
    *,
    topk: int = 10,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
    similarity_tau: float = 0.01,
) -> Dict[str, float | Tuple[int, int]]:
    """Calibrate similarity-aware kNN from cached n x k cosine values."""
    if not x_knn or not y_knn:
        raise ValueError("x_knn and y_knn must be non-empty")
    if len(x_knn) != len(x_sims) or len(y_knn) != len(y_sims):
        raise ValueError("kNN and similarity cache lists must have matching lengths")

    device = x_knn[0].device
    n = x_knn[0].shape[0]

    S = torch.empty((len(x_knn), len(y_knn)), device=device)
    for i, (knn_A, sim_A) in enumerate(zip(x_knn, x_sims)):
        for j, (knn_B, sim_B) in enumerate(zip(y_knn, y_sims)):
            S[i, j] = _knn_similarity_overlap_from_values(
                knn_A, knn_B, sim_A, sim_B, topk=topk, tau=similarity_tau
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
        }

    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)

    null_samples = []
    rows = torch.arange(n, device=device)
    for _ in range(num_permutations):
        perm = torch.randperm(n, generator=rng, device=device)
        perm_inv = torch.empty_like(perm)
        perm_inv[perm] = rows

        y_perm_cache = []
        for knn_B, sim_B in zip(y_knn, y_sims):
            knn_B_orig = knn_B[perm]
            knn_B_perm = perm_inv[knn_B_orig]
            sim_B_perm = sim_B[perm]
            y_perm_cache.append((knn_B_perm, sim_B_perm))

        S_perm = torch.empty_like(S)
        for i, (knn_A, sim_A) in enumerate(zip(x_knn, x_sims)):
            for j, (knn_B_perm, sim_B_perm) in enumerate(y_perm_cache):
                S_perm[i, j] = _knn_similarity_overlap_from_values(
                    knn_A,
                    knn_B_perm,
                    sim_A,
                    sim_B_perm,
                    topk=topk,
                    tau=similarity_tau,
                )
        null_samples.append(float(agg_max(S_perm).value))

    return _build_gated_summary(
        null_samples, T_obs=T_obs, best_indices=best_indices, alpha=alpha
    )

