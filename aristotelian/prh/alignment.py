"""PRH alignment helpers."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch

from ..experiments import layerwise_engine as lwe
from ..metrics.aggregation import (
    SimpleMetric,
    agg_max,
    compute_null_summary,
    gated_rescaled,
    permutation_null_aggregated,
)
from ..metrics.api import prh_metric_spec
from .layers import _as_layers, _normalize_layers

from .topology_alignment import (
    TOPOLOGY_METRICS,
    compute_alignment_gated_topology_cached,
)

def _is_topology_metric_name(metric: str) -> bool:
    return (
        metric in TOPOLOGY_METRICS
        or metric.startswith("mst_overlap_dist_tau")
        or metric.startswith("mst_overlap_sim_tau")
    )


def compute_alignment_gated_cached(
    x_layers: Sequence[torch.Tensor],
    y_layers: Sequence[torch.Tensor],
    x_masks: Sequence[torch.Tensor],
    y_masks: Sequence[torch.Tensor],
    *,
    topk: int = 10,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Dict[str, float | Tuple[int, int]]:
    return lwe.compute_alignment_gated_cached(
        x_layers,
        y_layers,
        x_masks,
        y_masks,
        topk=topk,
        num_permutations=num_permutations,
        alpha=alpha,
        seed=seed,
    )


def compute_alignment_gated_knn_cached(
    x_knn: Sequence[torch.Tensor],
    y_knn: Sequence[torch.Tensor],
    *,
    topk: int = 10,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Dict[str, float | Tuple[int, int]]:
    return lwe.compute_alignment_gated_knn_cached(
        x_knn,
        y_knn,
        topk=topk,
        num_permutations=num_permutations,
        alpha=alpha,
        seed=seed,
    )


def build_knn_log_distance_cache(
    layers,
    knn_indices,
    *,
    distance_value_geometry="ambient",
    geodesic_k=20,
    riemannian_k=30,
    riemannian_dim=10,
    riemannian_reg=1e-3,
    q=0.9,
    distance_normalization="quantile",
):
    return lwe.build_knn_log_distance_cache(
        layers,
        knn_indices,
        distance_value_geometry=distance_value_geometry,
        geodesic_k=geodesic_k,
        riemannian_k=riemannian_k,
        riemannian_dim=riemannian_dim,
        riemannian_reg=riemannian_reg,
        q=q,
        distance_normalization=distance_normalization,
    )

def build_knn_similarity_cache(layers, knn_indices):
    return lwe.build_knn_similarity_cache(layers, knn_indices)


def compute_alignment_gated_knn_similarity_values_cached(
    x_knn,
    y_knn,
    x_sims,
    y_sims,
    *,
    topk=10,
    num_permutations=200,
    alpha=0.05,
    seed=None,
    similarity_tau=0.01,
):
    return lwe.compute_alignment_gated_knn_similarity_values_cached(
        x_knn,
        y_knn,
        x_sims,
        y_sims,
        topk=topk,
        num_permutations=num_permutations,
        alpha=alpha,
        seed=seed,
        similarity_tau=similarity_tau,
    )


def compute_alignment_gated_knn_distance_values_cached(
    x_knn,
    y_knn,
    x_log_dists,
    y_log_dists,
    *,
    topk=10,
    num_permutations=200,
    alpha=0.05,
    seed=None,
    distance_tau=0.01,
):
    return lwe.compute_alignment_gated_knn_distance_values_cached(
        x_knn,
        y_knn,
        x_log_dists,
        y_log_dists,
        topk=topk,
        num_permutations=num_permutations,
        alpha=alpha,
        seed=seed,
        distance_tau=distance_tau,
    )


def compute_alignment_gated_knn_distance_cached(
    x_layers,
    y_layers,
    x_knn,
    y_knn,
    *,
    topk=10,
    num_permutations=200,
    alpha=0.05,
    seed=None,
    distance_tau=0.01,
    distance_value_geometry="ambient",
    distance_normalization="quantile",
    distance_normalization_q=0.9,
    geodesic_k=20,
    riemannian_k=30,
    riemannian_dim=10,
    riemannian_reg=1e-3,
):
    return lwe.compute_alignment_gated_knn_distance_cached(
        x_layers,
        y_layers,
        x_knn,
        y_knn,
        topk=topk,
        num_permutations=num_permutations,
        alpha=alpha,
        seed=seed,
        distance_tau=distance_tau,
        distance_value_geometry=distance_value_geometry,
        distance_normalization=distance_normalization,
        distance_normalization_q=distance_normalization_q,
        geodesic_k=geodesic_k,
        riemannian_k=riemannian_k,
        riemannian_dim=riemannian_dim,
        riemannian_reg=riemannian_reg,
    )

def compute_alignment_gated_knn_rank_cached(
    x_knn: Sequence[torch.Tensor],
    y_knn: Sequence[torch.Tensor],
    *,
    topk: int = 10,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Dict[str, float | Tuple[int, int]]:
    return lwe.compute_alignment_gated_knn_rank_cached(
        x_knn,
        y_knn,
        topk=topk,
        num_permutations=num_permutations,
        alpha=alpha,
        seed=seed,
    )


def compute_alignment_gated_cknna_cached(
    x_grams: Sequence[torch.Tensor],
    y_grams: Sequence[torch.Tensor],
    *,
    topk: int = 10,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
    unbiased: bool = True,
) -> Dict[str, float | Tuple[int, int]]:
    return lwe.compute_alignment_gated_cknna_cached(
        x_grams,
        y_grams,
        topk=topk,
        num_permutations=num_permutations,
        alpha=alpha,
        seed=seed,
        unbiased=unbiased,
    )


def compute_alignment_gated_cycle_knn_cached(
    x_knn: Sequence[torch.Tensor],
    y_knn: Sequence[torch.Tensor],
    *,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Dict[str, float | Tuple[int, int]]:
    return lwe.compute_alignment_gated_cycle_knn_cached(
        x_knn,
        y_knn,
        num_permutations=num_permutations,
        alpha=alpha,
        seed=seed,
    )


def compute_alignment_gated_svcca_cached(
    x_layers: Sequence[torch.Tensor],
    y_layers: Sequence[torch.Tensor],
    *,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Dict[str, float | Tuple[int, int]]:
    return lwe.compute_alignment_gated_svcca_cached(
        x_layers,
        y_layers,
        num_permutations=num_permutations,
        alpha=alpha,
        seed=seed,
    )


def compute_alignment_gated_pwcca_cached(
    x_layers: Sequence[torch.Tensor],
    y_layers: Sequence[torch.Tensor],
    *,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Dict[str, float | Tuple[int, int]]:
    return lwe.compute_alignment_gated_pwcca_cached(
        x_layers,
        y_layers,
        num_permutations=num_permutations,
        alpha=alpha,
        seed=seed,
    )


def compute_alignment_gated_procrustes_cached(
    x_layers: Sequence[torch.Tensor],
    y_layers: Sequence[torch.Tensor],
    *,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Dict[str, float | Tuple[int, int]]:
    return lwe.compute_alignment_gated_procrustes_cached(
        x_layers,
        y_layers,
        num_permutations=num_permutations,
        alpha=alpha,
        seed=seed,
    )


def compute_alignment_gated_cka_cached(
    x_grams: Sequence[torch.Tensor],
    y_grams: Sequence[torch.Tensor],
    *,
    num_permutations: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
    unbiased: bool = False,
) -> Dict[str, float | Tuple[int, int]]:
    return lwe.compute_alignment_gated_cka_cached(
        x_grams,
        y_grams,
        num_permutations=num_permutations,
        alpha=alpha,
        seed=seed,
        unbiased=unbiased,
    )


def compute_alignment_gated(
    x_feats: torch.Tensor | Sequence[torch.Tensor],
    y_feats: torch.Tensor | Sequence[torch.Tensor],
    *,
    metric: str = "mutual_knn",
    topk: int = 10,
    normalize: bool = True,
    num_permutations: int = 200,
    topology_dist_metric: str = "euclidean",
    distance_value_geometry: str = "ambient",
    geodesic_k: int = 20,
    riemannian_k: int = 30,
    riemannian_dim: int = 10,
    riemannian_reg: float = 1e-3,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Dict[str, float | Tuple[int, int]]:
    standard_metrics = {
        "cycle_knn",
        "knn",
        "mutual_knn",
        "cka",
        "cka_lin",
        "cka_rbf",
        "unbiased_cka",
        "cknna",
        "svcca",
        "pwcca",
        "procrustes",
        "cca",
        "rv_coefficient",
    }

    if metric not in standard_metrics and not _is_topology_metric_name(metric):
        raise ValueError(f"Unsupported metric {metric}")
    # if metric not in {
    #     "cycle_knn",
    #     "knn",
    #     "mutual_knn",
    #     "cka",
    #     "cka_lin",
    #     "cka_rbf",
    #     "unbiased_cka",
    #     "cknna",
    #     "svcca",
    #     "pwcca",
    #     "procrustes",
    #     "cca",
    #     "rv_coefficient",

    #     # topology
    #     "mst_overlap",
    #     "mst_overlap_dist",
    #     "mst_min_overlap",
    #     "rtd",
    #     "rtd_lite",
    #     "rtd_lite_tai",
    #     "pd0_bottleneck",
    #     "pd1_bottleneck",
    #     "pd01_bottleneck",
    #     "pd0_wasserstein",
    #     "pd1_wasserstein",
    #     "pd01_wasserstein",
    # }:
    #     raise ValueError(f"Unsupported metric {metric}")
    x_layers = _as_layers(x_feats)
    y_layers = _as_layers(y_feats)
    if normalize:
        x_layers = _normalize_layers(x_layers)
        y_layers = _normalize_layers(y_layers)
    if _is_topology_metric_name(metric):
        return compute_alignment_gated_topology_cached(
            x_layers,
            y_layers,
            metric=metric,
            num_permutations=num_permutations,
            alpha=alpha,
            seed=seed,
            normalize_distances=True,
            dist_metric=topology_dist_metric,
            q=0.9,
            distance_value_geometry=distance_value_geometry,
            geodesic_k=geodesic_k,
            riemannian_k=riemannian_k,
            riemannian_dim=riemannian_dim,
            riemannian_reg=riemannian_reg,
        )

    metric_compute, max_value = prh_metric_spec(metric, topk=topk)

    metric_fn = SimpleMetric(
        name=metric,
        max_value=max_value,
        compute=metric_compute,
    )

    S = torch.empty((len(x_layers), len(y_layers)), device=x_layers[0].device)
    for i, x in enumerate(x_layers):
        for j, y in enumerate(y_layers):
            S[i, j] = metric_fn.compute(x, y)

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

    null_samples = permutation_null_aggregated(
        x_layers,
        y_layers,
        metric_fn,
        agg_max,
        num_permutations=num_permutations,
        seed=seed,
    )
    summary = compute_null_summary(null_samples, T_obs=T_obs, alpha=alpha)
    g_score = gated_rescaled(
        T_obs, tau_alpha=summary["tau_alpha"], s_max=metric_fn.max_value
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
