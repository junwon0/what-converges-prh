"""PRH path helpers."""

from __future__ import annotations

import os


def _format_float_for_path(x: float) -> str:
    return f"{x:g}".replace(".", "p").replace("-", "m")


def prh_feature_filename(
    output_dir: str,
    dataset: str,
    subset: str,
    model_name: str,
    *,
    pool: str | None,
    prompt: bool | None,
    caption_idx: int | None,
    num_samples: int | None = None,
) -> str:
    save_name = model_name.replace("/", "_")
    if pool:
        save_name += f"_pool-{pool}"
    if prompt is not None:
        save_name += f"_prompt-{prompt}"
    if caption_idx is not None:
        save_name += f"_cid-{caption_idx}"
    if num_samples is not None:
        save_name += f"_n{num_samples}"

    return os.path.join(output_dir, dataset, subset, f"{save_name}.pt")


def prh_alignment_filename(
    output_dir: str,
    dataset: str,
    modelset: str,
    modality_x: str,
    pool_x: str | None,
    prompt_x: bool,
    modality_y: str,
    pool_y: str | None,
    prompt_y: bool,
    metric: str,
    topk: int,
    *,
    max_samples: int | None = None,
    num_permutations: int | None = None,
    topology_dist_metric: str | None = None,
    distance_value_geometry: str | None = None,
    distance_normalization: str | None = None,
    distance_normalization_q: float | None = None,
    geodesic_k: int | None = None,
    riemannian_k: int | None = None,
    riemannian_dim: int | None = None,
    riemannian_reg: float | None = None,
) -> str:
    metric_name = f"{metric}_k{topk}" if "knn" in metric else metric

    topology_metrics = {
        "mst_overlap",
        "mst_overlap_dist",
        "mst_overlap_sim",
        "mst_min_overlap",
        "rtd",
        "rtd_lite",
        "rtd_lite_tai",
        "pd0_bottleneck",
        "pd1_bottleneck",
        "pd01_bottleneck",
        "pd0_wasserstein",
        "pd1_wasserstein",
        "pd01_wasserstein",
    }

    is_topology_metric = (
        metric in topology_metrics
        or metric.startswith("mst_overlap_dist_tau")
        or metric.startswith("mst_overlap_sim_tau")
    )

    if is_topology_metric and topology_dist_metric is not None:
        metric_name += f"_dist-{topology_dist_metric}"

    is_distance_aware = (
        metric == "mutual_knn_dist"
        or metric.startswith("mutual_knn_dist_tau")
        or metric == "mst_overlap_dist"
        or metric.startswith("mst_overlap_dist_tau")
    )
    if is_distance_aware and distance_value_geometry == "geodesic":
        metric_name += "_value-geodesic"
        if geodesic_k is not None:
            metric_name += f"_gk{geodesic_k}"
    elif is_distance_aware and distance_value_geometry == "riemannian":
        metric_name += "_value-riemannian"
        if riemannian_k is not None:
            metric_name += f"_rk{riemannian_k}"
        if riemannian_dim is not None:
            metric_name += f"_rd{riemannian_dim}"
        if riemannian_reg is not None:
            metric_name += f"_rr{_format_float_for_path(riemannian_reg)}"

    if is_distance_aware and distance_normalization is not None:
        is_default_norm = (
            distance_normalization == "quantile"
            and (
                distance_normalization_q is None
                or abs(float(distance_normalization_q) - 0.9) < 1e-12
            )
        )
        if not is_default_norm:
            if distance_normalization == "quantile":
                metric_name += f"_norm-q{_format_float_for_path(float(distance_normalization_q))}"
            else:
                metric_name += f"_norm-{distance_normalization}"

    if max_samples is not None:
        metric_name += f"_n{max_samples}"

    if num_permutations is not None:
        metric_name += f"_perm{num_permutations}"

    return os.path.join(
        output_dir,
        dataset,
        modelset,
        f"{modality_x}_pool-{pool_x}_prompt-{prompt_x}_{modality_y}_pool-{pool_y}_prompt-{prompt_y}",
        f"{metric_name}.npy",
    )
