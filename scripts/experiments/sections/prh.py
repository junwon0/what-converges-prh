"""PRH (Platonic Representation Hypothesis) alignment experiments.

This module includes:
1. Cross-modal alignment (language <-> vision) following PRH main text
2. Video-to-text alignment extension
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from aristotelian.prh.prh_experiment import run_prh_experiment

from ..infra.io import save_array, should_skip

PRH_METRICS = (
    "mutual_knn",
    "mst_overlap",
)

TOPOLOGY_METRICS = (
    "mst_overlap",
    "mst_overlap_dist",
    "mst_overlap_sim",
)

PRH_NUM_PERMUTATIONS = 500
TOPOLOGY_MAX_SAMPLES = 1024
TOPOLOGY_NUM_PERMUTATIONS = 500
# k values for kNN-based metrics that sweep over k (mutual_knn, cknna)
# PRH_K_VALUES = (10, 20, 50, 100)
PRH_K_VALUES = (10,)
# sigma values for RBF kernel CKA (small=local, large=global)
# Default is 1.0; sweep explores local-to-global spectrum
PRH_SIGMA_VALUES = (0.1, 0.5, 1.0, 2.0, 5.0)
PRH_Q_OUTLIER = 0.95


def _format_float_for_path(x: float) -> str:
    return f"{x:g}".replace(".", "p").replace("-", "m")
# Metrics that use k parameter
# KNN_BASED_METRICS = ("mutual_knn", "cycle_knn", "cknna")
# KNN_BASED_METRICS = ("mutual_knn", "mutual_knn_dist", "mutual_knn_rank", "cycle_knn", "cknna")
def _is_knn_based_metric(metric: str) -> bool:
    return (
        metric in {
            "mutual_knn", "mutual_knn_dist", "mutual_knn_sim"
        }
        or metric.startswith("mutual_knn_dist_tau")
        or metric.startswith("mutual_knn_sim_tau")
    )
def _is_topology_metric(metric: str) -> bool:
    return (
        metric in TOPOLOGY_METRICS
        or metric.startswith("mst_overlap_dist_tau")
        or metric.startswith("mst_overlap_sim_tau")
    )
def _is_distance_aware_metric(metric: str) -> bool:
    return (
        metric == "mutual_knn_dist"
        or metric.startswith("mutual_knn_dist_tau")
        or metric == "mst_overlap_dist"
        or metric.startswith("mst_overlap_dist_tau")
    )

# Metrics that sweep over multiple k values (others use default k=10)
# K_SWEEP_METRICS = ("mutual_knn", "cknna")
# K_SWEEP_METRICS = ("mutual_knn", "mutual_knn_dist", "mutual_knn_rank", "cknna")
def _uses_k_sweep(metric: str) -> bool:
    return (
        metric in {"mutual_knn", "mutual_knn_dist", "mutual_knn_sim"}
        or metric.startswith("mutual_knn_dist_tau")
        or metric.startswith("mutual_knn_sim_tau")
    )
# Metrics that use sigma parameter (RBF kernel bandwidth)
RBF_SIGMA_METRICS = ("cka_rbf",)


def run_prh_alignment(
    assets_dir: Path,
    *,
    device: str,
    force: bool,
    force_features: bool = False,
    seed: int | None,
    num_workers: int = 1,
    prh_metrics: tuple = PRH_METRICS,
    prh_k_values: tuple = PRH_K_VALUES,
    prh_sigma_values: tuple = PRH_SIGMA_VALUES,
    prh_q_outlier: float = PRH_Q_OUTLIER,
    prh_max_samples: int | None = None,
    prh_num_permutations: int = PRH_NUM_PERMUTATIONS,
    topology_max_samples: int | None = TOPOLOGY_MAX_SAMPLES,
    topology_num_permutations: int = TOPOLOGY_NUM_PERMUTATIONS,
    topology_dist_metric: str = "euclidean",
    distance_value_geometry: str = "ambient",
    distance_normalization: str = "quantile",
    distance_normalization_q: float = 0.9,
    geodesic_k: int = 20,
    riemannian_k: int = 30,
    riemannian_dim: int = 10,
    riemannian_reg: float = 1e-3,
    **kwargs,
) -> None:
    """Run PRH alignment experiment across multiple metrics.

    For mutual_knn and cknna, experiments are run for each k value in
    prh_k_values. For RBF kernel metrics (cka_rbf), experiments are run for
    each sigma value in prh_sigma_values. Other metrics (including cycle_knn)
    use the default k=10.
    """
    for metric in prh_metrics:
        # Determine k values to use for this metric
        # Only mutual_knn and cknna sweep over k values; others use default k=10
        # if metric in K_SWEEP_METRICS:
        if _uses_k_sweep(metric):
            k_values = prh_k_values
        else:
            k_values = (10,)  # Default k for non-sweep metrics

        # Determine sigma values to use for this metric
        if metric in RBF_SIGMA_METRICS:
            sigma_values = prh_sigma_values
        else:
            sigma_values = (1.0,)  # Placeholder, sigma not used for non-RBF metrics

        for k in k_values:
            for sigma in sigma_values:

                is_topology_metric = _is_topology_metric(metric)

                effective_dist_metric = (
                    topology_dist_metric if is_topology_metric else "euclidean"
                )
                
                effective_max_samples = (
                    topology_max_samples if is_topology_metric else prh_max_samples
                )
                n_tag = "all" if effective_max_samples is None else str(effective_max_samples)

                effective_num_permutations = (
                    topology_num_permutations if is_topology_metric else prh_num_permutations
                )

                # Build output filename
                if metric == "mutual_knn" and k == 10:
                    # Default case for backward compatibility
                    output_name = f"prh_alignment_{metric}_k{k}_n{n_tag}_perm{effective_num_permutations}.npy"
                # elif metric in KNN_BASED_METRICS:
                elif _is_knn_based_metric(metric):
                    output_name = f"prh_alignment_{metric}_k{k}_n{n_tag}_perm{effective_num_permutations}.npy"
                elif metric in RBF_SIGMA_METRICS and sigma == 1.0:
                    # Default sigma=1.0 case for backward compatibility
                    output_name = f"prh_alignment_{metric}_n{n_tag}_perm{effective_num_permutations}.npy"
                elif metric in RBF_SIGMA_METRICS:
                    output_name = f"prh_alignment_{metric}_sigma{sigma}_n{n_tag}_perm{effective_num_permutations}.npy"
                else:
                    output_name = f"prh_alignment_{metric}_n{n_tag}_perm{effective_num_permutations}.npy"

                if is_topology_metric:
                    output_name = (
                        f"prh_alignment_{metric}"
                        f"_dist-{effective_dist_metric}"
                        f"_n{n_tag}"
                        f"_perm{effective_num_permutations}.npy"
                    )

                if _is_distance_aware_metric(metric) and distance_value_geometry == "geodesic":
                    output_name = output_name.replace(
                        ".npy", f"_value-geodesic_gk{geodesic_k}.npy"
                    )
                elif _is_distance_aware_metric(metric) and distance_value_geometry == "riemannian":
                    reg_tag = _format_float_for_path(riemannian_reg)
                    output_name = output_name.replace(
                        ".npy",
                        f"_value-riemannian_rk{riemannian_k}_rd{riemannian_dim}_rr{reg_tag}.npy",
                    )

                if _is_distance_aware_metric(metric):
                    is_default_norm = (
                        distance_normalization == "quantile"
                        and abs(float(distance_normalization_q) - 0.9) < 1e-12
                    )
                    if not is_default_norm:
                        if distance_normalization == "quantile":
                            norm_tag = f"q{_format_float_for_path(distance_normalization_q)}"
                        else:
                            norm_tag = distance_normalization
                        output_name = output_name.replace(".npy", f"_norm-{norm_tag}.npy")

                output = assets_dir / output_name
                if should_skip([output], force):
                    logger.info(
                        f"Skipping prh_alignment ({metric}, k={k}, sigma={sigma}, output exists: {output})"
                    )
                    continue

                out = run_prh_experiment(
                    dataset="minhuh/prh",
                    subset="wit_1024",
                    split="train",
                    modelset="val",
                    modality_x="language",
                    pool_x="avg",
                    prompt_x=False,
                    modality_y="vision",
                    pool_y="cls",
                    prompt_y=False,
                    caption_idx=0,
                    max_samples=effective_max_samples,
                    batch_size=4,
                    device=device,
                    output_dir="./results",
                    k=k,
                    metric=metric,
                    rbf_sigma=sigma,
                    num_permutations=effective_num_permutations,
                    alpha=0.05,
                    q_outlier=prh_q_outlier,
                    force_features=force_features,
                    num_workers=num_workers,
                    seed=seed,
                    topology_dist_metric=effective_dist_metric,
                    distance_value_geometry=distance_value_geometry,
                    distance_normalization=distance_normalization,
                    distance_normalization_q=distance_normalization_q,
                    geodesic_k=geodesic_k,
                    riemannian_k=riemannian_k,
                    riemannian_dim=riemannian_dim,
                    riemannian_reg=riemannian_reg,
                )
                save_array(output, out)


def run_v2t_alignment(
    assets_dir: Path,
    *,
    device: str,
    force: bool,
    force_features: bool = False,
    seed: int | None,
    num_workers: int = 1,
    v2t_metrics: tuple = PRH_METRICS,
    v2t_k_values: tuple = PRH_K_VALUES,
    v2t_sigma_values: tuple = PRH_SIGMA_VALUES,
    v2t_video_modelset: str = "videoprh",
    v2t_text_modelset: str = "videoprh",
    v2t_q_outlier: float = PRH_Q_OUTLIER,
    num_frames: int = 16,
    num_captions: int = 1,
    streaming_limit: int | None = 5000,
    topology_dist_metric: str = "euclidean",
    distance_value_geometry: str = "ambient",
    geodesic_k: int = 20,
    riemannian_k: int = 30,
    riemannian_dim: int = 10,
    riemannian_reg: float = 1e-3,
) -> None:
    """Run the video-to-text alignment experiment used in this study.

    This experiment compares video models to text models using the PE-Video (PVD)
    dataset, demonstrating PRH in the video-text domain.

    For mutual_knn and cknna, experiments are run for each k value in
    v2t_k_values. For RBF kernel metrics (cka_rbf), experiments are run for
    each sigma value in v2t_sigma_values. Other metrics (including cycle_knn)
    use the default k=10.
    """
    del seed  # unused but kept for consistent interface
    from aristotelian.prh.v2t_experiment import run_v2t_experiment

    for metric in v2t_metrics:
        # Determine k values to use for this metric
        # Only mutual_knn and cknna sweep over k values; others use default k=10
        # if metric in K_SWEEP_METRICS:
        if _uses_k_sweep(metric):
            k_values = v2t_k_values
        else:
            k_values = (10,)  # Default k for non-sweep metrics

        # Determine sigma values to use for this metric
        if metric in RBF_SIGMA_METRICS:
            sigma_values = v2t_sigma_values
        else:
            sigma_values = (1.0,)  # Placeholder, sigma not used for non-RBF metrics

        for k in k_values:
            for sigma in sigma_values:
                # Build output filename
                if metric == "mutual_knn" and k == 10:
                    output_name = (
                        f"v2t_alignment_{v2t_video_modelset}_{v2t_text_modelset}.npy"
                    )
                # elif metric in KNN_BASED_METRICS:
                elif _is_knn_based_metric(metric):
                    output_name = f"v2t_alignment_{v2t_video_modelset}_{v2t_text_modelset}_{metric}_k{k}.npy"
                elif metric in RBF_SIGMA_METRICS and sigma == 1.0:
                    output_name = f"v2t_alignment_{v2t_video_modelset}_{v2t_text_modelset}_{metric}.npy"
                elif metric in RBF_SIGMA_METRICS:
                    output_name = f"v2t_alignment_{v2t_video_modelset}_{v2t_text_modelset}_{metric}_sigma{sigma}.npy"
                else:
                    output_name = f"v2t_alignment_{v2t_video_modelset}_{v2t_text_modelset}_{metric}.npy"

                if _is_distance_aware_metric(metric) and distance_value_geometry == "geodesic":
                    output_name = output_name.replace(
                        ".npy", f"_value-geodesic_gk{geodesic_k}.npy"
                    )
                elif _is_distance_aware_metric(metric) and distance_value_geometry == "riemannian":
                    reg_tag = _format_float_for_path(riemannian_reg)
                    output_name = output_name.replace(
                        ".npy",
                        f"_value-riemannian_rk{riemannian_k}_rd{riemannian_dim}_rr{reg_tag}.npy",
                    )

                output = assets_dir / output_name
                if should_skip([output], force):
                    logger.info(
                        f"Skipping v2t_alignment ({metric}, k={k}, sigma={sigma}, output exists: {output})"
                    )
                    continue

                out = run_v2t_experiment(
                    dataset="pvd",
                    split="test",
                    video_modelset=v2t_video_modelset,
                    text_modelset=v2t_text_modelset,
                    pool_video="cls",
                    pool_text="avg",
                    num_frames=num_frames,
                    num_captions=num_captions,
                    max_samples=1024,  # Paper setting
                    streaming_limit=streaming_limit,  # Legacy name: cap on valid examples examined
                    batch_size=4,
                    device=device,
                    output_dir="./results",
                    k=k,
                    metric=metric,
                    rbf_sigma=sigma,
                    num_permutations=500,
                    alpha=0.05,
                    q_outlier=v2t_q_outlier,
                    force_features=force_features,
                    num_workers=num_workers,
                    topology_dist_metric=topology_dist_metric,
                    distance_value_geometry=distance_value_geometry,
                    geodesic_k=geodesic_k,
                    riemannian_k=riemannian_k,
                    riemannian_dim=riemannian_dim,
                    riemannian_reg=riemannian_reg,
                )
                save_array(output, out)
