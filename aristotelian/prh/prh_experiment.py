"""
PRH-style feature extraction and alignment with significance gating.

This mirrors the Platonic-Rep workflow while applying aggregation-aware null calibration.
"""

from __future__ import annotations

import gc
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from loguru import logger
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from ..experiments.multiple import bh_fdr
from ..utils.logging import ExperimentState, log_timing
from .alignment import (
    compute_alignment_gated,
    compute_alignment_gated_cka_cached,
    compute_alignment_gated_cknna_cached,
    compute_alignment_gated_cycle_knn_cached,
    compute_alignment_gated_knn_cached,
    compute_alignment_gated_knn_distance_cached,
    compute_alignment_gated_knn_distance_values_cached,
    compute_alignment_gated_knn_similarity_values_cached,
    build_knn_log_distance_cache,
    build_knn_similarity_cache,
    compute_alignment_gated_knn_rank_cached,
    compute_alignment_gated_procrustes_cached,
    compute_alignment_gated_pwcca_cached,
    compute_alignment_gated_svcca_cached,
)
from .cache import _cached_feats_match, build_gram_cache, build_knn_cache_with_indices
from .layers import _as_layers, _normalize_layers
from .paths import prh_alignment_filename, prh_feature_filename
from .preprocess import prepare_features
from .prh_data import iter_prh_samples
from .prh_models import get_models, load_text_model, load_vision_model
from .prh_pipeline import collect_text_activations, collect_vision_activations
from .topology_alignment import (
    compute_alignment_gated_mst_distance_from_caches,
    compute_alignment_gated_mst_similarity_from_caches,
    prepare_mst_overlap_cache,
    prepare_mst_similarity_cache,
)


def _stack_layers(layers: Sequence[np.ndarray]) -> torch.Tensor:
    tensors = [torch.tensor(x) for x in layers]
    return torch.stack(tensors, dim=1)

def _format_float_for_path(x: float) -> str:
    return f"{x:g}".replace(".", "p").replace("-", "m")


def _parse_value_sensitive_metric(metric: str) -> tuple[str, float | None]:
    """Parse tau-encoded distance/similarity-aware metrics."""
    prefixes = {
        "mutual_knn_dist_tau": "mutual_knn_dist",
        "mst_overlap_dist_tau": "mst_overlap_dist",
        "mutual_knn_sim_tau": "mutual_knn_sim",
        "mst_overlap_sim_tau": "mst_overlap_sim",
    }
    for prefix, base in prefixes.items():
        if metric.startswith(prefix):
            tau_str = metric[len(prefix):].replace("p", ".").replace("m", "-")
            return base, float(tau_str)

    defaults = {
        "mutual_knn_dist": 0.01,
        "mst_overlap_dist": 0.01,
        "mutual_knn_sim": 0.01,
        "mst_overlap_sim": 0.01,
    }
    if metric in defaults:
        return metric, defaults[metric]
    return metric, None

def _parse_knn_distance_metric(metric: str) -> tuple[str, float | None]:
    """Backward-compatible parser for existing video-text code."""
    return _parse_value_sensitive_metric(metric)

def _load_prh_samples(
    dataset_name: str,
    subset: str,
    *,
    split: str,
    max_samples: int | None,
    data_dir: str | None,
    caption_idx: int,
    prompt: bool,
) -> Tuple[List[str], List]:
    ds = load_dataset(dataset_name, revision=subset, split=split, data_dir=data_dir)
    texts = []
    images = []
    for i, row in enumerate(iter_prh_samples(ds)):
        if max_samples is not None and i >= max_samples:
            break
        if row["text"] is not None:
            text_entry = row["text"]
            if isinstance(text_entry, list):
                text_entry = text_entry[caption_idx]
            text_value = str(text_entry)
            if prompt:
                text_value = f"caption: {text_value}"
            texts.append(text_value)
        if row["image"] is not None:
            images.append(row["image"])
    return texts, images


def _extract_text_features(
    texts: Iterable[str],
    *,
    model_name: str,
    device: str,
    batch_size: int,
    pool: str,
) -> Tuple[torch.Tensor, int]:
    tokenizer, model = load_text_model(model_name, device=device)
    layers = collect_text_activations(
        texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        pool="mean" if pool == "avg" else ("cls" if pool == "cls" else "last"),
    )
    num_params = int(sum(p.numel() for p in model.parameters()))
    return _stack_layers(layers), num_params


def _extract_vision_features(
    images: Iterable,
    *,
    model_name: str,
    device: str,
    batch_size: int,
    pool: str,
) -> Tuple[torch.Tensor, int]:
    model = load_vision_model(model_name, device=device)
    num_params = int(sum(p.numel() for p in model.parameters()))
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )
    if "vit" in model_name:
        return_nodes = [f"blocks.{i}.add_1" for i in range(len(model.blocks))]
    else:
        raise NotImplementedError(f"unknown model {model_name}")
    feat_model = create_feature_extractor(model, return_nodes=return_nodes)
    layers = collect_vision_activations(
        images,
        model=feat_model,
        device=device,
        batch_size=batch_size,
        pool=pool,
        transform=transform,
    )
    return _stack_layers(layers), num_params


def _prh_compute_pair_generic(
    config: Tuple[
        int,
        int,  # i, j indices
        List,
        List,  # x_layers, y_layers
        str,
        int,
        int,
        float,  # metric, topk, num_permutations, alpha
        str,   # topology_dist_metric
        str,   # distance_value_geometry
        int,   # geodesic_k
        int,   # riemannian_k
        int,   # riemannian_dim
        float, # riemannian_reg
    ],
) -> Tuple[int, int, Dict[str, object]]:
    """Compute alignment for a single pair with generic metric (for ProcessPoolExecutor)."""
    (
        i,
        j,
        x_layers,
        y_layers,
        metric,
        topk,
        num_permutations,
        alpha,
        topology_dist_metric,
        distance_value_geometry,
        geodesic_k,
        riemannian_k,
        riemannian_dim,
        riemannian_reg,
    ) = config
    res = compute_alignment_gated(
        x_layers,
        y_layers,
        metric=metric,
        topk=topk,
        normalize=False,
        num_permutations=num_permutations,
        alpha=alpha,
        topology_dist_metric=topology_dist_metric,
        distance_value_geometry=distance_value_geometry,
        geodesic_k=geodesic_k,
        riemannian_k=riemannian_k,
        riemannian_dim=riemannian_dim,
        riemannian_reg=riemannian_reg,
    )
    return i, j, res


def _prh_compute_pair_cka_cached(
    config: Tuple[
        int,
        int,  # i, j indices
        List,
        List,  # x_grams, y_grams
        int,
        float,  # num_permutations, alpha
        bool,  # unbiased
    ],
) -> Tuple[int, int, Dict[str, object]]:
    """Compute CKA alignment for a single pair with cached Gram matrices."""
    i, j, x_grams, y_grams, num_permutations, alpha, unbiased = config
    res = compute_alignment_gated_cka_cached(
        x_grams,
        y_grams,
        num_permutations=num_permutations,
        alpha=alpha,
        unbiased=unbiased,
    )
    return i, j, res


def _prh_compute_pair_cknna_cached(
    config: Tuple[
        int,
        int,  # i, j indices
        List,
        List,  # x_grams, y_grams
        int,
        int,
        float,  # topk, num_permutations, alpha
    ],
) -> Tuple[int, int, Dict[str, object]]:
    """Compute CKNNA alignment for a single pair with cached Gram matrices."""
    i, j, x_grams, y_grams, topk, num_permutations, alpha = config
    res = compute_alignment_gated_cknna_cached(
        x_grams,
        y_grams,
        topk=topk,
        num_permutations=num_permutations,
        alpha=alpha,
    )
    return i, j, res


def _prh_compute_pair_cycle_knn_cached(
    config: Tuple[
        int,
        int,  # i, j indices
        List,
        List,  # x_knn, y_knn
        int,
        float,  # num_permutations, alpha
    ],
) -> Tuple[int, int, Dict[str, object]]:
    """Compute cycle-KNN alignment for a single pair with cached KNN indices."""
    i, j, x_knn, y_knn, num_permutations, alpha = config
    res = compute_alignment_gated_cycle_knn_cached(
        x_knn,
        y_knn,
        num_permutations=num_permutations,
        alpha=alpha,
    )
    return i, j, res


def _prh_compute_pair_knn_cached(
    config: Tuple[
        int,
        int,  # i, j indices
        List,
        List,  # x_knn, y_knn
        int,
        int,
        float,  # topk, num_permutations, alpha
    ],
) -> Tuple[int, int, Dict[str, object]]:
    """Compute KNN overlap alignment for a single pair with cached indices."""
    i, j, x_knn, y_knn, topk, num_permutations, alpha = config
    res = compute_alignment_gated_knn_cached(
        x_knn,
        y_knn,
        topk=topk,
        num_permutations=num_permutations,
        alpha=alpha,
    )
    return i, j, res


# def _prh_compute_pair_knn_distance_cached(
#     config: Tuple[
#         int,
#         int,  # i, j indices
#         List,
#         List,  # x_layers, y_layers
#         List,
#         List,  # x_knn, y_knn
#         int,
#         int,
#         float,  # topk, num_permutations, alpha
#     ],
# ) -> Tuple[int, int, Dict[str, object]]:
#     """Compute distance-aware kNN alignment for a single pair."""
#     i, j, x_layers, y_layers, x_knn, y_knn, topk, num_permutations, alpha = config
#     res = compute_alignment_gated_knn_distance_cached(
#         x_layers,
#         y_layers,
#         x_knn,
#         y_knn,
#         topk=topk,
#         num_permutations=num_permutations,
#         alpha=alpha,
#     )
#     return i, j, res

def _prh_compute_pair_knn_distance_cached(config):
    (
        i,
        j,
        x_layers,
        y_layers,
        x_knn,
        y_knn,
        topk,
        num_permutations,
        alpha,
        distance_tau,
        distance_value_geometry,
        geodesic_k,
        riemannian_k,
        riemannian_dim,
        riemannian_reg,
    ) = config

    res = compute_alignment_gated_knn_distance_cached(
        x_layers,
        y_layers,
        x_knn,
        y_knn,
        topk=topk,
        num_permutations=num_permutations,
        alpha=alpha,
        distance_tau=distance_tau,
        distance_value_geometry=distance_value_geometry,
        geodesic_k=geodesic_k,
        riemannian_k=riemannian_k,
        riemannian_dim=riemannian_dim,
        riemannian_reg=riemannian_reg,
    )
    return i, j, res


def _prh_compute_pair_knn_distance_values_cached(config):
    (
        i, j, x_knn, y_knn, x_log_dists, y_log_dists, topk,
        num_permutations, alpha, distance_tau, seed
    ) = config
    res = compute_alignment_gated_knn_distance_values_cached(
        x_knn, y_knn, x_log_dists, y_log_dists,
        topk=topk, num_permutations=num_permutations, alpha=alpha,
        seed=seed, distance_tau=distance_tau,
    )
    return i, j, res



def _prh_compute_pair_knn_similarity_values_cached(config):
    (
        i,
        j,
        x_knn,
        y_knn,
        x_sims,
        y_sims,
        topk,
        num_permutations,
        alpha,
        similarity_tau,
        seed,
    ) = config
    res = compute_alignment_gated_knn_similarity_values_cached(
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
    return i, j, res


def _prh_compute_pair_mst_distance_cached(config):
    (i, j, x_cache, y_cache, num_permutations, alpha, distance_tau, seed) = config
    res = compute_alignment_gated_mst_distance_from_caches(
        x_cache,
        y_cache,
        distance_tau=distance_tau,
        num_permutations=num_permutations,
        alpha=alpha,
        seed=seed,
    )
    return i, j, res


def _prh_compute_pair_mst_similarity_cached(config):
    (i, j, x_cache, y_cache, num_permutations, alpha, similarity_tau, seed) = config
    res = compute_alignment_gated_mst_similarity_from_caches(
        x_cache,
        y_cache,
        similarity_tau=similarity_tau,
        num_permutations=num_permutations,
        alpha=alpha,
        seed=seed,
    )
    return i, j, res

def _prh_compute_pair_knn_rank_cached(
    config: Tuple[
        int,
        int,  # i, j indices
        List,
        List,  # x_knn, y_knn
        int,
        int,
        float,  # topk, num_permutations, alpha
    ],
) -> Tuple[int, int, Dict[str, object]]:
    """Compute rank-aware kNN alignment for a single pair."""
    i, j, x_knn, y_knn, topk, num_permutations, alpha = config
    res = compute_alignment_gated_knn_rank_cached(
        x_knn,
        y_knn,
        topk=topk,
        num_permutations=num_permutations,
        alpha=alpha,
    )
    return i, j, res


def _prh_compute_pair_svcca_cached(
    config: Tuple[
        int,
        int,  # i, j indices
        List,
        List,  # x_layers, y_layers
        int,
        float,  # num_permutations, alpha
    ],
) -> Tuple[int, int, Dict[str, object]]:
    """Compute SVCCA alignment for a single pair with cached preprocessing."""
    i, j, x_layers, y_layers, num_permutations, alpha = config
    res = compute_alignment_gated_svcca_cached(
        x_layers,
        y_layers,
        num_permutations=num_permutations,
        alpha=alpha,
    )
    return i, j, res


def _prh_compute_pair_pwcca_cached(
    config: Tuple[
        int,
        int,  # i, j indices
        List,
        List,  # x_layers, y_layers
        int,
        float,  # num_permutations, alpha
    ],
) -> Tuple[int, int, Dict[str, object]]:
    """Compute PWCCA alignment for a single pair with cached preprocessing."""
    i, j, x_layers, y_layers, num_permutations, alpha = config
    res = compute_alignment_gated_pwcca_cached(
        x_layers,
        y_layers,
        num_permutations=num_permutations,
        alpha=alpha,
    )
    return i, j, res


def _prh_compute_pair_procrustes_cached(
    config: Tuple[
        int,
        int,  # i, j indices
        List,
        List,  # x_layers, y_layers
        int,
        float,  # num_permutations, alpha
    ],
) -> Tuple[int, int, Dict[str, object]]:
    """Compute Procrustes alignment for a single pair with cached preprocessing."""
    i, j, x_layers, y_layers, num_permutations, alpha = config
    res = compute_alignment_gated_procrustes_cached(
        x_layers,
        y_layers,
        num_permutations=num_permutations,
        alpha=alpha,
    )
    return i, j, res


def run_prh_experiment(
    *,
    dataset: str = "minhuh/prh",
    subset: str = "wit_1024",
    split: str = "train",
    modelset: str = "val",
    modality_x: str = "language",
    pool_x: str = "avg",
    prompt_x: bool = False,
    modality_y: str = "vision",
    pool_y: str = "cls",
    prompt_y: bool = False,
    caption_idx: int = 0,
    max_samples: int | None = None,
    batch_size: int = 4,
    device: str = "cpu",
    fallback_device: str = "cpu",
    output_dir: str = "./results",
    k: int = 10,
    metric: str = "mutual_knn",
    rbf_sigma: float = 1.0,
    num_permutations: int = 200,
    alpha: float = 0.05,
    q_outlier: float = 1.0,
    force_features: bool = False,
    num_workers: int = 1,
    seed: int | None = None,
    topology_dist_metric: str = "euclidean",
    distance_value_geometry: str = "ambient",
    distance_normalization: str = "quantile",
    distance_normalization_q: float = 0.9,
    geodesic_k: int = 20,
    riemannian_k: int = 30,
    riemannian_dim: int = 10,
    riemannian_reg: float = 1e-3,
) -> Dict[str, np.ndarray]:
    if device != "cpu" and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    metric_for_output = metric
    metric_for_compute, value_tau = _parse_value_sensitive_metric(metric)

    logger.info("=" * 80)
    logger.info("PRH Alignment Experiment")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset}/{subset} ({split} split)")
    logger.info(f"Modelset: {modelset}")
    logger.info(
        f"Modalities: {modality_x} (pool={pool_x}) → {modality_y} (pool={pool_y})"
    )
    if metric_for_compute in {
        "mutual_knn",
        "knn",
        "cycle_knn",
        "mutual_knn_dist",
        "mutual_knn_sim",
        "mutual_knn_rank",
    }:
        msg = f"Metric: {metric_for_output} (base={metric_for_compute}, k={k})"
        if value_tau is not None:
            msg += f", value_tau={value_tau:g}"
        logger.info(msg)
    elif metric_for_compute == "cka_rbf":
        logger.info(f"Metric: {metric_for_output} (sigma={rbf_sigma})")
    else:
        logger.info(f"Metric: {metric_for_output}")
    logger.info(f"Max samples: {max_samples}, permutations: {num_permutations}")
    if metric_for_compute in {"mutual_knn_dist", "mst_overlap_dist"}:
        geometry_detail = ""
        if distance_value_geometry == "geodesic":
            geometry_detail = f" (geodesic_k={geodesic_k})"
        elif distance_value_geometry == "riemannian":
            geometry_detail = (f" (riemannian_k={riemannian_k}, "
                               f"dim={riemannian_dim}, reg={riemannian_reg:g})")
        logger.info(f"Distance-value geometry: {distance_value_geometry}{geometry_detail}")
        norm_detail = (
            f"q={distance_normalization_q:g}"
            if distance_normalization == "quantile"
            else distance_normalization
        )
        logger.info(f"Distance normalization: {distance_normalization} ({norm_detail})")
    if metric_for_compute in {
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
    }:
        logger.info(f"Topology distance metric: {topology_dist_metric}")
    logger.info(f"Outlier quantile: {q_outlier}")
    logger.info("=" * 80)

    with log_timing("Loading data"):
        texts, images = _load_prh_samples(
            dataset,
            subset,
            split=split,
            max_samples=max_samples,
            data_dir=None,
            caption_idx=caption_idx,
            prompt=prompt_x,
        )
        logger.info(f"Loaded {len(texts)} texts and {len(images)} images")

    llm_models, lvm_models = get_models(modelset, modality="all")
    models_x = llm_models if modality_x == "language" else lvm_models
    models_y = llm_models if modality_y == "language" else lvm_models
    logger.info(f"Models X ({modality_x}): {len(models_x)}")
    logger.info(f"Models Y ({modality_y}): {len(models_y)}")

    logger.info(f"State: {ExperimentState.GENERATING_DATA} - extracting features")
    feats_x = []
    feats_y = []
    for model_name in tqdm(models_x, desc=f"Extracting {modality_x} features"):
        out_path = prh_feature_filename(
            os.path.join(output_dir, "features"),
            dataset,
            subset,
            model_name,
            pool=pool_x,
            prompt=prompt_x,
            caption_idx=caption_idx,
            num_samples=len(texts),
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if os.path.exists(out_path) and not force_features:
            payload = torch.load(out_path, map_location=device)
            if _cached_feats_match(payload, len(texts)):
                feats_x.append(payload["feats"])
                continue
            logger.warning(
                f"Cached features have {payload.get('feats').shape[0] if payload.get('feats') is not None else 'unknown'} "
                f"samples but expected {len(texts)}; recomputing {model_name}"
            )
        try:
            feats, num_params = _extract_text_features(
                texts,
                model_name=model_name,
                device=device,
                batch_size=batch_size,
                pool=pool_x,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            logger.warning(
                f"OOM while extracting {model_name} on {device}; retrying on {fallback_device}"
            )
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            gc.collect()
            feats, num_params = _extract_text_features(
                texts,
                model_name=model_name,
                device=fallback_device,
                batch_size=batch_size,
                pool=pool_x,
            )
        # torch.save({"feats": feats, "num_params": num_params}, out_path)
        torch.save(
            {
                "feats": feats,
                "num_params": num_params,
                "num_samples": len(texts),
                "dataset": dataset,
                "subset": subset,
                "split": split,
                "model_name": model_name,
                "pool": pool_x,
                "prompt": prompt_x,
                "caption_idx": caption_idx,
            },
            out_path,
        )
        feats_x.append(feats)
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            gc.collect()

    for model_name in tqdm(models_y, desc=f"Extracting {modality_y} features"):
        out_path = prh_feature_filename(
            os.path.join(output_dir, "features"),
            dataset,
            subset,
            model_name,
            pool=pool_y,
            prompt=prompt_y,
            caption_idx=None,
            num_samples=len(images),
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if os.path.exists(out_path) and not force_features:
            payload = torch.load(out_path, map_location=device)
            if _cached_feats_match(payload, len(images)):
                feats_y.append(payload["feats"])
                continue
            logger.warning(
                f"Cached features have {payload.get('feats').shape[0] if payload.get('feats') is not None else 'unknown'} "
                f"samples but expected {len(images)}; recomputing {model_name}"
            )
        feats, num_params = _extract_vision_features(
            images,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            pool=pool_y,
        )
        # torch.save({"feats": feats, "num_params": num_params}, out_path)
        torch.save(
            {
                "feats": feats,
                "num_params": num_params,
                "num_samples": len(images),
                "dataset": dataset,
                "subset": subset,
                "split": split,
                "model_name": model_name,
                "pool": pool_y,
                "prompt": prompt_y,
                "caption_idx": None,
            },
            out_path,
        )
        feats_y.append(feats)
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            gc.collect()

    logger.info(f"Extracted {len(feats_x)} feature sets for X modality")
    logger.info(f"Extracted {len(feats_y)} feature sets for Y modality")

    logger.info(f"State: {ExperimentState.COMPUTING_SIMILARITIES}")
    scores = np.zeros((len(feats_x), len(feats_y)))
    indices = np.zeros((len(feats_x), len(feats_y), 2), dtype=int)
    gated = np.zeros((len(feats_x), len(feats_y)))
    pvals = np.zeros((len(feats_x), len(feats_y)))
    taus = np.zeros((len(feats_x), len(feats_y)))

    x_prepped_list = [
        prepare_features(x, q=q_outlier, exact=False, device=device) for x in feats_x
    ]
    y_prepped_list = [
        prepare_features(y, q=q_outlier, exact=False, device=device) for y in feats_y
    ]

    x_layers_list = []
    y_layers_list = []
    x_masks_list = []
    y_masks_list = []
    x_knn_list = []
    y_knn_list = []
    x_grams_list = []
    y_grams_list = []
    x_log_dists_list = []
    y_log_dists_list = []
    x_sims_list = []
    y_sims_list = []
    x_mst_cache_list = []
    y_mst_cache_list = []

    # Metrics that can use cached Gram matrices
    cka_cached_metrics = {"cka", "cka_lin", "unbiased_cka", "cka_rbf"}
    # knn_cached_metrics = {"mutual_knn", "knn", "cycle_knn"}
    knn_cached_metrics = {
        "mutual_knn", "knn", "cycle_knn", "mutual_knn_dist",
        "mutual_knn_sim", "mutual_knn_rank"
    }
    extra_cached_metrics = {"svcca", "pwcca", "procrustes"}
    gram_cached_metrics = cka_cached_metrics | {"cknna"}

    if metric_for_compute in knn_cached_metrics:
        for x_prepped in x_prepped_list:
            layers, knn_idx, masks = build_knn_cache_with_indices(
                x_prepped, topk=k, normalize=True
            )
            x_layers_list.append(layers)
            x_knn_list.append(knn_idx)
            x_masks_list.append(masks)

        for y_prepped in y_prepped_list:
            layers, knn_idx, masks = build_knn_cache_with_indices(
                y_prepped, topk=k, normalize=True
            )
            y_layers_list.append(layers)
            y_knn_list.append(knn_idx)
            y_masks_list.append(masks)

        if metric_for_compute == "mutual_knn_dist":
            logger.info(f"Building distance-value caches once per model/layer (geometry={distance_value_geometry})")
            x_log_dists_list = [
                build_knn_log_distance_cache(
                    layers, knn_idx,
                    distance_value_geometry=distance_value_geometry,
                    q=distance_normalization_q,
                    distance_normalization=distance_normalization,
                    geodesic_k=geodesic_k,
                    riemannian_k=riemannian_k,
                    riemannian_dim=riemannian_dim,
                    riemannian_reg=riemannian_reg,
                )
                for layers, knn_idx in zip(x_layers_list, x_knn_list)
            ]
            y_log_dists_list = [
                build_knn_log_distance_cache(
                    layers, knn_idx,
                    distance_value_geometry=distance_value_geometry,
                    q=distance_normalization_q,
                    distance_normalization=distance_normalization,
                    geodesic_k=geodesic_k,
                    riemannian_k=riemannian_k,
                    riemannian_dim=riemannian_dim,
                    riemannian_reg=riemannian_reg,
                )
                for layers, knn_idx in zip(y_layers_list, y_knn_list)
            ]
        elif metric_for_compute == "mutual_knn_sim":
            logger.info("Building cosine-similarity value caches on fixed kNN support")
            x_sims_list = [
                build_knn_similarity_cache(layers, knn_idx)
                for layers, knn_idx in zip(x_layers_list, x_knn_list)
            ]
            y_sims_list = [
                build_knn_similarity_cache(layers, knn_idx)
                for layers, knn_idx in zip(y_layers_list, y_knn_list)
            ]
    elif metric in gram_cached_metrics:
        kernel = "rbf" if metric == "cka_rbf" else "linear"
        if metric == "cka_rbf":
            logger.info(
                f"Using cached Gram matrices for {metric} (optimized, kernel={kernel}, sigma={rbf_sigma})"
            )
        else:
            logger.info(
                f"Using cached Gram matrices for {metric} (optimized, kernel={kernel})"
            )
        for x_prepped in x_prepped_list:
            x_grams_list.append(
                build_gram_cache(
                    x_prepped, normalize=True, kernel=kernel, rbf_sigma=rbf_sigma
                )
            )
        for y_prepped in y_prepped_list:
            y_grams_list.append(
                build_gram_cache(
                    y_prepped, normalize=True, kernel=kernel, rbf_sigma=rbf_sigma
                )
            )
    else:
        for x_prepped in x_prepped_list:
            x_layers_list.append(_normalize_layers(_as_layers(x_prepped)))
        for y_prepped in y_prepped_list:
            y_layers_list.append(_normalize_layers(_as_layers(y_prepped)))

    if metric_for_compute == "mst_overlap_dist":
        if value_tau is None:
            value_tau = 0.01
        logger.info(
            "Building lean MST distance-value caches once per model/layer "
            f"(geometry={distance_value_geometry})"
        )
        x_mst_cache_list = [
            prepare_mst_overlap_cache(
                layers,
                dist_metric=topology_dist_metric,
                normalize=True,
                q=distance_normalization_q,
                distance_normalization=distance_normalization,
                distance_value_geometry=distance_value_geometry,
                geodesic_k=geodesic_k,
                riemannian_k=riemannian_k,
                riemannian_dim=riemannian_dim,
                riemannian_reg=riemannian_reg,
            )
            for layers in x_layers_list
        ]
        y_mst_cache_list = [
            prepare_mst_overlap_cache(
                layers,
                dist_metric=topology_dist_metric,
                normalize=True,
                q=distance_normalization_q,
                distance_normalization=distance_normalization,
                distance_value_geometry=distance_value_geometry,
                geodesic_k=geodesic_k,
                riemannian_k=riemannian_k,
                riemannian_dim=riemannian_dim,
                riemannian_reg=riemannian_reg,
            )
            for layers in y_layers_list
        ]

    if metric_for_compute == "mst_overlap_sim":
        if value_tau is None:
            value_tau = 0.01
        logger.info("Building lean MST cosine-similarity caches on fixed spanning support")
        x_mst_cache_list = [
            prepare_mst_similarity_cache(
                layers,
                dist_metric=topology_dist_metric,
                normalize_support=True,
                q=0.9,
            )
            for layers in x_layers_list
        ]
        y_mst_cache_list = [
            prepare_mst_similarity_cache(
                layers,
                dist_metric=topology_dist_metric,
                normalize_support=True,
                q=0.9,
            )
            for layers in y_layers_list
        ]

    # Build configs for parallel execution
    # if metric in {"mutual_knn", "knn"}:
    if metric_for_compute in {"mutual_knn", "knn"}:
        configs = [
            (
                i,
                j,
                x_knn_list[i],
                y_knn_list[j],
                k,
                num_permutations,
                alpha,
            )
            for i in range(len(x_knn_list))
            for j in range(len(y_knn_list))
        ]
        compute_fn = _prh_compute_pair_knn_cached

    elif metric_for_compute == "mutual_knn_dist":
        if value_tau is None:
            value_tau = 0.01

        configs = [
            (
                i, j,
                x_knn_list[i], y_knn_list[j],
                x_log_dists_list[i], y_log_dists_list[j],
                k, num_permutations, alpha, value_tau, seed,
            )
            for i in range(len(x_knn_list))
            for j in range(len(y_knn_list))
        ]
        compute_fn = _prh_compute_pair_knn_distance_values_cached

    elif metric_for_compute == "mutual_knn_sim":
        if value_tau is None:
            value_tau = 0.01
        configs = [
            (
                i, j,
                x_knn_list[i], y_knn_list[j],
                x_sims_list[i], y_sims_list[j],
                k, num_permutations, alpha, value_tau, seed,
            )
            for i in range(len(x_knn_list))
            for j in range(len(y_knn_list))
        ]
        compute_fn = _prh_compute_pair_knn_similarity_values_cached

    elif metric_for_compute == "mst_overlap_dist":
        configs = [
            (
                i, j, x_mst_cache_list[i], y_mst_cache_list[j],
                num_permutations, alpha, value_tau, seed,
            )
            for i in range(len(x_mst_cache_list))
            for j in range(len(y_mst_cache_list))
        ]
        compute_fn = _prh_compute_pair_mst_distance_cached

    elif metric_for_compute == "mst_overlap_sim":
        configs = [
            (
                i, j, x_mst_cache_list[i], y_mst_cache_list[j],
                num_permutations, alpha, value_tau, seed,
            )
            for i in range(len(x_mst_cache_list))
            for j in range(len(y_mst_cache_list))
        ]
        compute_fn = _prh_compute_pair_mst_similarity_cached

    elif metric_for_compute == "mutual_knn_rank":
        configs = [
            (
                i,
                j,
                x_knn_list[i],
                y_knn_list[j],
                k,
                num_permutations,
                alpha,
            )
            for i in range(len(x_knn_list))
            for j in range(len(y_knn_list))
        ]
        compute_fn = _prh_compute_pair_knn_rank_cached

    elif metric_for_compute == "cycle_knn":
        configs = [
            (
                i,
                j,
                x_knn_list[i],
                y_knn_list[j],
                num_permutations,
                alpha,
            )
            for i in range(len(x_knn_list))
            for j in range(len(y_knn_list))
        ]
        compute_fn = _prh_compute_pair_cycle_knn_cached

    elif metric_for_compute in cka_cached_metrics:
        use_unbiased = metric_for_compute == "unbiased_cka"
        configs = [
            (
                i,
                j,
                x_grams_list[i],
                y_grams_list[j],
                num_permutations,
                alpha,
                use_unbiased,
            )
            for i in range(len(x_grams_list))
            for j in range(len(y_grams_list))
        ]
        compute_fn = _prh_compute_pair_cka_cached

    elif metric_for_compute == "cknna":
        configs = [
            (
                i,
                j,
                x_grams_list[i],
                y_grams_list[j],
                k,
                num_permutations,
                alpha,
            )
            for i in range(len(x_grams_list))
            for j in range(len(y_grams_list))
        ]
        compute_fn = _prh_compute_pair_cknna_cached

    elif metric_for_compute in extra_cached_metrics:
        configs = [
            (
                i,
                j,
                x_layers_list[i],
                y_layers_list[j],
                num_permutations,
                alpha,
            )
            for i in range(len(x_layers_list))
            for j in range(len(y_layers_list))
        ]
        if metric_for_compute == "svcca":
            compute_fn = _prh_compute_pair_svcca_cached
        elif metric_for_compute == "pwcca":
            compute_fn = _prh_compute_pair_pwcca_cached
        else:
            compute_fn = _prh_compute_pair_procrustes_cached

    else:
        configs = [
            (
                i,
                j,
                x_layers_list[i],
                y_layers_list[j],
                metric_for_output if (
                    metric_for_output.startswith("mst_overlap_dist_tau")
                    or metric_for_output.startswith("mst_overlap_sim_tau")
                ) else metric_for_compute,
                k,
                num_permutations,
                alpha,
                topology_dist_metric,
                distance_value_geometry,
                geodesic_k,
                riemannian_k,
                riemannian_dim,
                riemannian_reg,
            )
            for i in range(len(x_layers_list))
            for j in range(len(y_layers_list))
        ]
        compute_fn = _prh_compute_pair_generic

    total_pairs = len(configs)
    if num_workers > 1 and device == "cpu":
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results_iter = executor.map(compute_fn, configs)
            for i, j, res in tqdm(
                results_iter, total=total_pairs, desc=f"Computing {metric}"
            ):
                scores[i, j] = res["raw_score"]
                indices[i, j] = res["best_indices"]
                gated[i, j] = res["g_score"]
                pvals[i, j] = res["p_value"]
                taus[i, j] = res["tau_alpha"]
    else:
        for config in tqdm(configs, desc=f"Computing {metric}"):
            i, j, res = compute_fn(config)
            scores[i, j] = res["raw_score"]
            indices[i, j] = res["best_indices"]
            gated[i, j] = res["g_score"]
            pvals[i, j] = res["p_value"]
            taus[i, j] = res["tau_alpha"]

    flat_p = pvals.reshape(-1)
    fdr_threshold, fdr_mask_flat = bh_fdr(flat_p, alpha=alpha)
    fdr_mask = fdr_mask_flat.reshape(pvals.shape)

    logger.info(
        f"Computed {len(feats_x)} × {len(feats_y)} = {len(feats_x) * len(feats_y)} alignments"
    )

    logger.info(f"State: {ExperimentState.SAVING_RESULTS}")
    save_path = prh_alignment_filename(
        os.path.join(output_dir, "alignment"),
        dataset,
        modelset,
        modality_x,
        pool_x,
        prompt_x,
        modality_y,
        pool_y,
        prompt_y,
        metric_for_output,
        k,
        max_samples=max_samples,
        num_permutations=num_permutations,
        topology_dist_metric=topology_dist_metric,
        distance_value_geometry=distance_value_geometry,
        distance_normalization=distance_normalization,
        distance_normalization_q=distance_normalization_q,
        geodesic_k=geodesic_k,
        riemannian_k=riemannian_k,
        riemannian_dim=riemannian_dim,
        riemannian_reg=riemannian_reg,
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    payload = {
        "scores": scores,
        "indices": indices,
        "gated": gated,
        "pvalues": pvals,
        "taus": taus,
        "metric": metric_for_output,
        "metric_base": metric_for_compute,
        "metric_name": metric_for_output,
        "topology_dist_metric": topology_dist_metric,
        "distance_value_geometry": distance_value_geometry,
        "distance_normalization": distance_normalization,
        "distance_normalization_q": float(distance_normalization_q),
        "geodesic_k": int(geodesic_k),
        "riemannian_k": int(riemannian_k),
        "riemannian_dim": int(riemannian_dim),
        "riemannian_reg": float(riemannian_reg),
        "max_samples": max_samples,
        "num_permutations": num_permutations,
        "fdr_threshold": fdr_threshold,
        "fdr_mask": fdr_mask,
    }
    if value_tau is not None:
        payload["value_tau"] = float(value_tau)
        if metric_for_compute in {"mutual_knn_dist", "mst_overlap_dist"}:
            payload["distance_tau"] = float(value_tau)
        elif metric_for_compute in {"mutual_knn_sim", "mst_overlap_sim"}:
            payload["similarity_tau"] = float(value_tau)
    np.save(save_path, payload)
    logger.success(f"Results saved to {save_path}")
    return payload
