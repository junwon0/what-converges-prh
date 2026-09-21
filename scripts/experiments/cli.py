#!/usr/bin/env python
"""Run the experiments used in the anonymous PRH submission."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch

from .sections import run_prh_alignment, run_v2t_alignment

SECTION_FUNCS = {
    "prh_alignment": run_prh_alignment,
    "v2t_alignment": run_v2t_alignment,
}


def _parse_sections(items: list[str]) -> list[str]:
    parts: list[str] = []
    for item in items:
        parts.extend(x.strip() for x in item.split(",") if x.strip())
    if not parts or "all" in parts:
        return list(SECTION_FUNCS)
    unknown = sorted(set(parts) - set(SECTION_FUNCS))
    if unknown:
        raise ValueError(f"Unknown sections: {', '.join(unknown)}")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper reproduction experiments.")
    parser.add_argument("--sections", nargs="*", default=["all"])
    parser.add_argument("--assets-dir", default="assets")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-prh-features", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)

    parser.add_argument(
        "--prh-metrics",
        default="mutual_knn,mst_overlap",
        help="Comma-separated paper metrics.",
    )
    parser.add_argument("--prh-q-outlier", type=float, default=0.95)
    parser.add_argument("--prh-max-samples", type=int, default=None)
    parser.add_argument("--prh-num-permutations", type=int, default=500)

    parser.add_argument("--topology-dist-metric", default="euclidean")
    parser.add_argument("--topology-max-samples", type=int, default=1024)
    parser.add_argument("--topology-num-permutations", type=int, default=500)

    parser.add_argument(
        "--distance-value-geometry",
        choices=["ambient", "riemannian"],
        default="ambient",
    )
    parser.add_argument(
        "--distance-normalization",
        choices=["quantile", "mean", "none"],
        default="quantile",
    )
    parser.add_argument("--distance-normalization-q", type=float, default=0.9)
    parser.add_argument("--riemannian-k", type=int, default=30)
    parser.add_argument("--riemannian-dim", type=int, default=10)
    parser.add_argument("--riemannian-reg", type=float, default=1e-3)

    parser.add_argument("--v2t-video-modelset", default="videoprh")
    parser.add_argument("--v2t-text-modelset", default="videoprh")
    parser.add_argument("--v2t-num-frames", type=int, default=16)
    parser.add_argument("--v2t-num-captions", type=int, default=1)

    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    metrics = tuple(x.strip() for x in args.prh_metrics.split(",") if x.strip())
    assets_dir = Path(args.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    common = {
        "assets_dir": assets_dir,
        "device": device,
        "force": args.force,
        "force_features": args.force_prh_features,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "topology_dist_metric": args.topology_dist_metric,
        "distance_value_geometry": args.distance_value_geometry,
        "distance_normalization": args.distance_normalization,
        "distance_normalization_q": args.distance_normalization_q,
        "riemannian_k": args.riemannian_k,
        "riemannian_dim": args.riemannian_dim,
        "riemannian_reg": args.riemannian_reg,
        "topology_max_samples": args.topology_max_samples,
        "topology_num_permutations": args.topology_num_permutations,
        "prh_max_samples": args.prh_max_samples,
        "prh_num_permutations": args.prh_num_permutations,
        "prh_q_outlier": args.prh_q_outlier,
        "v2t_video_modelset": args.v2t_video_modelset,
        "v2t_text_modelset": args.v2t_text_modelset,
        "num_frames": args.v2t_num_frames,
        "num_captions": args.v2t_num_captions,
    }

    for section in _parse_sections(args.sections):
        fn = SECTION_FUNCS[section]
        sig = inspect.signature(fn)
        kwargs = {k: v for k, v in common.items() if k in sig.parameters}
        if "prh_metrics" in sig.parameters:
            kwargs["prh_metrics"] = metrics
        if "v2t_metrics" in sig.parameters:
            kwargs["v2t_metrics"] = metrics
        if "v2t_q_outlier" in sig.parameters:
            kwargs["v2t_q_outlier"] = args.prh_q_outlier
        fn(**kwargs)


if __name__ == "__main__":
    main()
