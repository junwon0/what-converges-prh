"""Core helpers for the anonymous PRH reproduction package."""

from __future__ import annotations

import torch


def compute_nearest_neighbors(feats: torch.Tensor, topk: int = 1) -> torch.Tensor:
    """Compute cosine-similarity k-nearest-neighbor indices."""
    if feats.ndim != 2:
        raise ValueError(f"Expected feats to be 2D, got {feats.ndim}")
    from .metrics.knn import _compute_knn_indices

    return _compute_knn_indices(feats, topk)


__all__ = ["compute_nearest_neighbors"]
