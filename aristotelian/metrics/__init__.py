"""Metrics required by the PRH reproduction code."""

from __future__ import annotations

# Import the paper-relevant metric implementations so they register themselves.
from . import knn as _knn  # noqa: F401
from . import topology as _topology  # noqa: F401
from .base import BaseMetric, MetricConfig, MetricResult
from .registry import MetricRegistry, register_metric

__all__ = [
    "BaseMetric",
    "MetricConfig",
    "MetricResult",
    "MetricRegistry",
    "register_metric",
]
