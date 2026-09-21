"""Graph-geodesic distance helpers for PRH representation analysis.

The geodesic distance used here is the standard Isomap-style approximation:
construct an undirected k-nearest-neighbor graph using an ambient pairwise
metric, weight graph edges by their ambient distances, and compute all-pairs
shortest-path distances on that graph.

This module intentionally separates *support construction* from *distance
values*.  In the distance-aware PRH metrics, the original kNN/MST support can
therefore remain unchanged while only the numerical distance values are
replaced by graph-geodesic distances.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components, shortest_path
except ImportError:  # pragma: no cover
    csr_matrix = None
    connected_components = None
    shortest_path = None


def graph_geodesic_from_distances(
    D: np.ndarray,
    *,
    k: int = 20,
    eps: float = 1e-12,
) -> np.ndarray:
    """Approximate intrinsic distances with shortest paths on a symmetric kNN graph.

    Parameters
    ----------
    D:
        Square ambient pairwise-distance matrix.  It need not be normalized.
    k:
        Number of ambient nearest neighbors used to build the graph for each
        sample.  The graph is symmetrized by the union rule: an undirected edge
        is present if either endpoint selects the other.
    eps:
        Small positive floor for off-diagonal graph-edge weights.  This avoids
        sparse-matrix ambiguity for distinct samples with exactly zero distance.

    Returns
    -------
    np.ndarray
        Dense all-pairs graph-geodesic distance matrix.

    Notes
    -----
    We deliberately fail when the kNN graph is disconnected instead of silently
    changing ``k`` per layer.  This keeps ``k`` a controlled experimental
    parameter.  Increase ``geodesic_k`` (or sweep it) if disconnected layers
    occur.
    """
    if csr_matrix is None or connected_components is None or shortest_path is None:
        raise ImportError("scipy is required for graph-geodesic distances")

    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be square, got {D.shape}")
    if not np.isfinite(D).all():
        raise ValueError("Ambient distance matrix contains NaN or Inf")

    n = D.shape[0]
    if n <= 1:
        return D.copy()
    if not (1 <= int(k) < n):
        raise ValueError(f"geodesic k must satisfy 1 <= k < n={n}, got {k}")

    # Exclude self-distances when selecting neighbors.
    work = D.copy()
    np.fill_diagonal(work, np.inf)

    # argpartition avoids a full O(n log n) sort per row.
    nn = np.argpartition(work, kth=int(k) - 1, axis=1)[:, : int(k)]
    rows = np.repeat(np.arange(n, dtype=np.int64), int(k))
    cols = nn.reshape(-1).astype(np.int64, copy=False)
    vals = D[rows, cols]
    vals = np.maximum(vals, float(eps))

    graph = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
    # Union symmetrization.  Because the metric is symmetric, if both directed
    # edges exist they have the same weight; maximum therefore preserves it.
    graph = graph.maximum(graph.T)

    n_components, _ = connected_components(graph, directed=False, return_labels=True)
    if n_components != 1:
        raise ValueError(
            f"geodesic kNN graph is disconnected ({n_components} components) "
            f"for k={k}. Increase --geodesic-k or report a k sensitivity sweep."
        )

    G = shortest_path(graph, directed=False, unweighted=False, method="D")
    G = np.asarray(G, dtype=np.float64)
    if not np.isfinite(G).all():
        raise RuntimeError("Shortest-path computation produced NaN or Inf")
    np.fill_diagonal(G, 0.0)
    return G


def quantile_normalize_numpy(
    D: np.ndarray,
    *,
    q: float = 0.9,
    eps: float = 1e-12,
) -> np.ndarray:
    """Normalize a distance matrix by its positive off-diagonal q-quantile."""
    if not (0.0 < q <= 1.0):
        raise ValueError(f"q must be in (0, 1], got {q}")

    D = np.asarray(D, dtype=np.float64)
    vals = D[D > 0]
    if vals.size == 0:
        return D.copy()

    scale = max(float(np.quantile(vals, q)), float(eps))
    out = D / scale
    np.fill_diagonal(out, 0.0)
    return out
