"""Finite-set Pareto diagnostics, not continuous-front proofs."""

import numpy as np
from numpy.typing import NDArray
from .intervals import IntervalArray


def dominates(a: NDArray[np.float64], b: NDArray[np.float64]) -> bool:
    aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if aa.shape != bb.shape or aa.ndim != 1:
        raise ValueError("vectors must have the same one-dimensional shape")
    return bool(np.all(aa <= bb) and np.any(aa < bb))


def weakly_dominates(a: NDArray[np.float64], b: NDArray[np.float64]) -> bool:
    aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if aa.shape != bb.shape or aa.ndim != 1:
        raise ValueError("vectors must have the same one-dimensional shape")
    return bool(np.all(aa < bb))


def efficient_mask(values: NDArray[np.float64]) -> NDArray[np.bool_]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or not np.all(np.isfinite(arr)):
        raise ValueError("expected finite n-by-m matrix")
    n = arr.shape[0]
    return np.array([not any(dominates(arr[k], arr[i]) for k in range(n) if k != i)
                     for i in range(n)], dtype=bool)


def lu_efficient_mask(intervals: list[IntervalArray]) -> NDArray[np.bool_]:
    """Finite LU-efficient extraction by stacking all lower/upper endpoints."""
    if not intervals:
        raise ValueError("nonempty interval list required")
    shape = intervals[0].lower.shape
    if len(shape) != 1 or any(item.lower.shape != shape for item in intervals):
        raise ValueError("one-dimensional objective vectors with common shape required")
    values = np.asarray([np.stack([item.lower, item.upper], axis=1).ravel() for item in intervals])
    return efficient_mask(values)


def directed_hausdorff(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if aa.ndim != 2 or bb.ndim != 2 or aa.shape[1] != bb.shape[1] or not len(aa) or not len(bb):
        raise ValueError("nonempty point sets with common dimension required")
    return float(max(min(np.linalg.norm(x - y) for y in bb) for x in aa))


def hausdorff(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    return max(directed_hausdorff(a, b), directed_hausdorff(b, a))


def additive_coverage(front_values: NDArray[np.float64], set_values: NDArray[np.float64]) -> float:
    """Finite sup_front inf_set max_j(played_j-front_j)_+ proxy."""
    front, selected = np.asarray(front_values, float), np.asarray(set_values, float)
    if front.ndim != 2 or selected.ndim != 2 or front.shape[1] != selected.shape[1]:
        raise ValueError("value sets must have common objective dimension")
    if len(front) == 0 or len(selected) == 0:
        raise ValueError("nonempty value sets required")
    return float(max(min(np.max(np.maximum(played - target, 0.0)) for played in selected)
                     for target in front))
