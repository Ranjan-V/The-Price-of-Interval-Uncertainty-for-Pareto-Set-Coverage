"""Comparator and finite-set movement metrics."""

import numpy as np
from numpy.typing import NDArray
from .pareto import hausdorff


def path_variation(path: NDArray[np.float64]) -> float:
    arr = np.asarray(path, dtype=float)
    if arr.ndim != 2 or len(arr) == 0:
        raise ValueError("nonempty T-by-d path required")
    return float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())


def set_variation(sets: list[NDArray[np.float64]]) -> float:
    if not sets:
        raise ValueError("nonempty sequence required")
    return float(sum(hausdorff(sets[t - 1], sets[t]) for t in range(1, len(sets))))


def sampled_objective_variation(value_tables: NDArray[np.float64]) -> float:
    """Finite-probe max-norm variation; not the continuum supremum."""
    arr = np.asarray(value_tables, dtype=float)
    if arr.ndim != 3 or len(arr) == 0 or arr.shape[1] == 0 or arr.shape[2] == 0:
        raise ValueError("expected nonempty T-by-probes-by-objectives table")
    if not np.all(np.isfinite(arr)):
        raise ValueError("finite values required")
    return float(sum(np.max(np.abs(arr[t] - arr[t - 1])) for t in range(1, len(arr))))
