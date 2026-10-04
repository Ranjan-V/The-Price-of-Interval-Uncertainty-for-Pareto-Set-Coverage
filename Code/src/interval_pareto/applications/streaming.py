"""Convex streaming classification surrogates on supplied batches.

No statistical coverage claim is made for the caller-supplied interval radii.
"""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from ..intervals import IntervalArray


@dataclass(frozen=True)
class ClassificationBatch:
    features: NDArray[np.float64]  # n by d
    labels: NDArray[np.float64]    # {-1,+1}
    groups: NDArray[np.int64]      # {0,1}

    def __post_init__(self) -> None:
        x = np.asarray(self.features, float)
        y = np.asarray(self.labels, float)
        g = np.asarray(self.groups, int)
        if x.ndim != 2 or not len(x) or y.shape != (len(x),) or g.shape != (len(x),):
            raise ValueError("inconsistent batch shapes")
        if not np.all(np.isfinite(x)) or not np.all(np.isin(y, [-1.0, 1.0])) or not np.all(np.isin(g, [0, 1])):
            raise ValueError("features, labels or groups invalid")
        if not np.any(g == 0) or not np.any(g == 1):
            raise ValueError("both groups required for disparity surrogate")
        object.__setattr__(self, "features", x.copy())
        object.__setattr__(self, "labels", y.copy())
        object.__setattr__(self, "groups", g.copy())


def objective_and_subgradients(batch: ClassificationBatch, coefficients: NDArray[np.float64],
                               resource_scale: float = 1.0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Logistic loss, absolute group mean score gap, quadratic resource cost."""
    w = np.asarray(coefficients, float)
    x, y, group = batch.features, batch.labels, batch.groups
    if w.shape != (x.shape[1],) or resource_scale < 0:
        raise ValueError("coefficient dimension or resource scale invalid")
    margins = y * (x @ w)
    logistic = float(np.mean(np.logaddexp(0.0, -margins)))
    loss_grad = -(x.T @ (y / (1.0 + np.exp(np.clip(margins, -700, 700))))) / len(x)
    gap_direction = x[group == 0].mean(axis=0) - x[group == 1].mean(axis=0)
    gap = float(gap_direction @ w)
    disparity = abs(gap)
    fairness_subgradient = np.sign(gap) * gap_direction
    resource = 0.5 * resource_scale * float(w @ w)
    resource_gradient = resource_scale * w
    return np.array([logistic, disparity, resource]), np.stack([loss_grad, fairness_subgradient, resource_gradient])


def supplied_intervals(values: NDArray[np.float64], radii: NDArray[np.float64]) -> IntervalArray:
    center, radius = np.asarray(values, float), np.asarray(radii, float)
    if center.shape != (3,) or radius.shape != (3,) or np.any(radius < 0):
        raise ValueError("three nonnegative radii required")
    return IntervalArray(center - radius, center + radius)
