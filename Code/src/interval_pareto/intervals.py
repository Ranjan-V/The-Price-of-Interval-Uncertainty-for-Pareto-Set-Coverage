"""Endpoint representation and minimization orders."""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class IntervalArray:
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]

    def __post_init__(self) -> None:
        lo = np.asarray(self.lower, dtype=float)
        hi = np.asarray(self.upper, dtype=float)
        if lo.shape != hi.shape or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
            raise ValueError("finite endpoints must have matching shapes")
        if np.any(lo > hi):
            raise ValueError("lower endpoint exceeds upper endpoint")
        object.__setattr__(self, "lower", lo.copy())
        object.__setattr__(self, "upper", hi.copy())

    @property
    def width(self) -> NDArray[np.float64]:
        return self.upper - self.lower

    @property
    def midpoint(self) -> NDArray[np.float64]:
        return (self.lower + self.upper) / 2.0


def lu_leq(a: IntervalArray, b: IntervalArray) -> bool:
    if a.lower.shape != b.lower.shape:
        raise ValueError("shape mismatch")
    return bool(np.all(a.lower <= b.lower) and np.all(a.upper <= b.upper))


def lu_dominates(a: IntervalArray, b: IntervalArray) -> bool:
    return lu_leq(a, b) and bool(np.any(a.lower < b.lower) or np.any(a.upper < b.upper))


def weak_lu_dominates(a: IntervalArray, b: IntervalArray) -> bool:
    if a.lower.shape != b.lower.shape:
        raise ValueError("shape mismatch")
    return bool(np.all(a.lower < b.lower) and np.all(a.upper < b.upper))


def certainly_better(a: IntervalArray, b: IntervalArray) -> bool:
    if a.lower.shape != b.lower.shape:
        raise ValueError("shape mismatch")
    return bool(np.all(a.upper < b.lower))
