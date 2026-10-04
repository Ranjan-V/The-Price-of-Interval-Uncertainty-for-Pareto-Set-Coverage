"""Convex synthetic objectives with latent values hidden inside intervals."""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from .intervals import IntervalArray


@dataclass(frozen=True)
class QuadraticIntervalRound:
    centers: NDArray[np.float64]  # m by d, in [0,1]
    radii: NDArray[np.float64]    # m, nonnegative
    biases: NDArray[np.float64]   # m by d, each coordinate in [-1,1]
    width_slopes: NDArray[np.float64] | None = None  # m by d; affine half-width

    def __post_init__(self) -> None:
        c = np.asarray(self.centers, dtype=float)
        r = np.asarray(self.radii, dtype=float)
        b = np.asarray(self.biases, dtype=float)
        s = np.zeros_like(c) if self.width_slopes is None else np.asarray(self.width_slopes, dtype=float)
        if c.ndim != 2 or r.shape != (c.shape[0],) or b.shape != c.shape or s.shape != c.shape:
            raise ValueError("centers/biases must be m-by-d and radii length m")
        if c.size == 0 or not np.all(np.isfinite(c)) or not np.all(np.isfinite(r)) or not np.all(np.isfinite(b)) or not np.all(np.isfinite(s)):
            raise ValueError("finite nonempty arrays required")
        if np.any(c < 0) or np.any(c > 1) or np.any(r < 0) or np.any(np.abs(b) > 1):
            raise ValueError("centers in box, nonnegative radii, bounded biases required")
        if np.any(r < 0.5 * np.sum(np.abs(s), axis=1)):
            raise ValueError("affine half-width must stay nonnegative on unit box")
        object.__setattr__(self, "centers", c.copy())
        object.__setattr__(self, "radii", r.copy())
        object.__setattr__(self, "biases", b.copy())
        object.__setattr__(self, "width_slopes", s.copy())

    @property
    def m(self) -> int:
        return self.centers.shape[0]

    @property
    def d(self) -> int:
        return self.centers.shape[1]

    def _decision(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        xx = np.asarray(x, dtype=float)
        if xx.shape != (self.d,) or not np.all(np.isfinite(xx)) or np.any(xx < 0) or np.any(xx > 1):
            raise ValueError("decision outside unit box")
        return xx

    def midpoint(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        xx = self._decision(x)
        return np.sum((self.centers - xx) ** 2, axis=1)

    def interval(self, x: NDArray[np.float64]) -> IntervalArray:
        xx = self._decision(x)
        center = self.midpoint(xx)
        radius = self.radii + self.width_slopes @ (xx - 0.5)
        return IntervalArray(center - radius, center + radius)

    def latent(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        xx = self._decision(x)
        tilt = np.sum(self.biases * (2.0 * xx - 1.0), axis=1) / self.d
        minimum_radius = self.radii - 0.5 * np.sum(np.abs(self.width_slopes), axis=1)
        return self.midpoint(xx) + minimum_radius * tilt

    def midpoint_gradient(self, x: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        xx = self._decision(x)
        ww = _weights(weights, self.m)
        return 2.0 * (xx - ww @ self.centers)

    def latent_weighted_minimizer(self, weights: NDArray[np.float64]) -> NDArray[np.float64]:
        ww = _weights(weights, self.m)
        minimum_radius = self.radii - 0.5 * np.sum(np.abs(self.width_slopes), axis=1)
        return np.clip(ww @ self.centers - (ww * minimum_radius) @ self.biases / self.d, 0.0, 1.0)

    def upper_gradient(self, x: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        ww = _weights(weights, self.m)
        return self.midpoint_gradient(x, ww) + ww @ self.width_slopes

    def lower_gradient(self, x: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        ww = _weights(weights, self.m)
        return self.midpoint_gradient(x, ww) - ww @ self.width_slopes

    def maximum_width(self) -> NDArray[np.float64]:
        return 2.0 * (self.radii + 0.5 * np.sum(np.abs(self.width_slopes), axis=1))


def _weights(weights: NDArray[np.float64], m: int) -> NDArray[np.float64]:
    ww = np.asarray(weights, dtype=float)
    if ww.shape != (m,) or not np.all(np.isfinite(ww)) or np.any(ww < 0) or not np.isclose(ww.sum(), 1.0):
        raise ValueError("weights must lie in the m-simplex")
    return ww
