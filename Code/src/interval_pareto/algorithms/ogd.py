"""Box-projected online gradient methods using post-decision feedback."""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class MidpointOGD:
    dimension: int
    eta: float
    x: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if self.dimension < 1 or not np.isfinite(self.eta) or self.eta <= 0:
            raise ValueError("positive dimension and step size required")
        self.x = np.full(self.dimension, 0.5) if self.x is None else np.asarray(self.x, dtype=float).copy()
        if self.x.shape != (self.dimension,) or not np.all(np.isfinite(self.x)) or np.any(self.x < 0) or np.any(self.x > 1):
            raise ValueError("initial action must be in unit box")

    def act(self) -> NDArray[np.float64]:
        return self.x.copy()

    def observe_gradient(self, gradient: NDArray[np.float64]) -> None:
        grad = np.asarray(gradient, dtype=float)
        if grad.shape != (self.dimension,) or not np.all(np.isfinite(grad)):
            raise ValueError("finite gradient with decision dimension required")
        self.x = np.clip(self.x - self.eta * grad, 0.0, 1.0)


class UpperEndpointOGD(MidpointOGD):
    """Same update interface; caller supplies upper-endpoint weighted gradient."""


class LowerEndpointOGD(MidpointOGD):
    """Same update interface; caller supplies lower-endpoint weighted gradient."""
