"""One midpoint learner per fixed preference, matching THM-05's K-query model."""

from itertools import product
import numpy as np
from numpy.typing import NDArray
from .ogd import MidpointOGD
from ..objectives import QuadraticIntervalRound


def simplex_lattice(objectives: int, denominator: int) -> NDArray[np.float64]:
    """All nonnegative integer weights summing to q, divided by q.

    Its l1 covering radius is at most 2(m-1)/q for m>1; m=1 is exact.
    Cardinality grows combinatorially, so use modest q.
    """
    if objectives < 1 or denominator < 1:
        raise ValueError("positive objective count and denominator required")
    points = [part for part in product(range(denominator + 1), repeat=objectives)
              if sum(part) == denominator]
    return np.asarray(points, dtype=float) / denominator


class ParallelMidpointOGD:
    def __init__(self, weights: NDArray[np.float64], dimension: int, eta: float):
        grid = np.asarray(weights, dtype=float)
        if grid.ndim != 2 or not len(grid) or np.any(grid < 0) or not np.allclose(grid.sum(axis=1), 1.0):
            raise ValueError("nonempty simplex weight grid required")
        self.weights = grid.copy()
        self.learners = [MidpointOGD(dimension, eta) for _ in grid]

    def act(self) -> NDArray[np.float64]:
        return np.asarray([learner.act() for learner in self.learners])

    def observe(self, item: QuadraticIntervalRound, actions: NDArray[np.float64]) -> None:
        xx = np.asarray(actions, float)
        if item.m != self.weights.shape[1] or xx.shape != (len(self.learners), item.d):
            raise ValueError("round and action shapes mismatch")
        for learner, x, w in zip(self.learners, xx, self.weights):
            learner.observe_gradient(item.midpoint_gradient(x, w))
