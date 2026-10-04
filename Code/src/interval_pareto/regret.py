"""Offline evaluation of selected-preference regret and theorem terms."""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from .objectives import QuadraticIntervalRound, _weights
from .variation import path_variation


def scalar_static_regret(losses: NDArray[np.float64], comparator_losses: NDArray[np.float64]) -> float:
    """Regret against a supplied fixed comparator's per-round scalar losses."""
    played, fixed = np.asarray(losses, float), np.asarray(comparator_losses, float)
    if played.ndim != 1 or fixed.shape != played.shape or not np.all(np.isfinite(played)) or not np.all(np.isfinite(fixed)):
        raise ValueError("matching finite one-dimensional losses required")
    return float(np.sum(played - fixed))


def scalar_dynamic_regret(losses: NDArray[np.float64], comparator_losses: NDArray[np.float64]) -> float:
    """Regret against supplied moving comparator losses; same arithmetic as static."""
    return scalar_static_regret(losses, comparator_losses)


@dataclass(frozen=True)
class RegretReport:
    regret: float
    midpoint_regret: float
    path_variation: float
    two_location_width: float
    uniform_width: float
    online_term: float
    drift_term: float
    gradient_term: float
    theorem_bound: float


def evaluate(rounds: list[QuadraticIntervalRound], actions: NDArray[np.float64],
             comparators: NDArray[np.float64], weights: NDArray[np.float64], eta: float) -> RegretReport:
    if not rounds or not np.isfinite(eta) or eta <= 0:
        raise ValueError("nonempty rounds and positive step size required")
    xx, uu, ww = np.asarray(actions, float), np.asarray(comparators, float), np.asarray(weights, float)
    t_count, d, m = len(rounds), rounds[0].d, rounds[0].m
    if xx.shape != (t_count, d) or uu.shape != (t_count, d) or ww.shape != (t_count, m):
        raise ValueError("action, comparator, weight shapes inconsistent")
    if any(r.d != d or r.m != m for r in rounds):
        raise ValueError("round dimensions must match")
    regret = midpoint_regret = q = uniform_width = 0.0
    for t, item in enumerate(rounds):
        w = _weights(ww[t], m)
        regret += float(w @ (item.latent(xx[t]) - item.latent(uu[t])))
        midpoint_regret += float(w @ (item.midpoint(xx[t]) - item.midpoint(uu[t])))
        q += float(w @ (item.interval(xx[t]).width + item.interval(uu[t]).width)) / 2.0
        uniform_width += float(np.max(item.maximum_width()))
    p = path_variation(uu)
    diameter = np.sqrt(d)
    gradient_bound = 2.0 * np.sqrt(d)  # all centers and actions in [0,1]^d
    online_term = diameter ** 2 / (2.0 * eta)
    drift_term = diameter * p / eta
    gradient_term = eta * gradient_bound ** 2 * t_count / 2.0
    bound = online_term + drift_term + gradient_term + q
    return RegretReport(regret, midpoint_regret, p, q, uniform_width,
                        online_term, drift_term, gradient_term, bound)
