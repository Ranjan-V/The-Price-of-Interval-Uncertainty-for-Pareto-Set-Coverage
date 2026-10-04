"""Deterministic-seed drifting convex interval problems."""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from ..objectives import QuadraticIntervalRound


@dataclass(frozen=True)
class SyntheticConfig:
    horizon: int = 100
    dimension: int = 2
    objectives: int = 2
    drift: float = 0.1
    radius: float = 0.1
    width_slope_fraction: float = 0.0
    seed: int = 7

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.dimension < 1 or self.objectives < 1:
            raise ValueError("positive horizon/dimension/objectives required")
        if not (0 <= self.drift <= 0.5) or self.radius < 0:
            raise ValueError("drift must be in [0,.5] and radius nonnegative")
        if not (0 <= self.width_slope_fraction <= 1):
            raise ValueError("width slope fraction must be in [0,1]")


def generate(config: SyntheticConfig) -> list[QuadraticIntervalRound]:
    rng = np.random.default_rng(config.seed)
    m, d = config.objectives, config.dimension
    base = rng.uniform(0.25, 0.75, size=(m, d))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(m, d))
    biases = rng.choice(np.array([-1.0, 1.0]), size=(m, d))
    slope_signs = rng.choice(np.array([-1.0, 1.0]), size=(m, d))
    slope = slope_signs * (2.0 * config.radius * config.width_slope_fraction / d)
    rounds = []
    for t in range(config.horizon):
        centers = np.clip(base + config.drift * np.sin(2.0 * np.pi * t / max(config.horizon, 2) + phases), 0.0, 1.0)
        rounds.append(QuadraticIntervalRound(centers, np.full(m, config.radius), biases, slope))
    return rounds


def weight_schedule(horizon: int, objectives: int) -> NDArray[np.float64]:
    if horizon < 1 or objectives < 1:
        raise ValueError("positive dimensions required")
    if objectives == 1:
        return np.ones((horizon, 1))
    weights = np.full((horizon, objectives), 0.2 / (objectives - 1))
    weights[np.arange(horizon), np.arange(horizon) % objectives] = 0.8
    return weights
