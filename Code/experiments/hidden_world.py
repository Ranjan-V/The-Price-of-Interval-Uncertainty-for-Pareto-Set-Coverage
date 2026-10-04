"""Future EXP-10 diagnostic for THM-03's indistinguishable interval worlds."""

import argparse
from pathlib import Path
import csv
import numpy as np


def regret_pair(actions: np.ndarray, widths: np.ndarray) -> tuple[float, float, float]:
    x, a = np.asarray(actions, float), np.asarray(widths, float)
    if x.ndim != 1 or a.shape != x.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(a)) or np.any(x < 0) or np.any(x > 1) or np.any(a < 0):
        raise ValueError("actions in [0,1] and nonnegative matching widths required")
    plus = float(a @ x)
    minus = float(a @ (1.0 - x))
    return plus, minus, float(a.sum())


def prepare(widths: np.ndarray, seed: int, output: Path) -> None:
    """Use interval-only random actions; no latent-world signal reaches learner."""
    rng = np.random.default_rng(seed)
    actions = rng.uniform(size=len(widths))
    plus, minus, uncertainty = regret_pair(actions, widths)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "world_plus_regret", "world_minus_regret", "uniform_width"])
        writer.writeheader()
        writer.writerow({"seed": seed, "world_plus_regret": plus,
                         "world_minus_regret": minus, "uniform_width": uncertainty})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--width", type=float, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.horizon < 1 or not np.isfinite(args.width) or args.width < 0:
        raise ValueError("positive horizon and nonnegative width required")
    prepare(np.full(args.horizon, args.width), args.seed, args.output)
