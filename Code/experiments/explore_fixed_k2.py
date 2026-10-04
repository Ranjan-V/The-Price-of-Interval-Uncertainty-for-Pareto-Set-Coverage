"""Exploratory optimization for two-output fixed-regularity coverage.

This is a numerical investigation, not a theorem or certified optimizer.
"""

from __future__ import annotations

from scipy.optimize import differential_evolution


C = 0.5
R = 0.125


def coverage_pair(pair: tuple[float, float], theta: float) -> float:
    x1, x2 = sorted(pair)
    z1, z2 = x1 - theta, x2 - theta
    assert -R <= z1 <= z2 <= R
    left = (z1 + R) ** 2
    right = (R - z2) ** 2
    crossing = ((R + z2) ** 2 - (R - z1) ** 2) / (4 * R)
    middle = (R - z1) ** 2 - (R - crossing) ** 2
    return max(left, middle, right)


oracle = 9 * R * R / 16
for s in (0.001, 0.002, 0.005, 0.01, 0.02):
    worlds = (C - s, C + s)
    bounds = ((C - R / 2, C - 2 * s), (C + 2 * s, C + R / 2))
    for mode in ("mean", "worst"):
        def objective(pair):
            values = [coverage_pair(pair, theta) for theta in worlds]
            return sum(values) / 2 if mode == "mean" else max(values)

        result = differential_evolution(objective, bounds, tol=1e-12, polish=True, seed=17)
        x1, x2 = sorted(result.x)
        continuous_worst = max(
            coverage_pair((x1, x2), C - s + 2 * s * i / 1000)
            for i in range(1001)
        )
        print(
            f"s={s:.3g} {mode} x=({x1:.7f},{x2:.7f}) "
            f"excess={result.fun-oracle:.9f} excess/s={(result.fun-oracle)/s:.6f} "
            f"continuous_worst_excess={continuous_worst-oracle:.9f}"
        )
