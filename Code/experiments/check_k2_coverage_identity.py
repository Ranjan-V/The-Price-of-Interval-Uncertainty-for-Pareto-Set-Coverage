"""Numerical checks for the proved two-point coverage identity and upper bound."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Code" / "results" / "fixed_k2_coverage"
OUT.mkdir(parents=True, exist_ok=True)
C = 0.5
R = 0.125
S_VALUES = (0.001, 0.002, 0.005, 0.01, 0.015)
ORACLE = 9 * R * R / 16


def target_loss(x: float, u: float, theta: float) -> float:
    return max(
        0.0,
        (x - theta + R) ** 2 - (u - theta + R) ** 2,
        (x - theta - R) ** 2 - (u - theta - R) ** 2,
    )


def exact_pair(pair: tuple[float, float], theta: float) -> float:
    x1, x2 = sorted(pair)
    m, h = (x1 + x2) / 2, (x2 - x1) / 2
    z = m - theta
    assert abs(z) + h <= R + 1e-12
    return max((R - h + abs(z)) ** 2, (2 * R * h + h * h) * (1 - z * z / (R * R)))


rows = []
for s in S_VALUES:
    h = (R + s) ** 2 / (2 * (2 * R + s))
    pair = (C - h, C + h)
    assert h + s <= R
    robust = exact_pair(pair, C)
    endpoint = exact_pair(pair, C + s)
    assert abs(robust - endpoint) < 1e-12
    direct_errors = []
    for theta in (C - s, C, C + s):
        x1, x2 = pair
        z = C - theta
        crossing = theta + z * (1 + h / R)
        targets = (theta - R, crossing, theta + R)
        direct = max(min(target_loss(x1, u, theta), target_loss(x2, u, theta)) for u in targets)
        direct_errors.append(abs(direct - exact_pair(pair, theta)))
    assert max(direct_errors) < 1e-12
    rows.append(
        {
            "s": s,
            "half_gap": h,
            "oracle_two_point_floor": ORACLE,
            "robust_coverage_upper_per_round": robust,
            "robust_excess_upper_per_round": robust - ORACLE,
            "excess_upper_over_width": (robust - ORACLE) / (2.5 * s),
            "max_identity_error": max(direct_errors),
        }
    )

with (OUT / "fixed_k2_coverage.csv").open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
summary = {
    "cases": len(rows),
    "all_checks_passed": True,
    "max_identity_error": max(row["max_identity_error"] for row in rows),
    "upper_excess_over_width_range": [
        min(row["excess_upper_over_width"] for row in rows),
        max(row["excess_upper_over_width"] for row in rows),
    ],
    "note": "Only an achieved upper bound; a K=2 minimax lower bound is open.",
}
(OUT / "fixed_k2_coverage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
