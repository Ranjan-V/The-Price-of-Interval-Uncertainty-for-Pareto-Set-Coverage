"""Deterministic numerical diagnostics for THM-09 (not a proof)."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Code" / "results" / "fixed_regularity_information"
OUT.mkdir(parents=True, exist_ok=True)

C = 0.5
R = 0.125
T = 128
S_VALUES = (0.005, 0.01, 0.025, 0.05, 0.1, 0.125)
X_GRID = tuple(i / 1000 for i in range(1001))
U_GRID = tuple(i / 1000 for i in range(1001))


def coverage_sampled(x: float, theta: float) -> float:
    left, right = theta - R, theta + R
    return max(
        max(
            max(0.0, (x - theta + R) ** 2 - (u - theta + R) ** 2),
            max(0.0, (x - theta - R) ** 2 - (u - theta - R) ** 2),
        )
        for u in U_GRID
        if left - 1e-12 <= u <= right + 1e-12
    )


rows = []
for s in S_VALUES:
    theta_left, theta_right = C - s, C + s
    exact_per_round = 2 * R * s + s * s
    width = 2.5 * s
    # Independently optimize the two-world sampled minimax objective on an x-grid.
    grid_minimax = min(
        max((R + abs(x - theta_left)) ** 2, (R + abs(x - theta_right)) ** 2) - R * R
        for x in X_GRID
    )
    # Check the original sup_u min_x metric directly for K=1 at representative points.
    sampled_errors = [
        abs(coverage_sampled(x, theta) - (R + abs(x - theta)) ** 2)
        for x, theta in ((C, theta_left), (C, theta_right), (0.3, theta_left), (0.7, theta_right))
    ]
    # Sweep domain points to test the envelope's maximum width.
    for center in (C - R, C + R):
        sampled_width = max(
            (abs(x - center) + s) ** 2 - max(abs(x - center) - s, 0.0) ** 2
            for x in X_GRID
        )
        assert abs(sampled_width - width) < 1e-12
    assert abs(grid_minimax - exact_per_round) < 1e-12
    assert max(sampled_errors) < 1e-12
    rows.append(
        {
            "s": s,
            "width_per_round": width,
            "exact_minimax_excess_per_round": exact_per_round,
            "minimax_excess_over_width": exact_per_round / width,
            "grid_minimax_excess_per_round": grid_minimax,
            "max_sampled_coverage_identity_error": max(sampled_errors),
            "fixed_lipschitz_upper": 2.0,
            "fixed_strong_convexity": 2.0,
            "horizon": T,
        }
    )

with (OUT / "fixed_regularity_information.csv").open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

summary = {
    "cases": len(rows),
    "all_checks_passed": True,
    "min_excess_to_width_ratio": min(row["minimax_excess_over_width"] for row in rows),
    "max_excess_to_width_ratio": max(row["minimax_excess_over_width"] for row in rows),
    "max_coverage_identity_error": max(row["max_sampled_coverage_identity_error"] for row in rows),
    "note": "Grid checks probe exact algebra but do not prove minimaxity; see proof_thm09.md.",
}
(OUT / "fixed_regularity_information_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(json.dumps(summary, indent=2))
