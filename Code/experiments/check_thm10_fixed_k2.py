"""Deterministic diagnostic grids for THM-10's local and global bounds."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Code" / "results" / "fixed_k2_minimax"
OUT.mkdir(parents=True, exist_ok=True)
R = 0.125
E_STAR = 9 * R * R / 16


def excess(z: float, d: float) -> float:
    h = R / 4 + d
    return max((R - h + abs(z)) ** 2, (2 * R * h + h * h) * (1 - z * z / (R * R))) - E_STAR


checks = 0
min_slack = float("inf")
for iz in range(101):
    z = (iz - 50) * R / 200
    for idelta in range(101):
        d = (idelta - 50) * R / 400
        slack = excess(z, d) - (R / 3) * (abs(z) + abs(d))
        min_slack = min(min_slack, slack)
        assert slack > -1e-12
        checks += 1

global_checks = 0
minimum_global_slack = float("inf")
gamma = 3 * R * R / 32
for it in range(101):
    t = it * R / 100
    for ih in range(101):
        h = ih * (R - t) / 100
        d = h - R / 4
        if t <= R / 4 and abs(d) <= R / 8:
            continue
        slack = excess(t, d) - gamma
        minimum_global_slack = min(minimum_global_slack, slack)
        assert slack > -1e-12
        global_checks += 1

upper_cases = []
for q in (0.001, 0.01, 0.05, 0.1, 0.125):
    s = q * R
    h = (R + s) ** 2 / (2 * (2 * R + s))
    upper = 2 * R * h + h * h - E_STAR
    assert upper <= R * s + 1e-12
    upper_cases.append({"s_over_r": q, "upper_excess": upper, "upper_over_width": upper / (2.5 * s)})

summary = {
    "local_grid_checks": checks,
    "minimum_local_slack": min_slack,
    "global_grid_checks": global_checks,
    "minimum_global_slack": minimum_global_slack,
    "upper_cases": upper_cases,
    "all_checks_passed": True,
    "note": "Finite grids check the inequalities but do not prove the continuum theorem.",
}
(OUT / "thm10_fixed_k2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
