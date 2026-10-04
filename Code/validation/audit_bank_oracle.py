"""Independent, grid-based audit of the Bank finite-pool coverage study.

This deliberately does not call the experiment's coverage or oracle routines.
The target-grid objective is a lower approximation of continuous coverage;
differential evolution searches the two-output action square independently.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Code/data/bank_marketing/bank_stream.npz"
ROUNDS = ROOT / "Code/results/real_bank_pareto_coverage/bank_pareto_coverage_rounds.csv"
SUMMARY = ROOT / "Code/results/real_bank_pareto_coverage/bank_pareto_coverage_summary.json"
OUTPUT = ROOT / "Code/results/real_bank_pareto_coverage/independent_audit.json"
A = 0.01
B = 0.8


def grid_coverage(pair: tuple[float, float], p: float, targets: int = 4001) -> float:
    u = np.linspace(A, max(A, min(B, p)), targets)
    base_resource = (u - A) ** 2
    base_risk = -p * np.log(u) - (1 - p) * np.log1p(-u)
    losses = []
    for x in pair:
        resource_excess = (x - A) ** 2 - base_resource
        risk_excess = -p * np.log(x) - (1 - p) * np.log1p(-x) - base_risk
        losses.append(np.maximum(np.maximum(resource_excess, risk_excess), 0))
    return float(np.max(np.minimum(losses[0], losses[1])))


def audit() -> dict:
    with np.load(DATA, allow_pickle=False) as stream:
        labels = np.asarray(stream["labels"], int)
        groups = np.asarray(stream["groups"], int)
    rows = list(csv.DictReader(ROUNDS.open(newline="", encoding="utf-8")))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert len(rows) == 128 * 3 * 5
    max_rate_error = 0.0
    max_grid_gap = 0.0
    max_coverage_grid_gap = 0.0
    for row in rows:
        t = int(row["batch"]) - 1
        pool = labels[t, groups[t] == 1] > 0
        n = int(row["sample_count"])
        k_lo = float(row["interval_rate_lower"]) * len(pool)
        p = float(np.mean(pool))
        assert n == int(np.ceil(float(row["sample_fraction"]) * len(pool)))
        assert abs(k_lo - round(k_lo)) < 1e-8
        k = round(k_lo)
        assert k <= int(pool.sum()) <= k + len(pool) - n
        max_rate_error = max(max_rate_error, abs(p - float(row["population_positive_rate"])))
        q = float(row["query_point"])
        risk_lo = -float(row["interval_rate_lower"]) * np.log(q) - (1 - float(row["interval_rate_lower"])) * np.log1p(-q)
        risk_hi = -float(row["interval_rate_upper"]) * np.log(q) - (1 - float(row["interval_rate_upper"])) * np.log1p(-q)
        assert abs(min(risk_lo, risk_hi) - float(row["interval_query_lower"])) < 1e-12
        assert abs(max(risk_lo, risk_hi) - float(row["interval_query_upper"])) < 1e-12
        for prefix in ("blind", "adaptive"):
            pair = (float(row[prefix + "_left"]), float(row[prefix + "_right"]))
            sampled = grid_coverage(pair, p, targets=1001)
            reported = float(row[prefix + "_coverage"])
            assert sampled <= reported + 1e-9
            max_coverage_grid_gap = max(max_coverage_grid_gap, reported - sampled)
            assert reported - sampled < 2e-4

    # Independent optimizer on representative pool rates. Its gridded optimum
    # is a lower bound on the true continuous optimum, within numerical search error.
    pool_rates = np.array([np.mean(labels[t, groups[t] == 1] > 0) for t in range(128)])
    rates = [float(np.quantile(pool_rates, q)) for q in (0.1, 0.3, 0.5, 0.7, 0.9)]
    checks = []
    for p in rates:
        reference = next((r for r in rows if abs(float(r["population_positive_rate"]) - p) < 1e-12), None)
        # Quantiles may interpolate; select nearest observed pool.
        if reference is None:
            reference = min(rows, key=lambda r: abs(float(r["population_positive_rate"]) - p))
            p = float(reference["population_positive_rate"])
        opt = differential_evolution(
            lambda x: grid_coverage((float(x[0]), float(x[1])), p, targets=801),
            bounds=[(A, B), (A, B)], seed=177, maxiter=150, popsize=12,
            tol=1e-9, polish=True,
        )
        grid_opt = grid_coverage(tuple(map(float, opt.x)), p, targets=4001)
        floor = float(reference["oracle_floor"])
        gap = floor - grid_opt
        max_grid_gap = max(max_grid_gap, abs(gap))
        checks.append({"rate": p, "reported_oracle_floor": floor,
                       "independent_grid_minimum": grid_opt, "difference": gap})
        assert -2e-4 <= gap <= 2e-4, (p, floor, grid_opt)

    for item in summary["by_fraction"]:
        subset = [r for r in rows if float(r["sample_fraction"]) == item["sample_fraction"]]
        for key, column in (("mean_blind_excess", "blind_excess"),
                            ("mean_adaptive_excess", "adaptive_excess")):
            assert abs(np.mean([float(r[column]) for r in subset]) - item[key]) < 1e-12
    result = {"records_checked": len(rows), "max_pool_rate_error": max_rate_error,
              "max_continuous_minus_grid_coverage": max_coverage_grid_gap,
              "independent_optimizer_checks": checks, "max_abs_oracle_gap": max_grid_gap,
              "note": "Grid minimization is an independent numerical diagnostic, not a proof of global optimality."}
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
