"""Test APP-01 with known finite-window population objectives.

Each original chronological batch is a fixed finite population. Feedback is
sampled IID with replacement from that batch. This designed experiment meets
APP-01's within-round sampling assumption conditional on the fixed pools.
It is not an observational coverage claim for the original UCI stream.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from interval_pareto.algorithms import MidpointOGD
from interval_pareto.applications.streaming import ClassificationBatch, objective_and_subgradients
from certify_bank_population import gap_radius, loss_radius


def run(data_path: Path, output: Path, seeds: tuple[int, ...] = tuple(range(10)),
        eta: float = 0.05, delta: float = 0.05) -> dict:
    if not seeds or len(set(seeds)) != len(seeds) or eta <= 0 or not 0 < delta < 1:
        raise ValueError("invalid seeds, eta, or delta")
    with np.load(data_path, allow_pickle=False) as data:
        x = np.asarray(data["features"], float)
        y = np.asarray(data["labels"], float)
        g = np.asarray(data["groups"], int)
        weights = np.asarray(data["weights"], float)
    if x.ndim != 3 or x.shape[2] != 24 or y.shape != x.shape[:2] or g.shape != y.shape:
        raise ValueError("EXP-10 arrays required")
    horizon, n, dimension = x.shape
    risk_radius, spacing = loss_radius(n, horizon, delta / 2)
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        learner = MidpointOGD(dimension, eta)
        for t in range(horizon):
            action = learner.act()
            pool = ClassificationBatch(x[t], y[t], g[t])
            draw = rng.integers(0, n, size=n)
            if not np.any(g[t, draw] == 0) or not np.any(g[t, draw] == 1):
                raise RuntimeError("Rare empty sampled age group; choose a new predeclared seed set")
            sample = ClassificationBatch(x[t, draw], y[t, draw], g[t, draw])
            population, _ = objective_and_subgradients(pool, action)
            observed, gradients = objective_and_subgradients(sample, action)
            n0 = int(np.sum(sample.groups == 0))
            n1 = n - n0
            radii = np.array([risk_radius, gap_radius(n0, n1, horizon, delta / 2), 0.0])
            error = np.abs(observed - population)
            rows.append({
                "seed": seed, "batch": t + 1, "n0": n0, "n1": n1,
                "population_logloss": float(population[0]),
                "sample_logloss": float(observed[0]),
                "population_gap": float(population[1]),
                "sample_gap": float(observed[1]),
                "risk_error": float(error[0]), "gap_error": float(error[1]),
                "risk_radius": float(radii[0]), "gap_radius": float(radii[1]),
                "risk_covered": int(error[0] <= radii[0] + 1e-12),
                "gap_covered": int(error[1] <= radii[1] + 1e-12),
                "fixed_risk_covered": int(error[0] <= 0.05 + 1e-12),
                "fixed_gap_covered": int(error[1] <= 0.05 + 1e-12),
            })
            learner.observe_gradient(weights[t] @ gradients)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controlled_resampling_rounds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    per_seed = []
    for seed in seeds:
        subset = [row for row in rows if row["seed"] == seed]
        per_seed.append({
            "seed": seed,
            "all_played_actions_covered": all(row["risk_covered"] and row["gap_covered"] for row in subset),
            "risk_coverage": float(np.mean([row["risk_covered"] for row in subset])),
            "gap_coverage": float(np.mean([row["gap_covered"] for row in subset])),
            "fixed_risk_coverage": float(np.mean([row["fixed_risk_covered"] for row in subset])),
            "fixed_gap_coverage": float(np.mean([row["fixed_gap_covered"] for row in subset])),
        })
    summary = {
        "protocol": "IID-with-replacement feedback from each fixed chronological 256-row pool",
        "population_target": "exact objective of the original 256-row pool at each current action",
        "APP01_assumption_satisfied_by_design": True,
        "observational_UCI_coverage_claim": False,
        "confidence_per_seed_for_all_rounds_and_actions": 1 - delta,
        "seeds": list(seeds), "horizon": horizon, "sample_size": n,
        "risk_grid_spacing": spacing,
        "risk_radius": risk_radius,
        "mean_gap_radius": float(np.mean([row["gap_radius"] for row in rows])),
        "max_risk_error_at_played_actions": float(np.max([row["risk_error"] for row in rows])),
        "max_gap_error_at_played_actions": float(np.max([row["gap_error"] for row in rows])),
        "fraction_of_seeds_all_played_actions_covered": float(np.mean([r["all_played_actions_covered"] for r in per_seed])),
        "mean_fixed_005_risk_coverage": float(np.mean([r["fixed_risk_coverage"] for r in per_seed])),
        "mean_fixed_005_gap_coverage": float(np.mean([r["fixed_gap_coverage"] for r in per_seed])),
        "coverage_for_every_action_empirically_checked": False,
        "per_seed": per_seed,
    }
    (output / "controlled_resampling_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.output), indent=2))
