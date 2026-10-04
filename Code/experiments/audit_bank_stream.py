"""EXP-10 causal replay and empirical next-batch interval audit.

The interval target is the *observed next-batch empirical objective at the
chosen action*. This is distinct from an unobserved population objective.
Chronological residual quantiles are descriptive under this drifting stream;
no exchangeability or time-uniform coverage guarantee is asserted.
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


PREFERENCES = {
    "jcam_80_15_05": np.array([0.80, 0.15, 0.05]),
    "logistic_only": np.array([1.0, 0.0, 0.0]),
    "gap_heavy_65_30_05": np.array([0.65, 0.30, 0.05]),
    "resource_heavy_70_10_20": np.array([0.70, 0.10, 0.20]),
}


def _quantile(past: list[float], alpha: float) -> float:
    # Split-conformal order statistic applied to chronological residuals as
    # a *heuristic*. The order statistic alone does not give validity here.
    n = len(past)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.partition(np.asarray(past), min(rank, n) - 1)[min(rank, n) - 1])


def audit(data_path: Path, rounds_path: Path, output: Path, eta: float = 0.05,
          warmup: int = 20, alpha: float = 0.10) -> dict:
    if eta <= 0 or warmup < 1 or not 0 < alpha < 1:
        raise ValueError("invalid eta, warmup, or alpha")
    with np.load(data_path, allow_pickle=False) as data:
        x = np.asarray(data["features"], float)
        y = np.asarray(data["labels"], float)
        g = np.asarray(data["groups"], int)
        supplied_radii = np.asarray(data["radii"], float)
    with rounds_path.open(newline="", encoding="utf-8") as handle:
        original = list(csv.DictReader(handle))
    if len(original) != len(x) or warmup >= len(x):
        raise ValueError("round count mismatch or warmup exceeds stream")

    learners = {name: MidpointOGD(x.shape[2], eta) for name in PREFERENCES}
    residuals: list[list[float]] = [[], [], []]
    rows: list[dict] = []
    max_replay_action_error = 0.0
    for t in range(len(x)):
        batch = ClassificationBatch(x[t], y[t], g[t])
        for name, learner in learners.items():
            action = learner.act()
            values, gradients = objective_and_subgradients(batch, action)
            scores = x[t] @ action
            probability = 1 / (1 + np.exp(-np.clip(scores, -700, 700)))
            row = {
                "batch": t + 1, "method": name,
                "logloss": float(values[0]), "score_gap": float(values[1]),
                "resource": float(values[2]),
                "accuracy": float(np.mean((scores >= 0) == (y[t] > 0))),
                "brier": float(np.mean((probability - (y[t] > 0)) ** 2)),
            }
            if name == "jcam_80_15_05":
                logged = np.asarray(json.loads(original[t]["action"]), float)
                max_replay_action_error = max(max_replay_action_error, float(np.max(np.abs(action - logged))))
                if max_replay_action_error > 1e-9:
                    raise ValueError(f"Default replay differs from logged action at batch {t + 1}")
                if t > 0:
                    previous = ClassificationBatch(x[t - 1], y[t - 1], g[t - 1])
                    forecast, _ = objective_and_subgradients(previous, action)
                    error = np.abs(values - forecast)
                    row.update({f"forecast_{j}": float(forecast[j]) for j in range(3)})
                    row.update({f"error_{j}": float(error[j]) for j in range(3)})
                    if t >= warmup:
                        radius = np.array([_quantile(residuals[j], alpha) for j in range(3)])
                        row.update({f"radius_{j}": float(radius[j]) for j in range(3)})
                        row.update({f"covered_{j}": int(error[j] <= radius[j] + 1e-12) for j in range(3)})
                        row.update({f"fixed_covered_{j}": int(error[j] <= supplied_radii[t, j] + 1e-12) for j in range(3)})
                        row["all_covered"] = int(bool(np.all(error <= radius + 1e-12)))
                    for j in range(3):
                        residuals[j].append(float(error[j]))
            rows.append(row)
            learner.observe_gradient(PREFERENCES[name] @ gradients)

    output.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (output / "replay_and_intervals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    methods = {}
    for name in PREFERENCES:
        subset = [row for row in rows if row["method"] == name]
        methods[name] = {metric: float(np.mean([row[metric] for row in subset]))
                         for metric in ("logloss", "score_gap", "resource", "accuracy", "brier")}
    calibrated = [row for row in rows if row["method"] == "jcam_80_15_05" and "all_covered" in row]
    summary = {
        "target": "next observed batch empirical objectives at the current action",
        "forecast": "previous observed batch objectives evaluated at the current action",
        "radius_rule": "expanding past absolute forecast-error 90th percentile order statistic",
        "nominal_marginal_coverage": 1 - alpha,
        "warmup_batches": warmup,
        "evaluated_batches": len(calibrated),
        "guaranteed_population_or_sequential_coverage": False,
        "methods": methods,
        "max_replay_action_error": max_replay_action_error,
        "calibration": {
            f"objective_{j}": {
                "empirical_coverage": float(np.mean([row[f"covered_{j}"] for row in calibrated])),
                "fixed_band_coverage": float(np.mean([row[f"fixed_covered_{j}"] for row in calibrated])),
                "mean_radius": float(np.mean([row[f"radius_{j}"] for row in calibrated])),
                "mean_absolute_forecast_error": float(np.mean([row[f"error_{j}"] for row in calibrated])),
            } for j in range(3)
        },
        "joint_empirical_coverage": float(np.mean([row["all_covered"] for row in calibrated])),
    }
    (output / "audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--rounds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.data, args.rounds, args.output), indent=2))
