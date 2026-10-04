"""Conditional EXP-10 certificate against the predeclared neutral action.

The action-specific radii are valid only for past-measurable actions and
comparators under conditional within-batch IID sampling. The UCI chronology
does not establish that sampling condition.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from interval_pareto.algorithms import MidpointOGD
from interval_pareto.applications.streaming import ClassificationBatch, objective_and_subgradients


def action_radii(action: np.ndarray, n0: int, n1: int, horizon: int,
                 delta: float = 0.05) -> tuple[float, float]:
    """Two-sided risk and group-gap radii for a predictable action."""
    if action.shape != (24,) or np.any(action < 0) or np.any(action > 1):
        raise ValueError("expected a 24-dimensional unit-box action")
    if min(n0, n1, horizon) < 1 or not 0 < delta < 1:
        raise ValueError("invalid batch counts or confidence")
    theta = action[:7] - action[7:14]
    intercept = float(np.sum(action[14:19] - action[19:24]))
    amplitude = float(np.sum(np.abs(theta)))
    score_bound = amplitude + abs(intercept)
    risk = score_bound * math.sqrt(math.log(4 * horizon / delta) / (2 * (n0 + n1)))
    group_factor = math.sqrt(2 * math.log(8 * horizon / delta))
    gap = amplitude * group_factor * (1 / math.sqrt(n0) + 1 / math.sqrt(n1))
    return risk, gap


def run(data_path: Path, output: Path, eta: float = 0.05,
        delta: float = 0.05) -> dict:
    with np.load(data_path, allow_pickle=False) as data:
        x = np.asarray(data["features"], float)
        y = np.asarray(data["labels"], float)
        g = np.asarray(data["groups"], int)
        weights = np.asarray(data["weights"], float)
    if x.ndim != 3 or x.shape[2] != 24 or y.shape != x.shape[:2] or g.shape != y.shape:
        raise ValueError("EXP-10 arrays required")
    horizon, n, dimension = x.shape
    comparator = np.full(dimension, 0.5)
    learner = MidpointOGD(dimension, eta)
    rows = []
    gradient_square_sum = 0.0
    for t in range(horizon):
        action = learner.act()
        batch = ClassificationBatch(x[t], y[t], g[t])
        observed, gradients = objective_and_subgradients(batch, action)
        comparator_observed, _ = objective_and_subgradients(batch, comparator)
        n0 = int(np.sum(g[t] == 0))
        n1 = n - n0
        risk_radius, gap_radius = action_radii(action, n0, n1, horizon, delta)
        gradient = weights[t] @ gradients
        gradient_square_sum += float(gradient @ gradient)
        rows.append({
            "batch": t + 1, "n0": n0, "n1": n1,
            "observed_weighted_action": float(weights[t] @ observed),
            "observed_weighted_comparator": float(weights[t] @ comparator_observed),
            "risk_radius_at_action": risk_radius,
            "gap_radius_at_action": gap_radius,
            "weighted_transfer_radius": float(weights[t, 0] * risk_radius + weights[t, 1] * gap_radius),
            "gradient_squared_norm": float(gradient @ gradient),
        })
        learner.observe_gradient(gradient)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictable_comparator_rounds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    transfer = sum(row["weighted_transfer_radius"] for row in rows)
    optimization = eta * gradient_square_sum / 2  # comparator equals initial action
    summary = {
        "status": "conditional_predictable_comparator_certificate",
        "sampling_assumption": "current batch IID conditional on past; unverified for observational UCI chronology",
        "comparator": "predeclared neutral action w=(0.5,...,0.5)",
        "comparator_selected_using_current_or_future_data": False,
        "confidence_for_all_rounds_at_played_and_comparator_actions": 1 - delta,
        "horizon": horizon,
        "mean_risk_radius_at_played_action": float(np.mean([r["risk_radius_at_action"] for r in rows])),
        "mean_gap_radius_at_played_action": float(np.mean([r["gap_radius_at_action"] for r in rows])),
        "THM02_Q_T_bound": transfer,
        "realized_gradient_optimization_bound": optimization,
        "conditional_latent_regret_upper_bound": optimization + transfer,
        "observed_cumulative_action_minus_comparator": sum(
            r["observed_weighted_action"] - r["observed_weighted_comparator"] for r in rows),
        "observational_population_coverage_claim": False,
    }
    (output / "predictable_comparator_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.output), indent=2))
