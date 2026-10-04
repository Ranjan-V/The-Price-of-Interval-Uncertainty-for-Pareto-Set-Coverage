"""Controlled numerical check of THM-06/07's set-coverage information price.

This checks the theorem's exact quadratic family against a finite grid of
hidden worlds and a dense grid over each true Pareto interval. Explicit
Lipschitz corrections bound the continuous world and target errors.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


RADIUS = 1.0 / 20.0


def sampled_coverage(theta: np.ndarray, actions: np.ndarray, amplitude: float,
                     u_count: int = 101) -> np.ndarray:
    """Return grid approximation to THM-05 coverage for each hidden world."""
    offsets = np.linspace(-RADIUS, RADIUS, u_count)
    target = theta[:, None, None] + offsets[None, :, None]
    centers = np.stack((theta - RADIUS, theta + RADIUS), axis=-1)
    # actions has one row per world and a common set budget K.
    action = actions[:, None, :, None]
    target = target[:, :, :, None]
    centers = centers[:, None, None, :]
    deficit = amplitude * ((action - centers) ** 2 - (target - centers) ** 2)
    return np.maximum(deficit, 0.0).max(axis=-1).min(axis=-1).max(axis=-1)


def run(output: Path, world_count: int = 1024, u_count: int = 101,
        horizon: int = 128) -> dict:
    if world_count < 16 or world_count % 16 or u_count < 3 or u_count % 2 == 0:
        raise ValueError("world_count must be a multiple of 16; u_count must be odd")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    length = 1.0 - 2.0 * RADIUS
    theta = RADIUS + length * (np.arange(world_count) + 0.5) / world_count
    rows = []
    for amplitude in (0.125, 0.25, 0.5, 1.0, 2.0):
        for budget in (1, 2, 4, 8, 16):
            blind = RADIUS + length * (np.arange(budget) + 0.5) / budget
            blind_sets = np.broadcast_to(blind, (world_count, budget))
            universal = (np.arange(budget) + 0.5) / budget
            universal_sets = np.broadcast_to(universal, (world_count, budget))
            oracle_sets = theta[:, None] + RADIUS * (2 * (np.arange(budget) + 0.5) / budget - 1)
            blind_loss = sampled_coverage(theta, blind_sets, amplitude, u_count)
            universal_loss = sampled_coverage(theta, universal_sets, amplitude, u_count)
            oracle_loss = sampled_coverage(theta, oracle_sets, amplitude, u_count)
            excess = blind_loss - oracle_loss
            guaranteed = amplitude / (80 * budget)
            upper = amplitude / budget
            # For fixed theta and S, coverage as a function of target u is
            # 2a-Lipschitz. Every target is within RADIUS/(u_count-1) of a
            # sampled target. The blind-grid sampled coverage is 2a-Lipschitz
            # in theta; its midpoint quadrature mean error is a*length/(2N).
            target_error = 2 * amplitude * RADIUS / (u_count - 1)
            world_mean_error = amplitude * length / (2 * world_count)
            world_max_error = amplitude * length / world_count
            certified_excess = float(excess.mean() - target_error - world_mean_error)
            certified_universal_max = float(
                universal_loss.max() + target_error + world_max_error)
            # The oracle grid is feasible but need not be the true best K-set.
            # Replacing it by the true optimum only increases the excess.
            rows.append({
                "amplitude": amplitude, "K": budget, "T": horizon,
                "world_count": world_count, "u_count": u_count,
                "mean_blind_coverage": float(blind_loss.mean()),
                "mean_universal_grid_coverage": float(universal_loss.mean()),
                "max_universal_grid_coverage": float(universal_loss.max()),
                "theorem_uniform_upper_bound": upper,
                "certified_universal_grid_max_upper": certified_universal_max,
                "upper_numerical_check_holds": bool(certified_universal_max <= upper + 1e-12),
                "oracle_grid_coverage": float(oracle_loss.mean()),
                "mean_excess_over_oracle_grid": float(excess.mean()),
                "certified_continuous_excess_lower": certified_excess,
                "target_grid_error": target_error,
                "world_mean_quadrature_error": world_mean_error,
                "theorem_excess_lower_bound": guaranteed,
                "numerical_check_holds": bool(certified_excess + 1e-12 >= guaranteed),
                "mean_cumulative_excess": float(horizon * excess.mean()),
                "theorem_cumulative_lower_bound": horizon * guaranteed,
            })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "pareto_information_lower_bound.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "theorem": "THM-06 and THM-07",
        "protocol": "identical complete intervals [0,a]; hidden fixed translated Pareto interval",
        "metric": "THM-05 sup-inf-positive-part set coverage",
        "theoretical_statement": "U_T/(80K) <= minimax expected excess over optimal world-specific K-set <= U_T/K",
        "numerical_method": "midpoint world quadrature and Pareto-target grid with Lipschitz error certificates",
        "numerical_results_are_proof": False,
        "configurations": len(rows),
        "all_checked_configurations_meet_bound": all(row["numerical_check_holds"] for row in rows),
        "all_checked_configurations_meet_upper_bound": all(row["upper_numerical_check_holds"] for row in rows),
        "smallest_excess_to_bound_ratio": min(
            row["mean_excess_over_oracle_grid"] / row["theorem_excess_lower_bound"] for row in rows),
        "smallest_certified_excess_to_bound_ratio": min(
            row["certified_continuous_excess_lower"] / row["theorem_excess_lower_bound"] for row in rows),
        "mean_excess_to_bound_ratio": float(np.mean([
            row["mean_excess_over_oracle_grid"] / row["theorem_excess_lower_bound"] for row in rows])),
        "rows": rows,
    }
    (output / "pareto_information_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-count", type=int, default=1024)
    parser.add_argument("--u-count", type=int, default=101)
    parser.add_argument("--horizon", type=int, default=128)
    args = parser.parse_args()
    result = run(args.output, args.world_count, args.u_count, args.horizon)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
