"""Compute EXP-10 uniform population bands under conditional within-batch IID.

The UCI chronology does not verify this assumption. The script measures band
width, not population coverage, because the population targets are unknown.
See Math/streaming_population_intervals.md for the conditional proof.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


FEATURES = 7
SCORE_BOUND = 12.0
GAP_BOUND = 14.0


def loss_radius(n: int, horizon: int, delta_loss: float) -> tuple[float, float]:
    if n < 1 or horizon < 1 or not 0 < delta_loss < 1:
        raise ValueError("invalid certificate parameters")
    # Selection is deterministic from public sample size and confidence.
    spacings = np.geomspace(0.001, 1.0, 1000)
    best = (float("inf"), float("nan"))
    for spacing in spacings:
        log_net = (FEATURES * np.log(np.ceil(2 / spacing) + 1)
                   + np.log(np.ceil(10 / spacing) + 1))
        radius = SCORE_BOUND * np.sqrt((np.log(2 * horizon / delta_loss) + log_net) / (2 * n))
        radius += 8 * spacing
        if radius < best[0]:
            best = (float(radius), float(spacing))
    # Uniform Rademacher bound for the eight effective score parameters.
    # A one-sided symmetrization plus contraction bounds the expected
    # supremum by 2*(7+5)/sqrt(n); McDiarmid and a union over both tails
    # and all rounds contribute the second term. See APP-01 in Math/.
    complexity_radius = (2 * (FEATURES + 5) / np.sqrt(n)
                         + SCORE_BOUND * np.sqrt(np.log(2 * horizon / delta_loss) / (2 * n)))
    return min(SCORE_BOUND, best[0], float(complexity_radius)), best[1]


def gap_radius(n0: int, n1: int, horizon: int, delta_gap: float) -> float:
    if min(n0, n1) < 1:
        return GAP_BOUND
    if horizon < 1 or not 0 < delta_gap < 1:
        raise ValueError("invalid certificate parameters")
    log_term = np.log(4 * FEATURES * horizon / delta_gap)
    eps0 = np.sqrt(2 * log_term / n0)
    eps1 = np.sqrt(2 * log_term / n1)
    return min(GAP_BOUND, FEATURES * (eps0 + eps1))


def certify(data_path: Path, rounds_path: Path, output: Path, delta: float = 0.05) -> dict:
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0,1)")
    with np.load(data_path, allow_pickle=False) as data:
        groups = np.asarray(data["groups"], int)
        weights = np.asarray(data["weights"], float)
        shape = data["features"].shape
    if len(shape) != 3 or shape[2] != 24 or groups.shape != shape[:2] or weights.shape != (shape[0], 3):
        raise ValueError("This certificate requires the EXP-10 signed-pair feature map")
    with rounds_path.open(newline="", encoding="utf-8") as handle:
        rounds = list(csv.DictReader(handle))
    horizon, n, _ = shape
    if len(rounds) != horizon:
        raise ValueError("round count mismatch")
    risk_radius, spacing = loss_radius(n, horizon, delta / 2)
    rows = []
    for t in range(horizon):
        n0 = int(np.sum(groups[t] == 0))
        n1 = int(np.sum(groups[t] == 1))
        gap = gap_radius(n0, n1, horizon, delta / 2)
        center = np.asarray(json.loads(rounds[t]["midpoints"]), float)
        radius = np.array([risk_radius, gap, 0.0])
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError(f"invalid objective center at batch {t + 1}")
        rows.append({
            "batch": t + 1, "n0": n0, "n1": n1,
            "observed_logloss_center": float(center[0]),
            "observed_gap_center": float(center[1]),
            "risk_radius": float(radius[0]), "gap_radius": float(radius[1]),
            "resource_radius": 0.0,
            "weighted_half_width": float(weights[t] @ radius),
        })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "population_band_widths.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "conditional_certificate_only",
        "sampling_assumption": "current batch IID conditional on past; unverified for UCI Bank Marketing",
        "target": "conditional population objective functions, uniformly over all box actions",
        "simultaneous_confidence_if_assumption_holds": 1 - delta,
        "horizon": horizon, "batch_size": n,
        "risk_grid_spacing": spacing,
        "risk_radius_method": "minimum of grid-Hoeffding and uniform Rademacher-McDiarmid bounds",
        "risk_radius": risk_radius,
        "mean_gap_radius": float(np.mean([row["gap_radius"] for row in rows])),
        "max_gap_radius": float(np.max([row["gap_radius"] for row in rows])),
        "mean_observed_logloss_center": float(np.mean([row["observed_logloss_center"] for row in rows])),
        "mean_observed_gap_center": float(np.mean([row["observed_gap_center"] for row in rows])),
        "mean_weighted_half_width": float(np.mean([row["weighted_half_width"] for row in rows])),
        "THM02_U_T_upper_bound": float(2 * np.sum([row["weighted_half_width"] for row in rows])),
        "empirical_population_coverage_measurable": False,
    }
    if np.allclose(weights, [0.8, 0.15, 0.05]):
        gradient_bound = 0.85 * math.sqrt(24) + 0.15 * 2 * math.sqrt(14)
        ogd_static_part = 24 / (2 * 0.05) + 0.05 * gradient_bound**2 * horizon / 2
        summary["static_comparator_envelope_at_eta_0.05"] = {
            "gradient_norm_bound": gradient_bound,
            "optimization_part": ogd_static_part,
            "total_with_U_T": ogd_static_part + summary["THM02_U_T_upper_bound"],
            "interpretation": "conditional worst-case bound, not measured regret",
        }
    (output / "population_band_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--rounds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delta", type=float, default=0.05)
    args = parser.parse_args()
    print(json.dumps(certify(args.data, args.rounds, args.output, args.delta), indent=2))
