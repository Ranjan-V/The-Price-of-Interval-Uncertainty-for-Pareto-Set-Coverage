"""Real-data finite-pool, two-output Pareto coverage with exact label intervals.

Each chronological Bank Marketing batch is a fixed, hidden finite population.
One age>=40 log-loss objective depends on its positive-label rate; the second
objective is a known quadratic resource penalty. A uniform sample without
replacement reveals one containing log-loss interval at the right output.
The claim is exact *finite-pool* containment, not population generalization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_bank_marketing import load_rows


Q_MIN = 0.01
Q_MAX = 0.8
FRACTIONS = (0.5, 0.8, 0.95)
SEEDS = tuple(range(5))


def risk(q: float, p: float) -> float:
    return -p * math.log(q) - (1 - p) * math.log1p(-q)


def resource(q: float) -> float:
    return (q - Q_MIN) ** 2


def efficient_right(p: float) -> float:
    return max(Q_MIN, min(Q_MAX, p))


def coverage(pair: tuple[float, float], p: float) -> float:
    """Exact continuous worst-target two-output coverage on the efficient arc.

    On [Q_MIN,p], resource is increasing and log loss decreasing. Each
    output's resource-difference curve decreases in the target, and its
    log-loss-difference curve increases. All interior maxima of their lower
    envelope occur at one of four cross-objective equality targets.
    """
    hi = efficient_right(p)
    xs = tuple(sorted(pair))
    if any(x < Q_MIN - 1e-12 or x > Q_MAX + 1e-12 for x in xs):
        raise ValueError("output outside action domain")
    if hi == Q_MIN:
        targets = [Q_MIN]
    else:
        targets = [Q_MIN, hi]
        left = resource(Q_MIN) - risk(Q_MIN, p)
        right = resource(hi) - risk(hi, p)
        for x0 in xs:
            for x1 in xs:
                level = resource(x0) - risk(x1, p)
                if left < level < right:
                    targets.append(brentq(lambda u: resource(u) - risk(u, p) - level,
                                          Q_MIN, hi, xtol=1e-14))
    return max(min(max(resource(x) - resource(u), risk(x, p) - risk(u, p), 0.0)
                   for x in xs) for u in targets)


def oracle_pair(p: float) -> tuple[tuple[float, float], float]:
    """Solve the three-active-constraint K=2 optimum by nested scalar roots."""
    hi = efficient_right(p)
    if hi == Q_MIN:
        return (Q_MIN, Q_MIN), 0.0
    floor_risk = risk(hi, p)
    mid = brentq(lambda x: resource(x) - (risk(x, p) - floor_risk),
                 Q_MIN, hi, xtol=1e-14)
    e_single = resource(mid)

    def pair_for(e: float) -> tuple[float, float]:
        if e >= e_single - 1e-14:
            return mid, mid
        x0 = Q_MIN + math.sqrt(e)
        x1 = brentq(lambda x: risk(x, p) - floor_risk - e,
                    mid, hi, xtol=1e-14) if e > 0 else hi
        return x0, x1

    def crossing_loss(x0: float, x1: float) -> float:
        if x1 <= x0 + 1e-14:
            return 0.0
        u = brentq(lambda z: risk(x0, p) - risk(z, p)
                   - resource(x1) + resource(z), x0, x1, xtol=1e-14)
        return risk(x0, p) - risk(u, p)

    e_star = brentq(lambda e: crossing_loss(*pair_for(e)) - e,
                    0.0, e_single, xtol=1e-13)
    pair = pair_for(e_star)
    check = coverage(pair, p)
    if abs(check - e_star) > 1e-9:
        raise AssertionError((p, e_star, check))
    return pair, e_star


def feedback_interval(q: float, positives: int, sample_count: int,
                      pool_count: int) -> tuple[float, float, float, float]:
    """Deterministic finite-pool containment for all q, with no IID claim."""
    p_lo = positives / pool_count
    p_hi = (positives + pool_count - sample_count) / pool_count
    values = (risk(q, p_lo), risk(q, p_hi))
    return min(values), max(values), p_lo, p_hi


def run(data_path: Path, raw_path: Path | None, output: Path) -> dict:
    with np.load(data_path, allow_pickle=False) as data:
        y = np.asarray(data["labels"], float)
        groups = np.asarray(data["groups"], int)
    if y.shape != groups.shape or y.ndim != 2:
        raise ValueError("expected EXP-10 chronological batches")
    metadata = json.loads(data_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    source_sha = metadata["source_sha256"]
    prefix_rows = int(metadata["calibration_rows"])
    p_prior = (int(metadata["calibration_positive_count_age40plus"])
               / int(metadata["calibration_group_count_age40plus"]))
    if raw_path is not None and raw_path.exists():
        source_rows, observed_sha = load_rows(raw_path)
        prefix = source_rows[:prefix_rows]
        actual = [row["y"] == "yes" for row in prefix if int(row["age"]) >= 40]
        if observed_sha != source_sha or abs(float(np.mean(actual)) - p_prior) > 1e-15:
            raise ValueError("raw archive and prepared metadata do not match")
    blind_pair, _ = oracle_pair(p_prior)
    query_q = blind_pair[1]
    if abs(query_q - 0.5) < 1e-10:
        raise ValueError("query at 0.5 does not identify the label rate")
    records = []
    oracle_cache = {}
    containment_checks = 0
    oracle_checks = 0
    for t in range(y.shape[0]):
        labels = np.asarray(y[t, groups[t] == 1] > 0, int)
        count = len(labels)
        p_true = float(np.mean(labels))
        if p_true not in oracle_cache:
            oracle_cache[p_true] = oracle_pair(p_true)
        _, floor = oracle_cache[p_true]
        blind_cov = coverage(blind_pair, p_true)
        blind_excess = max(0.0, blind_cov - floor)
        for fraction in FRACTIONS:
            sample_count = max(1, math.ceil(fraction * count))
            for seed in SEEDS:
                rng = np.random.default_rng(1000003 * seed + 1009 * t + int(fraction * 100))
                draw = rng.choice(count, size=sample_count, replace=False)
                positives = int(labels[draw].sum())
                lo, hi, p_lo, p_hi = feedback_interval(query_q, positives,
                                                       sample_count, count)
                true_query = risk(query_q, p_true)
                if not lo - 1e-12 <= true_query <= hi + 1e-12:
                    raise AssertionError("finite-pool interval missed")
                containment_checks += 1
                # Recover only what this single interval reveals at query_q.
                slope = math.log1p(-query_q) - math.log(query_q)
                p_est = ((lo + hi) / 2 + math.log1p(-query_q)) / slope
                p_est = min(1.0, max(0.0, p_est))
                if abs(p_est - (p_lo + p_hi) / 2) > 1e-10:
                    raise AssertionError("interval inversion mismatch")
                adapted, _ = oracle_pair(p_est)
                adaptive_cov = coverage(adapted, p_true)
                adaptive_excess = max(0.0, adaptive_cov - floor)
                oracle_checks += 1
                records.append({
                    "batch": t + 1, "seed": seed, "sample_fraction": fraction,
                    "pool_size_age40plus": count, "sample_count": sample_count,
                    "population_positive_rate": p_true,
                    "interval_rate_lower": p_lo, "interval_rate_upper": p_hi,
                    "interval_query_lower": lo, "interval_query_upper": hi,
                    "interval_query_width": hi - lo,
                    "query_point": query_q, "oracle_floor": floor,
                    "blind_left": blind_pair[0], "blind_right": blind_pair[1],
                    "adaptive_left": adapted[0], "adaptive_right": adapted[1],
                    "blind_coverage": blind_cov, "adaptive_coverage": adaptive_cov,
                    "blind_excess": blind_excess, "adaptive_excess": adaptive_excess,
                    "adaptive_better": int(adaptive_excess < blind_excess - 1e-12),
                })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "bank_pareto_coverage_rounds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    by_fraction = []
    for fraction in FRACTIONS:
        subset = [r for r in records if r["sample_fraction"] == fraction]
        blind = float(np.mean([r["blind_excess"] for r in subset]))
        adaptive = float(np.mean([r["adaptive_excess"] for r in subset]))
        by_fraction.append({
            "sample_fraction": fraction,
            "mean_blind_excess": blind,
            "mean_adaptive_excess": adaptive,
            "adaptive_to_blind_ratio": adaptive / blind if blind else None,
            "fraction_adaptive_better": float(np.mean([r["adaptive_better"] for r in subset])),
            "mean_query_interval_width": float(np.mean([r["interval_query_width"] for r in subset])),
        })
    summary = {
        "protocol": "one post-output log-loss interval from age>=40 finite-pool labels sampled without replacement",
        "target": "exact finite-pool Pareto-set coverage, not superpopulation risk",
        "chronological_batches": int(y.shape[0]),
        "calibration_prefix_rows": prefix_rows,
        "calibration_positive_rate_age40plus": p_prior,
        "resource_objective": "(q-0.01)^2",
        "real_objective": "age>=40 Bernoulli log loss",
        "action_domain": [Q_MIN, Q_MAX],
        "output_budget": 2,
        "query_budget": "one objective-value interval at the right output point per batch",
        "finite_pool_containment": "deterministic for every action q; no sampling confidence parameter",
        "containment_checks": containment_checks,
        "oracle_checks": oracle_checks,
        "source_sha256": source_sha,
        "by_fraction": by_fraction,
        "limitation": "Each batch is a different hidden world; this is an application of the same metric and feedback format, not a test of THM-12's stationary symmetric-profile minimax rate.",
    }
    (output / "bank_pareto_coverage_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    xx = [100 * r["sample_fraction"] for r in by_fraction]
    ax.plot(xx, [r["mean_blind_excess"] for r in by_fraction], "o--",
            color="#6b7280", label="Calibration-prefix pair")
    ax.plot(xx, [r["mean_adaptive_excess"] for r in by_fraction], "o-",
            color="#136f63", label="Pair after one interval")
    ax.set_yscale("log")
    ax.set_xticks(xx)
    ax.set_xlabel("Age 40+ pool labels observed (%)")
    ax.set_ylabel("Mean worst-target excess coverage")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "bank_pareto_coverage.pdf")
    fig.savefig(output / "bank_pareto_coverage.png", dpi=170)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Code/data/bank_marketing/bank_stream.npz"))
    parser.add_argument("--raw", type=Path, default=Path("Code/data/bank_marketing/bank_marketing_uci.zip"))
    parser.add_argument("--output", type=Path, default=Path("Code/results/real_bank_pareto_coverage"))
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.raw, args.output), indent=2))
