"""Check the predictable-comparator certificate with known finite pools."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from interval_pareto.algorithms import MidpointOGD
from interval_pareto.applications.streaming import ClassificationBatch, objective_and_subgradients
from predictable_comparator_certificate import action_radii


def run(data_path: Path, output: Path, seeds: tuple[int, ...] = tuple(range(10)),
        eta: float = 0.05, delta: float = 0.05) -> dict:
    with np.load(data_path, allow_pickle=False) as data:
        x = np.asarray(data["features"], float)
        y = np.asarray(data["labels"], float)
        g = np.asarray(data["groups"], int)
        weights = np.asarray(data["weights"], float)
    horizon, n, dimension = x.shape
    comparator = np.full(dimension, 0.5)
    results = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        learner = MidpointOGD(dimension, eta)
        risk_hits = []
        gap_hits = []
        transfer = 0.0
        optimization = 0.0
        latent_regret = 0.0
        for t in range(horizon):
            action = learner.act()
            draw = rng.integers(0, n, size=n)
            sample = ClassificationBatch(x[t, draw], y[t, draw], g[t, draw])
            pool = ClassificationBatch(x[t], y[t], g[t])
            n0 = int(np.sum(sample.groups == 0))
            n1 = n - n0
            if min(n0, n1) < 1:
                raise RuntimeError("empty sampled age group")
            observed, gradients = objective_and_subgradients(sample, action)
            population, _ = objective_and_subgradients(pool, action)
            neutral, _ = objective_and_subgradients(pool, comparator)
            rr, gr = action_radii(action, n0, n1, horizon, delta)
            risk_hits.append(abs(observed[0] - population[0]) <= rr + 1e-12)
            gap_hits.append(abs(observed[1] - population[1]) <= gr + 1e-12)
            transfer += weights[t, 0] * rr + weights[t, 1] * gr
            gradient = weights[t] @ gradients
            optimization += eta * float(gradient @ gradient) / 2
            latent_regret += float(weights[t] @ (population - neutral))
            learner.observe_gradient(gradient)
        results.append({
            "seed": seed,
            "risk_covered": int(sum(risk_hits)),
            "gap_covered": int(sum(gap_hits)),
            "joint_all_rounds_covered": bool(all(risk_hits) and all(gap_hits)),
            "population_regret_vs_neutral": latent_regret,
            "conditional_bound": optimization + transfer,
            "bound_holds_for_realized_pool": bool(latent_regret <= optimization + transfer + 1e-10),
        })
    summary = {
        "protocol": "IID-with-replacement feedback from each fixed chronological pool",
        "APP01_sampling_assumption_satisfied_by_design": True,
        "confidence_per_seed_simultaneously_over_rounds_at_played_actions": 1 - delta,
        "seeds": list(seeds), "horizon": horizon,
        "all_seeds_all_played_actions_covered": all(r["joint_all_rounds_covered"] for r in results),
        "all_realized_pool_regret_bounds_hold": all(r["bound_holds_for_realized_pool"] for r in results),
        "per_seed": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictable_controlled_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.output), indent=2))
