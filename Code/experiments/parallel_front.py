"""Future THM-05 diagnostic on the exactly characterized one-dimensional front."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from interval_pareto.algorithms import ParallelMidpointOGD, simplex_lattice
from interval_pareto.pareto import additive_coverage
from interval_pareto.regret import evaluate
from interval_pareto.synthetic import SyntheticConfig, generate


def run_front_case(config: SyntheticConfig, denominator: int, eta: float,
                   front_grid: int = 201, record_trace: bool = False) -> dict:
    if config.dimension != 1 or config.objectives < 2 or front_grid < 2:
        raise ValueError("one-dimensional multiobjective case and front grid >=2 required")
    rounds = generate(config)
    weights = simplex_lattice(config.objectives, denominator)
    learner = ParallelMidpointOGD(weights, 1, eta)
    actions = []
    coverage = 0.0
    static_coverage = 0.0
    front_trace = []
    static_set = ((np.arange(len(weights)) + 0.5) / len(weights))[:, None]
    for item in rounds:
        selected = learner.act()
        actions.append(selected)
        individual_minima = np.array([item.latent_weighted_minimizer(np.eye(item.m)[j])[0]
                                       for j in range(item.m)])
        front_decisions = np.linspace(individual_minima.min(), individual_minima.max(), front_grid)[:, None]
        front_values = np.asarray([item.latent(x) for x in front_decisions])
        set_values = np.asarray([item.latent(x) for x in selected])
        static_values = np.asarray([item.latent(x) for x in static_set])
        coverage += additive_coverage(front_values, set_values)
        static_coverage += additive_coverage(front_values, static_values)
        if record_trace:
            front_trace.append({"t": len(actions), "front_min": float(individual_minima.min()),
                                "front_max": float(individual_minima.max()),
                                "actions": selected[:, 0].tolist()})
        learner.observe(item, selected)
    history = np.asarray(actions)  # T by K by d
    bounds = []
    for k, w in enumerate(weights):
        comparators = np.asarray([item.latent_weighted_minimizer(w) for item in rounds])
        repeated_weights = np.tile(w, (config.horizon, 1))
        bounds.append(evaluate(rounds, history[:, k, :], comparators, repeated_weights, eta).theorem_bound)
    minimum_radii = [item.radii - 0.5 * np.sum(np.abs(item.width_slopes), axis=1)
                     for item in rounds]
    max_min_radius = max(float(np.max(radius)) for radius in minimum_radii)
    lipschitz = 2.0 + 2.0 * max_min_radius  # d=1, bound on latent derivative
    stability = max(float(np.max(np.abs(item.centers[:, 0] - radius * item.biases[:, 0])))
                    for item, radius in zip(rounds, minimum_radii))
    # Clipped weighted-center map is l1-Lipschitz with this constant.
    delta = 2.0 * (config.objectives - 1) / denominator
    mu = 2.0
    representation = lipschitz * stability * delta * config.horizon
    previous_bound = representation + lipschitz * sum(
        np.sqrt(2.0 * config.horizon * bound / mu) for bound in bounds)
    theorem_bound = representation + lipschitz * np.sqrt(
        2.0 * config.horizon * sum(bounds) / mu)
    static_metric_bound = lipschitz * config.horizon / (2.0 * len(weights))
    result = {**asdict(config), "denominator": denominator, "grid_size": len(weights),
            "eta": eta, "finite_grid_coverage": coverage, "thm05_bound": theorem_bound,
            "thm05_previous_bound": previous_bound,
            "thm05_bound_improvement_factor": previous_bound / theorem_bound,
            "static_grid_sampled_coverage": static_coverage,
            "static_metric_net_bound": static_metric_bound,
            "static_sampled_below_metric_bound": bool(static_coverage <= static_metric_bound + 1e-9),
            "finite_grid_below_bound": bool(coverage <= theorem_bound + 1e-9),
            "note": "Finite front sampling underestimates the continuous supremum; theorem bound is continuous."}
    if record_trace:
        result["front_trace"] = front_trace
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--denominator", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    config = replace(SyntheticConfig(**settings["synthetic"]), dimension=1)
    result = run_front_case(config, args.denominator, float(settings["eta"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
