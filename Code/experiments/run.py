"""Future config-driven experiment driver. Writing this file does not execute it."""

import argparse
import csv
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from interval_pareto.algorithms import MidpointOGD, UpperEndpointOGD, LowerEndpointOGD
from interval_pareto.pareto import additive_coverage
from interval_pareto.regret import evaluate
from interval_pareto.objectives import _weights
from interval_pareto.synthetic import SyntheticConfig, generate, weight_schedule


def _run_algorithm(rounds, weights, eta: float, name: str):
    d = rounds[0].d
    if name == "upper":
        learner = UpperEndpointOGD(d, eta)
    elif name == "lower":
        learner = LowerEndpointOGD(d, eta)
    elif name in ("midpoint", "static"):
        learner = MidpointOGD(d, eta)
    else:
        raise ValueError("unknown algorithm")
    actions = []
    for item, weight in zip(rounds, weights):
        action = learner.act()
        actions.append(action)
        if name == "static":
            continue
        if name == "upper":
            gradient = item.upper_gradient(action, weight)
        elif name == "lower":
            gradient = item.lower_gradient(action, weight)
        else:
            gradient = item.midpoint_gradient(action, weight)
        learner.observe_gradient(gradient)
    return np.asarray(actions)


def _front_proxy_1d(item, count: int = 101):
    """Exact efficient decision interval sampled on a grid, d=1 only."""
    if item.d != 1:
        raise ValueError("front proxy requires d=1")
    minima = np.array([item.latent_weighted_minimizer(np.eye(item.m)[j])[0] for j in range(item.m)])
    decisions = np.linspace(minima.min(), minima.max(), count)[:, None]
    return decisions, np.asarray([item.latent(x) for x in decisions])


def run_case(config: SyntheticConfig, eta: float, algorithm: str, front_grid: int = 101,
             preference_mode: str = "fixed", fixed_weights=None,
             record_trace: bool = True) -> tuple[dict, list[dict]]:
    if front_grid < 2:
        raise ValueError("front_grid must be at least two")
    rounds = generate(config)
    if preference_mode == "fixed":
        selected = (np.full(config.objectives, 1.0 / config.objectives) if fixed_weights is None
                    else _weights(np.asarray(fixed_weights, float), config.objectives))
        weights = np.tile(selected, (config.horizon, 1))
    elif preference_mode == "cycling":
        weights = weight_schedule(config.horizon, config.objectives)
    else:
        raise ValueError("preference_mode must be fixed or cycling")
    actions = _run_algorithm(rounds, weights, eta, algorithm)
    comparators = np.asarray([item.latent_weighted_minimizer(w) for item, w in zip(rounds, weights)])
    report = evaluate(rounds, actions, comparators, weights, eta)
    coverage = None
    if config.dimension == 1:
        coverage = float(sum(additive_coverage(_front_proxy_1d(item, front_grid)[1],
                                                item.latent(actions[t])[None, :])
                             for t, item in enumerate(rounds)))
    trace = []
    if record_trace:
        for t, (item, action, comparator, weight) in enumerate(zip(rounds, actions, comparators, weights)):
            action_interval = item.interval(action)
            comparator_interval = item.interval(comparator)
            trace.append({"t": t + 1, "action": json.dumps(action.tolist()),
                          "comparator": json.dumps(comparator.tolist()),
                          "weights": json.dumps(weight.tolist()),
                          "latent_weighted_action": float(weight @ item.latent(action)),
                          "latent_weighted_comparator": float(weight @ item.latent(comparator)),
                          "midpoint_weighted_action": float(weight @ item.midpoint(action)),
                          "midpoint_weighted_comparator": float(weight @ item.midpoint(comparator)),
                          "action_widths": json.dumps(action_interval.width.tolist()),
                          "comparator_widths": json.dumps(comparator_interval.width.tolist()),
                          "width_term": float(weight @ (action_interval.width + comparator_interval.width)) / 2.0})
    summary = {**asdict(config), "algorithm": algorithm, "preference_mode": preference_mode,
               "fixed_weights": weights[0].tolist() if preference_mode == "fixed" else None,
               "eta": eta, **asdict(report),
               "finite_grid_coverage": coverage, "bound_respected": bool(report.regret <= report.theorem_bound + 1e-9)
               if algorithm == "midpoint" else None}
    return summary, trace


def cases(base: SyntheticConfig) -> list[tuple[str, SyntheticConfig, str]]:
    """Small diagnostic matrix; full EXP-01–10 definitions live in kaggle/."""
    out = []
    for radius in [0.0, base.radius, 2.0 * base.radius]:
        out.append(("EXP-01" if radius == 0 else "EXP-02", replace(base, radius=radius), "midpoint"))
    for drift in [0.0, base.drift, min(0.5, 2.0 * base.drift)]:
        out.append(("EXP-03", replace(base, drift=drift), "midpoint"))
    for horizon in [max(1, base.horizon // 2), base.horizon, 2 * base.horizon]:
        out.append(("EXP-04", replace(base, horizon=horizon), "midpoint"))
    for radius in [0.0, base.radius]:
        for drift in [0.0, base.drift]:
            out.append(("EXP-05", replace(base, radius=radius, drift=drift), "midpoint"))
    for m in sorted({1, base.objectives, base.objectives + 1}):
        out.append(("EXP-06", replace(base, objectives=m), "midpoint"))
    for d in sorted({1, base.dimension, base.dimension + 1}):
        out.append(("EXP-07", replace(base, dimension=d), "midpoint"))
    out.append(("DIAG-single-action-front", replace(base, dimension=1, objectives=2), "midpoint"))
    for algorithm in ["midpoint", "upper", "lower", "static"]:
        out.append(("EXP-09", replace(base, width_slope_fraction=0.5), algorithm))
    out.append(("DIAG-theorem", base, "midpoint"))
    out.append(("DIAG-preference-cycling", base, "midpoint_cycling"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    base = SyntheticConfig(**settings["synthetic"])
    eta = float(settings["eta"])
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be positive")
    output_root = Path(settings["output_root"])
    if not output_root.is_absolute():
        output_root = Path(__file__).resolve().parents[2] / output_root
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-seed{base.seed}"
    destination = output_root / run_id
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    traces = []
    for experiment_id, config, algorithm in cases(base):
        preference_mode = "cycling" if algorithm == "midpoint_cycling" else "fixed"
        method = "midpoint" if algorithm == "midpoint_cycling" else algorithm
        summary, trace = run_case(config, eta, method, preference_mode=preference_mode)
        case_id = len(rows) + 1
        rows.append({"case_id": case_id, "experiment_id": experiment_id, **summary})
        traces.extend({"case_id": case_id, "experiment_id": experiment_id, "algorithm": method,
                       "preference_mode": preference_mode, **point}
                      for point in trace)
    with (destination / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (destination / "rounds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(traces[0]))
        writer.writeheader()
        writer.writerows(traces)
    (destination / "metadata.json").write_text(json.dumps({"config": settings, "run_id": run_id,
        "code_version": "unexecuted-development-v1",
        "note": "Only midpoint rows carry THM-02 bound validity; coverage is finite-grid diagnostic."}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
