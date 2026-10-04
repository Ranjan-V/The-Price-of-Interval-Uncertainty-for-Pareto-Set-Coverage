"""Analytic sanity cases SANITY-01 through SANITY-06; CPU only."""

import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from interval_pareto.intervals import IntervalArray, lu_dominates, weak_lu_dominates
from interval_pareto.pareto import dominates, efficient_mask
from interval_pareto.objectives import QuadraticIntervalRound
from interval_pareto.regret import evaluate
from interval_pareto.synthetic import SyntheticConfig, generate
from interval_pareto.variation import path_variation


def check() -> dict:
    results = {}
    zero = generate(SyntheticConfig(horizon=8, dimension=2, objectives=2, drift=0, radius=0, seed=7))
    weight = np.array([0.5, 0.5])
    actions = np.tile([0.5, 0.5], (8, 1))
    comparators = np.asarray([item.latent_weighted_minimizer(weight) for item in zero])
    report = evaluate(zero, actions, comparators, np.tile(weight, (8, 1)), eta=0.1)
    assert report.uniform_width == 0 and report.two_location_width == 0
    assert all(np.array_equal(item.interval(actions[t]).lower, item.latent(actions[t]))
               and np.array_equal(item.interval(actions[t]).upper, item.latent(actions[t]))
               for t, item in enumerate(zero))
    results["SANITY-01"] = {"pass": True, "U_T": report.uniform_width, "Q_T": report.two_location_width}
    assert report.path_variation == 0
    results["SANITY-02"] = {"pass": True, "P_T": report.path_variation}
    assert np.isclose(report.regret, report.midpoint_regret)
    results["SANITY-03"] = {"pass": True, "regret_equals_midpoint": True}

    widths = [0.0, 0.01, 0.02, 0.05, 0.10]
    measured = []
    for width in widths:
        radius = width / 2.0
        rounds = generate(SyntheticConfig(horizon=8, dimension=2, objectives=2, drift=0,
                                          radius=radius, width_slope_fraction=0, seed=7))
        comparators = np.asarray([item.latent_weighted_minimizer(weight) for item in rounds])
        row = evaluate(rounds, actions, comparators, np.tile(weight, (8, 1)), eta=0.1)
        np.testing.assert_allclose(row.uniform_width, 8 * width, atol=1e-12)
        np.testing.assert_allclose(row.two_location_width, 8 * width, atol=1e-12)
        measured.append({"width": width, "U_T": row.uniform_width, "Q_T": row.two_location_width})
    results["SANITY-04"] = {"pass": True, "values": measured}

    delta, horizon = 0.1, 6
    moving = [QuadraticIntervalRound(np.array([[0.1 + delta * t], [0.1 + delta * t]]),
                                     np.zeros(2), np.zeros((2, 1))) for t in range(horizon)]
    moving_path = np.asarray([item.latent_weighted_minimizer(weight) for item in moving])
    actual_path = path_variation(moving_path)
    np.testing.assert_allclose(actual_path, delta * (horizon - 1), atol=1e-12)
    results["SANITY-05"] = {"pass": True, "P_T": actual_path, "expected": delta * (horizon - 1)}

    values = np.array([[0, 1], [1, 0], [1, 1], [0, 0]], float)
    np.testing.assert_array_equal(efficient_mask(values), [False, False, False, True])
    assert dominates(values[3], values[2]) and not dominates(values[0], values[1])
    ia = IntervalArray(np.array([0.0, 1.0]), np.array([2.0, 2.0]))
    ib = IntervalArray(np.array([1.0, 1.0]), np.array([3.0, 3.0]))
    assert lu_dominates(ia, ib) and not weak_lu_dominates(ia, ib)
    results["SANITY-06"] = {"pass": True, "finite_efficient_count": 1}
    return results


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "results" / "sanity" / "sanity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(check(), indent=2), encoding="utf-8")
    print(output)
