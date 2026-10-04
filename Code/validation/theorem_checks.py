"""Small CPU numerical checks of the exact THM-02 and THM-05 inequalities."""

import csv
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.run import run_case
from experiments.parallel_front import run_front_case
from experiments.hidden_world import regret_pair
from interval_pareto.synthetic import SyntheticConfig


def check() -> list[dict]:
    rows = []
    for seed in [3, 17]:
        for radius in [0.0, 0.02, 0.10]:
            for drift in [0.0, 0.1]:
                config = SyntheticConfig(horizon=32, dimension=2, objectives=2,
                                         radius=radius, drift=drift,
                                         width_slope_fraction=0.25, seed=seed)
                summary, _ = run_case(config, eta=0.08, algorithm="midpoint")
                assert summary["regret"] <= summary["theorem_bound"] + 1e-9
                assert summary["two_location_width"] <= summary["uniform_width"] + 1e-9
                rows.append({"claim": "THM-02", "seed": seed, "radius": radius,
                             "drift": drift, "lhs": summary["regret"],
                             "online": summary["online_term"], "movement": summary["drift_term"],
                             "gradient": summary["gradient_term"],
                             "uncertainty": summary["two_location_width"],
                             "rhs": summary["theorem_bound"], "pass": True})
    for seed in [3, 17]:
        front = run_front_case(SyntheticConfig(horizon=8, dimension=1, objectives=2,
                                              radius=0.05, drift=0.1, seed=seed),
                               denominator=3, eta=0.08, front_grid=21)
        assert front["finite_grid_below_bound"]
        rows.append({"claim": "THM-05 sampled metric", "seed": seed, "radius": 0.05,
                     "drift": 0.1, "lhs": front["finite_grid_coverage"],
                     "online": "", "movement": "", "gradient": "", "uncertainty": "",
                     "rhs": front["thm05_bound"], "pass": True})
    for width in [0.0, 0.01, 0.1]:
        plus, minus, uncertainty = regret_pair(np.full(8, 0.25), np.full(8, width))
        assert np.isclose(plus + minus, uncertainty)
        rows.append({"claim": "THM-03 pair identity", "seed": "", "radius": width / 2,
                     "drift": 0.0, "lhs": max(plus, minus), "online": "", "movement": "",
                     "gradient": "", "uncertainty": uncertainty,
                     "rhs": uncertainty / 2, "pass": max(plus, minus) >= uncertainty / 2 - 1e-12})
    return rows


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "results" / "theorem_checks"
    output.mkdir(parents=True, exist_ok=True)
    rows = check()
    with (output / "checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps({"total": len(rows),
        "passed": sum(bool(row["pass"]) for row in rows), "claims": sorted(set(row["claim"] for row in rows))}, indent=2), encoding="utf-8")
    print(json.dumps({"total": len(rows), "passed": sum(bool(row["pass"]) for row in rows)}))
