"""Compare the original and sharpened THM-05 bounds on parallel-grid runs."""

import csv
import json
from pathlib import Path

from parallel_front import run_front_case

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from interval_pareto.synthetic import SyntheticConfig


def run(output: Path) -> dict:
    rows = []
    for seed in (3, 17):
        for denominator in (3, 7, 15):
            case = run_front_case(
                SyntheticConfig(horizon=32, dimension=1, objectives=2,
                                radius=0.05, drift=0.1, seed=seed),
                denominator=denominator, eta=0.08, front_grid=101)
            rows.append({
                "seed": seed, "K": case["grid_size"],
                "sampled_coverage": case["finite_grid_coverage"],
                "previous_bound": case["thm05_previous_bound"],
                "sharpened_bound": case["thm05_bound"],
                "improvement_factor": case["thm05_bound_improvement_factor"],
                "sampled_below_sharpened": case["finite_grid_below_bound"],
                "static_grid_sampled_coverage": case["static_grid_sampled_coverage"],
                "static_metric_net_bound": case["static_metric_net_bound"],
                "static_sampled_below_metric_bound": case["static_sampled_below_metric_bound"],
            })
    summary = {
        "cases": len(rows),
        "all_sampled_below_sharpened": all(r["sampled_below_sharpened"] for r in rows),
        "all_static_sampled_below_metric_bound": all(r["static_sampled_below_metric_bound"] for r in rows),
        "minimum_improvement_factor": min(r["improvement_factor"] for r in rows),
        "maximum_improvement_factor": max(r["improvement_factor"] for r in rows),
        "note": "Sampled coverage underestimates the continuous supremum; proof establishes the bound.",
        "rows": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "thm05_sharpening.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "thm05_sharpening_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = run(Path(__file__).resolve().parents[1] / "results" / "thm05_sharpening")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
