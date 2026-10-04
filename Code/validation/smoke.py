"""One tiny CPU experiment, including deterministic files and a diagnostic figure."""

import argparse
import csv
import json
from pathlib import Path
import sys
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.run import run_case
from experiments.parallel_front import run_front_case
from interval_pareto.synthetic import SyntheticConfig, generate


def run(output: Path) -> float:
    started = time.perf_counter()
    config = SyntheticConfig(horizon=64, dimension=2, objectives=2, drift=0.1,
                             radius=0.05, width_slope_fraction=0.25, seed=17)
    summary, trace = run_case(config, eta=0.08, algorithm="midpoint", preference_mode="fixed")
    front = run_front_case(SyntheticConfig(horizon=24, dimension=1, objectives=2,
                                         drift=0.1, radius=0.05, width_slope_fraction=0.25,
                                         seed=17), denominator=4, eta=0.08, front_grid=41)
    assert summary["bound_respected"] and front["finite_grid_below_bound"]
    numeric = [summary[k] for k in ("regret", "path_variation", "two_location_width",
                                    "uniform_width", "theorem_bound")]
    assert all(np.isfinite(numeric)) and summary["two_location_width"] >= 0
    assert all(np.all((np.asarray(json.loads(row["action"])) >= 0) &
                      (np.asarray(json.loads(row["action"])) <= 1)) for row in trace)
    for item, row in zip(generate(config), trace):
        action = np.asarray(json.loads(row["action"]), float)
        interval = item.interval(action)
        latent = item.latent(action)
        assert np.all(interval.lower <= interval.upper)
        assert np.all(interval.lower <= latent + 1e-12)
        assert np.all(latent <= interval.upper + 1e-12)
        assert all(np.isfinite(row[key]) for key in ("latent_weighted_action",
            "latent_weighted_comparator", "midpoint_weighted_action",
            "midpoint_weighted_comparator", "width_term"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps({"scalar": summary, "front": front}, indent=2,
                                                    allow_nan=False), encoding="utf-8")
    with (output / "rounds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)
    losses = np.array([row["latent_weighted_action"] - row["latent_weighted_comparator"]
                       for row in trace])
    widths = np.array([row["width_term"] for row in trace])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    axes[0].plot(np.arange(1, len(losses) + 1), np.cumsum(losses), label="cumulative regret")
    axes[0].axhline(summary["theorem_bound"], color="black", linestyle="--", label="THM-02 horizon bound")
    axes[0].set(xlabel="round", ylabel="loss difference")
    axes[0].legend(fontsize=7)
    axes[1].plot(np.arange(1, len(widths) + 1), np.cumsum(widths), label="two-location width")
    axes[1].set(xlabel="round", ylabel="cumulative width term")
    axes[1].legend(fontsize=7)
    fig.savefig(output / "smoke_diagnostic.png", dpi=160)
    plt.close(fig)
    return time.perf_counter() - started


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    duration = run(args.output)
    print(json.dumps({"output": str(args.output), "duration_seconds": duration}))
