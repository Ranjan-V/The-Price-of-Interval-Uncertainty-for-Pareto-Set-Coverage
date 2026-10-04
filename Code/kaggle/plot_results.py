"""Aggregate designated seeds; produce figures only when matching results exist."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_results import collect
from common import IMPLEMENTATION_VERSION


def plot(output_root: Path) -> list[str]:
    per_seed, grouped = collect(output_root)
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    produced = []
    specs = [("EXP-02", "width", "Regret versus interval width", "regret_mean", "uncertainty.png"),
             ("EXP-02", "U_T_mean", "Regret versus cumulative uncertainty", "regret_mean", "regret_vs_U_T.png"),
             ("EXP-03", "P_T_mean", "Regret versus comparator movement", "regret_mean", "drift.png"),
             ("EXP-04", "horizon", "Regret versus horizon", "regret_mean", "horizon.png"),
             ("EXP-08", "horizon", "Pareto coverage", "coverage_mean", "pareto.png")]
    for experiment, xkey, title, ykey, filename in specs:
        rows = [r for r in grouped if r["experiment_id"] == experiment and r["preference_id"] in ("uniform", "grid2", "front_grid")
                and r[ykey] is not None]
        if not rows:
            continue
        rows.sort(key=lambda r: r[xkey])
        fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
        errors = [r[ykey.replace("_mean", "_ci95_halfwidth")] for r in rows]
        if all(error is not None for error in errors):
            ax.errorbar([r[xkey] for r in rows], [r[ykey] for r in rows], yerr=errors,
                        marker="o", capsize=3)
        else:
            ax.plot([r[xkey] for r in rows], [r[ykey] for r in rows], marker="o")
        ax.set(xlabel=xkey, ylabel=ykey, title=title)
        if experiment == "EXP-04" and len(rows) >= 3:
            positive = [(float(r[xkey]), float(r[ykey])) for r in rows if r[xkey] > 0 and r[ykey] > 0]
            if len(positive) >= 3:
                slope = float(np.polyfit(np.log([p[0] for p in positive]),
                                         np.log([p[1] for p in positive]), 1)[0])
                ax.text(0.03, 0.95, f"descriptive log-log slope: {slope:.2f}",
                        transform=ax.transAxes, va="top", fontsize=8)
        fig.savefig(figures / filename, dpi=220)
        plt.close(fig)
        produced.append(filename)
    joint = [r for r in grouped if r["experiment_id"] == "EXP-05" and r["preference_id"] in ("uniform", "grid2")]
    if joint:
        widths = sorted(set(r["width"] for r in joint))
        drifts = sorted(set(r["drift"] for r in joint))
        matrix = np.full((len(drifts), len(widths)), np.nan)
        for r in joint:
            matrix[drifts.index(r["drift"]), widths.index(r["width"])] = r["regret_mean"]
        fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
        im = ax.imshow(matrix, aspect="auto", origin="lower")
        ax.set(xticks=range(len(widths)), yticks=range(len(drifts)),
               xticklabels=widths, yticklabels=drifts, xlabel="interval width", ylabel="drift")
        fig.colorbar(im, ax=ax, label="mean cumulative regret")
        fig.savefig(figures / "joint_heatmap.png", dpi=220)
        plt.close(fig)
        produced.append("joint_heatmap.png")
    ablation = [r for r in grouped if r["experiment_id"] == "EXP-09" and
                r["preference_id"] in ("uniform", "grid2") and r["regret_mean"] is not None]
    if ablation:
        ablation.sort(key=lambda r: r["algorithm"])
        fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
        ax.bar([r["algorithm"] for r in ablation], [r["regret_mean"] for r in ablation])
        ax.set(ylabel="mean cumulative regret", title="Endpoint and static ablations")
        fig.savefig(figures / "ablations.png", dpi=220)
        plt.close(fig)
        produced.append("ablations.png")
    bound_rows = [r for r in per_seed if r["regret"] is not None and r["bound"] is not None
                  and r["algorithm"] == "midpoint"]
    if bound_rows:
        fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
        ax.scatter([r["bound"] for r in bound_rows], [r["regret"] for r in bound_rows], s=12, alpha=0.55)
        ax.set(xlabel="THM-02 bound", ylabel="observed cumulative regret",
               title="Regret versus analytic upper bound")
        fig.savefig(figures / "regret_vs_bound.png", dpi=220)
        plt.close(fig)
        produced.append("regret_vs_bound.png")
    front_paths = []
    for path in sorted((output_root / "synthetic" / "EXP-08").glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        if (entry.get("status") == "success" and
            entry.get("config", {}).get("implementation_version") == IMPLEMENTATION_VERSION and
            entry.get("metrics", {}).get("front_trace")):
            front_paths.append(entry)
    if front_paths:
        selected = min(front_paths, key=lambda entry: entry["config"]["synthetic"]["seed"])
        trace = selected["metrics"]["front_trace"]
        time = np.array([row["t"] for row in trace])
        lows = np.array([row["front_min"] for row in trace])
        highs = np.array([row["front_max"] for row in trace])
        actions = np.asarray([row["actions"] for row in trace])
        fig, ax = plt.subplots(figsize=(6.2, 3.7), constrained_layout=True)
        ax.fill_between(time, lows, highs, alpha=0.22, label="exact efficient interval")
        for k in range(actions.shape[1]):
            ax.plot(time, actions[:, k], linewidth=0.7, alpha=0.45)
        ax.set(xlabel="round", ylabel="decision in [0,1]",
               title=f"Front tracking illustration, prespecified seed {selected['config']['synthetic']['seed']}")
        ax.legend(fontsize=7)
        fig.savefig(figures / "front_tracking.png", dpi=220)
        plt.close(fig)
        produced.append("front_tracking.png")
    return produced


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(plot(args.output_root))
