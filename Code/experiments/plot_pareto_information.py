"""Plot the controlled THM-06/07 numerical check from its CSV output."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def run(table: Path, output: Path) -> None:
    with table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    selected = sorted((row for row in rows if float(row["amplitude"]) == 1.0),
                      key=lambda row: int(row["K"]))
    budgets = [int(row["K"]) for row in selected]
    axes[0].loglog(budgets, [float(row["mean_excess_over_oracle_grid"]) for row in selected],
                   "o-", label="Measured excess, blind grid")
    axes[0].loglog(budgets, [float(row["certified_continuous_excess_lower"]) for row in selected],
                   "d-.", label="Certified continuous excess")
    axes[0].loglog(budgets, [float(row["theorem_excess_lower_bound"]) for row in selected],
                   "s--", label="Theorem lower bound")
    axes[0].loglog(budgets, [float(row["theorem_uniform_upper_bound"]) for row in selected],
                   "^:", label="Theorem upper bound")
    axes[0].set(xlabel="Output budget K", ylabel="Excess coverage per round",
                title="Budget scaling (a=1)")
    axes[0].set_xticks(budgets, labels=[str(k) for k in budgets])
    axes[0].legend(frameon=False, fontsize=8)
    selected = sorted((row for row in rows if int(row["K"]) == 4),
                      key=lambda row: float(row["amplitude"]))
    amplitudes = [float(row["amplitude"]) for row in selected]
    axes[1].loglog(amplitudes, [float(row["mean_excess_over_oracle_grid"]) for row in selected],
                 "o-", label="Measured excess, blind grid")
    axes[1].loglog(amplitudes, [float(row["certified_continuous_excess_lower"]) for row in selected],
                 "d-.", label="Certified continuous excess")
    axes[1].loglog(amplitudes, [float(row["theorem_excess_lower_bound"]) for row in selected],
                 "s--", label="Theorem lower bound")
    axes[1].loglog(amplitudes, [float(row["theorem_uniform_upper_bound"]) for row in selected],
                 "^:", label="Theorem upper bound")
    axes[1].set(xlabel="Interval width a", ylabel="Excess coverage per round",
                title="Width scaling (K=4)")
    axes[1].set_xticks(amplitudes, labels=[str(a) for a in amplitudes])
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.table, args.output)
