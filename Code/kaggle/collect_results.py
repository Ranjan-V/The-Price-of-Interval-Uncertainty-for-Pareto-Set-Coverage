"""Collect per-run checkpoints and per-seed aggregate statistics."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics
import math
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import IMPLEMENTATION_VERSION


def collect(output_root: Path) -> tuple[list[dict], list[dict]]:
    records = []
    for path in sorted((output_root / "synthetic").glob("EXP-*/*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "success":
            continue
        cfg, metrics = data["config"], data["metrics"]
        if cfg.get("implementation_version") != IMPLEMENTATION_VERSION:
            continue
        synthetic = cfg["synthetic"]
        records.append({"run_id": data["run_id"], "experiment_id": cfg["experiment_id"],
                        "seed": synthetic["seed"], "algorithm": cfg["algorithm"],
                        "preference_id": cfg["preference_id"],
                        "weights": json.dumps(cfg["weights"]),
                        "horizon": synthetic["horizon"], "dimension": synthetic["dimension"],
                        "objectives": synthetic["objectives"], "drift": synthetic["drift"],
                        "width": 2 * synthetic["radius"],
                        "regret": metrics.get("regret"),
                        "average_regret": (metrics["regret"] / synthetic["horizon"]
                                           if metrics.get("regret") is not None else None),
                        "P_T": metrics.get("path_variation"), "Q_T": metrics.get("two_location_width"),
                        "U_T": metrics.get("uniform_width"),
                        "coverage": metrics.get("finite_grid_coverage"),
                        "bound": metrics.get("theorem_bound", metrics.get("thm05_bound")),
                        "runtime_seconds": data.get("runtime_seconds"), "device": data.get("device")})
    groups = defaultdict(list)
    for row in records:
        key = tuple(row[k] for k in ("experiment_id", "algorithm", "preference_id", "horizon",
                                     "dimension", "objectives", "drift", "width"))
        groups[key].append(row)
    aggregated = []
    for key, items in groups.items():
        out = dict(zip(("experiment_id", "algorithm", "preference_id", "horizon",
                        "dimension", "objectives", "drift", "width"), key))
        out["n_seeds"] = len(items)
        for field in ["regret", "average_regret", "P_T", "Q_T", "U_T", "coverage", "bound", "runtime_seconds"]:
            values = [float(row[field]) for row in items if row[field] is not None]
            out[f"{field}_mean"] = statistics.mean(values) if values else None
            out[f"{field}_sd"] = statistics.stdev(values) if len(values) > 1 else None
            out[f"{field}_median"] = statistics.median(values) if values else None
            out[f"{field}_ci95_halfwidth"] = (1.96 * statistics.stdev(values) / len(values) ** 0.5
                                                if len(values) > 1 else None)
        aggregated.append(out)
    return records, aggregated


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def worst_preferences(aggregate: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in aggregate:
        if row["regret_mean"] is not None:
            key = tuple(row[k] for k in ("experiment_id", "algorithm", "horizon", "dimension",
                                         "objectives", "drift", "width"))
            groups[key].append(row)
    output = []
    for key, rows in groups.items():
        worst = max(rows, key=lambda row: row["regret_mean"])
        output.append({**dict(zip(("experiment_id", "algorithm", "horizon", "dimension",
                                   "objectives", "drift", "width"), key)),
                       "worst_preference_id": worst["preference_id"],
                       "worst_mean_regret": worst["regret_mean"],
                       "preferences_compared": len(rows)})
    return output


def scaling_slopes(aggregate: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in aggregate:
        if row["experiment_id"] == "EXP-04" and row["regret_mean"] is not None:
            groups[(row["algorithm"], row["preference_id"])].append(row)
    output = []
    for (algorithm, preference_id), rows in groups.items():
        points = [(math.log(row["horizon"]), math.log(row["regret_mean"]))
                  for row in rows if row["horizon"] > 0 and row["regret_mean"] > 0]
        if len(points) < 3:
            continue
        xs, ys = zip(*points)
        x_mean, y_mean = statistics.mean(xs), statistics.mean(ys)
        denominator = sum((x - x_mean) ** 2 for x in xs)
        if denominator == 0:
            continue
        slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator
        output.append({"algorithm": algorithm, "preference_id": preference_id,
                       "points": len(points), "descriptive_loglog_slope": slope})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    per_seed, aggregate = collect(args.output_root)
    write_csv(args.output_root / "tables" / "per_seed.csv", per_seed)
    write_csv(args.output_root / "tables" / "aggregate.csv", aggregate)
    write_csv(args.output_root / "tables" / "worst_preference.csv", worst_preferences(aggregate))
    write_csv(args.output_root / "tables" / "scaling_slopes.csv", scaling_slopes(aggregate))
    print(json.dumps({"per_seed": len(per_seed), "groups": len(aggregate)}))
