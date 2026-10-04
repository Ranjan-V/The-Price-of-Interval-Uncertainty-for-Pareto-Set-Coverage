"""EXP-10 driver for supplied chronological NPZ classification batches.

Required arrays: features (T,n,d), labels (T,n), groups (T,n), radii (T,3),
weights (T,3). Data are deliberately not downloaded or bundled. The interval
radii are caller-supplied; containment of a population objective is unverified.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from interval_pareto.algorithms import MidpointOGD
from interval_pareto.applications.streaming import ClassificationBatch, objective_and_subgradients, supplied_intervals


def prepare_stream(data_path: Path, eta: float, output_path: Path) -> None:
    if eta <= 0:
        raise ValueError("positive eta required")
    with np.load(data_path, allow_pickle=False) as data:
        features = np.asarray(data["features"], float)
        labels = np.asarray(data["labels"], float)
        groups = np.asarray(data["groups"], int)
        radii = np.asarray(data["radii"], float)
        weights = np.asarray(data["weights"], float)
    if features.ndim != 3 or labels.shape != features.shape[:2] or groups.shape != features.shape[:2]:
        raise ValueError("expected chronological T-by-n-by-d features and T-by-n labels/groups")
    horizon, _, dimension = features.shape
    if horizon < 1 or dimension < 1:
        raise ValueError("nonempty stream and positive decision dimension required")
    if radii.shape != (horizon, 3) or weights.shape != (horizon, 3):
        raise ValueError("radii and weights must be T-by-3")
    learner = MidpointOGD(dimension, eta)
    rows = []
    for t in range(horizon):
        weight = weights[t]
        if np.any(weight < 0) or not np.isclose(weight.sum(), 1.0):
            raise ValueError("weights must lie in simplex")
        batch = ClassificationBatch(features[t], labels[t], groups[t])
        action = learner.act()
        values, gradients = objective_and_subgradients(batch, action)
        interval = supplied_intervals(values, radii[t])
        learner.observe_gradient(weight @ gradients)
        rows.append({"t": t + 1, "action": json.dumps(action.tolist()),
                     "midpoints": json.dumps(interval.midpoint.tolist()),
                     "widths": json.dumps(interval.width.tolist()),
                     "weighted_midpoint": float(weight @ values)})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_stream(args.data, args.eta, args.output)
