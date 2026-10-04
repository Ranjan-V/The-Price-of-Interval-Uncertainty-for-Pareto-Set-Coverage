"""Audit EXP-10's pre-update predictions against a causal prevalence baseline."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def evaluate(data_path: Path, rounds_path: Path, source_metadata: Path, output: Path) -> dict:
    with np.load(data_path, allow_pickle=False) as data:
        features = data["features"]
        labels = data["labels"]
        groups = data["groups"]
    with rounds_path.open(newline="", encoding="utf-8") as handle:
        rounds = list(csv.DictReader(handle))
    source = json.loads(source_metadata.read_text(encoding="utf-8"))
    if len(rounds) != len(features):
        raise ValueError("Round count does not match NPZ horizon")
    past_negative, past_positive = source["calibration_label_counts"]
    batch_rows = []
    total_model_loss = total_baseline_loss = total_brier = 0.0
    total_model_correct = total_baseline_correct = 0
    total_n = 0
    max_objective_difference = 0.0

    for t, row in enumerate(rounds):
        action = np.asarray(json.loads(row["action"]), dtype=float)
        if action.shape != (features.shape[2],):
            raise ValueError(f"Action dimension mismatch at batch {t}")
        x = np.asarray(features[t], dtype=float)
        y = np.asarray(labels[t], dtype=float)
        group = np.asarray(groups[t], dtype=int)
        scores = x @ action
        margins = y * scores
        model_loss = np.logaddexp(0, -margins)
        probability = 1 / (1 + np.exp(-np.clip(scores, -700, 700)))
        baseline_probability = np.clip(past_positive / (past_positive + past_negative), 1e-12, 1 - 1e-12)
        baseline_logit = float(np.log(baseline_probability / (1 - baseline_probability)))
        baseline_loss = np.logaddexp(0, -y * baseline_logit)
        observed_midpoint = json.loads(row["midpoints"])[0]
        max_objective_difference = max(max_objective_difference, abs(float(observed_midpoint) - float(model_loss.mean())))
        if max_objective_difference > 1e-6:
            raise ValueError("Logged prequential logistic objective does not match recomputation")
        n = len(y)
        gap = abs(float(scores[group == 0].mean() - scores[group == 1].mean()))
        batch_rows.append({
            "batch": t + 1, "n": n, "positives": int(np.sum(y > 0)),
            "group0": int(np.sum(group == 0)), "group1": int(np.sum(group == 1)),
            "model_logloss": float(model_loss.mean()),
            "causal_prevalence_baseline_logloss": float(baseline_loss.mean()),
            "model_accuracy": float(np.mean((scores >= 0) == (y > 0))),
            "baseline_accuracy": float(np.mean((baseline_logit >= 0) == (y > 0))),
            "model_brier": float(np.mean((probability - (y > 0)) ** 2)),
            "score_group_gap": gap,
            "baseline_pre_batch_positive_rate": baseline_probability,
        })
        total_n += n
        total_model_loss += float(model_loss.sum())
        total_baseline_loss += float(baseline_loss.sum())
        total_brier += float(np.sum((probability - (y > 0)) ** 2))
        total_model_correct += int(np.sum((scores >= 0) == (y > 0)))
        total_baseline_correct += int(np.sum((baseline_logit >= 0) == (y > 0)))
        past_positive += int(np.sum(y > 0))
        past_negative += int(np.sum(y < 0))

    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_batch.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(batch_rows[0]))
        writer.writeheader()
        writer.writerows(batch_rows)
    summary = {
        "status": "success", "interpretation": "exploratory, prequential empirical metrics",
        "n": total_n, "batches": len(batch_rows),
        "model_logloss": total_model_loss / total_n,
        "causal_prevalence_baseline_logloss": total_baseline_loss / total_n,
        "model_accuracy": total_model_correct / total_n,
        "baseline_accuracy": total_baseline_correct / total_n,
        "model_brier": total_brier / total_n,
        "mean_batch_score_group_gap": float(np.mean([r["score_group_gap"] for r in batch_rows])),
        "max_logged_logloss_difference": max_objective_difference,
        "population_interval_coverage_verified": False,
    }
    (output / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--rounds", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.data, args.rounds, args.source_metadata, args.output), indent=2))
