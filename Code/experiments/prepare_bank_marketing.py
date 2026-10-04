"""Prepare the UCI Bank Marketing full file for exploratory EXP-10.

The source file is ordered by date, but does not expose per-row timestamps.
Features for each evaluation batch are scaled using only an initial prefix and
earlier evaluation batches. The post-call duration field is never used.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np


SOURCE_URL = "https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip"
NUMERIC_COLUMNS = (
    "campaign", "previous", "emp.var.rate", "cons.price.idx",
    "cons.conf.idx", "euribor3m", "nr.employed",
)
AGE_CUTOFF = 40
CALIBRATION_FRACTION = 0.20
BATCH_SIZE = 256
BIAS_PAIRS = 5
WEIGHTS = (0.80, 0.15, 0.05)
RADII = (0.05, 0.05, 0.0)


def load_rows(source: Path) -> tuple[list[dict[str, str]], str]:
    source_bytes = source.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    with zipfile.ZipFile(io.BytesIO(source_bytes)) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read("bank-additional.zip"))) as inner:
            raw = inner.read("bank-additional/bank-additional-full.csv").decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(raw), delimiter=";"))
    if len(rows) != 41188 or "duration" not in rows[0]:
        raise ValueError("Unexpected UCI Bank Marketing full file")
    return rows, digest


def prepare(source: Path, destination: Path) -> dict:
    rows, digest = load_rows(source)
    cut = int(len(rows) * CALIBRATION_FRACTION)
    calibration = np.array([[float(row[col]) for col in NUMERIC_COLUMNS]
                            for row in rows[:cut]], dtype=np.float64)
    lower = calibration.min(axis=0)
    upper = calibration.max(axis=0)
    usable = (len(rows) - cut) // BATCH_SIZE * BATCH_SIZE
    stream_rows = rows[cut:cut + usable]
    horizon = usable // BATCH_SIZE
    dimension = 2 * len(NUMERIC_COLUMNS) + 2 * BIAS_PAIRS
    features = np.empty((horizon, BATCH_SIZE, dimension), dtype=np.float32)
    labels = np.empty((horizon, BATCH_SIZE), dtype=np.float32)
    groups = np.empty((horizon, BATCH_SIZE), dtype=np.int8)
    group_counts = []
    label_counts = []

    for t in range(horizon):
        batch = stream_rows[t * BATCH_SIZE:(t + 1) * BATCH_SIZE]
        raw = np.array([[float(row[col]) for col in NUMERIC_COLUMNS]
                        for row in batch], dtype=np.float64)
        span = upper - lower
        scaled = np.where(span > 0, (raw - lower) / np.where(span > 0, span, 1), 0.5)
        signed = 2 * np.clip(scaled, 0, 1) - 1
        bias = np.ones((BATCH_SIZE, BIAS_PAIRS), dtype=np.float64)
        features[t] = np.concatenate((signed, -signed, bias, -bias), axis=1)
        labels[t] = np.array([1 if row["y"] == "yes" else -1 for row in batch], dtype=np.float32)
        groups[t] = np.array([0 if int(row["age"]) < AGE_CUTOFF else 1 for row in batch], dtype=np.int8)
        counts = np.bincount(groups[t].astype(int), minlength=2)
        if np.any(counts == 0):
            raise ValueError(f"Batch {t} lacks an age group")
        group_counts.append(counts.tolist())
        label_counts.append(np.bincount((labels[t] > 0).astype(int), minlength=2).tolist())
        # The next block can use this block's covariates, never future blocks.
        lower = np.minimum(lower, raw.min(axis=0))
        upper = np.maximum(upper, raw.max(axis=0))

    radii = np.tile(np.array(RADII, dtype=np.float64), (horizon, 1))
    weights = np.tile(np.array(WEIGHTS, dtype=np.float64), (horizon, 1))
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, features=features, labels=labels,
                        groups=groups, radii=radii, weights=weights)
    metadata = {
        "source_url": SOURCE_URL,
        "source_sha256": digest,
        "source_file": "bank-additional-full.csv",
        "license": "CC BY 4.0; Moro, Rita, and Cortez (2014), UCI Bank Marketing",
        "chronology": "Published row order is date order; exact per-row timestamps unavailable",
        "total_rows": len(rows), "calibration_rows": cut,
        "calibration_label_counts": [sum(row["y"] == "no" for row in rows[:cut]),
                                     sum(row["y"] == "yes" for row in rows[:cut])],
        "calibration_group_count_age40plus": sum(int(row["age"]) >= AGE_CUTOFF for row in rows[:cut]),
        "calibration_positive_count_age40plus": sum(int(row["age"]) >= AGE_CUTOFF and row["y"] == "yes" for row in rows[:cut]),
        "evaluation_rows": usable, "discarded_tail_rows": len(rows) - cut - usable,
        "batch_size": BATCH_SIZE, "horizon": horizon, "dimension": dimension,
        "numeric_columns": list(NUMERIC_COLUMNS),
        "excluded_columns": ["duration", "age"],
        "age_group": f"0 if age < {AGE_CUTOFF}, else 1",
        "group_counts_min": np.min(group_counts, axis=0).tolist(),
        "group_counts_max": np.max(group_counts, axis=0).tolist(),
        "label_counts_total": np.sum(label_counts, axis=0).tolist(),
        "feature_transform": "past-only min-max, clipped to [-1,1], signed pairs and five bias pairs",
        "weights": list(WEIGHTS), "radii": list(RADII),
        "radius_interpretation": "fixed sensitivity bands only; population coverage unverified",
        "npz_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    destination.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
