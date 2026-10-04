"""Stage E: supplied chronological classification data; exploratory only."""

import argparse
import json
from pathlib import Path
import sys
import time

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(CODE_ROOT))
from common import atomic_json, canonical_id, valid_checkpoint, versions, git_commit, utc_now
from experiments.streaming_classification import prepare_stream


def run(data: Path, eta: float, output_root: Path) -> Path:
    if not data.is_file():
        raise FileNotFoundError(data)
    import hashlib
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    payload = {"experiment_id": "EXP-10", "data_sha256": digest, "eta": eta,
               "interpretation": "empirical surrogate; population interval containment unverified"}
    run_id = canonical_id(payload)
    destination = output_root / "application" / run_id
    checkpoint = destination / "metadata.json"
    if valid_checkpoint(checkpoint, run_id):
        return checkpoint
    began = time.perf_counter()
    table = destination / "rounds.csv"
    prepare_stream(data, eta, table)
    atomic_json(checkpoint, {"run_id": run_id, "status": "success", "experiment_id": "EXP-10",
                             "config": payload, "metrics": {"round_table": str(table)},
                             "timestamp_utc": utc_now(), "runtime_seconds": time.perf_counter() - began,
                             "device": "cpu", "packages": versions(), "git_commit": git_commit()})
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--output-root", type=Path, default=CODE_ROOT / "results")
    args = parser.parse_args()
    print(run(args.data, args.eta, args.output_root))
