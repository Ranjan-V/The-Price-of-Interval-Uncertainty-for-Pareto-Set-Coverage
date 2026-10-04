"""Resumable EXP-01–09 matrix. Per-seed JSON checkpoints; CPU or optional CUDA."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "src"))
from common import canonical_id, atomic_json, valid_checkpoint, versions, git_commit, utc_now, IMPLEMENTATION_VERSION
from experiments.run import run_case
from experiments.parallel_front import run_front_case
from interval_pareto.synthetic import SyntheticConfig


def preference_vectors(m: int) -> list[tuple[str, list[float]]]:
    if m == 2:
        return [(f"grid{i}", [i / 4, 1 - i / 4]) for i in range(5)]
    vectors = [("uniform", (np.ones(m) / m).tolist())]
    for j in range(min(m, 2)):
        vectors.append((f"vertex{j}", np.eye(m)[j].tolist()))
    rng = np.random.default_rng(421 + m)
    vectors.extend((f"simplex{i}", rng.dirichlet(np.ones(m)).tolist()) for i in range(2))
    return vectors


def settings(config: dict, experiment: str):
    base = SyntheticConfig(**config["base"], seed=0)
    if experiment == "EXP-01":
        yield replace(base, radius=0.0), "midpoint"
    elif experiment == "EXP-02":
        for width in config["widths"]:
            yield replace(base, radius=width / 2), "midpoint"
    elif experiment == "EXP-03":
        for drift in config["drifts"]:
            yield replace(base, drift=drift), "midpoint"
    elif experiment == "EXP-04":
        for horizon in config["horizons"]:
            yield replace(base, horizon=horizon), "midpoint"
    elif experiment == "EXP-05":
        for width in config["joint_widths"]:
            for drift in config["joint_drifts"]:
                yield replace(base, radius=width / 2, drift=drift), "midpoint"
    elif experiment == "EXP-06":
        for m in config["objective_counts"]:
            yield replace(base, objectives=m), "midpoint"
    elif experiment == "EXP-07":
        for d in config["dimensions"]:
            yield replace(base, dimension=d), "midpoint"
    elif experiment == "EXP-08":
        yield replace(base, dimension=1, objectives=2), "parallel"
    elif experiment == "EXP-09":
        for method in ["midpoint", "upper", "lower", "static"]:
            yield base, method
    else:
        raise ValueError(f"unknown synthetic experiment {experiment}")


def _checkpoint(output_root: Path, experiment: str, payload: dict) -> tuple[str, Path]:
    run_id = canonical_id(payload)
    return run_id, output_root / "synthetic" / experiment / f"{run_id}.json"


def _device_choice(requested: str, complexity: int, algorithm: str, d: int) -> str:
    if requested == "cpu" or algorithm != "midpoint" or d == 1:
        return "cpu"
    try:
        import torch
        available = bool(torch.cuda.is_available())
    except ImportError:
        available = False
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA requested but unavailable")
    return "cuda" if available and (requested == "cuda" or complexity >= 200000) else "cpu"


def _save(path: Path, run_id: str, payload: dict, metrics: dict, device: str,
          seconds: float, common_metadata: dict) -> None:
    if metrics.get("bound_respected") is False or metrics.get("finite_grid_below_bound") is False:
        raise AssertionError(f"bound check failed for {run_id}; scaling halted")
    atomic_json(path, {"run_id": run_id, "status": "success", "experiment_id": payload["experiment_id"],
                       "config": payload, "metrics": metrics, "device": device,
                       "timestamp_utc": utc_now(), "runtime_seconds": seconds, **common_metadata})


def run_matrix(config: dict, experiments: list[str], seed_mode: str, requested_device: str,
               output_root: Path, max_jobs: int | None = None,
               seeds_override: list[int] | None = None) -> dict:
    seeds = (config["seeds"][:1] if seed_mode == "one" else config["seeds"])
    if seeds_override is not None:
        seeds = [seed for seed in seeds if seed in seeds_override]
    metadata = {"packages": versions(), "git_commit": git_commit()}
    counts = {"considered": 0, "completed": 0, "skipped": 0, "failed": 0, "cuda_runs": 0}
    for experiment in experiments:
        for base, algorithm in settings(config, experiment):
            preferences = [("front_grid", None)] if algorithm == "parallel" else preference_vectors(base.objectives)
            for preference_id, weights in preferences:
                pending = []
                for seed in seeds:
                    if max_jobs is not None and counts["considered"] >= max_jobs:
                        return counts
                    counts["considered"] += 1
                    instance = replace(base, seed=seed)
                    payload = {"experiment_id": experiment, "synthetic": asdict(instance),
                               "algorithm": algorithm, "preference_id": preference_id,
                               "weights": weights, "eta": config["eta"],
                               "implementation_version": IMPLEMENTATION_VERSION,
                               "front_denominator": config["front_denominator"] if algorithm == "parallel" else None}
                    run_id, path = _checkpoint(output_root, experiment, payload)
                    if valid_checkpoint(path, run_id):
                        counts["skipped"] += 1
                    else:
                        pending.append((instance, payload, run_id, path))
                if not pending:
                    continue
                complexity = base.horizon * base.dimension * base.objectives * len(pending)
                device = _device_choice(requested_device, complexity, algorithm, base.dimension)
                if device == "cuda":
                    from batched_synthetic import evaluate_batch
                    for start in range(0, len(pending), 2):
                        chunk = pending[start:start + 2]
                        began = time.perf_counter()
                        try:
                            batch_metrics = evaluate_batch([item[0] for item in chunk], np.asarray(weights),
                                                           float(config["eta"]), "cuda")
                            elapsed = time.perf_counter() - began
                            for (_, payload, run_id, path), metrics in zip(chunk, batch_metrics):
                                _save(path, run_id, payload, metrics, "cuda", elapsed / len(chunk), metadata)
                                counts["completed"] += 1
                                counts["cuda_runs"] += 1
                        except Exception as error:
                            counts["failed"] += len(chunk)
                            atomic_json(output_root / "logs" / f"failed-{chunk[0][2]}.json",
                                        {"timestamp_utc": utc_now(), "runs": [item[2] for item in chunk],
                                         "configs": [item[1] for item in chunk], "device": "cuda",
                                         "packages": metadata["packages"], "git_commit": metadata["git_commit"],
                                         "error": repr(error), "runtime_seconds": time.perf_counter() - began})
                            raise
                else:
                    for instance, payload, run_id, path in pending:
                        began = time.perf_counter()
                        try:
                            if algorithm == "parallel":
                                metrics = run_front_case(instance, config["front_denominator"],
                                                         float(config["eta"]), front_grid=201,
                                                         record_trace=True)
                            else:
                                metrics, _ = run_case(instance, float(config["eta"]), algorithm,
                                                      fixed_weights=np.asarray(weights), record_trace=False)
                            _save(path, run_id, payload, metrics, "cpu", time.perf_counter() - began, metadata)
                            counts["completed"] += 1
                        except Exception as error:
                            counts["failed"] += 1
                            atomic_json(output_root / "logs" / f"failed-{run_id}.json",
                                        {"timestamp_utc": utc_now(), "run_id": run_id,
                                         "config": payload, "device": "cpu", "packages": metadata["packages"],
                                         "git_commit": metadata["git_commit"], "error": repr(error),
                                         "runtime_seconds": time.perf_counter() - began})
                            raise
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+", default=[f"EXP-{i:02d}" for i in range(1, 10)])
    parser.add_argument("--seed-mode", choices=["one", "full"], default="one")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--seed-mod", type=int, help="partition independent seeds across workers")
    parser.add_argument("--seed-rem", type=int, help="remainder for --seed-mod partition")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_root = Path(config["output_root"])
    if not output_root.is_absolute():
        output_root = CODE_ROOT / output_root
    if (args.seed_mod is None) != (args.seed_rem is None):
        raise ValueError("--seed-mod and --seed-rem must be supplied together")
    if args.seed_mod is not None and (args.seed_mod < 1 or not 0 <= args.seed_rem < args.seed_mod):
        raise ValueError("invalid seed partition")
    selected_seeds = ([seed for seed in config["seeds"] if seed % args.seed_mod == args.seed_rem]
                      if args.seed_mod is not None else None)
    print(json.dumps(run_matrix(config, args.experiments, args.seed_mode, args.device,
                                output_root, args.max_jobs, selected_seeds), indent=2))
