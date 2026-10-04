"""Staged Kaggle entry point A through H. Stages are explicit; no auto full sweep."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "src"))
from common import atomic_json, utc_now
from verify_environment import inspect
from run_synthetic import run_matrix
from run_application import run as run_application
from collect_results import collect, write_csv, worst_preferences, scaling_slopes
from plot_results import plot
from validation.smoke import run as smoke_run
from experiments.run import run_case
from interval_pareto.synthetic import SyntheticConfig


def stage_b(output_root: Path) -> dict:
    smoke_seconds = smoke_run(output_root / "smoke" / "kaggle-stageB")
    outcome = {"smoke_seconds": smoke_seconds, "gpu_parity": "not_available", "gpu_devices_tested": []}
    try:
        import torch
        if torch.cuda.is_available():
            from batched_synthetic import evaluate_batch
            configs = [SyntheticConfig(horizon=16, dimension=2, objectives=2, radius=0.05,
                                       drift=0.1, width_slope_fraction=0.25, seed=seed)
                       for seed in [3, 17]]
            weights = np.array([0.4, 0.6])
            cpu = [run_case(config, 0.08, "midpoint", fixed_weights=weights)[0] for config in configs]
            keys = ["regret", "midpoint_regret", "path_variation", "two_location_width",
                    "uniform_width", "theorem_bound"]
            for index in range(torch.cuda.device_count()):
                device = f"cuda:{index}"
                gpu = evaluate_batch(configs, weights, 0.08, device)
                for a, b in zip(cpu, gpu):
                    for key in keys:
                        if not np.isclose(a[key], b[key], rtol=1e-8, atol=1e-8):
                            raise AssertionError(f"CPU/{device} parity failed for {key}: {a[key]} versus {b[key]}")
                outcome["gpu_devices_tested"].append(torch.cuda.get_device_name(index))
            outcome["gpu_parity"] = "passed"
    except ImportError:
        pass
    atomic_json(output_root / "logs" / "stageB.json",
                {"timestamp_utc": utc_now(), "status": "success", **outcome})
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=list("ABCDEFGH"), required=True)
    parser.add_argument("--config", type=Path, default=HERE / "configs" / "full.json")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_root = Path(config["output_root"])
    if not output_root.is_absolute():
        output_root = CODE_ROOT / output_root
    stage = args.stage
    if stage == "A":
        result = inspect()
        atomic_json(output_root / "logs" / "kaggle_environment.json", result)
    elif stage == "B":
        result = stage_b(output_root)
    elif stage == "C":
        result = {exp: run_matrix(config, [exp], "one", args.device, output_root, max_jobs=1)
                  for exp in [f"EXP-{i:02d}" for i in range(1, 10)]}
    elif stage == "D":
        result = run_matrix(config, ["EXP-01", "EXP-02", "EXP-03", "EXP-05",
                                     "EXP-06", "EXP-07", "EXP-08"], "full", args.device, output_root)
    elif stage == "E":
        if args.data is None:
            raise ValueError("Stage E needs --data path to supplied chronological NPZ")
        result = {"checkpoint": str(run_application(args.data, config["eta"], output_root))}
    elif stage == "F":
        result = run_matrix(config, ["EXP-09"], "full", args.device, output_root)
    elif stage == "G":
        result = run_matrix(config, ["EXP-04"], "full", args.device, output_root)
    else:
        per_seed, aggregate = collect(output_root)
        incomplete = [row for row in aggregate if row["n_seeds"] != len(config["seeds"])]
        if incomplete:
            raise RuntimeError(f"{len(incomplete)} synthetic groups lack the configured seed count; finish or resume prior stages")
        write_csv(output_root / "tables" / "per_seed.csv", per_seed)
        write_csv(output_root / "tables" / "aggregate.csv", aggregate)
        write_csv(output_root / "tables" / "worst_preference.csv", worst_preferences(aggregate))
        write_csv(output_root / "tables" / "scaling_slopes.csv", scaling_slopes(aggregate))
        result = {"per_seed": len(per_seed), "groups": len(aggregate), "figures": plot(output_root)}
    print(json.dumps({"stage": stage, "result": result}, indent=2))


if __name__ == "__main__":
    main()
