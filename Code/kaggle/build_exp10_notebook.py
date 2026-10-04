"""Build the importable, cell-by-cell Kaggle EXP-10 notebook."""

import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "JCAM_EXP10_Bank_Marketing.ipynb"
EVALUATOR = (ROOT / "Code" / "experiments" / "evaluate_bank_stream.py").read_text(encoding="utf-8")
EVALUATOR = EVALUATOR.split('if __name__ == "__main__":')[0].rstrip()

CELLS = [
    ("1. Locate source code and the Bank Marketing stream", r'''
from pathlib import Path
import os, shutil, zipfile, json, sys, subprocess, time

INPUT_ROOT = Path(os.environ.get("JCAM_INPUT_ROOT", "/kaggle/input"))
WORK_ROOT = Path(os.environ.get("JCAM_WORK_ROOT", "/kaggle/working"))
CODE = WORK_ROOT / "JCAM" / "Code"
RESULTS = CODE / "results" / "exp10_bank"
BOOTSTRAP_ERROR = None

def extract_matching_zip(target_name, destination):
    matches = []
    for path in INPUT_ROOT.rglob("*.zip"):
        try:
            with zipfile.ZipFile(path) as archive:
                if any(name.replace("\\", "/").endswith(target_name) for name in archive.namelist()):
                    matches.append(path)
        except zipfile.BadZipFile:
            continue
    if len(matches) != 1:
        raise RuntimeError(f"Expected one archive containing {target_name}, found {matches}")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(matches[0]) as archive:
        for member in archive.infolist():
            if not (destination / member.filename).resolve().is_relative_to(destination.resolve()):
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
        archive.extractall(destination)

try:
    source_roots = sorted({path.parent.parent for path in INPUT_ROOT.rglob("run_application.py")
                           if path.parent.name == "kaggle" and path.parent.parent.name == "Code"})
    if not source_roots:
        extracted_source = WORK_ROOT / "_jcam_source"
        extract_matching_zip("Code/kaggle/run_application.py", extracted_source)
        source_roots = sorted({path.parent.parent for path in extracted_source.rglob("run_application.py")
                               if path.parent.name == "kaggle" and path.parent.parent.name == "Code"})
    if len(source_roots) != 1:
        raise RuntimeError(f"Expected one Code source, found {source_roots}")
    shutil.copytree(source_roots[0], CODE, dirs_exist_ok=True)
    inputs = sorted(INPUT_ROOT.rglob("bank_stream.npz"))
    if not inputs:
        extracted_data = WORK_ROOT / "_jcam_bank_input"
        extract_matching_zip("bank_stream.npz", extracted_data)
        inputs = sorted(extracted_data.rglob("bank_stream.npz"))
    if len(inputs) != 1:
        raise RuntimeError(f"Expected one bank_stream.npz, found {inputs}")
    DATA = inputs[0]
    SOURCE_META = DATA.with_suffix(".metadata.json")
    if not SOURCE_META.is_file():
        raise FileNotFoundError(SOURCE_META)
    RESULTS.mkdir(parents=True, exist_ok=True)
    print("Code:", CODE, "Data:", DATA)
except Exception as error:
    BOOTSTRAP_ERROR = repr(error)
    RESULTS.mkdir(parents=True, exist_ok=True)
    print("Bootstrap failed:", BOOTSTRAP_ERROR)
'''),
    ("2. Validate the chronological NPZ", r'''
import hashlib
import numpy as np

RUN_OK = BOOTSTRAP_ERROR is None
STAGES = []
ROUND_PATH = None

def record(stage, outcome, detail):
    STAGES.append({"stage": stage, "outcome": outcome, "detail": detail})
    print(stage, outcome, detail)

if RUN_OK:
    try:
        source = json.loads(SOURCE_META.read_text(encoding="utf-8"))
        if hashlib.sha256(DATA.read_bytes()).hexdigest() != source["npz_sha256"]:
            raise ValueError("NPZ SHA-256 mismatch")
        with np.load(DATA, allow_pickle=False) as data:
            x, y, g, r, w = [data[k] for k in ("features", "labels", "groups", "radii", "weights")]
            if x.ndim != 3 or y.shape != x.shape[:2] or g.shape != y.shape:
                raise ValueError("Invalid stream shapes")
            if r.shape != (len(x), 3) or w.shape != r.shape:
                raise ValueError("Invalid objective metadata shapes")
            if not np.all(np.isfinite(x)) or not np.all(np.isin(y, [-1, 1])):
                raise ValueError("Invalid features or labels")
            if not np.all((g == 0).any(axis=1) & (g == 1).any(axis=1)):
                raise ValueError("Both age groups are required in every batch")
            if not np.allclose(w.sum(axis=1), 1) or np.any(w < 0) or np.any(r < 0):
                raise ValueError("Invalid weights or radii")
        record("validate", "passed", f"T={len(x)}, n={x.shape[1]}, d={x.shape[2]}")
    except Exception as error:
        RUN_OK = False
        record("validate", "failed", repr(error))
else:
    record("validate", "skipped", BOOTSTRAP_ERROR)
'''),
    ("3. Run EXP-10 in chronological order", r'''
if RUN_OK:
    started = time.perf_counter()
    command = [sys.executable, str(CODE / "kaggle" / "run_application.py"),
               "--data", str(DATA), "--eta", "0.05", "--output-root", str(RESULTS)]
    outcome = subprocess.run(command, cwd=CODE, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(outcome.stdout[-5000:])
    if outcome.returncode == 0:
        checkpoints = sorted((RESULTS / "application").rglob("metadata.json"))
        if len(checkpoints) != 1:
            RUN_OK = False
            record("application", "failed", f"Expected one checkpoint, found {checkpoints}")
        else:
            ROUND_PATH = checkpoints[0].with_name("rounds.csv")
            record("application", "passed", f"seconds={time.perf_counter()-started:.2f}")
    else:
        RUN_OK = False
        record("application", "failed", f"exit={outcome.returncode}")
else:
    record("application", "skipped", "earlier gate failed")
'''),
    ("4. Evaluate pre-update predictions against a causal baseline", EVALUATOR + r'''

if RUN_OK:
    try:
        SUMMARY = evaluate(DATA, ROUND_PATH, SOURCE_META, RESULTS / "evaluation")
        print(json.dumps(SUMMARY, indent=2))
        record("evaluation", "passed", "prequential metrics saved")
    except Exception as error:
        RUN_OK = False
        record("evaluation", "failed", repr(error))
else:
    record("evaluation", "skipped", "earlier gate failed")
'''),
    ("5. Plot diagnostics and zip all EXP-10 outputs", r'''
import csv, zipfile

if RUN_OK:
    try:
        import matplotlib.pyplot as plt
        with (RESULTS / "evaluation" / "per_batch.csv").open(newline="", encoding="utf-8") as handle:
            per_batch = list(csv.DictReader(handle))
        steps = [int(row["batch"]) for row in per_batch]
        fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        ax[0].plot(steps, [float(row["model_logloss"]) for row in per_batch], label="Online midpoint")
        ax[0].plot(steps, [float(row["causal_prevalence_baseline_logloss"]) for row in per_batch], label="Past-prevalence baseline")
        ax[0].set_ylabel("Pre-update log loss")
        ax[0].legend()
        ax[1].plot(steps, [float(row["score_group_gap"]) for row in per_batch])
        ax[1].set_ylabel("Absolute age-group mean score gap")
        ax[1].set_xlabel("Chronological batch")
        fig.tight_layout()
        fig.savefig(RESULTS / "exp10_diagnostics.png", dpi=160)
        plt.close(fig)
        record("plot", "passed", "exp10_diagnostics.png")
    except Exception as error:
        RUN_OK = False
        record("plot", "failed", repr(error))
else:
    record("plot", "skipped", "earlier gate failed")

(RESULTS / "notebook_stages.json").write_text(json.dumps({"run_ok": RUN_OK, "stages": STAGES}, indent=2), encoding="utf-8")
if BOOTSTRAP_ERROR is None:
    shutil.copy2(SOURCE_META, RESULTS / "bank_stream.metadata.json")
FINAL_ZIP = WORK_ROOT / "result_jcam_exp10.zip"
with zipfile.ZipFile(FINAL_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(RESULTS.rglob("*")):
        if path.is_file():
            archive.write(path, arcname=(Path("exp10_bank") / path.relative_to(RESULTS)).as_posix())
print("Output archive:", FINAL_ZIP, "bytes:", FINAL_ZIP.stat().st_size, "run_ok:", RUN_OK)
'''),
]


def build() -> Path:
    cells = []
    for heading, code in CELLS:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [f"## {heading}\n"]})
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                      "outputs": [], "source": textwrap.dedent(code).lstrip("\n").splitlines(keepends=True)})
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                               "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    print(build())
