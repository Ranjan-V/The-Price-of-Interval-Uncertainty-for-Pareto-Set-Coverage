"""Build a cell-by-cell Kaggle notebook for the THM-06/07 controlled check."""

import base64
import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "JCAM_THM06_Pareto_Information.ipynb"


def encoded(path: Path) -> str:
    value = base64.b64encode(path.read_bytes()).decode("ascii")
    return "\n".join(f"    {value[i:i + 96]!r}," for i in range(0, len(value), 96))


EXPERIMENT = encoded(ROOT / "Code" / "experiments" / "pareto_information_lower_bound.py")
PLOT = encoded(ROOT / "Code" / "experiments" / "plot_pareto_information.py")
PROOF = encoded(ROOT / "Math" / "proofs" / "proof_thm06.md")
PROOF_UPPER = encoded(ROOT / "Math" / "proofs" / "proof_thm07.md")
COMPARISON = encoded(ROOT / "Theory" / "closest_theorem_comparison.md")
MANUSCRIPT = encoded(ROOT / "Paper" / "interval_pareto_information_draft.tex")
PAPER_PDF = encoded(ROOT / "Paper" / "interval_pareto_information_draft.pdf")
JOURNAL_PLAN = encoded(ROOT / "Paper" / "journal_and_submission_plan.md")

CELLS = [
    ("1. Locate a zipped or automatically unzipped input package", r'''
from pathlib import Path
import base64, hashlib, json, os, shutil, subprocess, sys, time, zipfile
INPUT_ROOT = Path(os.environ.get("JCAM_INPUT_ROOT", "/kaggle/input"))
WORK_ROOT = Path(os.environ.get("JCAM_WORK_ROOT", "/kaggle/working"))
RUN_ROOT = WORK_ROOT / "jcam_thm06"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
STAGES = []
RUN_OK = True
PACKAGE_MODE = "embedded_notebook_sources"
def record(stage, outcome, detail):
    STAGES.append({"stage": stage, "outcome": outcome, "detail": detail})
    print(stage, outcome, detail)
try:
    candidates = sorted(INPUT_ROOT.rglob("pareto_information_lower_bound.py")) if INPUT_ROOT.exists() else []
    if candidates:
        PACKAGE_MODE = "kaggle_auto_unzipped"
    elif INPUT_ROOT.exists():
        for path in sorted(INPUT_ROOT.rglob("*.zip")):
            try:
                with zipfile.ZipFile(path) as archive:
                    members = [name for name in archive.namelist() if name.replace("\\", "/").endswith("pareto_information_lower_bound.py")]
                    if not members:
                        continue
                    destination = RUN_ROOT / "input_package"
                    destination.mkdir(parents=True, exist_ok=True)
                    for member in archive.infolist():
                        target = (destination / member.filename).resolve()
                        if not target.is_relative_to(destination.resolve()):
                            raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
                    archive.extractall(destination)
                    PACKAGE_MODE = "kaggle_zip_extracted"
                    break
            except zipfile.BadZipFile:
                continue
    record("bootstrap", "passed", PACKAGE_MODE)
except Exception as error:
    RUN_OK = False
    record("bootstrap", "failed", repr(error))
'''),
    ("2. Install the exact experiment and proof sources from notebook cells", f'''
EXPERIMENT_SOURCE = base64.b64decode("".join([\n{EXPERIMENT}\n]))
PLOT_SOURCE = base64.b64decode("".join([\n{PLOT}\n]))
PROOF_SOURCE = base64.b64decode("".join([\n{PROOF}\n]))
PROOF_UPPER_SOURCE = base64.b64decode("".join([\n{PROOF_UPPER}\n]))
COMPARISON_SOURCE = base64.b64decode("".join([\n{COMPARISON}\n]))
MANUSCRIPT_SOURCE = base64.b64decode("".join([\n{MANUSCRIPT}\n]))
PAPER_PDF_SOURCE = base64.b64decode("".join([\n{PAPER_PDF}\n]))
JOURNAL_PLAN_SOURCE = base64.b64decode("".join([\n{JOURNAL_PLAN}\n]))
if RUN_OK:
    (RUN_ROOT / "pareto_information_lower_bound.py").write_bytes(EXPERIMENT_SOURCE)
    (RUN_ROOT / "plot_pareto_information.py").write_bytes(PLOT_SOURCE)
    (RUN_ROOT / "proof_thm06.md").write_bytes(PROOF_SOURCE)
    (RUN_ROOT / "proof_thm07.md").write_bytes(PROOF_UPPER_SOURCE)
    (RUN_ROOT / "closest_theorem_comparison.md").write_bytes(COMPARISON_SOURCE)
    (RUN_ROOT / "interval_pareto_information_draft.tex").write_bytes(MANUSCRIPT_SOURCE)
    (RUN_ROOT / "interval_pareto_information_draft.pdf").write_bytes(PAPER_PDF_SOURCE)
    (RUN_ROOT / "journal_and_submission_plan.md").write_bytes(JOURNAL_PLAN_SOURCE)
    record("sources", "passed", "experiment, plotting code, two theorem proofs, comparison, paper source and PDF")
else:
    record("sources", "skipped", "bootstrap failed")
'''),
    ("3. Check certified Pareto-information lower and upper bounds", r'''
if RUN_OK:
    started = time.perf_counter()
    commands = [
        [sys.executable, str(RUN_ROOT / "pareto_information_lower_bound.py"),
         "--output", str(RUN_ROOT / "results")],
        [sys.executable, str(RUN_ROOT / "plot_pareto_information.py"),
         "--table", str(RUN_ROOT / "results" / "pareto_information_lower_bound.csv"),
         "--output", str(RUN_ROOT / "results" / "pareto_information_scaling.png")],
    ]
    for command in commands:
        outcome = subprocess.run(command, cwd=RUN_ROOT, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(outcome.stdout[-5000:])
        if outcome.returncode:
            RUN_OK = False
            record("experiment", "failed", f"exit={outcome.returncode}")
            break
    if RUN_OK:
        summary = json.loads((RUN_ROOT / "results" / "pareto_information_summary.json").read_text())
        if not (summary["all_checked_configurations_meet_bound"] and
                summary["all_checked_configurations_meet_upper_bound"]):
            RUN_OK = False
            record("experiment", "failed", "numerical bound check failed")
        else:
            record("experiment", "passed", f"25 configurations; seconds={time.perf_counter()-started:.2f}")
else:
    record("experiment", "skipped", "earlier stage failed")
'''),
    ("4. Zip every THM-06/07 output", r'''
(RUN_ROOT / "notebook_stages.json").write_text(
    json.dumps({"run_ok": RUN_OK, "stages": STAGES}, indent=2), encoding="utf-8")
FINAL_ZIP = WORK_ROOT / "result_jcam_thm06_pareto_information.zip"
with zipfile.ZipFile(FINAL_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(RUN_ROOT.rglob("*")):
        if path.is_file():
            archive.write(path, arcname=(Path("thm06_pareto_information") / path.relative_to(RUN_ROOT)).as_posix())
print("Output archive:", FINAL_ZIP, "bytes:", FINAL_ZIP.stat().st_size, "run_ok:", RUN_OK)
'''),
]


def build() -> None:
    notebook = {
        "cells": [],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for title, source in CELLS:
        notebook["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [f"## {title}\n"]})
        notebook["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {},
                                  "outputs": [], "source": textwrap.dedent(source).strip().splitlines(keepends=True)})
    OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    build()
