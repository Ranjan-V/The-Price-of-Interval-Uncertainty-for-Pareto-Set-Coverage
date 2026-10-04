"""Build a self-contained four-code-cell Kaggle notebook for THM-05/08 checks."""

import base64
import io
import json
from pathlib import Path
import textwrap
import zipfile


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "JCAM_THM05_Sharpening.ipynb"


def source_bundle() -> bytes:
    paths = [
        *sorted((ROOT / "Code" / "src" / "interval_pareto").rglob("*.py")),
        ROOT / "Code" / "experiments" / "parallel_front.py",
        ROOT / "Code" / "experiments" / "check_thm05_sharpening.py",
        ROOT / "Math" / "proofs" / "proof_thm05.md",
        ROOT / "Math" / "proofs" / "proof_thm08.md",
        ROOT / "Paper" / "interval_pareto_information_draft.tex",
        ROOT / "Paper" / "interval_pareto_information_draft.pdf",
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, arcname=path.relative_to(ROOT).as_posix())
    return buffer.getvalue()


payload = base64.b64encode(source_bundle()).decode("ascii")
chunks = "\n".join(f"    {payload[i:i + 96]!r}," for i in range(0, len(payload), 96))

CELLS = [
    ("1. Detect zipped or automatically unzipped input", r'''
from pathlib import Path
import base64, json, os, subprocess, sys, time, zipfile
INPUT_ROOT = Path(os.environ.get("JCAM_INPUT_ROOT", "/kaggle/input"))
WORK_ROOT = Path(os.environ.get("JCAM_WORK_ROOT", "/kaggle/working"))
RUN_ROOT = WORK_ROOT / "jcam_thm05_sharpening"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
RUN_OK = True
STAGES = []
def record(stage, outcome, detail):
    STAGES.append({"stage": stage, "outcome": outcome, "detail": detail})
    print(stage, outcome, detail)
mode = "embedded_exact_sources"
if INPUT_ROOT.exists():
    if any(INPUT_ROOT.rglob("check_thm05_sharpening.py")):
        mode = "kaggle_auto_unzipped_detected; embedded exact sources selected"
    elif any(INPUT_ROOT.rglob("*.zip")):
        mode = "kaggle_zip_detected; embedded exact sources selected"
record("bootstrap", "passed", mode)
'''),
    ("2. Install exact source package from notebook cell", f'''
SOURCE_ARCHIVE = base64.b64decode("".join([\n{chunks}\n]))
(RUN_ROOT / "source_package.zip").write_bytes(SOURCE_ARCHIVE)
with zipfile.ZipFile(RUN_ROOT / "source_package.zip") as archive:
    for member in archive.infolist():
        target = (RUN_ROOT / member.filename).resolve()
        if not target.is_relative_to(RUN_ROOT.resolve()):
            raise RuntimeError(f"Unsafe ZIP member: {{member.filename}}")
    archive.extractall(RUN_ROOT)
record("sources", "passed", "experiment code, library, proofs, manuscript source and PDF")
'''),
    ("3. Run six THM-05 and metric-net comparisons", r'''
if RUN_OK:
    started = time.perf_counter()
    command = [sys.executable, str(RUN_ROOT / "Code" / "experiments" / "check_thm05_sharpening.py")]
    result = subprocess.run(command, cwd=RUN_ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout[-5000:])
    if result.returncode:
        RUN_OK = False
        record("experiment", "failed", f"exit={result.returncode}")
    else:
        summary_path = RUN_ROOT / "Code" / "results" / "thm05_sharpening" / "thm05_sharpening_summary.json"
        summary = json.loads(summary_path.read_text())
        RUN_OK = bool(summary["cases"] == 6 and summary["all_sampled_below_sharpened"]
                      and summary["all_static_sampled_below_metric_bound"]
                      and summary["minimum_improvement_factor"] > 1)
        record("experiment", "passed" if RUN_OK else "failed",
               f"six cases; seconds={time.perf_counter()-started:.2f}")
'''),
    ("4. Zip outputs and sources", r'''
(RUN_ROOT / "notebook_stages.json").write_text(
    json.dumps({"run_ok": RUN_OK, "stages": STAGES}, indent=2), encoding="utf-8")
FINAL_ZIP = WORK_ROOT / "result_jcam_thm05_sharpening.zip"
with zipfile.ZipFile(FINAL_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(RUN_ROOT.rglob("*")):
        if path.is_file() and path.name != "source_package.zip":
            archive.write(path, arcname=(Path("thm05_sharpening") / path.relative_to(RUN_ROOT)).as_posix())
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
