"""Build a self-contained, cell-by-cell Kaggle notebook for THM-11/PROP-03."""

import base64
import io
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "JCAM_THM10_Fixed_K2_Minimax.ipynb"
OUTPUT = ROOT / "JCAM_THM11_Profile_Class_Feedback.ipynb"
paths = [
    ROOT / "Code" / "experiments" / "check_profile_class_feedback.py",
    ROOT / "Math" / "proofs" / "proof_thm11_profile_class.md",
    ROOT / "Math" / "proofs" / "proof_prop03_informative_feedback.md",
    ROOT / "Math" / "proofs" / "proof_thm10.md",
    ROOT / "Math" / "theorem_index.md",
    ROOT / "Math" / "proof_audit.md",
    ROOT / "Theory" / "closest_theorem_comparison.md",
    ROOT / "Paper" / "interval_pareto_information_draft.tex",
    ROOT / "Paper" / "interval_pareto_information_draft.pdf",
    ROOT / "Paper" / "profile_class_feedback.pdf",
    ROOT / "Paper" / "journal_and_submission_plan.md",
    ROOT / "RESEARCH_STATUS.md",
]
bundle = io.BytesIO()
with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in paths:
        archive.write(path, arcname=path.relative_to(ROOT).as_posix())
payload = base64.b64encode(bundle.getvalue()).decode("ascii")
chunks = "\n".join(f"    {payload[i:i + 96]!r}," for i in range(0, len(payload), 96))

notebook = json.loads(TEMPLATE.read_text(encoding="utf-8"))
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
notebook["cells"][0]["source"] = ["## 1. Detect zipped or automatically unzipped Kaggle input\n"]
notebook["cells"][1]["source"] = [
    "from pathlib import Path\n",
    "import base64, json, os, subprocess, sys, time, zipfile\n",
    'INPUT_ROOT = Path(os.environ.get("JCAM_INPUT_ROOT", "/kaggle/input"))\n',
    'WORK_ROOT = Path(os.environ.get("JCAM_WORK_ROOT", "/kaggle/working"))\n',
    'RUN_ROOT = WORK_ROOT / "jcam_thm11_profile_class"\n',
    "RUN_ROOT.mkdir(parents=True, exist_ok=True)\n",
    "RUN_OK = True\n",
    "STAGES = []\n",
    "def record(stage, outcome, detail):\n",
    '    STAGES.append({"stage": stage, "outcome": outcome, "detail": detail})\n',
    "    print(stage, outcome, detail)\n",
    'mode = "embedded_exact_sources"\n',
    "if INPUT_ROOT.exists():\n",
    '    if any(INPUT_ROOT.rglob("check_profile_class_feedback.py")):\n',
    '        mode = "kaggle_auto_unzipped_detected; embedded exact sources selected"\n',
    '    elif any(INPUT_ROOT.rglob("*.zip")):\n',
    '        mode = "kaggle_zip_detected; embedded exact sources selected"\n',
    'record("bootstrap", "passed", mode)\n',
]
notebook["cells"][2]["source"] = ["## 2. Restore exact proof, experiment, and draft sources\n"]
notebook["cells"][3]["source"] = (
    f'SOURCE_ARCHIVE = base64.b64decode("".join([\n{chunks}\n]))\n'
    '(RUN_ROOT / "source_package.zip").write_bytes(SOURCE_ARCHIVE)\n'
    'with zipfile.ZipFile(RUN_ROOT / "source_package.zip") as archive:\n'
    '    for member in archive.infolist():\n'
    '        target = (RUN_ROOT / member.filename).resolve()\n'
    '        if not target.is_relative_to(RUN_ROOT.resolve()):\n'
    '            raise RuntimeError(f"Unsafe ZIP member: {member.filename}")\n'
    '    archive.extractall(RUN_ROOT)\n'
    'record("sources", "passed", "THM-11/PROP-03 proof, experiment, paper and audit")\n'
).splitlines(keepends=True)
notebook["cells"][4]["source"] = ["## 3. Run direct Pareto-set coverage and interval-feedback checks\n"]
notebook["cells"][5]["source"] = [
    "started = time.perf_counter()\n",
    'script = RUN_ROOT / "Code" / "experiments" / "check_profile_class_feedback.py"\n',
    "result = subprocess.run([sys.executable, str(script)], cwd=RUN_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)\n",
    "print(result.stdout[-6000:])\n",
    "if result.returncode:\n",
    "    RUN_OK = False\n",
    '    record("coverage_check", "failed", f"exit={result.returncode}")\n',
    "else:\n",
    '    summary_path = RUN_ROOT / "Code" / "results" / "profile_class_feedback" / "profile_class_feedback_summary.json"\n',
    "    summary = json.loads(summary_path.read_text())\n",
    '    RUN_OK = bool(summary["all_checks_passed"] and summary["rows"] == 64 and summary["endpoint_prior_grid_checks"] == 1936 and summary["quadratic_identity_checks"] == 315)\n',
    '    record("coverage_check", "passed" if RUN_OK else "failed", f"rows={summary[\'rows\']}; prior_pairs={summary[\'endpoint_prior_grid_checks\']}; seconds={time.perf_counter()-started:.2f}")\n',
]
notebook["cells"][6]["source"] = ["## 4. Zip all outputs and exact sources\n"]
notebook["cells"][7]["source"] = [
    '(RUN_ROOT / "notebook_stages.json").write_text(json.dumps({"run_ok": RUN_OK, "stages": STAGES}, indent=2), encoding="utf-8")\n',
    'FINAL_ZIP = WORK_ROOT / "result_jcam_thm11_profile_class.zip"\n',
    'with zipfile.ZipFile(FINAL_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:\n',
    '    for path in sorted(RUN_ROOT.rglob("*")):\n',
    '        if path.is_file() and path.name != "source_package.zip":\n',
    '            archive.write(path, arcname=(Path("thm11_profile_class") / path.relative_to(RUN_ROOT)).as_posix())\n',
    'print("Output archive:", FINAL_ZIP, "bytes:", FINAL_ZIP.stat().st_size, "run_ok:", RUN_OK)\n',
]
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        compile("".join(cell["source"]), "notebook_cell", "exec")
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT)
