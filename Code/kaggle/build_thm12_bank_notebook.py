"""Build a self-contained cell-by-cell Kaggle notebook for THM-12 and APP-02."""

import base64
import io
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "JCAM_THM11_Profile_Class_Feedback.ipynb"
OUTPUT = ROOT / "JCAM_THM12_Bank_Coverage.ipynb"
paths = [
    ROOT / "Code" / "experiments" / "real_bank_pareto_coverage.py",
    ROOT / "Code" / "experiments" / "prepare_bank_marketing.py",
    ROOT / "Code" / "data" / "bank_marketing" / "bank_stream.npz",
    ROOT / "Code" / "data" / "bank_marketing" / "bank_stream.metadata.json",
    ROOT / "Code" / "data" / "bank_marketing" / "README.md",
    ROOT / "Math" / "proofs" / "proof_thm12_unified_interval_query.md",
    ROOT / "Math" / "proofs" / "proof_bank_finite_pool_coverage.md",
    ROOT / "Math" / "proofs" / "proof_thm11_profile_class.md",
    ROOT / "Math" / "theorem_index.md",
    ROOT / "Math" / "proof_audit.md",
    ROOT / "Theory" / "closest_theorem_comparison.md",
    ROOT / "Paper" / "interval_pareto_information_draft.tex",
    ROOT / "Paper" / "interval_pareto_information_draft.pdf",
    ROOT / "Paper" / "profile_class_feedback.pdf",
    ROOT / "Paper" / "bank_pareto_coverage.pdf",
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
    'RUN_ROOT = WORK_ROOT / "jcam_thm12_bank_coverage"\n',
    "RUN_ROOT.mkdir(parents=True, exist_ok=True)\n",
    "RUN_OK = True\n",
    "STAGES = []\n",
    "def record(stage, outcome, detail):\n",
    '    STAGES.append({"stage": stage, "outcome": outcome, "detail": detail})\n',
    "    print(stage, outcome, detail)\n",
    'mode = "embedded_processed_source_with_verified_UCI_hash"\n',
    "if INPUT_ROOT.exists():\n",
    '    if any(INPUT_ROOT.rglob("bank_marketing_uci.zip")):\n',
    '        mode = "kaggle_zip_detected; raw source will be verified"\n',
    '    elif any(INPUT_ROOT.rglob("bank-additional-full.csv")):\n',
    '        mode = "kaggle_auto_unzipped_detected; embedded processed source selected"\n',
    '    elif any(INPUT_ROOT.rglob("*.zip")):\n',
    '        mode = "other_kaggle_zip_detected; embedded processed source selected"\n',
    'record("bootstrap", "passed", mode)\n',
]
notebook["cells"][2]["source"] = ["## 2. Restore dataset, proof, code, and draft\n"]
notebook["cells"][3]["source"] = (
    f'SOURCE_ARCHIVE = base64.b64decode("".join([\n{chunks}\n]))\n'
    '(RUN_ROOT / "source_package.zip").write_bytes(SOURCE_ARCHIVE)\n'
    'with zipfile.ZipFile(RUN_ROOT / "source_package.zip") as archive:\n'
    '    for member in archive.infolist():\n'
    '        target = (RUN_ROOT / member.filename).resolve()\n'
    '        if not target.is_relative_to(RUN_ROOT.resolve()):\n'
    '            raise RuntimeError(f"Unsafe ZIP member: {member.filename}")\n'
    '    archive.extractall(RUN_ROOT)\n'
    'record("sources", "passed", "UCI data, THM-12/APP-02 proofs, experiment, draft and audit")\n'
).splitlines(keepends=True)
notebook["cells"][4]["source"] = ["## 3. Run exact finite-pool Pareto coverage experiment\n"]
notebook["cells"][5]["source"] = [
    "started = time.perf_counter()\n",
    'script = RUN_ROOT / "Code" / "experiments" / "real_bank_pareto_coverage.py"\n',
    'data = RUN_ROOT / "Code" / "data" / "bank_marketing" / "bank_stream.npz"\n',
    'raw = next(INPUT_ROOT.rglob("bank_marketing_uci.zip"), None) if INPUT_ROOT.exists() else None\n',
    'output = RUN_ROOT / "Code" / "results" / "real_bank_pareto_coverage"\n',
    'cmd = [sys.executable, str(script), "--data", str(data), "--output", str(output)]\n',
    'if raw is not None:\n',
    '    cmd.extend(["--raw", str(raw)])\n',
    'result = subprocess.run(cmd, cwd=RUN_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)\n',
    'print(result.stdout[-6000:])\n',
    'if result.returncode:\n',
    '    RUN_OK = False\n',
    '    record("bank_coverage", "failed", f"exit={result.returncode}")\n',
    'else:\n',
    '    summary = json.loads((output / "bank_pareto_coverage_summary.json").read_text())\n',
    '    RUN_OK = bool(summary["chronological_batches"] == 128 and summary["containment_checks"] == 1920 and summary["oracle_checks"] == 1920 and len(summary["by_fraction"]) == 3)\n',
    '    record("bank_coverage", "passed" if RUN_OK else "failed", f"checks={summary[\'containment_checks\']}; ratios={[round(r[\'adaptive_to_blind_ratio\'], 4) for r in summary[\'by_fraction\']]}; seconds={time.perf_counter()-started:.2f}")\n',
]
notebook["cells"][6]["source"] = ["## 4. Zip outputs, exact data, and manuscript sources\n"]
notebook["cells"][7]["source"] = [
    '(RUN_ROOT / "notebook_stages.json").write_text(json.dumps({"run_ok": RUN_OK, "stages": STAGES}, indent=2), encoding="utf-8")\n',
    'FINAL_ZIP = WORK_ROOT / "result_jcam_thm12_bank_coverage.zip"\n',
    'with zipfile.ZipFile(FINAL_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:\n',
    '    for path in sorted(RUN_ROOT.rglob("*")):\n',
    '        if path.is_file() and path.name != "source_package.zip":\n',
    '            archive.write(path, arcname=(Path("thm12_bank_coverage") / path.relative_to(RUN_ROOT)).as_posix())\n',
    'print("Output archive:", FINAL_ZIP, "bytes:", FINAL_ZIP.stat().st_size, "run_ok:", RUN_OK)\n',
]
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        compile("".join(cell["source"]), "notebook_cell", "exec")
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT)
