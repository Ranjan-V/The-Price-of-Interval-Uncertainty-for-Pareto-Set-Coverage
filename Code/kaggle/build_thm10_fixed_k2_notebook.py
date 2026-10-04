"""Build a self-contained Kaggle notebook for THM-10 and its K=2 lemma."""

import base64
import io
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "JCAM_K2_Fixed_Regularity_Upper.ipynb"
OUTPUT = ROOT / "JCAM_THM10_Fixed_K2_Minimax.ipynb"
paths = [
    ROOT / "Code" / "experiments" / "check_k2_coverage_identity.py",
    ROOT / "Code" / "experiments" / "check_thm10_fixed_k2.py",
    ROOT / "Math" / "proofs" / "proof_k2_coverage_identity.md",
    ROOT / "Math" / "proofs" / "proof_thm09.md",
    ROOT / "Math" / "proofs" / "proof_thm10.md",
    ROOT / "Math" / "theorem_index.md",
    ROOT / "Math" / "proof_audit.md",
    ROOT / "Paper" / "interval_pareto_information_draft.tex",
    ROOT / "Paper" / "interval_pareto_information_draft.pdf",
    ROOT / "Paper" / "journal_and_submission_plan.md",
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
notebook["cells"][0]["source"] = ["## 1. Detect zipped or automatically unzipped input\n"]
notebook["cells"][1]["source"] = [
    'from pathlib import Path\n',
    'import base64, json, os, subprocess, sys, time, zipfile\n',
    'INPUT_ROOT = Path(os.environ.get("JCAM_INPUT_ROOT", "/kaggle/input"))\n',
    'WORK_ROOT = Path(os.environ.get("JCAM_WORK_ROOT", "/kaggle/working"))\n',
    'RUN_ROOT = WORK_ROOT / "jcam_thm10_fixed_k2"\n',
    'RUN_ROOT.mkdir(parents=True, exist_ok=True)\n',
    'RUN_OK = True\n',
    'STAGES = []\n',
    'def record(stage, outcome, detail):\n',
    '    STAGES.append({"stage": stage, "outcome": outcome, "detail": detail})\n',
    '    print(stage, outcome, detail)\n',
    'mode = "embedded_exact_sources"\n',
    'if INPUT_ROOT.exists():\n',
    '    if any(INPUT_ROOT.rglob("check_thm10_fixed_k2.py")):\n',
    '        mode = "kaggle_auto_unzipped_detected; embedded exact sources selected"\n',
    '    elif any(INPUT_ROOT.rglob("*.zip")):\n',
    '        mode = "kaggle_zip_detected; embedded exact sources selected"\n',
    'record("bootstrap", "passed", mode)\n',
]
notebook["cells"][2]["source"] = ["## 2. Install exact source package from notebook cell\n"]
notebook["cells"][3]["source"] = (
    f'SOURCE_ARCHIVE = base64.b64decode("".join([\n{chunks}\n]))\n'
    '(RUN_ROOT / "source_package.zip").write_bytes(SOURCE_ARCHIVE)\n'
    'with zipfile.ZipFile(RUN_ROOT / "source_package.zip") as archive:\n'
    '    for member in archive.infolist():\n'
    '        target = (RUN_ROOT / member.filename).resolve()\n'
    '        if not target.is_relative_to(RUN_ROOT.resolve()):\n'
    '            raise RuntimeError(f"Unsafe ZIP member: {member.filename}")\n'
    '    archive.extractall(RUN_ROOT)\n'
    'record("sources", "passed", "THM-10 checks and proof, manuscript source and PDF")\n'
).splitlines(keepends=True)
notebook["cells"][4]["source"] = ["## 3. Run K=2 coverage and THM-10 diagnostics\n"]
notebook["cells"][5]["source"] = [
    'for script, summary_relpath, expected_cases in [\n',
    '    ("check_k2_coverage_identity.py", "fixed_k2_coverage/fixed_k2_coverage_summary.json", 5),\n',
    '    ("check_thm10_fixed_k2.py", "fixed_k2_minimax/thm10_fixed_k2_summary.json", 10201),\n',
    ']:\n',
    '    if not RUN_OK:\n',
    '        break\n',
    '    started = time.perf_counter()\n',
    '    command = [sys.executable, str(RUN_ROOT / "Code" / "experiments" / script)]\n',
    '    result = subprocess.run(command, cwd=RUN_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)\n',
    '    print(result.stdout[-5000:])\n',
    '    if result.returncode:\n',
    '        RUN_OK = False\n',
    '        record(script, "failed", f"exit={result.returncode}")\n',
    '    else:\n',
    '        summary = json.loads((RUN_ROOT / "Code" / "results" / summary_relpath).read_text())\n',
    '        count = summary.get("cases", summary.get("local_grid_checks"))\n',
    '        RUN_OK = bool(summary["all_checks_passed"] and count == expected_cases)\n',
    '        record(script, "passed" if RUN_OK else "failed", f"checks={count}; seconds={time.perf_counter()-started:.2f}")\n',
]
notebook["cells"][6]["source"] = ["## 4. Zip outputs and sources\n"]
notebook["cells"][7]["source"] = [
    '(RUN_ROOT / "notebook_stages.json").write_text(\n',
    '    json.dumps({"run_ok": RUN_OK, "stages": STAGES}, indent=2), encoding="utf-8")\n',
    'FINAL_ZIP = WORK_ROOT / "result_jcam_thm10_fixed_k2.zip"\n',
    'with zipfile.ZipFile(FINAL_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:\n',
    '    for path in sorted(RUN_ROOT.rglob("*")):\n',
    '        if path.is_file() and path.name != "source_package.zip":\n',
    '            archive.write(path, arcname=(Path("thm10_fixed_k2") / path.relative_to(RUN_ROOT)).as_posix())\n',
    'print("Output archive:", FINAL_ZIP, "bytes:", FINAL_ZIP.stat().st_size, "run_ok:", RUN_OK)\n',
]
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT)
