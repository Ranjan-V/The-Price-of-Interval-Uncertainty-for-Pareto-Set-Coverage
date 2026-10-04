"""Generate the importable Kaggle notebook with one explicit cell per stage."""

import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "JCAM_GPU_Experiments.ipynb"


CELLS = [
    ("1. Locate zipped or auto-extracted Kaggle input", r'''
from pathlib import Path
import os, shutil, zipfile

INPUT_ROOT = Path(os.environ.get("JCAM_INPUT_ROOT", "/kaggle/input"))
WORK_ROOT = Path(os.environ.get("JCAM_WORK_ROOT", "/kaggle/working"))
DEST = WORK_ROOT / "JCAM"
CODE = DEST / "Code"
RESULTS = CODE / "results"
BOOTSTRAP_ERROR = None

def find_code_roots(base):
    return sorted({path.parent.parent for path in base.rglob("run_all.py")
                   if path.parent.name == "kaggle" and path.parent.parent.name == "Code"})

try:
    candidates = find_code_roots(INPUT_ROOT)
    if not candidates:
        archives = sorted(INPUT_ROOT.rglob("*.zip"))
        matching = []
        for archive_path in archives:
            with zipfile.ZipFile(archive_path) as archive:
                if any(name.replace("\\", "/").endswith("Code/kaggle/run_all.py")
                       for name in archive.namelist()):
                    matching.append(archive_path)
        if len(matching) != 1:
            raise RuntimeError(f"Expected one JCAM archive, found {matching}")
        staging = WORK_ROOT / "_jcam_input_extracted"
        staging.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(matching[0]) as archive:
            for member in archive.infolist():
                target = (staging / member.filename).resolve()
                if not target.is_relative_to(staging.resolve()):
                    raise RuntimeError(f"Unsafe archive path: {member.filename}")
            archive.extractall(staging)
        candidates = find_code_roots(staging)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one extracted Code root, found {candidates}")
    source_code = candidates[0]
    source_root = source_code.parent
    DEST.mkdir(parents=True, exist_ok=True)
    for name in ("Code", "Math", "Theory"):
        source = source_root / name
        if source.exists():
            shutil.copytree(source, DEST / name, dirs_exist_ok=True)
    status_file = source_root / "RESEARCH_STATUS.md"
    if status_file.is_file():
        shutil.copy2(status_file, DEST / status_file.name)
    if not (CODE / "kaggle" / "run_all.py").is_file():
        raise RuntimeError("JCAM code was not copied into Kaggle working storage")
    RESULTS.mkdir(parents=True, exist_ok=True)
    PACKAGE_SOURCE = str(source_root)
    print("Prepared", CODE, "from", PACKAGE_SOURCE)
except Exception as error:
    BOOTSTRAP_ERROR = repr(error)
    RESULTS.mkdir(parents=True, exist_ok=True)
    print("Bootstrap failed:", BOOTSTRAP_ERROR)
'''),
    ("2. Set up gated background-run helpers", r'''
import json, subprocess, sys, time

RUN_OK = BOOTSTRAP_ERROR is None
STAGES = []
CONFIG = CODE / "kaggle" / "configs" / "full.json"
GPU_COUNT = 0

def record(label, outcome, seconds=0.0, detail=None):
    STAGES.append({"stage": label, "outcome": outcome, "seconds": seconds, "detail": detail})
    print(label, outcome, detail or "")

def run_command(label, command, env=None):
    global RUN_OK
    if not RUN_OK:
        record(label, "skipped", detail="earlier gate failed")
        return False
    started = time.perf_counter()
    finished = subprocess.run(command, cwd=CODE, env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(finished.stdout[-5000:])
    ok = finished.returncode == 0
    record(label, "passed" if ok else "failed", time.perf_counter()-started,
           f"exit={finished.returncode}")
    RUN_OK = RUN_OK and ok
    return ok

def staged_command(stage):
    return [sys.executable, str(CODE / "kaggle" / "run_all.py"),
            "--stage", stage, "--config", str(CONFIG)]

def two_gpu_sweep(label, experiments):
    global RUN_OK
    if not RUN_OK:
        record(label, "skipped", detail="earlier gate failed")
        return
    if GPU_COUNT < 2:
        run_command(label, staged_command(label) + ["--device", "cuda"])
        return
    logs = RESULTS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    running = []
    for index in (0, 1):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(index)
        command = [sys.executable, str(CODE / "kaggle" / "run_synthetic.py"),
                   "--config", str(CONFIG), "--experiments", *experiments,
                   "--seed-mode", "full", "--device", "cuda",
                   "--seed-mod", "2", "--seed-rem", str(index)]
        path = logs / f"stage{label}_gpu{index}.log"
        handle = path.open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=CODE, env=environment,
                                   stdout=handle, stderr=subprocess.STDOUT)
        running.append((process, handle, path))
    outcomes = []
    for process, handle, path in running:
        status = process.wait()
        handle.close()
        outcomes.append((str(path), status))
    ok = all(status == 0 for _, status in outcomes)
    record(label, "passed" if ok else "failed", time.perf_counter()-started, outcomes)
    RUN_OK = RUN_OK and ok

print("Notebook gates initialized; full configuration:", CONFIG)
'''),
    ("3. Stage A — inspect T4 environment", r'''
if run_command("A", staged_command("A")):
    try:
        import torch
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        GPU_COUNT = sum("T4" in name.upper() for name in names)
        if GPU_COUNT < 1 or not torch.cuda.is_available():
            raise RuntimeError(f"Tesla T4 CUDA device unavailable: {names}")
        print("T4 devices:", names, "using", min(GPU_COUNT, 2))
        GPU_COUNT = min(GPU_COUNT, 2)
    except Exception as error:
        RUN_OK = False
        record("A-T4", "failed", detail=repr(error))
'''),
    ("4. Stage B — smoke and CPU/CUDA parity on every visible GPU", r'''
if run_command("B", staged_command("B")):
    report = json.loads((RESULTS / "logs" / "stageB.json").read_text(encoding="utf-8"))
    if report.get("gpu_parity") != "passed" or len(report.get("gpu_devices_tested", [])) < GPU_COUNT:
        RUN_OK = False
        record("B-parity", "failed", detail=report)
    else:
        record("B-parity", "passed", detail=report["gpu_devices_tested"])
'''),
    ("5. Stage C — one representative seed per experiment", r'''
run_command("C", staged_command("C") + ["--device", "auto"])
'''),
    ("6. Stage D — full synthetic seed sweep", r'''
two_gpu_sweep("D", ["EXP-01", "EXP-02", "EXP-03", "EXP-05",
                    "EXP-06", "EXP-07", "EXP-08"])
'''),
    ("7. Stage E — real-data experiment deferred", r'''
# The project owner confirmed that no real chronological NPZ dataset is supplied.
# Do not substitute toy data for a real application result.
reason = "No supplied real dataset; EXP-10 deferred by user instruction."
skip = RESULTS / "logs" / "stageE_skipped.json"
skip.parent.mkdir(parents=True, exist_ok=True)
skip.write_text(json.dumps({"reason": reason}, indent=2), encoding="utf-8")
record("E", "skipped", detail=reason)
'''),
    ("8. Stage F — endpoint and static ablations", r'''
two_gpu_sweep("F", ["EXP-09"])
'''),
    ("9. Stage G — horizon scaling", r'''
two_gpu_sweep("G", ["EXP-04"])
'''),
    ("10. Stage H — complete-seed tables and figures", r'''
run_command("H", staged_command("H"))
'''),
    ("11. Bundle outputs as result_jcam.zip", r'''
import zipfile

RESULTS.mkdir(parents=True, exist_ok=True)
logs = RESULTS / "logs"
logs.mkdir(parents=True, exist_ok=True)
(logs / "notebook_stages.json").write_text(json.dumps({"run_ok": RUN_OK, "stages": STAGES},
                                             indent=2), encoding="utf-8")
FINAL_ZIP = WORK_ROOT / "result_jcam.zip"
with zipfile.ZipFile(FINAL_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(RESULTS.rglob("*")):
        if path.is_file():
            archive.write(path, arcname=(Path("results") / path.relative_to(RESULTS)).as_posix())
print("Output archive:", FINAL_ZIP, "bytes:", FINAL_ZIP.stat().st_size,
      "pipeline_ok:", RUN_OK)
'''),
]


def build() -> Path:
    cells = []
    for title, source in CELLS:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [f"## {title}\n"]})
        code = textwrap.dedent(source).strip() + "\n"
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                      "outputs": [], "source": code.splitlines(keepends=True)})
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
        "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5}
    OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    print(build())
