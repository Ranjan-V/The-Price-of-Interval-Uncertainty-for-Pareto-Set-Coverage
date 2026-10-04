"""Build a cell-by-cell Kaggle follow-up for EXP-10's causal audit."""

import base64
import json
from pathlib import Path
import textwrap

from build_exp10_notebook import CELLS as BASE_CELLS


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "JCAM_EXP10_Chronological_Audit.ipynb"
AUDITOR = (ROOT / "Code" / "experiments" / "audit_bank_stream.py").read_bytes()
ENCODED = base64.b64encode(AUDITOR).decode("ascii")
ENCODED_LINES = "\n".join(f"    {ENCODED[i:i + 96]!r}," for i in range(0, len(ENCODED), 96))

CELLS = BASE_CELLS[:3] + [
    ("4. Replay matched learners and audit chronological coverage", f'''
import base64
AUDITOR_SOURCE = base64.b64decode("".join([
{ENCODED_LINES}
]))
(CODE / "experiments" / "audit_bank_stream.py").write_bytes(AUDITOR_SOURCE)
if RUN_OK:
    command = [sys.executable, str(CODE / "experiments" / "audit_bank_stream.py"),
               "--data", str(DATA), "--rounds", str(ROUND_PATH),
               "--output", str(RESULTS / "audit")]
    outcome = subprocess.run(command, cwd=CODE, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(outcome.stdout[-9000:])
    if outcome.returncode == 0:
        AUDIT = json.loads((RESULTS / "audit" / "audit.json").read_text(encoding="utf-8"))
        record("audit", "passed", f"evaluated={{AUDIT['evaluated_batches']}} batches")
    else:
        RUN_OK = False
        record("audit", "failed", f"exit={{outcome.returncode}}")
else:
    record("audit", "skipped", "earlier gate failed")
'''),
    ("5. Zip the complete audit output", r'''
import zipfile
(RESULTS / "notebook_stages.json").write_text(
    json.dumps({"run_ok": RUN_OK, "stages": STAGES}, indent=2), encoding="utf-8")
if BOOTSTRAP_ERROR is None:
    shutil.copy2(SOURCE_META, RESULTS / "bank_stream.metadata.json")
FINAL_ZIP = WORK_ROOT / "result_jcam_exp10_audit.zip"
with zipfile.ZipFile(FINAL_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(RESULTS.rglob("*")):
        if path.is_file():
            archive.write(path, arcname=(Path("exp10_bank_audit") / path.relative_to(RESULTS)).as_posix())
print("Output archive:", FINAL_ZIP, "bytes:", FINAL_ZIP.stat().st_size, "run_ok:", RUN_OK)
'''),
]


def build() -> Path:
    cells = []
    for heading, code in CELLS:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [f"## {heading}\n"]})
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                      "outputs": [], "source": textwrap.dedent(code).lstrip("\n").splitlines(keepends=True)})
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    print(build())
