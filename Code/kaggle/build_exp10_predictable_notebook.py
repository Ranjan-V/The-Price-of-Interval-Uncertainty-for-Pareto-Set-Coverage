"""Build the seven-code-cell EXP-10 predictable-comparator follow-up."""

import base64
import json
from pathlib import Path
import textwrap

from build_exp10_controlled_notebook import CELLS as BASE_CELLS


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "JCAM_EXP10_Predictable_Comparator.ipynb"


def encoded_lines(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return "\n".join(f"    {encoded[i:i + 96]!r}," for i in range(0, len(encoded), 96))


CERT_SOURCE = encoded_lines(ROOT / "Code" / "experiments" / "predictable_comparator_certificate.py")
CHECK_SOURCE = encoded_lines(ROOT / "Code" / "experiments" / "check_predictable_controlled.py")
PROOF = encoded_lines(ROOT / "Math" / "predictable_comparator_intervals.md")

CELLS = BASE_CELLS[:5] + [
    ("6. Check the predictable-comparator certificate", f'''
PREDICTABLE_SOURCE = base64.b64decode("".join([
{CERT_SOURCE}
]))
CHECK_SOURCE = base64.b64decode("".join([
{CHECK_SOURCE}
]))
PREDICTABLE_PROOF = base64.b64decode("".join([
{PROOF}
]))
(CODE / "experiments" / "predictable_comparator_certificate.py").write_bytes(PREDICTABLE_SOURCE)
(CODE / "experiments" / "check_predictable_controlled.py").write_bytes(CHECK_SOURCE)
(RESULTS / "predictable_comparator_intervals.md").write_bytes(PREDICTABLE_PROOF)
if RUN_OK:
    for stage, script, destination in [
        ("predictable_certificate", "predictable_comparator_certificate.py", "predictable_comparator"),
        ("predictable_controlled", "check_predictable_controlled.py", "predictable_controlled"),
    ]:
        command = [sys.executable, str(CODE / "experiments" / script),
                   "--data", str(DATA), "--output", str(RESULTS / destination)]
        outcome = subprocess.run(command, cwd=CODE, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(outcome.stdout[-7000:])
        if outcome.returncode == 0:
            record(stage, "passed", destination)
        else:
            RUN_OK = False
            record(stage, "failed", f"exit={{outcome.returncode}}")
            break
else:
    record("predictable_certificate", "skipped", "earlier gate failed")
    record("predictable_controlled", "skipped", "earlier gate failed")
'''),
    ("7. Zip all EXP-10 outputs", r'''
import zipfile
(RESULTS / "notebook_stages.json").write_text(
    json.dumps({"run_ok": RUN_OK, "stages": STAGES}, indent=2), encoding="utf-8")
if BOOTSTRAP_ERROR is None:
    shutil.copy2(SOURCE_META, RESULTS / "bank_stream.metadata.json")
FINAL_ZIP = WORK_ROOT / "result_jcam_exp10_predictable_comparator.zip"
with zipfile.ZipFile(FINAL_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(RESULTS.rglob("*")):
        if path.is_file():
            archive.write(path, arcname=(Path("exp10_predictable_comparator") / path.relative_to(RESULTS)).as_posix())
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
