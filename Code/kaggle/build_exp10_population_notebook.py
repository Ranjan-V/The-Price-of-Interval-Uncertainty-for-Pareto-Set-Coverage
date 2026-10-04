"""Build a five-code-cell conditional population-band Kaggle follow-up."""

import base64
import json
from pathlib import Path
import textwrap

from build_exp10_notebook import CELLS as BASE_CELLS


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "JCAM_EXP10_Population_Bands.ipynb"


def encoded_lines(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return "\n".join(f"    {encoded[i:i + 96]!r}," for i in range(0, len(encoded), 96))


SCRIPT = encoded_lines(ROOT / "Code" / "experiments" / "certify_bank_population.py")
PROOF = encoded_lines(ROOT / "Math" / "streaming_population_intervals.md")

CELLS = BASE_CELLS[:3] + [
    ("4. Compute conditional population-band widths", f'''
import base64
CERTIFIER_SOURCE = base64.b64decode("".join([
{SCRIPT}
]))
PROOF_TEXT = base64.b64decode("".join([
{PROOF}
]))
(CODE / "experiments" / "certify_bank_population.py").write_bytes(CERTIFIER_SOURCE)
RESULTS.mkdir(parents=True, exist_ok=True)
(RESULTS / "streaming_population_intervals.md").write_bytes(PROOF_TEXT)
if RUN_OK:
    command = [sys.executable, str(CODE / "experiments" / "certify_bank_population.py"),
               "--data", str(DATA), "--rounds", str(ROUND_PATH),
               "--output", str(RESULTS / "population_certificate"), "--delta", "0.05"]
    outcome = subprocess.run(command, cwd=CODE, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(outcome.stdout[-6000:])
    if outcome.returncode == 0:
        CERTIFICATE = json.loads((RESULTS / "population_certificate" /
                                  "population_band_summary.json").read_text(encoding="utf-8"))
        record("population_certificate", "passed", "conditional widths saved; IID assumption unverified")
    else:
        RUN_OK = False
        record("population_certificate", "failed", f"exit={{outcome.returncode}}")
else:
    record("population_certificate", "skipped", "earlier gate failed")
'''),
    ("5. Zip the proof and outputs", r'''
import zipfile
(RESULTS / "notebook_stages.json").write_text(
    json.dumps({"run_ok": RUN_OK, "stages": STAGES}, indent=2), encoding="utf-8")
if BOOTSTRAP_ERROR is None:
    shutil.copy2(SOURCE_META, RESULTS / "bank_stream.metadata.json")
FINAL_ZIP = WORK_ROOT / "result_jcam_exp10_population_bands.zip"
with zipfile.ZipFile(FINAL_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(RESULTS.rglob("*")):
        if path.is_file():
            archive.write(path, arcname=(Path("exp10_population_bands") / path.relative_to(RESULTS)).as_posix())
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
