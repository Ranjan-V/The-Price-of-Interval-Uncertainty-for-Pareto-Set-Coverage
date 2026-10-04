"""Build the Kaggle input archive from source, excluding local execution state.

Paper/ is intentionally neither read nor packaged.
"""

from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "jcam_input_package.zip"
EXCLUDED_DIRS = {".venv", "results", "data", "__pycache__", ".pytest_cache", ".git"}


def build() -> Path:
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for folder in ["Code", "Math", "Theory"]:
            source = ROOT / folder
            for path in sorted(source.rglob("*")):
                if not path.is_file() or any(part in EXCLUDED_DIRS for part in path.relative_to(source).parts):
                    continue
                archive.write(path, arcname=(Path("JCAM") / path.relative_to(ROOT)).as_posix())
        for filename in ["RESEARCH_STATUS.md", "JCAM_GPU_Experiments.ipynb"]:
            path = ROOT / filename
            if path.is_file():
                archive.write(path, arcname=(Path("JCAM") / filename).as_posix())
    return OUTPUT


if __name__ == "__main__":
    print(build())
