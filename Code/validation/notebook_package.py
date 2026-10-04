"""Validate import notebook syntax and both Kaggle input layouts locally."""

import json
import os
from pathlib import Path
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "JCAM_GPU_Experiments.ipynb"
ARCHIVE = ROOT / "jcam_input_package.zip"


def check() -> dict:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    if len(sources) != 11:
        raise AssertionError("expected eleven executable notebook cells")
    for index, source in enumerate(sources, start=1):
        compile(source, f"notebook-cell-{index}", "exec")
    with zipfile.ZipFile(ARCHIVE) as package:
        names = package.namelist()
        assert "JCAM/Code/kaggle/run_all.py" in names
        assert "JCAM/JCAM_GPU_Experiments.ipynb" in names
        assert not any(name.startswith("JCAM/Paper/") for name in names)
        assert not any("/.venv/" in name or "/results/" in name for name in names)
        for mode in ("zip", "auto_extracted"):
            with tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                input_root, work_root = base / "input", base / "working"
                input_root.mkdir()
                work_root.mkdir()
                if mode == "zip":
                    (input_root / "jcam_input_package.zip").write_bytes(ARCHIVE.read_bytes())
                else:
                    package.extractall(input_root / "dataset")
                old_input = os.environ.get("JCAM_INPUT_ROOT")
                old_work = os.environ.get("JCAM_WORK_ROOT")
                try:
                    os.environ["JCAM_INPUT_ROOT"] = str(input_root)
                    os.environ["JCAM_WORK_ROOT"] = str(work_root)
                    namespace = {}
                    exec(compile(sources[0], "notebook-cell-1", "exec"), namespace)
                    assert namespace["BOOTSTRAP_ERROR"] is None
                    assert (work_root / "JCAM" / "Code" / "kaggle" / "run_all.py").is_file()
                finally:
                    if old_input is None:
                        os.environ.pop("JCAM_INPUT_ROOT", None)
                    else:
                        os.environ["JCAM_INPUT_ROOT"] = old_input
                    if old_work is None:
                        os.environ.pop("JCAM_WORK_ROOT", None)
                    else:
                        os.environ["JCAM_WORK_ROOT"] = old_work
    return {"cells_compiled": len(sources), "input_layouts": ["zip", "auto_extracted"],
            "paper_excluded": True}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
