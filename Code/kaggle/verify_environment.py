"""Stage A: Kaggle environment and optional CUDA inventory."""

import json
import platform
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import versions, utc_now, atomic_json, CODE_ROOT


def inspect() -> dict:
    result = {"timestamp_utc": utc_now(), "python": platform.python_version(),
              "platform": platform.platform(), "packages": versions(), "cuda_available": False,
              "cuda_device": None, "cpu_count": __import__("os").cpu_count()}
    try:
        import torch
        result["cuda_available"] = bool(torch.cuda.is_available())
        if result["cuda_available"]:
            result["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return result


if __name__ == "__main__":
    outcome = inspect()
    output = CODE_ROOT / "results" / "logs" / "kaggle_environment.json"
    atomic_json(output, outcome)
    print(json.dumps(outcome, indent=2))
