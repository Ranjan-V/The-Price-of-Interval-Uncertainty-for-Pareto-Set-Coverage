"""Metadata and atomic checkpoint helpers."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone


CODE_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_VERSION = "2026-10-01-v2"


def canonical_id(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def versions() -> dict:
    names = ["numpy", "matplotlib", "torch"]
    out = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def git_commit() -> str | None:
    try:
        run = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE_ROOT.parent,
                             capture_output=True, text=True, timeout=3, check=False)
        return run.stdout.strip() if run.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def valid_checkpoint(path: Path, run_id: str) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("run_id") == run_id and data.get("status") == "success" and isinstance(data.get("metrics"), dict)
    except (OSError, ValueError):
        return False
