"""Remove embedded manuscript artifacts from public Kaggle notebooks.

The original notebooks used base64 ZIPs to make Kaggle runs self-contained.
This script keeps their executable code and data but omits Paper/ members.
"""

from __future__ import annotations

import ast
import base64
import io
import json
from pathlib import Path
import re
import zipfile


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
ENCODED = re.compile(r'base64\.b64decode\(""\.join\(\[(.*?)\]\)\)', re.S)
DIRECT_MANUSCRIPT = {"MANUSCRIPT_SOURCE", "PAPER_PDF_SOURCE", "JOURNAL_PLAN_SOURCE"}


def encode(data: bytes) -> str:
    payload = base64.b64encode(data).decode("ascii")
    chunks = ",\n".join("    " + repr(payload[i:i + 96]) for i in range(0, len(payload), 96))
    return 'base64.b64decode("".join([\n' + chunks + '\n]))'


def strip_paper_zip(data: bytes) -> tuple[bytes, list[str]]:
    if not zipfile.is_zipfile(io.BytesIO(data)):
        return data, []
    removed: list[str] = []
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(output, "w") as destination:
        for member in source.infolist():
            if member.filename.replace("\\", "/").startswith("Paper/"):
                removed.append(member.filename)
                continue
            destination.writestr(member, source.read(member.filename))
    return (output.getvalue() if removed else data), removed


def sanitize_source(source: str) -> tuple[str, list[str]]:
    removed: list[str] = []

    def replace(match: re.Match[str]) -> str:
        chunks = ast.literal_eval("[" + match.group(1) + "]")
        data = base64.b64decode("".join(chunks))
        sanitized, members = strip_paper_zip(data)
        removed.extend(members)
        if data.startswith(b"%PDF") or data.lstrip().startswith(b"\\documentclass"):
            removed.append("standalone embedded manuscript")
            return "b''"
        return encode(sanitized) if members else match.group(0)

    source = ENCODED.sub(replace, source)
    source, plan_count = re.subn(
        r'(?ms)^JOURNAL_PLAN_SOURCE = base64\.b64decode\(""\.join\(\[.*?\]\)\)\n?',
        "", source,
    )
    if plan_count:
        removed.append("standalone embedded journal plan")
    if "standalone embedded manuscript" in removed or plan_count:
        source = "\n".join(
            line for line in source.split("\n")
            if not (
                any(name in line and "write_bytes" in line for name in DIRECT_MANUSCRIPT)
                or re.match(r"^(MANUSCRIPT_SOURCE|PAPER_PDF_SOURCE) = b''$", line)
            )
        )
    return source, removed


def verify_public_source(source: str) -> None:
    for match in ENCODED.finditer(source):
        chunks = ast.literal_eval("[" + match.group(1) + "]")
        data = base64.b64decode("".join(chunks))
        if data.startswith(b"%PDF") or data.lstrip().startswith(b"\\documentclass"):
            raise RuntimeError("A notebook still contains a standalone manuscript payload")
        if zipfile.is_zipfile(io.BytesIO(data)):
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                if any(name.replace("\\", "/").startswith("Paper/") for name in archive.namelist()):
                    raise RuntimeError("A notebook ZIP still contains Paper/ material")


def main() -> None:
    total = 0
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        removed: list[str] = []
        for cell in notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            source, deleted = sanitize_source(source)
            verify_public_source(source)
            if deleted:
                cell["source"] = source.splitlines(keepends=True)
                removed.extend(deleted)
            if cell.get("cell_type") == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
                compile("".join(cell.get("source", [])), str(path), "exec")
        if removed:
            path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            total += len(removed)
            print(path.name, ":", ", ".join(removed))
    print("removed embedded manuscript artifacts:", total)


if __name__ == "__main__":
    main()
