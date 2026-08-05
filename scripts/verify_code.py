#!/usr/bin/env python3
"""Validate the code-only release without loading data or model weights."""

from __future__ import annotations

import hashlib
import json
import py_compile
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOFTWARE = ROOT / "software"
MANIFEST = SOFTWARE / "MANIFEST.sha256"
ENTRY_POINTS = [
    SOFTWARE / "reproduce_key_results.py",
    SOFTWARE / "recurrence" / "scripts" / "analyze_e31.py",
    SOFTWARE / "cross_dataset" / "scripts" / "analyze_e32.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest() -> int:
    checked = 0
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        path = SOFTWARE / relative.strip()
        if not path.is_file():
            raise FileNotFoundError(f"Missing source file: {relative}")
        actual = sha256(path)
        if actual != digest:
            raise AssertionError(f"Checksum mismatch for {relative}: {actual} != {digest}")
        checked += 1
    return checked


def compile_python() -> int:
    paths = sorted(SOFTWARE.rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    for path in paths:
        py_compile.compile(str(path), doraise=True)
    return len(paths)


def parse_json_configs() -> int:
    paths = sorted(SOFTWARE.rglob("*.json"))
    for path in paths:
        json.loads(path.read_text(encoding="utf-8"))
    return len(paths)


def check_entry_points() -> int:
    for path in ENTRY_POINTS:
        subprocess.run(
            [sys.executable, str(path), "--help"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return len(ENTRY_POINTS)


def main() -> None:
    forbidden = [ROOT / "paper"]
    present = [str(path.relative_to(ROOT)) for path in forbidden if path.exists()]
    if present:
        raise AssertionError(f"Code-only release contains forbidden paths: {present}")

    result = {
        "status": "PASS",
        "manifest_files": verify_manifest(),
        "python_files_compiled": compile_python(),
        "json_configs_parsed": parse_json_configs(),
        "entry_points_checked": check_entry_points(),
        "paper_included": False,
        "sanitized_data_included": True,
        "model_loaded": False,
        "generation_run": False,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
