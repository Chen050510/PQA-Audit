#!/usr/bin/env python3
"""Verify the sanitized outputs and reproduce selected paired summaries."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ARCHIVE = ROOT / "data" / "anonymous_data.zip"
CHECKSUMS = ROOT / "SHA256SUMS"
OUTPUT_DIR = ROOT / "reproduced"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums() -> int:
    checked = 0
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        path = ROOT / relative.strip()
        if not path.is_file():
            raise FileNotFoundError(f"Missing release file: {relative}")
        actual = sha256(path)
        if actual != digest:
            raise AssertionError(f"Checksum mismatch for {relative}: {actual} != {digest}")
        checked += 1
    return checked


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if target != destination and destination not in target.parents:
            raise ValueError(f"Unsafe archive path: {member.filename}")
    archive.extractall(destination)


def reproduce() -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    output = OUTPUT_DIR / "reproduced_key_results.json"
    with tempfile.TemporaryDirectory(prefix="direction4-results-") as tmp:
        extract_root = Path(tmp)
        with zipfile.ZipFile(DATA_ARCHIVE) as archive:
            safe_extract(archive, extract_root)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "software" / "reproduce_key_results.py"),
                "--data-root",
                str(extract_root / "anonymous_data"),
                "--output",
                str(output),
            ],
            check=True,
        )
    return output


def main() -> None:
    checked = verify_checksums()
    output = reproduce()
    result = json.loads(output.read_text(encoding="utf-8"))
    if result.get("status") != "PASS":
        raise AssertionError(f"Result reproduction did not pass: {result.get('status')}")
    print(
        json.dumps(
            {
                "status": "PASS",
                "checksums_verified": checked,
                "data_archive_sha256": sha256(DATA_ARCHIVE),
                "reproduced_output": str(output.relative_to(ROOT)),
                "result_status": result["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
