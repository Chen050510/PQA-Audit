"""Load and validate the frozen 1,168-item ARC-Challenge input."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io_utils import sha256_file

LABELS = "ABCD"


def load_dataset(path: str | Path, *, expected_n: int, expected_sha256: str) -> list[dict[str, Any]]:
    target = Path(path)
    if sha256_file(target) != expected_sha256:
        raise ValueError("dataset SHA256 mismatch")
    items: list[dict[str, Any]] = []
    with target.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["qid"])
            question = str(row["question"]).strip()
            choices_raw = row["choices"]
            if not isinstance(choices_raw, dict) or set(choices_raw) != set(LABELS):
                raise ValueError(f"qid={qid} must have exactly A-D choices")
            choices = {label: str(choices_raw[label]).strip() for label in LABELS}
            gold = str(row.get("answerKey", row.get("gold_label", ""))).strip().upper()
            if not qid or not question or any(not value for value in choices.values()) or gold not in LABELS:
                raise ValueError(f"invalid item at line {line_number}")
            items.append({"qid": qid, "question": question, "choices": choices, "gold_label": gold})
    if len(items) != expected_n:
        raise ValueError(f"dataset count mismatch: {len(items)} != {expected_n}")
    qids = [item["qid"] for item in items]
    if len(set(qids)) != expected_n:
        raise ValueError("dataset qids are not unique")
    return items
