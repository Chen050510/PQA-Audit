"""Frozen E31 prompt fixture loader and renderer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io_utils import sha256_file, sha256_text

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "config/prompt_contract.json"
EXPECTED_SHA256 = "9d03fa8a7d9145f63c7ef26b054d96257d99441b37a55a0508c44806451d73af"
CONDITIONS = ("D", "P0", "P1", "P2")
REPLICAS = (1, 2)


def load_contract() -> dict[str, Any]:
    if sha256_file(FIXTURE) != EXPECTED_SHA256:
        raise ValueError("E31 prompt fixture SHA256 mismatch")
    contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if tuple(contract["conditions"]) != CONDITIONS:
        raise ValueError("E31 prompt condition order mismatch")
    tails = []
    for condition in ("P0", "P1", "P2"):
        template = contract["conditions"][condition]["template"]
        tails.append("\n".join(template.splitlines()[1:]))
    if len(set(tails)) != 1:
        raise ValueError("Process variants differ beyond the action line")
    return contract


def render_prompt(item: dict[str, Any], condition: str) -> str:
    contract = load_contract()
    if condition not in CONDITIONS:
        raise ValueError(f"unknown E31 condition: {condition}")
    return contract["conditions"][condition]["template"].format(
        question=item["question"],
        choice_A=item["choices"]["A"],
        choice_B=item["choices"]["B"],
        choice_C=item["choices"]["C"],
        choice_D=item["choices"]["D"],
    )


def messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def prompt_hash(prompt: str) -> str:
    return sha256_text(prompt)
