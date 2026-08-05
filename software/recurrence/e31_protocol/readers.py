"""Frozen strict and post-hoc angle-only E31 final-field readers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

STRICT_RE = re.compile(r"^Final answer:\s*([A-D])\s*$")
ANGLE_RE = re.compile(r"^Final answer:\s*<([A-D])>\s*$")
CANDIDATE_RE = re.compile(r"^\s*final\s+answer\s*:", re.IGNORECASE)


@dataclass(frozen=True)
class ReadResult:
    status: str
    label: str | None
    candidate_count: int
    accepted_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_final(
    raw_output: str | None,
    *,
    finish_reason: str | None,
    hit_max_new_tokens: bool | None,
    mode: str = "STRICT",
) -> ReadResult:
    if mode not in {"STRICT", "RELAXED_ANGLE_ONLY"}:
        raise ValueError(f"unknown reader mode: {mode}")
    text = "" if raw_output is None else str(raw_output)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidate_lines = [line for line in lines if CANDIDATE_RE.match(line)]
    accepted: list[str] = []
    for line in lines:
        strict_match = STRICT_RE.fullmatch(line)
        angle_match = ANGLE_RE.fullmatch(line) if mode == "RELAXED_ANGLE_ONLY" else None
        match = strict_match or angle_match
        if match:
            accepted.append(match.group(1))
    unique = set(accepted)
    if finish_reason is None or hit_max_new_tokens is None:
        return ReadResult("MISSING_FINISH_METADATA", None, len(candidate_lines), len(accepted))
    if not lines:
        return ReadResult("EMPTY_OUTPUT", None, 0, 0)
    if len(unique) > 1:
        return ReadResult("CONFLICTING_FINAL_FIELDS", None, len(candidate_lines), len(accepted))
    if len(candidate_lines) != len(accepted):
        label = accepted[-1] if accepted else None
        return ReadResult("MALFORMED_FINAL_FIELD", label, len(candidate_lines), len(accepted))
    if accepted and (STRICT_RE.fullmatch(lines[-1]) or (mode == "RELAXED_ANGLE_ONLY" and ANGLE_RE.fullmatch(lines[-1]))):
        return ReadResult("VALID_FINAL", accepted[-1], len(candidate_lines), len(accepted))
    if accepted:
        return ReadResult("MALFORMED_FINAL_FIELD", accepted[-1], len(candidate_lines), len(accepted))
    if hit_max_new_tokens or str(finish_reason).lower() in {"length", "max_tokens", "max_new_tokens"}:
        return ReadResult("NO_FINAL_MAX_TOKENS", None, len(candidate_lines), 0)
    return ReadResult("NO_FINAL_NORMAL_END", None, len(candidate_lines), 0)
