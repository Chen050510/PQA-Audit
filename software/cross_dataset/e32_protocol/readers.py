"""Strict terminal-field reader parameterized only by dataset label universe."""
from __future__ import annotations
from dataclasses import asdict,dataclass
import re
from typing import Any

@dataclass(frozen=True)
class ReadResult:
    status:str; label:str|None; candidate_count:int; accepted_count:int
    def to_dict(self)->dict[str,Any]: return asdict(self)


def read_final(raw_output: str|None, *, finish_reason: str|None, hit_max_new_tokens: bool|None, labels: str) -> ReadResult:
    if not labels or len(set(labels))!=len(labels): raise ValueError("invalid label universe")
    exact=re.compile(rf"^Final answer:\s*([{re.escape(labels)}])\s*$")
    candidate=re.compile(r"^\s*final\s+answer\s*:",re.I)
    lines=[x.strip() for x in ("" if raw_output is None else str(raw_output)).splitlines() if x.strip()]
    candidates=[x for x in lines if candidate.match(x)]; accepted=[m.group(1) for x in lines if (m:=exact.fullmatch(x))]; unique=set(accepted)
    if finish_reason is None or hit_max_new_tokens is None: return ReadResult("MISSING_FINISH_METADATA",None,len(candidates),len(accepted))
    if not lines: return ReadResult("EMPTY_OUTPUT",None,0,0)
    if len(unique)>1: return ReadResult("CONFLICTING_FINAL_FIELDS",None,len(candidates),len(accepted))
    if len(candidates)!=len(accepted): return ReadResult("MALFORMED_FINAL_FIELD",accepted[-1] if accepted else None,len(candidates),len(accepted))
    if accepted and exact.fullmatch(lines[-1]): return ReadResult("VALID_FINAL",accepted[-1],len(candidates),len(accepted))
    if accepted: return ReadResult("MALFORMED_FINAL_FIELD",accepted[-1],len(candidates),len(accepted))
    if hit_max_new_tokens or str(finish_reason).lower() in {"length","max_tokens","max_new_tokens"}: return ReadResult("NO_FINAL_MAX_TOKENS",None,len(candidates),0)
    return ReadResult("NO_FINAL_NORMAL_END",None,len(candidates),0)
