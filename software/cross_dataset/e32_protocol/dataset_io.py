"""Load frozen CommonsenseQA and OpenBookQA manifests."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .io_utils import sha256_file


def load_dataset(path: str|Path, *, expected_n: int, expected_sha256: str, labels: str, dataset: str) -> list[dict[str,Any]]:
    target=Path(path)
    if sha256_file(target)!=expected_sha256: raise ValueError(f"{dataset} SHA256 mismatch")
    items=[]
    for line_no,line in enumerate(target.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        row=json.loads(line); qid=str(row["qid"]); question=str(row["question"]).strip(); raw=row["choices"]
        if not isinstance(raw,list) or len(raw)!=len(labels): raise ValueError(f"{qid}: choice count mismatch")
        choices={label:str(text).strip() for label,text in zip(labels,raw)}
        gold=str(row.get("answer",row.get("answerKey",""))).strip().upper()
        if not qid or not question or any(not x for x in choices.values()) or gold not in labels: raise ValueError(f"invalid row {line_no}")
        items.append({"qid":qid,"question":question,"choices":choices,"gold_label":gold,"dataset":dataset})
    if len(items)!=expected_n or len({x["qid"] for x in items})!=expected_n: raise ValueError(f"{dataset} count/qid mismatch")
    return items
