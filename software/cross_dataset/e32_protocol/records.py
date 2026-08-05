"""Duplicate-safe E32 writer."""
from __future__ import annotations
from pathlib import Path
from typing import Any,Iterable,Mapping
from .io_utils import atomic_jsonl,read_jsonl
REQUIRED={"qid","dataset","condition","model","rendered_prompt_hash","raw_output","generated_token_ids","finish_reason","hit_max_new_tokens","gold_label"}

def validate_record(row:Mapping[str,Any])->None:
    missing=REQUIRED-set(row)
    if missing: raise ValueError(f"missing: {sorted(missing)}")
    if row["condition"] not in {"D","P"} or row["dataset"] not in {"commonsenseqa","openbookqa"}: raise ValueError("bad cell")

class IncrementalWriter:
    def __init__(self,run_dir:str|Path,*,resume:bool)->None:
        self.run_dir=Path(run_dir); self.partial_path=self.run_dir/"records.partial.jsonl"; self.complete_path=self.run_dir/"records.complete.jsonl"
        if self.complete_path.exists(): raise FileExistsError(self.complete_path)
        if self.partial_path.exists() and not resume: raise FileExistsError("use --resume")
        self.rows=read_jsonl(self.partial_path) if resume else []; self.keys=set()
        for row in self.rows: validate_record(row); self.keys.add((row["qid"],row["condition"]))
        if len(self.keys)!=len(self.rows): raise ValueError("duplicate existing keys")
    def append(self,rows:Iterable[Mapping[str,Any]])->None:
        batch=[dict(x) for x in rows]; new=set()
        for row in batch:
            validate_record(row); key=(row["qid"],row["condition"])
            if key in self.keys or key in new: raise ValueError(f"duplicate {key}")
            new.add(key)
        self.rows.extend(batch); self.keys.update(new); atomic_jsonl(self.partial_path,self.rows)
    def finalize(self,qids:list[str],condition:str)->Path:
        if {q for q,c in self.keys if c==condition}!=set(qids) or len(self.rows)!=len(qids): raise ValueError("incomplete")
        atomic_jsonl(self.complete_path,self.rows); self.partial_path.unlink(missing_ok=True); return self.complete_path
