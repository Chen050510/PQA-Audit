"""Duplicate-safe E31 incremental writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from .io_utils import atomic_jsonl, read_jsonl
from .prompts import CONDITIONS, REPLICAS

REQUIRED = {
    "experiment_id", "run_id", "qid", "condition", "replica", "model",
    "model_revision", "dataset_sha256", "config_sha256",
    "prompt_contract_sha256", "preflight_sha256", "cell_contract_sha256",
    "user_prompt_hash", "rendered_prompt", "rendered_prompt_hash",
    "raw_output", "generated_token_ids", "finish_reason",
    "hit_max_new_tokens", "generation_eos_token_ids",
    "additional_eos_token_ids", "termination_token_id", "gold_label",
    "generated_token_count", "generated_sequence_token_count",
    "post_termination_padding_count", "post_termination_nonpad_token_ids",
    "termination_followed_only_by_padding", "process_uuid", "model_load_uuid",
}


def validate_record(row: Mapping[str, Any]) -> None:
    missing = REQUIRED - set(row)
    if missing:
        raise ValueError(f"record missing fields: {sorted(missing)}")
    if row["condition"] not in CONDITIONS or int(row["replica"]) not in REPLICAS:
        raise ValueError("invalid condition or replica")
    if row["gold_label"] not in set("ABCD"):
        raise ValueError("invalid gold label")
    if row["post_termination_nonpad_token_ids"] != []:
        raise ValueError("non-padding token recorded after termination")
    if row["termination_followed_only_by_padding"] is not True:
        raise ValueError("termination padding audit did not pass")


class IncrementalWriter:
    def __init__(self, run_dir: str | Path, *, resume: bool) -> None:
        self.run_dir = Path(run_dir)
        self.partial_path = self.run_dir / "records.partial.jsonl"
        self.complete_path = self.run_dir / "records.complete.jsonl"
        if self.complete_path.exists():
            raise FileExistsError(f"complete cell exists: {self.complete_path}")
        if self.partial_path.exists() and not resume:
            raise FileExistsError("partial cell exists; use --resume")
        self.rows = read_jsonl(self.partial_path) if resume else []
        self.keys: set[tuple[str, str, int]] = set()
        for row in self.rows:
            validate_record(row)
            key = (str(row["qid"]), str(row["condition"]), int(row["replica"]))
            if key in self.keys:
                raise ValueError(f"duplicate existing key: {key}")
            self.keys.add(key)

    def append(self, rows: Iterable[Mapping[str, Any]]) -> None:
        batch = [dict(row) for row in rows]
        new_keys: set[tuple[str, str, int]] = set()
        for row in batch:
            validate_record(row)
            key = (str(row["qid"]), str(row["condition"]), int(row["replica"]))
            if key in self.keys or key in new_keys:
                raise ValueError(f"duplicate key: {key}")
            new_keys.add(key)
        self.rows.extend(batch)
        self.keys.update(new_keys)
        atomic_jsonl(self.partial_path, self.rows)

    def finalize(self, expected_qids: list[str], condition: str, replica: int) -> Path:
        actual = {qid for qid, cond, rep in self.keys if cond == condition and rep == replica}
        if actual != set(expected_qids) or len(self.rows) != len(expected_qids):
            raise ValueError("cannot finalize incomplete cell")
        atomic_jsonl(self.complete_path, self.rows)
        self.partial_path.unlink(missing_ok=True)
        return self.complete_path
