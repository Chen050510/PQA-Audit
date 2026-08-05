"""Fail-closed provenance checks for one completed E31 formal cell."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io_utils import read_jsonl, sha256_text
from .prompts import prompt_hash, render_prompt


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def build_cell_contract(
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    prompt_contract_sha256: str,
    preflight_sha256: str,
    run_id: str,
    model: str,
    model_path: str,
    model_revision: str,
    attention_backend: str,
    condition: str,
    replica: int,
    batch_size: int,
    qid_scope: str,
    target_items: Sequence[Mapping[str, Any]],
    termination_contract: Mapping[str, Any],
) -> dict[str, Any]:
    qid_gold = [[str(row["qid"]), str(row["gold_label"])] for row in target_items]
    payload = {
        "experiment_id": config["experiment_id"],
        "run_id": run_id,
        "model": model,
        "model_path": str(Path(model_path).resolve()),
        "model_revision": model_revision,
        "attention_backend": attention_backend,
        "condition": condition,
        "replica": int(replica),
        "batch_size": int(batch_size),
        "qid_scope": qid_scope,
        "dataset_sha256": config["dataset"]["sha256"],
        "config_sha256": config_sha256,
        "prompt_contract_sha256": prompt_contract_sha256,
        "preflight_sha256": preflight_sha256,
        "qid_gold_sha256": canonical_hash(qid_gold),
        "termination_contract": dict(termination_contract),
    }
    payload["cell_contract_sha256"] = canonical_hash(payload)
    return payload


def validate_complete_cell(
    complete_path: str | Path,
    manifest_path: str | Path,
    *,
    expected: Mapping[str, Any],
    target_items: Sequence[Mapping[str, Any]],
    generation_config: Mapping[str, Any],
) -> list[str]:
    """Return every provenance/termination mismatch; an empty list means reusable."""
    complete = Path(complete_path)
    manifest_file = Path(manifest_path)
    errors: list[str] = []
    if not complete.is_file():
        return [f"missing complete file: {complete}"]
    if not manifest_file.is_file():
        return [f"missing manifest: {manifest_file}"]
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        rows = read_jsonl(complete)
    except Exception as exc:  # noqa: BLE001
        return [f"cannot read cell: {exc}"]

    manifest_fields = (
        "experiment_id", "run_id", "model", "model_path", "model_revision",
        "attention_backend", "condition", "replica", "batch_size", "qid_scope",
        "dataset_sha256", "config_sha256", "prompt_contract_sha256",
        "preflight_sha256", "qid_gold_sha256", "termination_contract",
        "cell_contract_sha256",
    )
    for field in manifest_fields:
        if manifest.get(field) != expected.get(field):
            errors.append(f"manifest {field} mismatch")

    expected_qids = [str(item["qid"]) for item in target_items]
    expected_gold = {str(item["qid"]): str(item["gold_label"]) for item in target_items}
    if len(rows) != len(target_items):
        errors.append(f"row count mismatch: {len(rows)} != {len(target_items)}")
    actual_qids = [str(row.get("qid")) for row in rows]
    if actual_qids != expected_qids:
        errors.append("qid order/content mismatch")

    process_ids = {row.get("process_uuid") for row in rows}
    load_ids = {row.get("model_load_uuid") for row in rows}
    if len(process_ids) != 1 or process_ids != {manifest.get("process_uuid")} or None in process_ids:
        errors.append("process_uuid mismatch")
    if len(load_ids) != 1 or load_ids != {manifest.get("model_load_uuid")} or None in load_ids:
        errors.append("model_load_uuid mismatch")

    stop_spec = expected["termination_contract"]
    all_eos = [int(value) for value in stop_spec["all_eos_token_ids"]]
    additional_eos = [int(value) for value in stop_spec["additional_eos_token_ids"]]
    eos_set = set(all_eos)
    max_new_tokens = int(generation_config["max_new_tokens"])
    row_identity = {
        "experiment_id": expected["experiment_id"],
        "run_id": expected["run_id"],
        "model": expected["model"],
        "model_revision": expected["model_revision"],
        "condition": expected["condition"],
        "replica": expected["replica"],
        "dataset_sha256": expected["dataset_sha256"],
        "config_sha256": expected["config_sha256"],
        "prompt_contract_sha256": expected["prompt_contract_sha256"],
        "preflight_sha256": expected["preflight_sha256"],
        "cell_contract_sha256": expected["cell_contract_sha256"],
    }
    for index, row in enumerate(rows):
        prefix = f"row[{index}]"
        for field, value in row_identity.items():
            if row.get(field) != value:
                errors.append(f"{prefix} {field} mismatch")
        qid = str(row.get("qid"))
        if row.get("gold_label") != expected_gold.get(qid):
            errors.append(f"{prefix} gold_label mismatch")
        expected_user = render_prompt(target_items[index], str(expected["condition"])) if index < len(target_items) else ""
        if row.get("user_prompt_hash") != prompt_hash(expected_user):
            errors.append(f"{prefix} user_prompt_hash mismatch")
        rendered = row.get("rendered_prompt")
        if not isinstance(rendered, str) or row.get("rendered_prompt_hash") != prompt_hash(rendered):
            errors.append(f"{prefix} rendered_prompt hash mismatch")
        raw = row.get("raw_output")
        if not isinstance(raw, str) or row.get("raw_output_hash") != sha256_text(raw):
            errors.append(f"{prefix} raw_output hash mismatch")
        if row.get("generation_eos_token_ids") != all_eos:
            errors.append(f"{prefix} full EOS set mismatch")
        if row.get("additional_eos_token_ids") != additional_eos:
            errors.append(f"{prefix} additional EOS set mismatch")
        if row.get("decoding_config") != dict(generation_config) or row.get("precision") != "bf16":
            errors.append(f"{prefix} generation config mismatch")
        ids = row.get("generated_token_ids")
        if not isinstance(ids, list) or any(not isinstance(value, int) for value in ids):
            errors.append(f"{prefix} generated_token_ids invalid")
            ids = []
        if row.get("generated_token_count") != len(ids):
            errors.append(f"{prefix} generated_token_count mismatch")
        padding_count = row.get("post_termination_padding_count")
        sequence_count = row.get("generated_sequence_token_count")
        if not isinstance(padding_count, int) or padding_count < 0:
            errors.append(f"{prefix} padding count invalid")
            padding_count = 0
        if sequence_count != len(ids) + padding_count:
            errors.append(f"{prefix} generated sequence count mismatch")
        if row.get("post_termination_nonpad_token_ids") != [] or row.get("termination_followed_only_by_padding") is not True:
            errors.append(f"{prefix} non-padding token after termination")
        termination = row.get("termination_token_id")
        finish = row.get("finish_reason")
        hit_max = row.get("hit_max_new_tokens")
        if finish == "eos_token":
            if not ids or termination not in eos_set or ids[-1] != termination or hit_max is not False:
                errors.append(f"{prefix} invalid EOS termination")
        elif finish == "max_tokens":
            if termination is not None or hit_max is not True or len(ids) != max_new_tokens or padding_count != 0:
                errors.append(f"{prefix} invalid max-token termination")
        else:
            errors.append(f"{prefix} invalid finish_reason")
    return errors
