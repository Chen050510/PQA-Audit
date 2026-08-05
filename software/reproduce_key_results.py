#!/usr/bin/env python3
"""Recompute paired counts from a prepared analysis-data directory."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def valid_label(row: dict[str, Any], model: str) -> tuple[bool, str | None]:
    if model.startswith("Qwen"):
        status = row.get("final_field_status")
        label = row.get("final_field_label")
    else:
        status = row.get("strict_status_at_generation")
        label = row.get("strict_label_at_generation")
    label = str(label).strip().upper() if label is not None else None
    valid = status == "VALID_FINAL" and label in set("ABCDE")
    return valid, label if valid else None


def index_rows(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    rows = read_jsonl(path)
    order: list[str] = []
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row.get("qid", ""))
        if not qid or qid in index:
            raise ValueError(f"{path}: missing or duplicate qid={qid!r}")
        order.append(qid)
        index[qid] = row
    return order, index


def main_arc(data_root: Path) -> list[dict[str, Any]]:
    base = data_root / "main_arc" / "raw_outputs"
    models = ("Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "gemma-3-4b-it")
    output: list[dict[str, Any]] = []
    for model in models:
        d_order, direct = index_rows(base / model / "D" / "records.complete.jsonl")
        p_order, process = index_rows(base / model / "P" / "records.complete.jsonl")
        if d_order != p_order or len(d_order) != 1168:
            raise ValueError(f"{model}: expected identical ordered 1,168-qid sets")
        counts = defaultdict(int)
        direct_correct = process_correct = 0
        for qid in d_order:
            d = direct[qid]
            p = process[qid]
            if d.get("gold_label") != p.get("gold_label"):
                raise ValueError(f"{model}/{qid}: gold mismatch")
            gold = str(d["gold_label"]).strip().upper()
            d_valid, d_label = valid_label(d, model)
            p_valid, p_label = valid_label(p, model)
            dc = d_valid and d_label == gold
            pc = p_valid and p_label == gold
            direct_correct += int(dc)
            process_correct += int(pc)
            counts["direct_valid"] += int(d_valid)
            counts["process_valid"] += int(p_valid)
            counts["both_valid"] += int(d_valid and p_valid)
            counts["all_correctness_changes"] += int(dc != pc)
            if not (d_valid and p_valid):
                continue
            counts["answer_changes"] += int(d_label != p_label)
            if dc and not pc:
                counts["loss"] += 1
            elif not dc and pc:
                counts["gain"] += 1
            elif not dc and not pc and d_label != p_label:
                counts["wrong_to_wrong"] += 1
        output.append(
            {
                "model": model,
                "n": len(d_order),
                **dict(counts),
            }
        )
    return output


def recurrence(data_root: Path) -> list[dict[str, Any]]:
    path = data_root / "recurrence" / "analysis" / "recurrence_qid_level.csv"
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            by_model[row["model"]].append(row)
    output: list[dict[str, Any]] = []
    for model, rows in sorted(by_model.items()):
        if len(rows) != 1168 or len({r["qid"] for r in rows}) != 1168:
            raise ValueError(f"{model}: recurrence qid count mismatch")
        eligible = [r for r in rows if as_bool(r["common_valid"]) and as_bool(r["stable_direct"])]
        recurring = fixed = spoiled = 0
        for row in eligible:
            directions: list[int] = []
            for condition in ("P0", "P1", "P2"):
                d1 = as_bool(row["D_r1_correct"])
                d2 = as_bool(row["D_r2_correct"])
                p1 = as_bool(row[f"{condition}_r1_correct"])
                p2 = as_bool(row[f"{condition}_r2_correct"])
                if d1 != p1 and d2 != p2:
                    directions.append(1 if (not d1 and p1) else -1)
            if len(directions) >= 2:
                recurring += 1
                if sum(x == 1 for x in directions) >= 2:
                    fixed += 1
                elif sum(x == -1 for x in directions) >= 2:
                    spoiled += 1
        output.append(
            {
                "model": model,
                "eligible_n": len(eligible),
                "recurring_n": recurring,
                "recurring_rate": recurring / len(eligible),
                "repeatedly_fixed_n": fixed,
                "repeatedly_spoiled_n": spoiled,
            }
        )
    return output


def cross_dataset(data_root: Path) -> list[dict[str, Any]]:
    path = data_root / "cross_dataset" / "analysis" / "cross_dataset_record_level.csv"
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            groups[(row["dataset"], row["model"])].append(row)
    output: list[dict[str, Any]] = []
    for (dataset, model), rows in sorted(groups.items()):
        expected = 1221 if dataset == "commonsenseqa" else 500
        if len(rows) != expected or len({r["qid"] for r in rows}) != expected:
            raise ValueError(f"{dataset}/{model}: qid count mismatch")
        direct_correct = sum(as_bool(r["direct_correct"]) for r in rows)
        process_correct = sum(as_bool(r["process_correct"]) for r in rows)
        gain = sum(
            as_bool(r["direct_valid"])
            and as_bool(r["process_valid"])
            and r["transition"] == "gained"
            for r in rows
        )
        loss = sum(
            as_bool(r["direct_valid"])
            and as_bool(r["process_valid"])
            and r["transition"] == "lost"
            for r in rows
        )
        answer_changes = sum(
            as_bool(r["direct_valid"])
            and as_bool(r["process_valid"])
            and r["direct_label"] != r["process_label"]
            for r in rows
        )
        output.append(
            {
                "dataset": dataset,
                "model": model,
                "n": expected,
                "direct_accuracy": direct_correct / expected,
                "process_accuracy": process_correct / expected,
                "net_accuracy_change": (process_correct - direct_correct) / expected,
                "gain": gain,
                "loss": loss,
                "answer_changes": answer_changes,
                "correctness_change_rate": (gain + loss) / expected,
            }
        )
    return output


def close(a: Any, b: Any, tolerance: float = 1e-12) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= tolerance
    return a == b


def compare(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    if set(actual) != set(expected):
        raise AssertionError(f"{label}: key mismatch: {set(actual) ^ set(expected)}")
    for key in actual:
        if not close(actual[key], expected[key]):
            raise AssertionError(f"{label}.{key}: {actual[key]!r} != {expected[key]!r}")


def compare_nested(actual: Any, expected: Any, label: str) -> None:
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f"{label}: key mismatch: {set(actual) ^ set(expected)}")
        for key in actual:
            compare_nested(actual[key], expected[key], f"{label}.{key}")
        return
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"{label}: length mismatch: {len(actual)} != {len(expected)}")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            compare_nested(actual_item, expected_item, f"{label}[{index}]")
        return
    if not close(actual, expected):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def recurrence_resampling(data_root: Path) -> dict[str, Any]:
    software_root = Path(__file__).resolve().parent
    analyzer = software_root / "recurrence" / "scripts" / "analyze_e31.py"
    config = software_root / "recurrence" / "config" / "e31_config.json"
    run_dir = data_root / "recurrence" / "run"
    expected_path = data_root / "recurrence" / "analysis" / "recurrence_summary.json"
    if not all(path.is_file() for path in (analyzer, config, expected_path)):
        raise FileNotFoundError("recurrence analyzer, config, or frozen summary is missing")
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "recurrence_analysis"
        subprocess.run(
            [
                sys.executable,
                str(analyzer),
                "--run-dir",
                str(run_dir),
                "--config",
                str(config),
                "--output-dir",
                str(output_dir),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        actual = json.loads((output_dir / "e31_decision.json").read_text(encoding="utf-8"))
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    compare_nested(actual, expected, "recurrence_resampling")
    analysis = json.loads(config.read_text(encoding="utf-8"))["analysis"]
    return {
        "status": "PASS",
        "bootstrap_resamples": int(analysis["bootstrap_resamples"]),
        "permutation_resamples": int(analysis["permutation_resamples"]),
        "frozen_summary_match": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {
        "main_arc": main_arc(args.data_root),
        "recurrence": recurrence(args.data_root),
        "cross_dataset": cross_dataset(args.data_root),
    }
    expected_path = args.data_root / "expected_key_results.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    for section, rows in result.items():
        if len(rows) != len(expected[section]):
            raise AssertionError(f"{section}: row count mismatch")
        for index, row in enumerate(rows):
            compare(row, expected[section][index], f"{section}[{index}]")
    result["recurrence_resampling"] = recurrence_resampling(args.data_root)
    result["status"] = "PASS"
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
