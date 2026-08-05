#!/usr/bin/env python3
"""Offline E31 replay and process-family recurrence analysis."""

from __future__ import annotations

import argparse, csv, json, math, sys
from pathlib import Path
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from e31_protocol.io_utils import atomic_json, read_jsonl
from e31_protocol.readers import read_final

MODELS = ("Qwen2.5-3B-Instruct", "gemma-3-4b-it")
CONDS = ("D", "P0", "P1", "P2")


def phi(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or np.all(a == a[0]) or np.all(b == b[0]): return 0.0
    return float(np.corrcoef(a.astype(float), b.astype(float))[0, 1])


def bootstrap_ci(values: np.ndarray, n: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed); size = len(values)
    means = np.empty(n)
    for i in range(n): means[i] = values[rng.integers(0, size, size)].mean()
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def load_cells(run: Path, model: str) -> tuple[list[str], dict[tuple[str,int], dict[str,dict[str,Any]]]]:
    cells = {}
    qids = None
    for cond in CONDS:
        for rep in (1,2):
            path = run / "raw_outputs" / model / cond / f"replica_{rep}" / "records.complete.jsonl"
            rows = read_jsonl(path)
            if len(rows) != 1168: raise ValueError(f"{path}: expected 1168 records")
            mapping = {}
            for row in rows:
                key = str(row["qid"])
                if key in mapping: raise ValueError(f"duplicate qid: {path}: {key}")
                parsed = read_final(row.get("raw_output"), finish_reason=row.get("finish_reason"), hit_max_new_tokens=row.get("hit_max_new_tokens"), mode="STRICT")
                if parsed.status != row.get("strict_status_at_generation") or parsed.label != row.get("strict_label_at_generation"):
                    raise ValueError(f"frozen parser mismatch: {model}/{cond}/r{rep}/{key}")
                row = dict(row); row["valid"] = parsed.status == "VALID_FINAL"; row["label"] = parsed.label
                row["correct"] = bool(row["valid"] and parsed.label == row["gold_label"])
                mapping[key] = row
            order = [str(x["qid"]) for x in rows]
            if qids is None: qids = order
            if order != qids: raise ValueError(f"qid order mismatch: {model}/{cond}/r{rep}")
            cells[(cond,rep)] = mapping
    assert qids is not None
    golds = [{cells[key][qid]["gold_label"] for key in cells} for qid in qids]
    if any(len(x) != 1 for x in golds): raise ValueError(f"gold mismatch: {model}")
    return qids, cells


def analyze_model(model: str, qids: list[str], cells: dict[tuple[str,int],dict[str,dict[str,Any]]], cfg: dict[str,Any], out_rows: list[dict[str,Any]]) -> tuple[dict[str,Any], list[dict[str,Any]]]:
    keys = [(c,r) for c in CONDS for r in (1,2)]
    vfc = {f"{c}_r{r}": np.mean([cells[(c,r)][q]["valid"] for q in qids]) for c,r in keys}
    common = np.array([all(cells[key][q]["valid"] for key in keys) for q in qids], dtype=bool)
    cv_idx = np.flatnonzero(common)
    common_rate = float(common.mean())
    per_variant = []
    delta_matrix = []
    all_delta_matrix = []
    for p in ("P0","P1","P2"):
        d1=np.array([cells[("D",1)][q]["correct"] for q in qids]); d2=np.array([cells[("D",2)][q]["correct"] for q in qids])
        p1=np.array([cells[(p,1)][q]["correct"] for q in qids]); p2=np.array([cells[(p,2)][q]["correct"] for q in qids])
        values=((d1!=p1).astype(float)+(d2!=p2).astype(float)-(d1!=d2).astype(float)-(p1!=p2).astype(float))/2
        delta_matrix.append(values[common]); all_delta_matrix.append(values)
        per_variant.append({"model":model,"variant":p,"common_valid_n":int(common.sum()),"cross_rate":float(((d1[common]!=p1[common]).mean()+(d2[common]!=p2[common]).mean())/2),"replay_rate":float(((d1[common]!=d2[common]).mean()+(p1[common]!=p2[common]).mean())/2),"delta":float(values[common].mean())})
    delta=np.mean(np.vstack(delta_matrix),axis=0); all_delta=np.mean(np.vstack(all_delta_matrix),axis=0)
    ci=bootstrap_ci(delta,cfg["analysis"]["bootstrap_resamples"],cfg["analysis"]["seed"])
    all_ci=bootstrap_ci(all_delta,cfg["analysis"]["bootstrap_resamples"],cfg["analysis"]["seed"]+1)
    validity_ok=min(vfc.values())>=cfg["analysis"]["cell_vfc_gate"] and common_rate>=cfg["analysis"]["common_valid_gate"]
    if not validity_ok: h1="INCONCLUSIVE"
    elif ci[0]>0: h1="PASS"
    elif all_ci[0]>0: h1="INCONCLUSIVE"
    else: h1="FAIL"
    stable_direct=np.array([cells[("D",1)][q]["correct"]==cells[("D",2)][q]["correct"] for q in qids])
    eligible=common & stable_direct; idx=np.flatnonzero(eligible)
    moves=[]
    for p in ("P0","P1","P2"):
        x=np.array([(cells[("D",1)][q]["correct"]!=cells[(p,1)][q]["correct"] and cells[("D",2)][q]["correct"]!=cells[(p,2)][q]["correct"]) for q in qids],dtype=bool)
        moves.append(x[eligible])
    arr=np.vstack(moves); recur=(arr.sum(axis=0)>=2); pairs=((0,1),(0,2),(1,2)); phis=[phi(arr[a],arr[b]) for a,b in pairs]; stat=float(np.mean(phis))
    rng=np.random.default_rng(cfg["analysis"]["seed"]); perm_n=cfg["analysis"]["permutation_resamples"]
    baseline=np.array([cells[("D",1)][q]["correct"] for q in qids])[eligible]
    null=np.empty(perm_n); null_recur=np.empty(perm_n)
    strata=[np.flatnonzero(baseline==value) for value in (False,True)]
    for j in range(perm_n):
        perm=np.empty_like(arr)
        for k in range(3):
            for ids in strata: perm[k,ids]=rng.permutation(arr[k,ids])
        null[j]=np.mean([phi(perm[a],perm[b]) for a,b in pairs])
        null_recur[j]=np.sum(perm.sum(axis=0)>=2)
    p=(1+int(np.sum(null>=stat)))/(perm_n+1)
    null_recurrence_95th=float(np.quantile(null_recur,.95))
    stable_rate=float(eligible.mean())
    if stable_rate<cfg["analysis"]["stable_direct_gate"]: h2="INCONCLUSIVE"
    elif stat>0 and p<.05 and int(recur.sum())>null_recurrence_95th: h2="PASS"
    else: h2="FAIL"
    for pos,q in enumerate(qids):
        row={"model":model,"qid":q,"common_valid":bool(common[pos]),"stable_direct":bool(stable_direct[pos])}
        for c,r in keys:
            x=cells[(c,r)][q]; row[f"{c}_r{r}_valid"]=x["valid"]; row[f"{c}_r{r}_label"]=x["label"]; row[f"{c}_r{r}_correct"]=x["correct"]
        out_rows.append(row)
    summary={"model":model,"vfc":vfc,"common_valid_n":int(common.sum()),"common_valid_rate":common_rate,"h31_1_delta":float(delta.mean()),"h31_1_ci_low":ci[0],"h31_1_ci_high":ci[1],"all_sample_delta":float(all_delta.mean()),"all_sample_ci_low":all_ci[0],"all_sample_ci_high":all_ci[1],"h31_1_status":h1,"stable_direct_common_valid_n":int(eligible.sum()),"stable_direct_rate":stable_rate,"recurrence_at_least_2_n":int(recur.sum()),"recurrence_at_least_2_rate":float(recur.mean()) if len(recur) else 0.0,"null_recurrence_95th_n":null_recurrence_95th,"weighted_phi":stat,"permutation_p":p,"h31_2_status":h2}
    return summary, per_variant


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--run-dir","--run-root",dest="run_dir",required=True); p.add_argument("--config",default=str(ROOT/"config/e31_config.json")); p.add_argument("--output-dir",required=True); a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text()); run=Path(a.run_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    summaries=[]; variants=[]; records=[]
    for model in MODELS:
        qids,cells=load_cells(run,model); s,v=analyze_model(model,qids,cells,cfg,records); summaries.append(s); variants.extend(v)
    h1s=[x["h31_1_status"] for x in summaries]; h2s=[x["h31_2_status"] for x in summaries]
    if "INCONCLUSIVE" in h1s+h2s: overall="INCONCLUSIVE"
    elif all(x=="PASS" for x in h1s+h2s): overall="PASS"
    elif all(x=="FAIL" for x in h1s) or all(x=="FAIL" for x in h2s): overall="FAIL"
    else: overall="MIXED"
    write_csv(out/"e31_qid_level.csv",records); write_csv(out/"e31_replay_vs_cross_protocol.csv",variants)
    atomic_json(out/"e31_decision.json",{"experiment_id":cfg["experiment_id"],"models":summaries,"overall_status":overall})
    print(json.dumps({"status":"COMPLETE","overall":overall,"output_dir":str(out)})); return 0


if __name__=="__main__": raise SystemExit(main())
