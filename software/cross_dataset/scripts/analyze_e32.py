#!/usr/bin/env python3
"""Offline paired crosswalk analysis for E32."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from e32_protocol.io_utils import atomic_json,read_jsonl
from e32_protocol.readers import read_final

def write_csv(path:Path,rows:list[dict[str,Any]]):
    path.parent.mkdir(parents=True,exist_ok=True); fields=list(rows[0]) if rows else []
    with path.open("w",encoding="utf-8",newline="") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
def ci(values:np.ndarray,n:int,seed:int):
    rng=np.random.default_rng(seed); size=len(values); means=np.array([values[rng.integers(0,size,size)].mean() for _ in range(n)]); return float(np.quantile(means,.025)),float(np.quantile(means,.975))
def load_cell(run:Path,model:str,dataset:dict,cond:str):
    path=run/"raw_outputs"/model/dataset["name"]/cond/"records.complete.jsonl"; rows=read_jsonl(path)
    if len(rows)!=dataset["expected_n"]: raise ValueError(f"{path}: count mismatch")
    result={}
    for row in rows:
        q=str(row["qid"]); parsed=read_final(row.get("raw_output"),finish_reason=row.get("finish_reason"),hit_max_new_tokens=row.get("hit_max_new_tokens"),labels=dataset["labels"])
        if parsed.status!=row.get("strict_status_at_generation") or parsed.label!=row.get("strict_label_at_generation"): raise ValueError(f"parser mismatch {path}:{q}")
        if q in result: raise ValueError(f"duplicate {q}")
        x=dict(row); x["valid"]=parsed.status=="VALID_FINAL"; x["label"]=parsed.label; x["correct"]=bool(x["valid"] and x["label"]==x["gold_label"]); result[q]=x
    return result,[str(x["qid"]) for x in rows]
def model_status(vfc_d,vfc_p,net,lost,gained,gross,cancel,share,lo,cfg):
    if min(vfc_d,vfc_p)<cfg["vfc_gate"]: return "INCONCLUSIVE"
    ok=abs(net)<=cfg["max_abs_net"] and lost>0 and gained>0 and gross>=cfg["min_gross"] and cancel>=cfg["min_cancellation"] and share>=cfg["min_cancellation_share"] and lo>0
    return "PASS" if ok else "FAIL"
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--run-dir","--run-root",dest="run_dir",required=True); p.add_argument("--config",default=str(ROOT/"config/e32_config.json")); p.add_argument("--output-dir",required=True); a=p.parse_args(); cfg=json.loads(Path(a.config).read_text()); run=Path(a.run_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    summaries=[]; records=[]; dataset_states={}
    for d in cfg["datasets"]:
        states=[]
        for model in [x["name"] for x in cfg["models"]]:
            direct,order=load_cell(run,model,d,"D"); process,porder=load_cell(run,model,d,"P")
            if order!=porder or set(direct)!=set(process): raise ValueError(f"qid order mismatch {model}/{d['name']}")
            n=len(order); lost=gained=wrong=stable_c=stable_w=answer_change=0; d_corr=[]; p_corr=[]; cancel_unit=[]
            for q in order:
                x,y=direct[q],process[q]
                if x["gold_label"]!=y["gold_label"]: raise ValueError(f"gold mismatch {q}")
                dc=int(x["correct"]); pc=int(y["correct"]); d_corr.append(dc); p_corr.append(pc)
                if x["valid"] and y["valid"]:
                    answer_change+=int(x["label"]!=y["label"])
                    if dc and not pc: lost+=1
                    elif not dc and pc: gained+=1
                    elif dc: stable_c+=1
                    else: wrong+=int(x["label"]!=y["label"]); stable_w+=int(x["label"]==y["label"])
                records.append({"dataset":d["name"],"model":model,"qid":q,"gold":x["gold_label"],"direct_valid":x["valid"],"process_valid":y["valid"],"direct_label":x["label"],"process_label":y["label"],"direct_correct":bool(dc),"process_correct":bool(pc),"transition":"lost" if dc and not pc else "gained" if not dc and pc else "stable_correct" if dc else "stable_wrong"})
            d_corr=np.array(d_corr); p_corr=np.array(p_corr); vfc_d=np.mean([direct[q]["valid"] for q in order]); vfc_p=np.mean([process[q]["valid"] for q in order]); net=float(p_corr.mean()-d_corr.mean()); gross=(lost+gained)/n; cancel=2*min(lost,gained)/n; share=0.0 if gross==0 else cancel/gross
            lost_vec=np.array([int(direct[q]["valid"] and process[q]["valid"] and direct[q]["correct"] and not process[q]["correct"]) for q in order]); gain_vec=np.array([int(direct[q]["valid"] and process[q]["valid"] and not direct[q]["correct"] and process[q]["correct"]) for q in order])
            rng=np.random.default_rng(cfg["analysis"]["seed"]); boots=np.empty(cfg["analysis"]["bootstrap_resamples"])
            for bi in range(len(boots)):
                ids=rng.integers(0,n,n); boots[bi]=2*min(int(lost_vec[ids].sum()),int(gain_vec[ids].sum()))/n
            lo=float(np.quantile(boots,.025)); hi=float(np.quantile(boots,.975))
            status=model_status(vfc_d,vfc_p,net,lost,gained,gross,cancel,share,lo,cfg["analysis"]); states.append(status)
            summaries.append({"dataset":d["name"],"model":model,"n":n,"direct_vfc":vfc_d,"process_vfc":vfc_p,"direct_accuracy":float(d_corr.mean()),"process_accuracy":float(p_corr.mean()),"net":net,"answer_change_n":answer_change,"lost":lost,"gained":gained,"wrong_to_wrong":wrong,"gross":gross,"cancellation":cancel,"cancellation_share":share,"cancellation_ci_low":lo,"cancellation_ci_high":hi,"model_status":status})
        if "INCONCLUSIVE" in states: ds="INCONCLUSIVE"
        elif states==["PASS","PASS"]: ds="PASS"
        elif states==["FAIL","FAIL"]: ds="FAIL"
        else: ds="MIXED"
        dataset_states[d["name"]]=ds
    vals=list(dataset_states.values())
    overall="INCONCLUSIVE" if "INCONCLUSIVE" in vals else "PASS" if vals==["PASS","PASS"] else "FAIL" if vals==["FAIL","FAIL"] else "MIXED"
    write_csv(out/"e32_dataset_model_summary.csv",summaries); write_csv(out/"e32_record_level.csv",records); atomic_json(out/"e32_decision.json",{"dataset_status":dataset_states,"overall_status":overall})
    print(json.dumps({"status":"COMPLETE","overall":overall,"output_dir":str(out)})); return 0
if __name__=="__main__": raise SystemExit(main())
