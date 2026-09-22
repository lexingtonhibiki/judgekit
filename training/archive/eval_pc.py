# -*- coding: utf-8 -*-
"""提示词条件化臂（pc06）专用评测：所有数据集的 instructions 加 "Task: {family}\\n" 前缀，
直连 DecisionPredictor（不经 serve）。数据集规格：dev2/dev1 从 task.yaml，econ 从混训记录抽。
用法：python training/archive/eval_pc.py --ckpt D:/Models/NanoJev-zh/pc06
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, r"C:/Users/14970/AppData/Local/Temp/NanoJev/scripts")
from predict_toy_decisions import DecisionPredictor  # noqa: E402

V2 = {"dev2_route_zh": ("route", "dev_route"), "dev2_route_en": ("route", "dev2_routeen"),
      "dev2_gate_en": ("risk", "dev2_gateen"), "dev2_triage_mixed": ("triage", "dev2_triage"),
      "dev2_commit_zh": ("ctype", "dev_committype"), "dev2_code_review": ("issue", "dev2_codereview"),
      "dev2_num_slo": ("slo", "dev2_numslo"), "dev2_num_calc": ("calc", "dev2_numcalc")}
DEV1 = {"dev_route_zh": ("route", "dev_route"), "cmd_risk_zh": ("risk", "dev_cmdrisk"),
        "err_triage": ("triage", "dev_errtriage"), "commit_type": ("ctype", "dev_committype")}
ECON = {"intent_zh": ("route", "zh_intent"), "sentiment_zh": ("polarity", "zh_sentiment"),
        "spam_zh": ("spam", "zh_spam"), "urgency_zh": ("urgent", "zh_urgency")}


def data_file(ds: str, suffix: str = ".jsonl") -> Path:
    flat = ROOT / "benchmarks" / "data" / f"{ds}{suffix}"
    if flat.exists():
        return flat
    hits = [p for p in (ROOT / "benchmarks" / "data").rglob(f"{ds}{suffix}")
            if p.parent.name != "deprecated_econ_v1"]
    return hits[0]


def spec_from_taskyaml(ds, fam):
    ty = yaml.safe_load(open(data_file(ds, ".task.yaml"), encoding="utf-8"))
    return ty["instruction"], ty["label_descriptions"], fam


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--no-prefix", action="store_true", help="不加 Task: 前缀（对照用）")
    args = ap.parse_args()
    eng = DecisionPredictor(args.ckpt, precision="bf16")

    # econ 规格：从混训记录按 family 抽一条
    econ_spec = {}
    for l in open(ROOT / "training" / "archive" / "mixed_records_split.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["family_id"] in {f for _, f in ECON.values()} and r["family_id"] not in econ_spec:
            qid = next(iter(r["questions"]))
            econ_spec[r["family_id"]] = (qid, r["questions"][qid]["instructions"], r["questions"][qid]["criteria"])

    hits = n = 0
    by_ds = {}
    all_jobs = []
    for ds, (qid, fam) in {**V2, **DEV1}.items():
        ins, crit, _ = spec_from_taskyaml(ds, fam)
        all_jobs.append((ds, qid, ins, crit, fam, data_file(ds)))
    for ds, (qid, fam) in ECON.items():
        q, ins, crit = econ_spec[fam]
        all_jobs.append((ds, q, ins, crit, fam, ROOT / "benchmarks" / "data" / f"{ds}.jsonl"))

    for ds, qid, ins, crit, fam, path in all_jobs:
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        prefix = "" if args.no_prefix else f"Task: {fam}\n"
        payload = {"states": [{"id": r["id"],
                               "state": json.dumps({"text": r["text"]}, ensure_ascii=False),
                               "questions": {qid: {"type": "choice",
                                                   "instructions": prefix + ins,
                                                   "criteria": crit}}} for r in rows]}
        res = eng.predict(payload)
        ans = {s["id"]: s["answers"][qid]["probabilities"] for s in res["states"]}
        h = t = 0
        for r in rows:
            p = ans[r["id"]]
            t += 1
            h += max(p, key=p.get) == r["label"]
        by_ds[ds] = (h, t)
        hits += h
        n += t
    for ds, (h, t) in by_ds.items():
        print(f"  {ds}: {h}/{t} = {h/t:.1%}")
    print(f"TOTAL: {hits}/{n} = {hits/n:.1%}  ({'no-prefix' if args.no_prefix else 'prefixed'})")


if __name__ == "__main__":
    main()
