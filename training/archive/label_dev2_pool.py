# -*- coding: utf-8 -*-
"""dev2_pool 140 条 Jev 教师软标签（复用 label_dev_pool 模式；task 规格从 dev2 task.yaml 读）。"""
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ARCH = ROOT / "training" / "archive"
sys.path.insert(0, str(ROOT))
from judgekit.engine import Task, run_task           # noqa: E402
from judgekit.providers import load_providers        # noqa: E402

QID = {"dev2_code_review": "issue", "dev2_num_slo": "slo", "dev2_num_calc": "calc",
       "dev2_route_en": "route", "dev2_gate_en": "risk", "dev2_triage_mixed": "triage"}
FAMILY = {"dev2_code_review": "dev2_codereview", "dev2_num_slo": "dev2_numslo",
          "dev2_num_calc": "dev2_numcalc", "dev2_route_en": "dev2_routeen",
          "dev2_gate_en": "dev2_gateen", "dev2_triage_mixed": "dev2_triage"}

specs = {}
for ds in QID:
    ty = yaml.safe_load(open(ROOT / "benchmarks" / "data" / f"{ds}.task.yaml", encoding="utf-8"))
    specs[ds] = {"criteria": ty["label_descriptions"], "instruction": ty.get("instruction", "")}
providers = load_providers(str(ROOT / "benchmarks" / "models.yaml"))
rows = [json.loads(l) for l in open(ARCH / "dev2_pool.jsonl", encoding="utf-8") if l.strip()]
out = open(ARCH / "dev2_labels.jsonl", "w", encoding="utf-8", newline="\n")
dis = n_ok = 0
for i, r in enumerate(rows):
    sp = specs[r["task"]]
    t = Task(name=f"label-{r['id']}", primitive="classify", labels=list(sp["criteria"]),
             label_descriptions=sp["criteria"], instruction=sp["instruction"])
    t.provider = "typesafe"
    d = None
    for attempt in range(4):
        try:
            d = run_task(t, {"text": r["text"]}, providers)
            if d.ok:
                break
        except Exception as e:
            if attempt == 3:
                print(f"FAIL {r['id']}: {e}", flush=True)
        time.sleep(3 * (attempt + 1))
    if d is None or not d.ok:
        continue
    probs = {str(k): float(v) for k, v in (d.probabilities or {}).items() if v is not None}
    if not probs:
        probs = {lb: (1.0 if lb == d.value else 0.0) for lb in sp["criteria"]}
    for lb in sp["criteria"]:
        probs.setdefault(lb, 0.0)
    s = sum(probs.values())
    probs = {k: round(v / s, 6) for k, v in probs.items()}
    out.write(json.dumps({"id": r["id"], "task": r["task"], "probs": probs}, ensure_ascii=False) + "\n")
    out.flush()
    n_ok += 1
    top = max(probs, key=probs.get)
    if top != r["intended"]:
        dis += 1
        print(f"DIS {r['id']} teacher={top} intended={r['intended']} conf={probs[top]:.2f}", flush=True)
    if (i + 1) % 25 == 0:
        print(f"progress {i+1}/{len(rows)} ok={n_ok} disagreement={dis}", flush=True)
out.close()
print(f"DONE ok={n_ok}/{len(rows)} disagreement={dis}")
