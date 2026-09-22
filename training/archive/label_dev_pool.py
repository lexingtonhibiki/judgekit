# -*- coding: utf-8 -*-
"""dev_pool 200 条 Jev 教师软标签（judgekit typesafe 通道，一次调用拿原生分布）。"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCH = ROOT / "training" / "archive"
sys.path.insert(0, str(ROOT))
from judgekit.engine import Task, run_task           # noqa: E402
from judgekit.providers import load_providers        # noqa: E402

specs = json.load(open(ARCH / "dev_task_specs.json", encoding="utf-8"))
providers = load_providers(str(ROOT / "benchmarks" / "models.yaml"))
rows = [json.loads(l) for l in open(ARCH / "dev_pool.jsonl", encoding="utf-8") if l.strip()]
out = open(ARCH / "dev_labels.jsonl", "w", encoding="utf-8", newline="\n")
dis = n_ok = 0
for i, r in enumerate(rows):
    sp = specs[r["task"]]
    t = Task(name=f"label-{r['id']}", primitive="classify", labels=list(sp["criteria"]),
             label_descriptions=sp["criteria"], instruction=sp["instructions"])
    t.provider = "typesafe"
    d = None
    for attempt in range(3):
        try:
            d = run_task(t, {"text": r["text"]}, providers)
            if d.ok:
                break
        except Exception as e:
            if attempt == 2:
                print(f"FAIL {r['id']}: {e}", flush=True)
        time.sleep(2 * (attempt + 1))
    if d is None or not d.ok:
        continue
    probs = {str(k): float(v) for k, v in (d.probabilities or {}).items() if v is not None}
    if not probs:  # 分布缺失 → argmax one-hot 兜底
        probs = {lb: (1.0 if lb == d.value else 0.0) for lb in sp["criteria"]}
    for lb in sp["criteria"]:
        probs.setdefault(lb, 0.0)
    s = sum(probs.values())
    probs = {k: round(v / s, 6) for k, v in probs.items()} if s else probs
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
print(f"DONE ok={n_ok}/{len(rows)} teacher-intent-disagreement={dis}")
