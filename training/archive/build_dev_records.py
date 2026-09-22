# -*- coding: utf-8 -*-
"""dev_pool + Jev 教师标签 → NanoJev 训练记录（gold_distribution schema）+ 四段切分。"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCH = ROOT / "training" / "archive"
specs = json.load(open(ARCH / "dev_task_specs.json", encoding="utf-8"))
pool = {r["id"]: r for r in (json.loads(l) for l in open(ARCH / "dev_pool.jsonl", encoding="utf-8") if l.strip())}
labels = {o["id"]: o for o in (json.loads(l) for l in open(ARCH / "dev_labels.jsonl", encoding="utf-8") if l.strip())}

recs, missing = [], []
for rid, r in pool.items():
    lab = labels.get(rid)
    if lab is None:
        missing.append(rid)
        continue
    sp = specs[r["task"]]
    probs = lab["probs"]
    assert set(probs) == set(sp["criteria"]), rid
    assert abs(sum(probs.values()) - 1.0) <= 1e-3, rid
    recs.append({
        "id": rid, "state_id": rid, "family_id": sp["family"], "split": "train",
        "state": json.dumps({"text": r["text"]}, ensure_ascii=False),
        "questions": {sp["qid"]: {"type": "choice", "instructions": sp["instructions"],
                                  "criteria": sp["criteria"]}},
        "gold_probs": {sp["qid"]: probs},
        "gold_probs_kind": {sp["qid"]: "programmatic_conditional_distribution"},
        "intended": r["intended"],
    })
if missing:
    print(f"WARN missing labels: {missing}")

rng = random.Random(17)
by_task = defaultdict(list)
for rec in recs:
    by_task[rec["family_id"]].append(rec)
out = []
for fam, rs in sorted(by_task.items()):
    keys = [r["state_id"] for r in rs]
    rng.shuffle(keys)
    n = len(keys)
    cuts = [("train", n * 70 // 100), ("dev", n * 12 // 100), ("calibration", n * 11 // 100)]
    idx = 0
    for split, cnt in cuts:
        for k in keys[idx:idx + cnt]:
            next(r for r in rs if r["state_id"] == k)["split"] = split
        idx += cnt
    for k in keys[idx:]:
        next(r for r in rs if r["state_id"] == k)["split"] = "test"
    out.extend(rs)
# 保持 pool 原顺序写出
order = {rid: i for i, rid in enumerate(pool)}
recs.sort(key=lambda r: order[r["id"]])
with open(ARCH / "dev_records_split.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("splits:", dict(Counter(r["split"] for r in recs)), "| by_family:", dict(Counter(r["family_id"] for r in recs)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
