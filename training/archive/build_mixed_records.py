# -*- coding: utf-8 -*-
"""三域混训数据：电商(zh_v2_train 352) + dev v1(146) + dev v2 新域(dev2_pool+labels) → 合并重切四段。

Agent A 建议：dev/calibration 各 ≥50（dev06 的 16/14 太小，best 选择噪声大）。
校验：state_id 全局唯一（训练器硬约束）、每条 probs 和=1、keys=criteria。
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ARCH = ROOT / "training" / "archive"

QID = {"dev2_code_review": ("issue", "dev2_codereview"), "dev2_num_slo": ("slo", "dev2_numslo"),
       "dev2_num_calc": ("calc", "dev2_numcalc"), "dev2_route_en": ("route", "dev2_routeen"),
       "dev2_gate_en": ("risk", "dev2_gateen"), "dev2_triage_mixed": ("triage", "dev2_triage")}

recs = []
# A. 电商域（zhft 300 + smoke 52，富描述，已带 source/eval_overlap/hard 标记）
for l in open(ARCH / "zh_v2_train.jsonl", encoding="utf-8"):
    if l.strip():
        recs.append(json.loads(l))

# B. dev v1（146，来自已构建的切分文件，剥掉旧 split）
for l in open(ARCH / "dev_records_split.jsonl", encoding="utf-8"):
    if l.strip():
        r = json.loads(l)
        r.pop("split", None)
        recs.append(r)

# C. dev v2 新域（dev2_pool + dev2_labels → 训练记录）
pool = {r["id"]: r for r in (json.loads(l) for l in open(ARCH / "dev2_pool.jsonl", encoding="utf-8") if l.strip())}
labels = {o["id"]: o for o in (json.loads(l) for l in open(ARCH / "dev2_labels.jsonl", encoding="utf-8") if l.strip())}
missing = []
for rid, r in pool.items():
    lab = labels.get(rid)
    if lab is None:
        missing.append(rid)
        continue
    ds = r["task"]
    qid, fam = QID[ds]
    ty = yaml.safe_load(open(ROOT / "benchmarks" / "data" / f"{ds}.task.yaml", encoding="utf-8"))
    criteria = ty["label_descriptions"]
    probs = lab["probs"]
    assert set(probs) == set(criteria), rid
    assert abs(sum(probs.values()) - 1.0) <= 1e-3, rid
    recs.append({
        "id": rid, "state_id": rid, "family_id": fam, "split": "train",
        "state": json.dumps({"text": r["text"]}, ensure_ascii=False),
        "questions": {qid: {"type": "choice", "instructions": ty["instruction"], "criteria": criteria}},
        "gold_probs": {qid: probs},
        "gold_probs_kind": {qid: "programmatic_conditional_distribution"},
        "intended": r["intended"],
    })
if missing:
    print(f"WARN missing dev2 labels: {len(missing)} -> {missing[:5]}")

# 全局唯一性 + 合法性
ids = [r["state_id"] for r in recs]
assert len(ids) == len(set(ids)), "state_id 重复！"
for r in recs:
    qid = next(iter(r["questions"]))
    assert set(r["gold_probs"][qid]) == set(r["questions"][qid]["criteria"]), r["id"]
    assert abs(sum(r["gold_probs"][qid].values()) - 1.0) <= 1e-3, r["id"]

# 分层重切（seed 17，dev/calibration 加大）
rng = random.Random(17)
by_fam = defaultdict(list)
for r in recs:
    by_fam[r["family_id"]].append(r)
for fam, rs in sorted(by_fam.items()):
    keys = [r["state_id"] for r in rs]
    rng.shuffle(keys)
    n = len(keys)
    cuts = [("train", n * 70 // 100), ("dev", max(n * 12 // 100, 8)), ("calibration", max(n * 11 // 100, 8))]
    idx = 0
    for split, cnt in cuts:
        for k in keys[idx:idx + cnt]:
            next(x for x in rs if x["state_id"] == k)["split"] = split
        idx += cnt
    for k in keys[idx:]:
        next(x for x in rs if x["state_id"] == k)["split"] = "test"

order = {sid: i for i, sid in enumerate(ids)}
recs.sort(key=lambda r: order[r["state_id"]])
with open(ARCH / "mixed_records_split.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("total:", len(recs))
print("splits:", dict(Counter(r["split"] for r in recs)))
print("families:", dict(Counter(r["family_id"] for r in recs)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
