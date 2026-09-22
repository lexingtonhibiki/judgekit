# -*- coding: utf-8 -*-
"""dev2 公平性加固靶场自检：类目均衡 / 长度 / v1 关键词重合率 / label 合法性 / jsonl 可解析 / BOM / 池评测重复。"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "benchmarks", "data")
TRAIN = os.path.join(ROOT, "training", "archive")

EXPECTED = {
    "dev2_route_zh": 24, "dev2_route_en": 24, "dev2_gate_en": 24,
    "dev2_triage_mixed": 24, "dev2_commit_zh": 25, "dev2_code_review": 24,
    "dev2_num_slo": 24, "dev2_num_calc": 24,
}
# v2 数据集 -> 同任务 v1 关键词表（dev2_route_en 与 dev2_route_zh 同表）
V1_RULES = {
    "dev2_route_zh": "dev_route_zh", "dev2_route_en": "dev_route_zh",
    "dev2_gate_en": "cmd_risk_zh", "dev2_triage_mixed": "err_triage",
    "dev2_commit_zh": "commit_type",
}
POOL_EXPECTED = {"dev2_code_review": 30, "dev2_num_slo": 30, "dev2_num_calc": 20,
                 "dev2_route_en": 20, "dev2_gate_en": 20, "dev2_triage_mixed": 20}
PREFIX = {"dev2_route_zh": "dv2rz", "dev2_route_en": "dv2re", "dev2_gate_en": "dv2gt",
          "dev2_triage_mixed": "dv2tg", "dev2_commit_zh": "dv2cm", "dev2_code_review": "dv2cr",
          "dev2_num_slo": "dv2slo", "dev2_num_calc": "dv2calc"}


def load_rules_keywords(name):
    """从 v1 <ds>.rules.yaml 提取全部 fallback 关键词（引号内的含空格词保留）。"""
    kws = []
    with open(os.path.join(DATA, name + ".rules.yaml"), encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^\s*[^:\[]+:\s*\[(.*)\]\s*$", line)
            if m:
                for tok in m.group(1).split(","):
                    tok = tok.strip().strip('"').strip("'")
                    if tok:
                        kws.append(tok)
    return kws


def load_task_labels(ds):
    """从 <ds>.task.yaml 提取 label_descriptions 的键集合。"""
    labels, in_block = [], False
    with open(os.path.join(DATA, ds + ".task.yaml"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("label_descriptions:"):
                in_block = True
                continue
            if in_block:
                if line[:2] == "  " and ":" in line:
                    labels.append(line.strip().split(":")[0].strip())
                elif line.strip():
                    break
    return set(labels)


def load_jsonl(path):
    with open(path, "rb") as f:
        raw_bytes = f.read()
    assert not raw_bytes.startswith(b"\xef\xbb\xbf"), path + " 含 BOM"
    raw = raw_bytes.decode("utf-8")
    assert "\r" not in raw, path + " 含 CR（应为 LF）"
    rows = []
    for i, line in enumerate(raw.splitlines(), 1):
        if line.strip():
            rows.append(json.loads(line))  # 不可解析直接抛异常
    return rows


def overlap_rate(rows, kws):
    hits = sum(1 for r in rows if any(k in r["text"] for k in kws))
    return hits, hits / len(rows)


failures = []


def check(cond, msg):
    if cond:
        print("  [PASS]", msg)
    else:
        print("  [FAIL]", msg)
        failures.append(msg)


print("== v2 评测集自检 ==")
eval_texts_by_ds = {}
for ds, n in EXPECTED.items():
    print("\n--- %s (期望 %d 条) ---" % (ds, n))
    rows = load_jsonl(os.path.join(DATA, ds + ".jsonl"))
    check(len(rows) == n, "总条数 %d == %d" % (len(rows), n))
    ids = [r["id"] for r in rows]
    check(len(set(ids)) == len(ids), "id 唯一")
    check(all(i.startswith(PREFIX[ds]) for i in ids), "id 前缀 %s" % PREFIX[ds])
    check(all(set(r) == {"id", "text", "label"} for r in rows), "字段恰为 {id,text,label}")
    dist = {}
    for r in rows:
        dist[r["label"]] = dist.get(r["label"], 0) + 1
    print("  类目分布:", dict(sorted(dist.items())))
    check(len(set(dist.values())) == 1, "类目严格均衡 %s" % sorted(set(dist.values())))
    keys = load_task_labels(ds)
    check(set(dist) == keys, "label 集合 == task.yaml label_descriptions 键 %s" % sorted(keys))
    lens = [len(r["text"]) for r in rows]
    print("  文本长度 min=%d max=%d mean=%.1f" % (min(lens), max(lens), sum(lens) / len(lens)))
    check(all(r["text"].strip() for r in rows), "无空文本")
    check(not any(w in r["text"] for r in rows for w in ("样本", "选项")), "无出题痕迹词(样本/选项)")
    warn_hits = [r["id"] for r in rows if "测试" in r["text"] and ds != "dev2_commit_zh"]
    if warn_hits:
        print("  [WARN] 含'测试'字样（请人工复核是否域内自然用法）:", warn_hits)
    if ds in V1_RULES:
        kws = load_rules_keywords(V1_RULES[ds])
        hits, ov = overlap_rate(rows, kws)
        print("  v1 关键词重合率: %.1f%%（命中 %d/%d，目标 <40%%）" % (ov * 100, hits, len(rows)))
        check(ov < 0.4, "重合率 <40%%")
        if ds == "dev2_route_zh":
            check(ov == 0, "route_zh 完全禁用 v1 原词（0%）")
    else:
        print("  v1 关键词重合率: 无同任务 v1 数据集，不适用")
    eval_texts_by_ds[ds] = {re.sub(r"\s+", "", r["text"]) for r in rows}

print("\n== 训练池自检 ==")
pool = load_jsonl(os.path.join(TRAIN, "dev2_pool.jsonl"))
check(len(pool) == 140, "训练池总条数 %d == 140" % len(pool))
check(all(set(r) == {"id", "task", "text", "intended"} for r in pool), "字段恰为 {id,task,text,intended}")
check(all(r["id"].startswith("p2_") for r in pool), "id 前缀 p2_")
check(len(set(r["id"] for r in pool)) == len(pool), "id 唯一")
by_task = {}
for r in pool:
    by_task.setdefault(r["task"], []).append(r)
check(set(by_task) == set(POOL_EXPECTED), "task 覆盖 6 个数据集")
pool_norm = {re.sub(r"\s+", "", r["text"]) for r in pool}
for task in sorted(by_task):
    rows = by_task[task]
    print("\n--- pool: %s (%d 条，期望 %d) ---" % (task, len(rows), POOL_EXPECTED[task]))
    check(len(rows) == POOL_EXPECTED[task], "条数 %d == %d" % (len(rows), POOL_EXPECTED[task]))
    dist = {}
    for r in rows:
        dist[r["intended"]] = dist.get(r["intended"], 0) + 1
    print("  类目分布:", dict(sorted(dist.items())))
    keys = load_task_labels(task)
    check(set(dist) <= keys, "intended 均在 %s.task.yaml 键内" % task)
    check(max(dist.values()) - min(dist.values()) <= 1,
          "均衡 ±1（极差 %d）" % (max(dist.values()) - min(dist.values())))
    lens = [len(r["text"]) for r in rows]
    print("  文本长度 min=%d max=%d mean=%.1f" % (min(lens), max(lens), sum(lens) / len(lens)))
    check(not any(w in r["text"] for r in rows for w in ("样本", "选项")), "无出题痕迹词(样本/选项)")
    if task in V1_RULES:
        kws = load_rules_keywords(V1_RULES[task])
        hits, ov = overlap_rate(rows, kws)
        print("  v1 关键词重合率: %.1f%%（命中 %d/%d，目标 <40%%）" % (ov * 100, hits, len(rows)))
        check(ov < 0.4, "重合率 <40%%")
    else:
        print("  v1 关键词重合率: 无同任务 v1 数据集，不适用")

print("\n== 训练池 × 评测集 重复检查 ==")
all_eval = set()
for s in eval_texts_by_ds.values():
    all_eval |= s
inter = all_eval & pool_norm
check(len(inter) == 0, "池与评测集无文本级重复（交集 %d）" % len(inter))
if inter:
    for t in sorted(inter)[:5]:
        print("  重复示例:", t[:60])

print()
if failures:
    print("自检未通过：%d 项" % len(failures))
    for m in failures:
        print(" -", m)
    sys.exit(1)
print("自检全部通过")
