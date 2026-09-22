# -*- coding: utf-8 -*-
"""靶场分析：无技能基线（多数类/首标签）+ 每类召回（坍缩检测）+ 汇总表。

用法：python training/archive/analyze_bench.py dev2  # 分析 nanojev-local__dev2_*__*.jsonl
"""
import collections
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks" / "data"


def dataset_rows(ds):
    flat = DATA / f"{ds}.jsonl"
    p = flat if flat.exists() else next(
        q for q in DATA.rglob(f"{ds}.jsonl") if q.parent.name != "deprecated_econ_v1")
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    prefix = sys.argv[1] if len(sys.argv) > 1 else "dev2"
    files = sorted(glob.glob(str(ROOT / "benchmarks" / "results" / f"nanojev-local__{prefix}_*__*.jsonl")))
    by_tag = collections.defaultdict(lambda: collections.defaultdict(dict))
    for f in files:
        parts = Path(f).name.replace(".jsonl", "").split("__")
        ds, tag = parts[1], parts[2]
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            by_tag[tag][ds][r["id"]] = r
    print(f"== 无技能基线（{prefix}）==")
    floors = {}
    for ds in sorted({ds for tag in by_tag for ds in by_tag[tag]}):
        rows = dataset_rows(ds)
        n = len(rows)
        cnt = collections.Counter(r["label"] for r in rows)
        maj = max(cnt.values()) / n
        first_label = rows[0]["label"]
        first = sum(1 for r in rows if r["label"] == first_label) / n
        floors[ds] = {"majority": maj, "first_label": first}
        print(f"  {ds}: n={n} 多数类={maj:.1%} 首标签={first:.1%} 随机={1/len(cnt):.1%}")
    print(f"\n== 模型汇总（相对无技能基线）==")
    for tag in sorted(by_tag):
        accs, details = [], []
        for ds in sorted(by_tag[tag]):
            preds = by_tag[tag][ds]
            if not preds:
                continue
            acc = sum(1 for r in preds.values() if r["correct"]) / len(preds)
            gold_cnt = collections.Counter(r["gold"] for r in preds.values())
            rec = {g: sum(1 for r in preds.values() if r["gold"] == g and r["correct"]) / c
                   for g, c in gold_cnt.items()}
            collapsed = max(rec.values()) >= 0.95 and min(rec.values()) <= 0.1
            floor = floors.get(ds, {}).get("majority", 0.0)
            accs.append(acc)
            flag = " ⚠坍缩" if collapsed else ""
            details.append(f"{ds.replace(prefix + '_', '')}:{acc:.0%}(地板{floor:.0%}){flag}")
        if accs:
            avg = sum(accs) / len(accs)
            avg_floor = sum(floors.get(ds, {}).get("majority", 0) for ds in by_tag[tag]) / len(accs)
            print(f"[{tag}] 均分={avg:.1%} vs 地板={avg_floor:.1%} 增益={avg - avg_floor:+.1%}")
            for d in details:
                print(f"    {d}")


if __name__ == "__main__":
    main()
