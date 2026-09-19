# -*- coding: utf-8 -*-
"""benchmarks 跑分器。

用法（仓库根目录执行）：
  python benchmarks/run_bench.py --models rules --limit 6            # 零成本基线
  python benchmarks/run_bench.py --models router-free-auto --limit 6 # 走本地路由器
  python benchmarks/run_bench.py --models all --limit 12             # 全量（烧额度，慎用）

设计约束：默认串行（并发额度 2 的环境留余量，--workers 最多给 2）；
每个样本一次判断调用；供应商整体不可达时自动跳过并记 unavailable。
输出 results/<provider>__<dataset>.jsonl。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from judgekit.engine import Task, rules_fallback, run_task   # noqa: E402
from judgekit.providers import load_providers                # noqa: E402


def load_dataset(path: Path, limit: int) -> tuple[list[str], list[dict]]:
    labels, rows = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["label"] not in labels:
                    labels.append(r["label"])
                rows.append(r)
    return labels, rows[:limit] if limit else rows


def bench_one(task: Task, providers: dict, row: dict) -> dict:
    dec = run_task(task, {"text": row["text"]}, providers) if task.provider in providers \
        else rules_fallback(task, {"text": row["text"]})
    return {"id": row["id"], "gold": row["label"], "pred": dec.value if dec.ok else None,
            "confidence": dec.confidence, "correct": bool(dec.ok and dec.value == row["label"]),
            "latency_ms": dec.latency_ms, "cost": round(dec.cost, 6),
            "provider": dec.provider, "ok": dec.ok, "error": dec.error}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="rules",
                    help="逗号分隔的供应商名，或 all（models.yaml 全部）")
    ap.add_argument("--datasets", default="intent_zh,sentiment_zh")
    ap.add_argument("--limit", type=int, default=12, help="每个数据集取前 N 条（控制额度消耗）")
    ap.add_argument("--workers", type=int, default=1, choices=[1, 2],
                    help="并发数；环境并发额度 2，默认串行")
    ap.add_argument("--providers-file", default=str(ROOT / "benchmarks" / "models.yaml"))
    ap.add_argument("--out-dir", default=str(ROOT / "benchmarks" / "results"))
    args = ap.parse_args()

    providers = load_providers(args.providers_file)
    names = list(providers) if args.models == "all" else [m.strip() for m in args.models.split(",")]
    bad = [m for m in names if m not in providers]
    if bad:
        sys.exit(f"models.yaml 里没有这些供应商: {bad}；可用: {list(providers)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks: dict[str, Task] = {}
    tasks_rows: dict[str, list] = {}
    import yaml
    for ds in [d.strip() for d in args.datasets.split(",")]:
        labels, rows = load_dataset(ROOT / "benchmarks" / "data" / f"{ds}.jsonl", args.limit)
        rules_p = ROOT / "benchmarks" / "data" / f"{ds}.rules.yaml"
        fb = yaml.safe_load(open(rules_p, encoding="utf-8"))["fallback_rules"] if rules_p.exists() else {}
        tasks[ds] = Task(name=f"bench-{ds}", primitive="classify", labels=labels,
                         instruction="中文分类基准测评。", fallback_rules=fb)
        tasks_rows[ds] = rows

    for name in names:
        for ds, task in tasks.items():
            task.provider = name  # 每个供应商都要在同一任务上跑，绑定后再执行
            rows = tasks_rows[ds]
            t0 = time.monotonic()
            if args.workers > 1:
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    recs = list(ex.map(lambda r: bench_one(task, providers, r), rows))
            else:
                recs = [bench_one(task, providers, r) for r in rows]
            fatal = sum(1 for r in recs if not r["ok"])
            acc = sum(r["correct"] for r in recs) / max(1, len(recs))
            cost = sum(r["cost"] for r in recs)
            lat = sum(r["latency_ms"] for r in recs) / max(1, len(recs))
            out = out_dir / f"{name}__{ds}.jsonl"
            with open(out, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[{name} × {ds}] n={len(recs)} acc={acc:.2%} cost=¥{cost:.4f} "
                  f"lat={lat:.0f}ms err={fatal} took={time.monotonic()-t0:.0f}s -> {out.name}",
                  flush=True)
            if fatal == len(recs) and fatal > 0:
                print(f"    ⚠ {name} 在 {ds} 上全部失败，疑似供应商不可达，跳过该模型剩余数据集",
                      flush=True)
                break


if __name__ == "__main__":
    main()
