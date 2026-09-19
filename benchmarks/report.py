# -*- coding: utf-8 -*-
"""benchmarks 报告器：汇总 results/*.jsonl → 每元准确率 Pareto。

  python benchmarks/report.py
输出：results/report.md（汇总表）、results/pareto.csv、有 matplotlib 时 results/pareto.png。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (small-sample honest)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rows = []
    import re
    for f in sorted(RES.glob("*__*.jsonl")):
        if re.search(r"__r\d+$", f.stem):   # 多轮重复文件（--tag r2/r3）不进单轮汇总
            continue
        provider, ds = f.stem.rsplit("__", 1)
        recs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        if not recs:
            continue
        n = len(recs)
        acc = sum(r["correct"] for r in recs) / n
        cost = sum(r["cost"] for r in recs)
        lat = sum(r["latency_ms"] for r in recs) / n
        err = sum(1 for r in recs if not r["ok"])
        rows.append({"provider": provider, "dataset": ds, "n": n,
                     "accuracy": round(acc, 4), "cost_total": round(cost, 6),
                     "cost_per_1k": round(cost / n * 1000, 4),
                     "avg_latency_ms": round(lat), "errors": err})

    # 汇总表（按数据集分组、按准确率降序）
    lines = ["# judge-econ 结果", "",
             "协议：temperature=0，每样本 1 次判断调用，无 few-shot；CI 为 Wilson 95%。", "",
             "| provider | dataset | n | accuracy (95% CI) | ¥/1000次 | 平均延迟ms | err |",
             "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["dataset"], -x["accuracy"])):
        k = round(r["accuracy"] * r["n"])
        lo, hi = wilson_ci(k, r["n"])
        lines.append(f"| {r['provider']} | {r['dataset']} | {r['n']} | "
                     f"{r['accuracy']:.1%} [{lo:.0%}–{hi:.0%}] | {r['cost_per_1k']:.4f} | "
                     f"{r['avg_latency_ms']} | {r['errors']} |")
    (RES / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Pareto 数据：准确率按样本合并（micro 平均，比 macro 更真实）；成本 = 总成本/总次数（与 README 同口径）
    agg: dict[str, dict] = {}
    for r in rows:
        a = agg.setdefault(r["provider"], {"k": 0, "n": 0, "cost": 0.0})
        a["k"] += round(r["accuracy"] * r["n"])
        a["n"] += r["n"]
        a["cost"] += r["cost_total"]
    pareto = [{"provider": p, "accuracy": round(v["k"] / max(1, v["n"]), 4),
               "cost_per_1k": round(v["cost"] / max(1, v["n"]) * 1000, 4)}
              for p, v in agg.items()]
    pareto.sort(key=lambda x: x["cost_per_1k"])
    with open(RES / "pareto.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["provider", "accuracy", "cost_per_1k"])
        w.writeheader()
        w.writerows(pareto)

    # 错误样本清单（误判明细，供 README 错误分析节引用）
    errs = []
    for f in sorted(RES.glob("*__*.jsonl")):
        for l in open(f, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            if not r["correct"]:
                errs.append({"file": f.stem, "id": r["id"], "gold": r["gold"],
                             "pred": r["pred"], "conf": r["confidence"]})
    with open(RES / "errors.json", "w", encoding="utf-8") as f:
        json.dump(errs, f, ensure_ascii=False, indent=1)

    # ASCII Pareto：x=成本(对数), y=准确率
    lines = ["", "## Cost-Accuracy Pareto（x=¥/1000次判断, y=accuracy）", "", "```"]
    for p in pareto:
        c = max(p["cost_per_1k"], 1e-6)
        import math
        col = min(60, max(1, int(math.log10(c * 10) * 20))) if c > 0 else 1
        lines.append(f"{p['accuracy']:6.1%} |{' ' * col}● {p['provider']} (¥{p['cost_per_1k']}/k)")
    lines.append("```")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        for p in pareto:
            plt.scatter(max(p["cost_per_1k"], 1e-6), p["accuracy"])
            plt.annotate(p["provider"], (max(p["cost_per_1k"], 1e-6), p["accuracy"]),
                         fontsize=8, xytext=(4, 2), textcoords="offset points")
        plt.xscale("symlog", linthresh=0.01)
        plt.xlabel("cost ¥ / 1000 decisions")
        plt.ylabel("accuracy")
        plt.title("benchmarks: cost-accuracy Pareto")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(RES / "pareto.png", dpi=150)
        lines.append("")
        lines.append("图：results/pareto.png")
    except ImportError:
        lines.append("(装 matplotlib 可生成 pareto.png：pip install matplotlib)")
    with open(RES / "report.md", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nreport → {RES / 'report.md'}")


if __name__ == "__main__":
    main()
