# -*- coding: utf-8 -*-
"""后处理温度缩放：在 checkpoint 的 predictions_calibration.jsonl 上拟合 T（最小化目标分布 CE）。

用法：python training/archive/fit_temperature.py --ckpt D:/Models/NanoJev-zh/dev06 [--tmax 3.0]
Agent A 结论：训练器无温度拟合（temperature_fitted 恒 False），但 DecisionPredictor.predict(temperature=T)
是现成钩子——本脚本只做拟合与验证，部署时把 T 传给 predict/serve。
注意：n<30 时 T 方差大，仅作参考。
"""
import argparse
import json
import math
import sys
from pathlib import Path


def softmax(xs, t):
    m = max(x / t for x in xs)
    ex = [math.exp(x / t - m) for x in xs]
    s = sum(ex)
    return [e / s for e in ex]


def ece(pairs, bins=10):
    """pairs: [(top_conf, correct_bool)]"""
    if not pairs:
        return float("nan")
    tot = len(pairs)
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [p for p in pairs if lo <= p[0] < hi or (b == bins - 1 and p[0] == hi)]
        if bucket:
            e += len(bucket) / tot * abs(sum(p[1] for p in bucket) / len(bucket)
                                         - sum(p[0] for p in bucket) / len(bucket))
    return e


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tmax", type=float, default=3.0)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(Path(args.ckpt) / "predictions_calibration.jsonl", encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("student_logits") and r.get("gold_distribution_probs")]
    if not rows:
        sys.exit("no calibration rows")
    grid = [round(0.20 + 0.05 * i, 2) for i in range(int((args.tmax - 0.20) / 0.05) + 1)]
    ce = {}
    for t in grid:
        s = 0.0
        for r in rows:
            p = softmax(r["student_logits"], t)
            s += -sum(tq * math.log(max(pi, 1e-12)) for tq, pi in zip(r["gold_distribution_probs"], p))
        ce[t] = s / len(rows)
    t_star = min(ce, key=ce.get)
    base = [max(p) for p in (softmax(r["student_logits"], 1.0) for r in rows)]
    fit = [max(p) for p in (softmax(r["student_logits"], t_star) for r in rows)]
    ok = lambda r: max(range(len(r["student_probs"])), key=lambda i: r["student_probs"][i]) == \
        max(range(len(r["gold_distribution_probs"])), key=lambda i: r["gold_distribution_probs"][i])
    pairs = [(c, ok(r)) for c, r in zip(base, rows)]
    pairs_f = [(c, ok(r)) for c, r in zip(fit, rows)]
    print(f"ckpt={args.ckpt} n={len(rows)}")
    print(f"CE(T=1)={ce[1.0]:.4f}  CE(T*={t_star})={ce[t_star]:.4f}")
    print(f"ECE(T=1)={ece(pairs):.4f}  ECE(T*)={ece(pairs_f):.4f}  (argmax 不受 T 影响，acc 不变)")
    print(f"建议 serve/predict temperature={t_star}")


if __name__ == "__main__":
    main()
