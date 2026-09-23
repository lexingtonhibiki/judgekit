# -*- coding: utf-8 -*-
"""T18 概率阈值重标spam120：同T17单任务逐字重跑120取choice全量probabilities扫阈值。

做法：
  Task/提示词逐字复用T17（import同一对象，禁复制禁改字）；
  decide后取Decision.probabilities记p_spam（键名“垃圾”winsorize到[0,1]，
  缺分布即停线NEEDS_CONTEXT，不编数）；
  ~120 calls typesafe，落spam120_probs.jsonl（id/gold/p_spam/pred_argmax）；
  sweep τ∈{0.05..0.95 step0.05} verdict=垃圾 iff p_spam≥τ，
  输出阈值表（acc/recall/precision/F1/与argmax翻转数）+推荐τ
  （Youden=recall+specificity-1最大，F1次之，并列取小τ）+提升Δvs60.0%。
不碰gold/models.yaml与提示词一字。

用法：
  python training/eval_spam_probs.py
  python training/eval_spam_probs.py --limit 2   # 冒烟（只跑前2条）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import eval_spam120 as t17  # noqa: E402  T17单任务与提示词逐字复用，禁复制
from training import eval_jev_v4 as jev  # noqa: E402
from judgekit.engine import run_task  # noqa: E402

XLSX_PATH = t17.XLSX_PATH
OUT_PATH = ROOT / "training" / "abc_out" / "spam120_probs.jsonl"
PROVIDER_NAME = t17.PROVIDER_NAME
TRIES = t17.TRIES

# T17单任务提示词逐字同一对象（禁复制禁改字）。
SPAM_PROMPT = t17.SPAM_PROMPT

# 阈值网格 0.05..0.95 step0.05（19档）；基线为T17 argmax 60.0%（72/120）。
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]
BASE_ACC = 0.60


def build_spam_task():
    """T17单任务逐字复用（委托同一构造，不重打字）。"""
    return t17.build_spam_task()


def extract_p_spam(dec) -> float:
    """decide后取Decision.probabilities记p_spam（键名“垃圾”winsorize到[0,1]）。

    缺分布（None/非dict/缺“垃圾”键/转float失败）即停线NEEDS_CONTEXT，不编数。
    """
    probs = getattr(dec, "probabilities", None)
    if not isinstance(probs, dict) or "垃圾" not in probs:
        raise SystemExit("NEEDS_CONTEXT: 缺choice全量probabilities（或缺'垃圾'键），停线不编数")
    try:
        p = float(probs["垃圾"])
    except (TypeError, ValueError):
        raise SystemExit("NEEDS_CONTEXT: p_spam非数值，停线不编数")
    if p != p:  # NaN
        raise SystemExit("NEEDS_CONTEXT: p_spam为NaN，停线不编数")
    return max(0.0, min(1.0, p))


def sweep_thresholds(records: list[dict],
                     thresholds: list[float] | None = None) -> list[dict]:
    """扫阈值：verdict=垃圾 iff p_spam≥τ；垃圾为正类。

    records项：{id,gold,p_spam,pred_argmax}；返回每τ一行
    {tau,n,acc,recall,specificity,precision,f1,youden,flips,tp,tn,fp,fn}。
    零除（无正例/无负例/无预测正例）对应指标记0.0，不抛错。
    """
    ts = list(THRESHOLDS if thresholds is None else thresholds)
    table: list[dict] = []
    for tau in ts:
        tp = tn = fp = fn = flips = 0
        for r in records:
            verdict = "垃圾" if float(r["p_spam"]) >= float(tau) else "正常"
            gold = r["gold"]
            if verdict == "垃圾" and gold == "垃圾":
                tp += 1
            elif verdict == "正常" and gold == "正常":
                tn += 1
            elif verdict == "垃圾" and gold == "正常":
                fp += 1
            else:
                fn += 1
            if verdict != r["pred_argmax"]:
                flips += 1
        n = len(records)
        acc = (tp + tn) / n if n else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        youden = recall + specificity - 1
        table.append({"tau": float(tau), "n": n, "acc": acc,
                      "recall": recall, "specificity": specificity,
                      "precision": precision, "f1": f1, "youden": youden,
                      "flips": flips, "tp": tp, "tn": tn, "fp": fp, "fn": fn})
    return table


def recommend_threshold(table: list[dict]) -> dict:
    """推荐τ：Youden最大，F1次之，并列取小τ（确定性，无随机）。"""
    if not table:
        raise ValueError("空阈值表，无可推荐τ")
    return max(table, key=lambda r: (r["youden"], r["f1"], -r["tau"]))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def judge_with_probs(row: dict, task, providers: dict,
                     tries: int = TRIES, backoff: float = 2.0) -> dict:
    """判一条并取p_spam：ok+非兜底+有分布才成功，否则重试耗尽后停线NEEDS_CONTEXT。"""
    x = {"text": row.get("text", "")}
    last_err = ""
    for i in range(tries):
        try:
            dec = run_task(task, x, providers, fallback=True)
        except Exception as e:  # noqa: BLE001 兜底之外的意外也重试
            last_err = f"{type(e).__name__}: {e}"
            if i < tries - 1 and backoff > 0:
                time.sleep(backoff * (2 ** i))
            continue
        if dec is not None and dec.ok and not jev.is_fallback_decision(dec):
            try:
                p = extract_p_spam(dec)
            except SystemExit:
                last_err = "missing-probabilities"
                if i < tries - 1 and backoff > 0:
                    time.sleep(backoff * (2 ** i))
                continue
            return {"id": row["id"], "gold": row.get("gold"),
                    "p_spam": p, "pred_argmax": dec.value}
        last_err = (getattr(dec, "error", "") or last_err or "not-ok-or-fallback")
        if i < tries - 1 and backoff > 0:
            time.sleep(backoff * (2 ** i))
    raise SystemExit(f"NEEDS_CONTEXT: 缺choice全量probabilities（行{row.get('id')}："
                     f"{last_err}），停线不编数")


def main() -> None:
    ap = argparse.ArgumentParser(description="T18 概率阈值重标spam120（同T17逐字，取probabilities扫阈值）")
    ap.add_argument("--xlsx", default=str(XLSX_PATH))
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--tries", type=int, default=TRIES)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    print(f"[spam-probs提示词全文] {SPAM_PROMPT}", flush=True)
    rows = t17.load_spam120_rows(Path(a.xlsx))
    print(f"[spam-probs集合] 05垃圾_120 n={len(rows)}（与T17同源同序；"
          f"直标垃圾{sum(1 for r in rows if r['gold'] == '垃圾')}"
          f"+正常{sum(1 for r in rows if r['gold'] == '正常')}）", flush=True)
    eval_rows = rows[:a.limit] if a.limit else rows
    providers = jev.load_providers_strict()  # 缺key在此停线，零调用
    task = build_spam_task()

    out = Path(a.out)
    done = {r["id"]: r for r in load_jsonl(out)}  # 断点续跑：已落盘id跳过
    n_skip = sum(1 for r in eval_rows if r["id"] in done)
    if n_skip:
        print(f"[spam-probs] 断点续跑：已落盘{n_skip}行跳过", flush=True)
    with open(out, "a", encoding="utf-8") as f:
        for i, row in enumerate(eval_rows):
            if row["id"] in done:
                continue
            rec = judge_with_probs(row, task, providers, tries=a.tries)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[rec["id"]] = rec
            if (i + 1) % 20 == 0 or len(done) == len(eval_rows):
                print(f"[spam-probs] 进度 {len(done)}/{len(eval_rows)}", flush=True)
    recs = [done[r["id"]] for r in eval_rows if r["id"] in done]
    n = len(recs)
    arg_k = sum(1 for r in recs if r["pred_argmax"] == r["gold"])
    print(f"[spam-probs落盘] n={n} -> {out}（字段id/gold/p_spam/pred_argmax；"
          f"argmax acc={arg_k}/{n}={arg_k / n * 100:.1f}%）", flush=True)

    table = sweep_thresholds(recs)
    print("[spam-probs阈值表] tau acc recall precision F1 flips (verdict=垃圾 iff p_spam≥τ)", flush=True)
    for r in table:
        print(f"  τ={r['tau']:.2f} acc={r['acc'] * 100:.1f}% "
              f"recall={r['recall'] * 100:.1f}% prec={r['precision'] * 100:.1f}% "
              f"F1={r['f1']:.3f} flips_vs_argmax={r['flips']}", flush=True)
    best = recommend_threshold(table)
    delta = (best["acc"] - BASE_ACC) * 100
    print(f"[spam-probs推荐] τ={best['tau']:.2f} acc={best['acc'] * 100:.1f}% "
          f"recall={best['recall'] * 100:.1f}% prec={best['precision'] * 100:.1f}% "
          f"F1={best['f1']:.3f} Youden={best['youden']:.3f} "
          f"flips_vs_argmax={best['flips']} "
          f"Δvs60.0%={delta:+.1f}pt（Youden最大，F1次之，并列取小τ）", flush=True)


if __name__ == "__main__":
    main()
