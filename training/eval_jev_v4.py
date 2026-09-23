# -*- coding: utf-8 -*-
"""Jev v4 gold榜单续集(T13)：gold_frozen.jsonl 190行，原生typesafe Jev逐条一判。

输入：training/abc_out/gold_frozen.jsonl（{id,text,task,gold,source}，
  task∈{spam,handoff}；spam120=JD刷单60+FakeReview60，handoff70=CSDS70）
任务：两个classify Task（labels+富描述≤40字，T7泛化口径：spam含刷单三语义+
  单特征禁令，handoff含升级语义），criteria/instruction沿用ABC rubric语义。
供应商：只载typesafe（.env/环境变量TYPESAFE_API_KEY；缺key即停线NEEDS_CONTEXT，
  不硬跑）。逐条run_task(fallback=True；兜底命中记ok=False、pred=None，不充正确)。
输出：training/abc_out/jev_v4_scores.jsonl（{id,task,source,gold,pred,conf,
  correct,latency_ms,cost,ok,error,provider}）+ stdout榜单行（分源acc+
  Wilson95%CI，小样本n=60/70如实标+latency均值+¥/千次）。
约190 calls；失败每条重试3次；断点续跑（读已落盘id跳过）。

用法：
  python training/eval_jev_v4.py
  python training/eval_jev_v4.py --limit 2   # 冒烟（只跑前2条）
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from judgekit.engine import Task, run_task  # noqa: E402
from judgekit.providers.typesafe import TypeSafe  # noqa: E402

GOLD_PATH = ROOT / "training" / "abc_out" / "gold_frozen.jsonl"
OUT_PATH = ROOT / "training" / "abc_out" / "jev_v4_scores.jsonl"
MODELS_YAML = ROOT / "benchmarks" / "models.yaml"
ENV_PATH = ROOT / ".env"
PROVIDER_NAME = "typesafe"
TRIES = 3

# ---- 富描述（每条≤40字，T7泛化口径：刷单三语义+单特征禁令/升级语义）----
SPAM_LABEL_DESC = {
    "垃圾": "营销引流/刷屏/刷单，或无细节断言夸赞情绪倒挂零增量",
    "正常": "抱怨咨询事实陈述不判；单特征不定罪看可验证事实",
}
HANDOFF_LABEL_DESC = {
    "转人工": "辱骂威胁重复催≥2次或情绪崩溃才转",
    "不转": "投诉冷静有诉求无失控情绪则不转",
}
SPAM_CRITERIA = ("垃圾判定：营销引流/刷屏才判，抱怨差评驳回；刷单须举一反三类推——"
                 "无可验证细节的断言式夸赞、情绪强度与事实密度倒挂、对下单决策零信息增量，"
                 "联想同类夸词/品牌极端词/无细节夸张/维度标点轰炸形态；"
                 "禁单特征定罪或脱罪，须结合可验证具体事实综合判")
HANDOFF_CRITERIA = ("转人工判定：辱骂/威胁/重复催≥2次或情绪崩溃才转；"
                    "投诉但冷静、有具体诉求、无失控情绪不转")
SPAM_INSTRUCTION = "举一反三类推，禁单特征定罪脱罪；只看text字段独立判定。"
HANDOFF_INSTRUCTION = "只看text字段独立判定。"

# 规则兜底关键词（仅失败托底用；命中仍记ok=False，不充正确）
SPAM_FALLBACK = {
    "垃圾": ["加微", "微信", "链接", "首存", "发票", "贷款", "赌博", "钓鱼", "返现"],
    "正常": ["排队", "差评", "咨询", "请问"],
}
HANDOFF_FALLBACK = {
    "转人工": ["辱骂", "威胁", "催", "哭", "绝望", "失控"],
    "不转": ["咨询", "请问", "谢谢", "查询"],
}


def build_tasks() -> dict:
    """组两个classify Task（provider钉typesafe，不钉名会滑入rules兜底假绿）。"""
    spam = Task(name="jev-v4-spam", primitive="classify",
                labels=["垃圾", "正常"],
                label_descriptions=dict(SPAM_LABEL_DESC),
                criteria=SPAM_CRITERIA, instruction=SPAM_INSTRUCTION,
                input_field="text", provider=PROVIDER_NAME,
                fallback_rules={k: list(v) for k, v in SPAM_FALLBACK.items()})
    handoff = Task(name="jev-v4-handoff", primitive="classify",
                   labels=["转人工", "不转"],
                   label_descriptions=dict(HANDOFF_LABEL_DESC),
                   criteria=HANDOFF_CRITERIA, instruction=HANDOFF_INSTRUCTION,
                   input_field="text", provider=PROVIDER_NAME,
                   fallback_rules={k: list(v) for k, v in HANDOFF_FALLBACK.items()})
    return {"spam": spam, "handoff": handoff}


def load_api_key() -> str:
    """环境变量优先，其次 repo 根 .env（TYPESAFE_API_KEY=…）；缺key返回空串。"""
    v = (os.environ.get("TYPESAFE_API_KEY") or "").strip()
    if v:
        return v
    try:
        with open(ENV_PATH, encoding="utf-8-sig") as f:
            for ln in f:
                s = ln.strip()
                if s.startswith("TYPESAFE_API_KEY") and "=" in s:
                    return s.split("=", 1)[1].strip().strip("'\"")
    except OSError:
        pass
    return ""


def load_providers_strict() -> dict:
    """只载typesafe；缺key即停线NEEDS_CONTEXT，不硬跑。"""
    key = load_api_key()
    if not key:
        raise SystemExit("NEEDS_CONTEXT: 缺TYPESAFE_API_KEY（.env/环境变量），停线不硬跑")
    model, pin, pout, timeout = "jev-latest", 0.0003, 0.0, 60
    try:
        import yaml
        with open(MODELS_YAML, encoding="utf-8") as f:
            c = (yaml.safe_load(f).get("providers") or {}).get(PROVIDER_NAME) or {}
        model = c.get("model", model)
        pin = float(c.get("price_in_per_1k", pin))
        pout = float(c.get("price_out_per_1k", pout))
        timeout = int(c.get("timeout", timeout))
    except Exception:
        pass  # models.yaml缺失即用默认，不停线
    ts = TypeSafe(name=PROVIDER_NAME, model=model, api_key=key,
                  price_in_per_1k=pin, price_out_per_1k=pout, timeout=timeout)
    return {PROVIDER_NAME: ts}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI（n=0→(0,0)；钳制[0,1]；k=0下界为0，k=n上界为1）。"""
    if n <= 0:
        return (0.0, 0.0)
    k = max(0, min(n, k))
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def is_fallback_decision(dec) -> bool:
    """是否走了规则兜底（命中/未命中均属非Jev判定，不充正确）。"""
    if dec is None:
        return True
    if str(getattr(dec, "provider", "")).startswith("rules"):
        return True
    return "fallback-ok" in (getattr(dec, "error", "") or "")


def judge_row(row: dict, tasks: dict, providers: dict,
              tries: int = TRIES, backoff: float = 2.0) -> dict:
    """判一条gold行。成功→Jev判定；失败/兜底→ok=False、pred=None，不充正确。"""
    task = tasks.get(row.get("task", ""))
    if task is None:
        raise ValueError(f"未知task: {row.get('task')!r}（行{row.get('id')}）")
    x = {"text": row.get("text", "")}
    gold = row.get("gold")
    dec, err = None, ""
    for i in range(tries):
        try:
            d = run_task(task, x, providers, fallback=True)
        except Exception as e:  # noqa: BLE001 兜底之外的意外也重试
            d, err = None, f"{type(e).__name__}: {e}"
        if d is not None and d.ok and not is_fallback_decision(d):
            dec = d
            break
        dec = d  # 末轮状态（失败或兜底），循环耗尽后记失败
        if i < tries - 1 and backoff > 0:
            time.sleep(backoff * (2 ** i))
    if dec is not None and dec.ok and not is_fallback_decision(dec):
        pred = dec.value
        return {"id": row["id"], "task": row["task"], "source": row.get("source", ""),
                "gold": gold, "pred": pred, "conf": round(float(dec.confidence), 3),
                "correct": bool(pred == gold),
                "latency_ms": int(dec.latency_ms), "cost": float(dec.cost),
                "ok": True, "error": "", "provider": dec.provider}
    if dec is not None:
        lat, cost, prov, err = (int(dec.latency_ms), float(dec.cost),
                                dec.provider, dec.error or err)
    else:
        lat, cost, prov = 0, 0.0, "exception"
    return {"id": row["id"], "task": row["task"], "source": row.get("source", ""),
            "gold": gold, "pred": None, "conf": 0.0, "correct": False,
            "latency_ms": lat, "cost": cost,
            "ok": False, "error": err, "provider": prov}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def report_line(recs: list[dict]) -> str:
    """stdout榜单行：分源acc+Wilson95%CI（小样本如实标）+latency均值+¥/千次。"""
    parts = []
    for src in ("JD刷单", "FakeReview", "CSDS"):
        rs = [r for r in recs if r.get("source") == src]
        n = len(rs)
        if not n:
            parts.append(f"{src} n=0：无行")
            continue
        k = sum(1 for r in rs if r.get("correct"))
        lo, hi = wilson(k, n)
        lat = sum(r.get("latency_ms", 0) for r in rs) / n
        cny_k = sum(r.get("cost", 0.0) for r in rs) / n * 1000
        parts.append(f"{src} n={n}(小样本) acc={k}/{n}={k / n * 100:.1f}% "
                     f"95%CI[{lo * 100:.1f}%,{hi * 100:.1f}%] "
                     f"lat均值={lat:.0f}ms ¥/千次={cny_k:.4f}")
    n = len(recs)
    k = sum(1 for r in recs if r.get("correct"))
    ok = sum(1 for r in recs if r.get("ok"))
    lo, hi = wilson(k, n)
    parts.append(f"合计 n={n} acc={k}/{n}={k / n * 100:.1f}% "
                 f"95%CI[{lo * 100:.1f}%,{hi * 100:.1f}%] Jev直判ok={ok}/{n}")
    return "[jev-v4榜单] " + " | ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Jev v4 gold榜单：原生typesafe逐条一判")
    ap.add_argument("--gold", default=str(GOLD_PATH))
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--tries", type=int, default=TRIES)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = load_jsonl(Path(a.gold))
    if not rows:
        raise SystemExit(f"gold为空或缺失：{a.gold}")
    if a.limit:
        rows = rows[:a.limit]
    providers = load_providers_strict()  # 缺key在此停线，零调用
    tasks = build_tasks()

    out = Path(a.out)
    done = {r["id"]: r for r in load_jsonl(out)}  # 断点续跑：已落盘id跳过
    n_skip = sum(1 for r in rows if r["id"] in done)
    if n_skip:
        print(f"[jev-v4] 断点续跑：已落盘{n_skip}行跳过", flush=True)
    with open(out, "a", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            if row["id"] in done:
                continue
            rec = judge_row(row, tasks, providers, tries=a.tries)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[rec["id"]] = rec
            if (i + 1) % 20 == 0 or (i + 1) == len(rows):
                print(f"[jev-v4] 进度 {len(done)}/{len(rows)}", flush=True)
    recs = [done[r["id"]] for r in rows if r["id"] in done]
    print(report_line(recs), flush=True)


if __name__ == "__main__":
    main()
