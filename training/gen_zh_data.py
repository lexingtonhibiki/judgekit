# -*- coding: utf-8 -*-
"""中文判官训练数据生成器：GLM 自一致性软标签 → NanoJev 训练记录 schema。

两阶段（全部走 GLM coding-plan 订阅，并发上限 2）：
  阶段1 meta-gen：批量生成多样本文本（每批 20 条，均衡覆盖候选类）
  阶段2 标注：每样本 k=3 温度采样自一致性投票 → 软分布 q（+0.5 平滑，和=1）
输出：training/zh_records.jsonl（NanoJev train_pipeline_decisions 训练 schema）
用法：
  export ZHIPU_CODING_KEY=... && python training/gen_zh_data.py            # 全量（约2-4小时）
  python training/gen_zh_data.py --meta-batches 1 --samples 6 --k 3       # 烟测
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "zh_records.jsonl"
DONE_LOG = Path(__file__).resolve().parent / "gen_done.txt"

TASKS = {
    "intent": {
        "family": "zh_intent",
        "qid": "route",
        "type": "choice",
        "instructions": "判断该电商求助内容应流转到哪个部门",
        "criteria": {"退款售后": "退货、退款、换货、发票、赔偿",
                      "物流查询": "快递、运单、发货进度、签收",
                      "投诉建议": "对服务态度、商家或平台的不满与举报",
                      "技术故障": "APP/网页/支付等功能异常",
                      "咨询其他": "一般性业务咨询"},
        "gen_hint": "电商购物售前售后场景的用户求助或咨询，语气多样（着急/平静/口语/带错别字），长短混合",
    },
    "sentiment": {
        "family": "zh_sentiment",
        "qid": "polarity",
        "type": "choice",
        "instructions": "判断这条商品评论的情感倾向",
        "criteria": {"正面": "好评、满意、推荐",
                      "负面": "差评、不满、不推荐"},
        "gen_hint": "电商平台商品评论（数码/服装/食品/家居等品类轮换），好坏均衡，含反讽与马虎表达",
    },
    "spam": {
        "family": "zh_spam",
        "type": "choice",
        "qid": "spam",
        "instructions": "判断这条评论是正常内容还是垃圾广告",
        "criteria": {"垃圾": "广告导流、联系方式、刷单兼职、灌水",
                      "正常": "真实体验、提问、吐槽"},
        "gen_hint": "评论区内容，垃圾类含软广/导流/兼职/灌水变体，正常类含真实好评差评与提问，垃圾样本要含高仿真软广",
    },
    "urgency": {
        "family": "zh_urgency",
        "qid": "urgent",
        "type": "choice",
        "instructions": "判断这条客服消息的紧急程度",
        "criteria": {"紧急": "涉及安全、人身、财产损失的即时风险，或时间窗极窄",
                      "非紧急": "常规咨询、建议、一般问题"},
        "gen_hint": "客服/热线消息，紧急类含安全与资损场景，非紧急类为日常咨询，避免每条都以感叹号结尾的刻板模式",
    },
}


def glm(messages: list, temperature: float, max_retries: int = 6) -> str:
    key = os.environ.get("ZHIPU_CODING_KEY", "")
    body = json.dumps({"model": "glm-5.3-flash", "messages": messages,
                       "temperature": temperature}).encode("utf-8")
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions", data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.loads(r.read().decode("utf-8"))
            content = data["choices"][0]["message"].get("content") or ""
            if content.strip():
                return content
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                time.sleep(20 + 10 * attempt)   # 并发额度被会话占用，长退避
                continue
            if attempt == max_retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("glm empty content")


def parse_json_objects(text: str) -> list:
    """从回复里抠 JSON 对象（容忍 ```json 包裹与逐行输出）。"""
    out = []
    for m in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            out.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            continue
    return out


def meta_generate(task_key: str, task: dict, batches: int) -> list:
    """阶段1：批量生成文本+意向标签。"""
    labels = list(task["criteria"])
    seen, rows = set(), []
    per_call = 20
    for b in range(batches):
        ask = (f"生成{per_call}条中文样本用于训练文本分类器。场景：{task['gen_hint']}。"
               f"类别体系（必须均衡覆盖，每类约{per_call // len(labels)}条）：{json.dumps(task['criteria'], ensure_ascii=False)}。"
               f"要求：自然真实、不要出现'样本'二字、不要编号。只输出 JSON 行，每行："
               f'{{"text": "样本内容", "label": "类别原文"}}')
        try:
            text = glm([{"role": "user", "content": ask}], temperature=1.0)
        except Exception as e:
            print(f"[meta {task_key} batch{b}] fail: {e}", file=sys.stderr, flush=True)
            continue
        got = 0
        for o in parse_json_objects(text):
            t, lb = str(o.get("text", "")).strip(), str(o.get("label", "")).strip()
            if t and lb in labels and t not in seen and 4 < len(t) < 300:
                seen.add(t)
                rows.append({"text": t, "intended": lb})
                got += 1
        print(f"[meta {task_key} batch{b}] +{got} (total {len(rows)})", flush=True)
    return rows


def soft_label(task_key: str, task: dict, text: str, k: int) -> dict | None:
    """阶段2：k=3 自一致性投票 → 平滑软分布（keys=全部候选，和=1）。"""
    votes = []
    for _ in range(k):
        try:
            ask = (f"判断引擎。{task['instructions']}。类别："
                   f"{json.dumps(task['criteria'], ensure_ascii=False)}。\n\n输入：{json.dumps({'text': text}, ensure_ascii=False)}\n\n"
                   '只输出 JSON：{"label": "<类别原文>"}')
            raw = glm([{"role": "system", "content": "你是判断引擎（System One）。只输出一个 JSON 对象。"},
                       {"role": "user", "content": ask}], temperature=0.9)
            m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
            if not m:
                continue
            lb = str(json.loads(m.group(0)).get("label", "")).strip()
            if lb in task["criteria"]:
                votes.append(lb)
        except Exception:
            continue
    if not votes:
        return None
    labels = list(task["criteria"])
    counts = {lb: votes.count(lb) for lb in labels}
    eps = 0.5
    q = {lb: (counts[lb] + eps) / (len(votes) + eps * len(labels)) for lb in labels}
    s = sum(q.values())
    return {lb: q[lb] / s for lb in labels}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-batches", type=int, default=28, help="每任务 meta-gen 批数（×20条）")
    ap.add_argument("--samples", type=int, default=0, help="每任务最多标注样本数（0=全部）")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--tasks", default="intent,sentiment,spam,urgency")
    ap.add_argument("--workers", type=int, default=1,
                    help="默认 1：ZCode 会话自身占用并发额度 2 中的一份")
    args = ap.parse_args()

    done = set()
    if DONE_LOG.exists():
        done = set(DONE_LOG.read_text(encoding="utf-8").split())
    out_f = open(OUT, "a", encoding="utf-8")

    for task_key in [t.strip() for t in args.tasks.split(",")]:
        task = TASKS[task_key]
        rows = meta_generate(task_key, task, args.meta_batches)
        if args.samples:
            rows = rows[:args.samples]
        rows = [r for r in rows if f"zh_{task_key}_{hashlib.md5(r['text'].encode()).hexdigest()[:6]}" not in done]
        print(f"[label {task_key}] 开始标注 {len(rows)} 条 × k={args.k}", flush=True)
        bar = {"n_done": 0}

        def work(r):
            q = soft_label(task_key, task, r["text"], args.k)
            return r, q

        with ThreadPoolExecutor(max_workers=args.workers) as ex:   # 默认 1，给会话留额度
            futs = [ex.submit(work, r) for r in rows]
            for fut in as_completed(futs):
                r, q = fut.result()
                if q is None:
                    continue
                rec_id = f"zh_{task_key}_{hashlib.md5(r['text'].encode()).hexdigest()[:6]}"
                rec = {
                    "id": rec_id,
                    "state_id": rec_id,
                    "family_id": task["family"],
                    "split": "train",
                    "state": json.dumps({"text": r["text"]}, ensure_ascii=False),
                    "questions": {task["qid"]: {"type": task["type"],
                                                "instructions": task["instructions"],
                                                "criteria": task["criteria"]}},
                    "gold_probs": {task["qid"]: q},
                    "gold_probs_kind": {task["qid"]: "programmatic_conditional_distribution"},
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                with open(DONE_LOG, "a", encoding="utf-8") as df:
                    df.write(rec["id"] + "\n")
                bar["n_done"] += 1
                if bar["n_done"] % 25 == 0:
                    print(f"[label {task_key}] {bar['n_done']}/{len(rows)}", flush=True)
    out_f.close()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
