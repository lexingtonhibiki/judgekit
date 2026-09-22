# -*- coding: utf-8 -*-
"""T4 CSDS派生：20对话截用户追问窗 → 转人工/不转 pre-label（brief派生规则）。

派生规则（确定性，brief原文）：
  H1 同topic复现≥2轮 → 转人工（同一intent出现≥2个QA段，或两用户问句内容词Jaccard≥0.5）
  H2 客服空洞回复 + 用户继续追问 → 转人工（AnsSummShort命中空洞模式且其后还有QA段）
  否则 → 不转
用户追问窗 = 触发轮的用户原文utterance（按QueSummUttIDs回查Dialogue），不转时取末2轮用户话。
空洞模式（HOLLOW，显式清单）：无法/暂时无法/不能/不支持/没有该功能/–/做不到/解决不了/
  建议(等待|咨询|联系|反馈)/请(耐心|等待|稍后|谅解)/已(记录|反馈|上报)/系统(问题|升级|繁忙)/
  权限不足/超出.*范围/不归.*管

brief文件名 derive_cscs.py 即本文件（CSDS之误植，此处用正确拼写）。

用法：
  python training/derive_csds.py [--in training/abc_out/probe_csds_raw.jsonl]
                                 [--out training/abc_out/probe_handoff.jsonl]
  python training/derive_csds.py --gates --scores training/abc_out/probe_scores.jsonl
      --derive training/abc_out/probe_handoff.jsonl --report training/abc_out/probe_gates.json
验收门（--gates自动算）：
  任一源 需人工率（C.action=需人工复核占比；删除率待xlsx人工复核后填，自动化阶段=0）超40% → 该源OUT
  CSDS派生-终判一致率（pre orig_label vs C.final，仅C.ok行）<70% → 回炉
  超标的源标出局但不停线。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CSDS_PAGE = "https://github.com/xiaolinAndy/CSDS"
CSDS_LIC = "研究使用(原仓仅代码；数据走作者GDrive直链，Baidu盘为镜像；注明来源+引用EMNLP21)"

HOLLOW = re.compile(
    "无法|暂时无法|不能|不支持|没有该|做不到|解决不了|办不了|"
    "建议.*(等待|咨询|联系|反馈|换|重)|请(耐心|等待|稍后|谅解|理解)|"
    "已(记录|反馈|上报|登记)|系统(问题|升级|繁忙|异常)|权限不足|"
    "超出.*范围|不归.*管|无此功能|暂未开通")
TOK = re.compile(r"[\u4e00-\u9fff_a-zA-Z0-9]+")


def content_words(s: str) -> set[str]:
    return {w for w in TOK.findall(s or "") if len(w) >= 2}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def utter_map(dlg: dict) -> dict[int, dict]:
    return {m.get("turn"): m for m in dlg.get("Dialogue", []) if isinstance(m.get("turn"), int)}


def all_user_texts(dlg: dict) -> list[str]:
    return [(m.get("utterance") or "").strip()
            for m in dlg.get("Dialogue", [])
            if m.get("speaker") == "Q" and (m.get("utterance") or "").strip()]


def user_texts(dlg: dict, tids: list) -> list[str]:
    um = utter_map(dlg)
    out = []
    for t in tids or []:
        m = um.get(t)
        if m and m.get("speaker") == "Q" and (m.get("utterance") or "").strip():
            out.append(m["utterance"].strip())
    return out


def derive_one(dlg: dict) -> dict:
    qas = dlg.get("QA", []) or []
    intents = [(q.get("intent") or "").strip() for q in qas]
    # H1：同intent复现，或两问句内容词Jaccard≥0.5
    h1_ev = ""
    seen: dict[str, int] = {}
    for it in intents:
        if it:
            seen[it] = seen.get(it, 0) + 1
    dup_intent = next((k for k, v in seen.items() if v >= 2), "")
    if dup_intent:
        h1_ev = f"同intent复现≥2轮:{dup_intent}"
    else:
        ques = [q.get("QueSumm", "") for q in qas]
        for i in range(len(ques)):
            for j in range(i + 1, len(ques)):
                if jaccard(content_words(ques[i]), content_words(ques[j])) >= 0.5:
                    h1_ev = f"问句复现≥2轮:QA{i + 1}&QA{j + 1}"
                    break
            if h1_ev:
                break
    # H2：空洞回复+继续追问
    h2_ev, h2_idx = "", -1
    for i, q in enumerate(qas):
        if HOLLOW.search(q.get("AnsSummShort", "") or "") and i + 1 < len(qas):
            h2_ev, h2_idx = f"QA{i + 1}空洞+继续追问", i
            break
    if h1_ev or h2_ev:
        pre, rule = "转人工", "|".join(x for x in (f"H1:{h1_ev}" if h1_ev else "",
                                                  f"H2:{h2_ev}" if h2_ev else "") if x)
        tids: list[int] = []
        if h1_ev:
            for q in qas:
                tids += [t for t in (q.get("QueSummUttIDs") or []) if t not in tids]
        else:
            for q in qas[h2_idx + 1:]:
                tids += [t for t in (q.get("QueSummUttIDs") or []) if t not in tids]
        win = user_texts(dlg, tids) or user_texts(
            dlg, [t for q in qas for t in (q.get("QueSummUttIDs") or [])][-2:])
    else:
        pre, rule = "不转", "无触发"
        allq = [t for q in qas for t in (q.get("QueSummUttIDs") or [])]
        win = user_texts(dlg, allq[-2:]) or user_texts(dlg, allq[-1:])
    if not win:  # 标注uttID错位（如指向客服轮）→ 回退全对话末2轮用户话
        win = all_user_texts(dlg)[-2:]
        rule += "|回退Q末2轮"
    if not win:  # 极端：无用户话 → 用问句摘要，标记
        win = [q.get("QueSumm", "") for q in qas if q.get("QueSumm")]
        rule += "|回退QueSumm"
    text = "\n".join(win)[:600]
    assert text.strip(), f"空窗：DialogueID={dlg.get('DialogueID')}"
    return {"pre": pre, "rule": rule,
            "ev": (h1_ev + ";" + h2_ev).strip(";") or "-",
            "hollow_hit": bool(h2_ev), "window_turns": len(win), "text": text}


def derive(in_p: Path, out_p: Path) -> list[dict]:
    dlgs = [json.loads(l) for l in open(in_p, encoding="utf-8") if l.strip()]
    rows = []
    for i, d in enumerate(dlgs):
        r = derive_one(d)
        rows.append({
            "id": f"csds_{i + 1:04d}", "text": r["text"], "task": "handoff",
            "orig_label": r["pre"], "source": "CSDS", "url": CSDS_PAGE,
            "license": CSDS_LIC, "pre_rule": r["rule"], "pre_ev": r["ev"],
            "dialogue_id": d.get("DialogueID"),
            "qa_n": len(d.get("QA", []) or []),
            "window_turns": r["window_turns"]})
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_h = sum(1 for r in rows if r["orig_label"] == "转人工")
    print(f"derive: {len(rows)}行（转人工预{n_h}/不转预{len(rows) - n_h}）-> {out_p.name}")
    return rows


def gates(scores_p: Path, derive_p: Path, report_p: Path) -> dict:
    scores = [json.loads(l) for l in open(scores_p, encoding="utf-8") if l.strip()]
    pre = {json.loads(l)["id"]: json.loads(l) for l in open(derive_p, encoding="utf-8")
           if l.strip()} if derive_p.exists() else {}
    rep: dict[str, dict] = {}
    for src in sorted({r.get("source", "?") for r in scores}):
        rs = [r for r in scores if r.get("source") == src]
        ab_ok = [r for r in rs if r.get("A", {}).get("ok") and r.get("B", {}).get("ok")]
        need = sum(1 for r in rs if r.get("C", {}).get("action") == "需人工复核")
        rate = round(need / len(rs), 3) if rs else 0.0
        rep[src] = {"n": len(rs), "ab_ok": len(ab_ok), "c_need": need,
                    "need_rate": rate, "delete_rate": 0.0,
                    "verdict": "OUT" if rate > 0.4 else "PASS"}
    # CSDS一致率：pre vs C.final（仅C.ok行）
    cs = [r for r in scores if r.get("source") == "CSDS" and r.get("C", {}).get("ok")]
    agree = sum(1 for r in cs if pre.get(r["id"], {}).get("orig_label") == r["C"].get("final"))
    rate = round(agree / len(cs), 3) if cs else 0.0
    rep["CSDS_agree"] = {"n": len(cs), "agree": agree, "agree_rate": rate,
                         "verdict": "回炉" if rate < 0.7 else "PASS"}
    with open(report_p, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    for k, v in rep.items():
        print(f"  gate[{k}]: {v}")
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="training/abc_out/probe_csds_raw.jsonl")
    ap.add_argument("--out", default="training/abc_out/probe_handoff.jsonl")
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("--scores", default="training/abc_out/probe_scores.jsonl")
    ap.add_argument("--derive", default="training/abc_out/probe_handoff.jsonl")
    ap.add_argument("--report", default="training/abc_out/probe_gates.json")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.gates:
        gates(ROOT / args.scores, ROOT / args.derive, ROOT / args.report)
    else:
        derive(ROOT / args.inp, ROOT / args.out)


if __name__ == "__main__":
    main()
