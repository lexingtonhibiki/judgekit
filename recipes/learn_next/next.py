# -*- coding: utf-8 -*-
"""learn-next — 学习导航判官。

给定学生状态（掌握度/近期错题）与课程知识点图谱（先修关系+考试权重），
判断"下一个最应该学的知识点"，目标可配置（pass_exam / interest）。

零成本默认：--local 纯启发式（先修完整度/缺口/考试权重加权）。
判官模式：--model <providers名> 用 score 原语给每个候选打"当前学它的综合收益"，
再与启发式按 0.5/0.5 融合。候选 = 未掌握且先修完整度最高的节点（可 --top 控制参与判官的候选数，省额度）。

用法（仓库根目录）：
  python learn-next/next.py                                # 本地启发式
  python learn-next/next.py --model router-free-auto --top 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from judgekit.engine import Task, run_task                 # noqa: E402
from judgekit.providers import load_providers              # noqa: E402


def heuristic(kg: dict, mastery: dict, goal: str) -> list[dict]:
    w = kg["goal_weights"][goal]
    scored = []
    for n in kg["nodes"]:
        m = mastery.get(n["id"], 0.0)
        if m >= 0.8:
            continue  # 已掌握
        pr = [mastery.get(p, 0.0) for p in n["prereqs"]]
        prereq_ready = sum(pr) / len(pr) if pr else 1.0
        score = (w["连贯性"] * prereq_ready
                 + w["过考概率"] * n["exam_weight"]
                 + w["掌握缺口"] * (1 - m))
        scored.append({"node": n["name"], "score": round(score, 3), "prereq_ready": round(prereq_ready, 2),
                       "why": f"先修完整度{prereq_ready:.0%}，考频权重{n['exam_weight']}，掌握度{m:.0%}"})
    scored.sort(key=lambda x: -x["score"])
    return scored


def judge_rescore(kg: dict, mastery: dict, goal: str, cands: list[dict],
                  providers: dict, provider_name: str) -> tuple[list[dict], float]:
    task = Task(name="learn-next", primitive="score",
                criteria=("以「通过该课程考试」为首要目标（goal=" + goal + "），"
                          "综合考虑：知识连贯性（先修是否扎实）、考试出现频率、当前掌握缺口，"
                          "评估「现在就学这个知识点」的收益。"),
                instruction="只对给定候选打分，不要发散。", provider=provider_name)
    total_cost = 0.0
    out = []
    snap = {n["name"]: mastery.get(n["id"], 0.0) for n in kg["nodes"]}
    for c in cands:
        x = {"课程": kg["course"], "目标": goal, "各点掌握度": snap, "候选知识点": c["node"],
             "候选详情": c["why"]}
        dec = run_task(task, x, providers)
        total_cost += dec.cost
        j = dec.value if dec.ok else c["score"]
        blended = round(0.5 * j + 0.5 * c["score"], 3)
        out.append({"node": c["node"], "judge": j, "heuristic": c["score"], "score": blended,
                    "confidence": dec.confidence, "ok": dec.ok, "error": dec.error})
    out.sort(key=lambda x: -x["score"])
    return out, total_cost


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default=str(here / "kg_signal_systems.json"))
    ap.add_argument("--state", default=str(here / "student_sample.json"))
    ap.add_argument("--model", default="", help="providers 名（models.yaml）；留空=纯本地启发式")
    ap.add_argument("--providers-file", default=str(ROOT / "benchmarks" / "models.yaml"))
    ap.add_argument("--top", type=int, default=3, help="送判官精排的候选数（控额度）")
    args = ap.parse_args()

    kg = json.load(open(args.kg, encoding="utf-8"))
    st = json.load(open(args.state, encoding="utf-8"))
    ranked = heuristic(kg, st["mastery"], st["goal"])
    cost = 0.0
    if args.model:
        providers = load_providers(args.providers_file)
        top, cost = judge_rescore(kg, st["mastery"], st["goal"],
                                  ranked[:args.top], providers, args.model)
        ranked = top + ranked[args.top:]

    print(f"课程《{kg['course']}》 学生：{st['student']} 目标：{st['goal']}"
          + (f" 模式：判官({args.model})" if args.model else " 模式：本地启发式(0成本)"))
    if st.get("recent_errors"):
        print(f"近期错题：{'; '.join(st['recent_errors'])}")
    print("-" * 64)
    for i, r in enumerate(ranked[:5], 1):
        tag = " ← 下一步就学这个" if i == 1 else ""
        extra = f" judge={r.get('judge')}" if "judge" in r else ""
        print(f"{i}. {r['node']}  score={r['score']}{extra}  conf={r.get('confidence', '-')}")
        print(f"   理由：{r.get('why', '')}{tag}")
    print(f"\n本次判官成本：¥{cost:.4f}")


if __name__ == "__main__":
    main()
