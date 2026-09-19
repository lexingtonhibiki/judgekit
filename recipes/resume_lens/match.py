# -*- coding: utf-8 -*-
"""resume-lens — 简历×岗位双向匹配判官（带意愿问卷前置）。

双向：fit(简历→JD 匹配度) 与 attract(JD→简历 吸引力)，外加 JD 红旗
（与意愿问卷冲突的强度信号：大小周/驻场/单休…）。人先填问卷，机器再判匹配——
定位是求职者侧辅助工具，不做企业侧淘汰。

零成本默认：--local 纯启发式（技能 Jaccard + 薪资/强度红线）。
判官模式：--model 用 score 原语双向打分 + verify 原语核对红旗，与启发式融合。

用法（仓库根目录）：
  python resume-lens/match.py
  python resume-lens/match.py --model router-free-auto --top 2
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from judgekit.engine import Task, run_task     # noqa: E402
from judgekit.providers import load_providers  # noqa: E402

OVERTIME_KW = ["大小周", "单休", "996", "加班", "驻场", "外派", "高压", "轮班"]
GREEN_KW = ["双休", "弹性工作", "六险一金", "五险一金", "远程", "体检"]


def parse_salary(s: str) -> tuple[int, int]:
    m = re.search(r"(\d+)\s*-\s*(\d+)\s*k", s.lower())
    return (int(m.group(1)) * 1000, int(m.group(2)) * 1000) if m else (0, 0)


def jaccard(a: list[str], b: list[str]) -> float:
    A = {x.lower() for x in a}
    B = {x.lower() for x in b}
    return len(A & B) / max(1, len(A | B))


def flags_of(jd: dict) -> tuple[list[str], list[str]]:
    text = jd["extras"] + " " + jd["title"]
    return ([k for k in OVERTIME_KW if k in text], [k for k in GREEN_KW if k in text])


def heuristic(resume: dict, jds: list, survey: dict) -> list[dict]:
    out = []
    for jd in jds:
        ot, green = flags_of(jd)
        skill_fit = jaccard(resume["skills"], jd["requirements"] + jd.get("nice_to_have", []))
        lo, hi = parse_salary(jd["salary_range"])
        salary_ok = 0.1 if lo >= survey["expected_salary_min"] else \
            (-0.2 if hi < survey["expected_salary_min"] else 0.0)
        ot_conflict = survey["overtime_tolerance"] <= 2 and ot
        industry_hit = any(k in jd["company"] or k in jd["title"] for k in survey["industry_preference"])
        fit = round(min(1.0, skill_fit * 1.2 + salary_ok + (0.05 if industry_hit else 0)), 3)
        attract = round(min(1.0, len(green) * 0.15 + (0.15 if industry_hit else 0)), 3)
        final = round(max(0.0, fit * 0.7 + attract * 0.3 - (0.25 if ot_conflict else 0)), 3)
        out.append({"jd": f'{jd["company"]}·{jd["title"]}', "fit": fit, "attract": attract,
                    "final": final, "overtime_flags": ot, "green_flags": green,
                    "ot_conflict": bool(ot_conflict)})
    out.sort(key=lambda x: -x["final"])
    return out


def judge_rescore(resume: dict, jds: list, survey: dict, base: list[dict],
                  providers: dict, provider_name: str) -> tuple[list[dict], float]:
    fit_task = Task(name="resume-fit", primitive="score",
                    criteria="评估这份简历与该岗位的综合匹配度（技能栈、行业、职级、薪资区间）。",
                    provider=provider_name, instruction="求职者侧匹配辅助，给出校准概率。")
    flag_task = Task(name="jd-redflag", primitive="verify",
                     criteria="判断该岗位的福利/强度描述是否与求职者的加班容忍度明显冲突。",
                     provider=provider_name, instruction="问卷加班容忍度 1-5，≤2 视为低容忍。")
    total_cost = 0.0
    out = []
    for b in base:
        jd = next(j for j in jds if f'{j["company"]}·{j["title"]}' == b["jd"])
        d1 = run_task(fit_task, {"简历": resume, "岗位": jd, "意愿问卷": survey}, providers)
        d2 = run_task(flag_task, {"岗位描述": jd["extras"], "问卷加班容忍度": survey["overtime_tolerance"]},
                      providers)
        total_cost += d1.cost + d2.cost
        jfit = d1.value if d1.ok else b["fit"]
        jconflict = d2.value if d2.ok else b["ot_conflict"]
        final = round(max(0.0, 0.4 * b["final"] + 0.6 * jfit - (0.15 if jconflict else 0)), 3)
        out.append({**b, "judge_fit": jfit, "judge_conflict": jconflict,
                    "confidence": d1.confidence, "final": final, "cost": round(d1.cost + d2.cost, 6)})
    out.sort(key=lambda x: -x["final"])
    return out, total_cost


def verdict(final: float) -> str:
    return "建议投递" if final >= 0.55 else ("可以一试" if final >= 0.35 else "不建议")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default=str(here / "sample_resume.json"))
    ap.add_argument("--jds", default=str(here / "sample_jds.json"))
    ap.add_argument("--survey", default=str(here / "sample_survey.json"))
    ap.add_argument("--model", default="", help="providers 名；留空=本地启发式")
    ap.add_argument("--providers-file", default=str(ROOT / "benchmarks" / "models.yaml"))
    args = ap.parse_args()

    resume = json.load(open(args.resume, encoding="utf-8"))
    jds = json.load(open(args.jds, encoding="utf-8"))
    survey = json.load(open(args.survey, encoding="utf-8"))
    ranked = heuristic(resume, jds, survey)
    cost = 0.0
    if args.model:
        providers = load_providers(args.providers_file)
        ranked, cost = judge_rescore(resume, jds, survey, ranked, providers, args.model)

    mode = f"判官({args.model})" if args.model else "本地启发式(0成本)"
    print(f"求职者：{resume['name']}（{resume['title']}，{resume['years']}年）  意愿："
          f"期望≥{survey['expected_salary_min']}元 加班容忍{survey['overtime_tolerance']}/5  模式：{mode}")
    print("-" * 72)
    for i, r in enumerate(ranked, 1):
        extra = f" judge_fit={r.get('judge_fit')} conflict={r.get('judge_conflict')}" if "judge_fit" in r else ""
        warn = " ⚠强度红旗:" + ",".join(r["overtime_flags"]) if r["overtime_flags"] else ""
        print(f"{i}. {r['jd']}  final={r['final']} fit={r['fit']} attract={r['attract']}{extra}"
              f"  → {verdict(r['final'])}{warn}")
    print(f"\n本次判官成本：¥{cost:.4f}（定位：求职者侧参考，非企业淘汰工具）")


if __name__ == "__main__":
    main()
