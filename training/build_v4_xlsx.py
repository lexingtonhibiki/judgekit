# -*- coding: utf-8 -*-
"""按省审设计出 数据审核_v4.xlsx（ABC三方打分审阅簿）。

输入：training/abc_score.py 产出的 scores jsonl（默认 training/abc_out/abc_scores_smoke.jsonl）
7 sheets：00指南 / 01总览 / 02路由 / 03转人工 / 04情感 / 05垃圾 / 06待审汇总
规则：
  我的最终无空值预填：(一致或分差≤2)且C置信≥0.75 → C action（通过→✓通过/需人工复核→⚠人工），否则待定
  红黄绿语义：红=待定（需你裁定）/黄=⚠人工/C低置信或失败/绿=✓通过
  排序：红上（待定>⚠人工>✓通过），组内分差降序（失败置顶）
  顶部摘要：每个明细sheet标题下3行摘要（总数/三色分布/平均分差/C重判数）
  理由压缩：A/B理由≤18字，C理由≤35字（模板化截断）
样式沿用 training/build_review_xlsx.py 的 xlsx skill（templates/base.py 同款token/函数）。

用法：
  python training/build_v4_xlsx.py [--in training/abc_out/abc_scores_smoke.jsonl] [--out training/abc_out/数据审核_v4.xlsx]
  python training/build_v4_xlsx.py --in training/abc_out/probe_scores.jsonl --out training/abc_out/数据审核_v4_probe.xlsx --gold-mismatch
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

XLSX_SKILL_DIR = r"C:\Users\14970\.zcode\cli\plugins\cache\zcode-plugins-official\spreadsheets\0.1.7\skills\xlsx"
for sub in [XLSX_SKILL_DIR, XLSX_SKILL_DIR + r"\templates"]:
    if sub not in sys.path:
        sys.path.insert(0, sub)
from base import (FONT_NAME, NEUTRAL_600, NEUTRAL_900, setup_sheet, style_header_row,  # noqa: E402
                  style_data_row, auto_fit_row_heights)
import base as _xlsx_base  # noqa: E402  writer层色值FF归一用（根因：6位hex被补00透明通道）
from openpyxl import Workbook  # noqa: E402
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side  # noqa: E402
from openpyxl.worksheet.datavalidation import DataValidation  # noqa: E402


def _ff6(h: str) -> str:
    h = str(h or "")
    return ("FF" + h) if len(h) == 6 else h


# writer层所有色值alpha归一FF：base模板token在建表前即打FF补丁，避免openpyxl补00透明通道
for _k in ("PRIMARY", "PRIMARY_LIGHT", "SECONDARY", "ACCENT_POSITIVE",
           "ACCENT_NEGATIVE", "ACCENT_WARNING", "NEUTRAL_900", "NEUTRAL_600",
           "NEUTRAL_200", "NEUTRAL_100", "NEUTRAL_50", "NEUTRAL_0", "HEADER_TEXT"):
    try:
        _v = getattr(_xlsx_base, _k, "")
        if isinstance(_v, str) and len(_v) == 6:
            setattr(_xlsx_base, _k, _ff6(_v))
    except Exception:  # noqa: BLE001
        pass
try:
    _xlsx_base.CHART_COLORS = [_ff6(c) for c in getattr(_xlsx_base, "CHART_COLORS", [])]
except Exception:  # noqa: BLE001
    pass
NEUTRAL_600, NEUTRAL_900 = _ff6(NEUTRAL_600), _ff6(NEUTRAL_900)

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TASK_CN = {"route": "路由", "handoff": "转人工", "sentiment": "情感", "spam": "垃圾"}
TASK_SHEET = {"route": "02路由", "handoff": "03转人工", "sentiment": "04情感", "spam": "05垃圾"}

RED_FILL = PatternFill("solid", fgColor="FFFDEDEC", bgColor="FFFDEDEC")
RED_FONT = Font(name=FONT_NAME, size=11, color="FFC0392B", bold=True)
YEL_FILL = PatternFill("solid", fgColor="FFFEF9E7", bgColor="FFFEF9E7")
YEL_FONT = Font(name=FONT_NAME, size=11, color="FFD4820A", bold=True)
GRN_FILL = PatternFill("solid", fgColor="FFE8F5E9", bgColor="FFE8F5E9")
GRN_FONT = Font(name=FONT_NAME, size=11, color="FF1B7D46", bold=True)
VERDICT_STYLE = {"待定": (RED_FILL, RED_FONT), "⚠人工": (YEL_FILL, YEL_FONT),
                 "✓通过": (GRN_FILL, GRN_FONT),
                 "✗删除": (RED_FILL, RED_FONT), "改标": (YEL_FILL, YEL_FONT)}


def _fix_argb(rgb: str | None) -> str | None:
    """6位hex被openpyxl补00透明通道的根因修复：alpha归一FF。

    仅处理8位00头非全零色（00XXXXXX→FFXXXXXX）；00000000默认底留原样由调用方按需对齐。
    """
    if not isinstance(rgb, str) or len(rgb) != 8 or not rgb.startswith("00"):
        return rgb
    if rgb == "00000000":
        return rgb
    return "FF" + rgb[2:]


def fix_workbook_colors(wb) -> int:
    """writer层所有色值alpha归一FF（含base.py模板色）。返回修复计数。

    安全实现：整体替换fill/font/border对象，不原地改共享Color（原地改会污染
    默认00000000共享对象，导致数据区底色被染成表头蓝）。默认00000000底保留，
    Excel对此兼容；校验时仅要求调色板色无00头。
    """
    n = 0
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for c in row:
                try:
                    fg = c.fill.fgColor.rgb if c.fill.fgColor else None
                    bg = c.fill.bgColor.rgb if c.fill.bgColor else None
                except Exception:  # noqa: BLE001
                    continue
                new_fg = _fix_argb(fg) if isinstance(fg, str) else fg
                new_bg = _fix_argb(bg) if isinstance(bg, str) else bg
                if new_bg == "00000000":
                    new_bg = bg  # 默认底保留，不碰共享对象
                if (isinstance(new_fg, str) and new_fg != fg) or \
                   (isinstance(new_bg, str) and new_bg != bg):
                    try:
                        c.fill = PatternFill(patternType=c.fill.patternType or "solid",
                                             fgColor=new_fg or "00000000",
                                             bgColor=new_bg or "00000000")
                        n += 1
                    except Exception:  # noqa: BLE001
                        pass
                # font（整体替换，避免共享Color污染）
                try:
                    frgb = c.font.color.rgb if c.font.color else None
                except Exception:  # noqa: BLE001
                    frgb = None
                new_f = _fix_argb(frgb) if isinstance(frgb, str) else frgb
                if isinstance(new_f, str) and new_f != frgb:
                    try:
                        c.font = Font(name=c.font.name, size=c.font.size,
                                      bold=c.font.bold, italic=c.font.italic,
                                      color=new_f)
                        n += 1
                    except Exception:  # noqa: BLE001
                        pass
                # border（重建Border，避免改共享Side）
                try:
                    sides = []
                    dirty = False
                    for side in (c.border.left, c.border.right,
                                 c.border.top, c.border.bottom):
                        srgb = side.color.rgb if side.color and isinstance(
                            side.color.rgb, str) else None
                        nsrgb = _fix_argb(srgb) if isinstance(srgb, str) else srgb
                        if isinstance(nsrgb, str) and nsrgb != srgb and srgb != "00000000":
                            dirty = True
                            sides.append(Side(style=side.style, color=nsrgb))
                        else:
                            sides.append(side)
                    if dirty:
                        c.border = Border(left=sides[0], right=sides[1],
                                          top=sides[2], bottom=sides[3])
                        n += 1
                except Exception:  # noqa: BLE001
                    pass
    return n
RANK = {"待定": 0, "⚠人工": 1, "✓通过": 2, "✗删除": 0, "改标": 1}

HEADERS = ["id", "文本", "原标签", "来源", "A判定", "A理由≤18", "B判定", "B理由≤18",
           "C终判", "C理由≤35", "分差", "C置信", "我的最终", "端点"]
WIDTHS = [16, 60, 14, 12, 10, 20, 10, 20, 12, 36, 7, 8, 10, 30]
FINAL_COL = len(HEADERS)  # B列起算：我的最终idx12 -> column 2+12=14（len=14）


def verdict_of(r: dict, gold_check: bool = False) -> str:
    """无空值预填：(一致或分差≤2)且C置信≥0.75→C action，否则待定。

    gold_check（T4-R1探针开关，默认关）：C终判与原标签不一致→强制待定，
    保证gold-mismatch行进06汇总；其他预填逻辑不变。
    """
    if gold_check and gold_mismatch(r):
        return "待定"
    c = r.get("C", {})
    agree = (r.get("delta", -1) in (0,)) or (isinstance(r.get("delta"), (int, float))
                                             and 0 <= r["delta"] <= 2)
    if agree and (c.get("confidence") or 0) >= 0.75 and c.get("ok"):
        return "⚠人工" if c.get("action") == "需人工复核" else "✓通过"
    return "待定"


# ---- gold对照（T4-R1）：C终判 vs 原标签。探针orig有多写法名（刷评spam/刷单spam），先归一到C标签空间 ----
GOLD_ALIAS = {"刷评spam": "垃圾", "刷单spam": "垃圾", "刷评": "垃圾", "刷单": "垃圾"}


def gold_final_of(r: dict) -> str:
    """C终判（情感任务取band）；ok=False或无终判返回空串。"""
    c = r.get("C", {})
    if not c.get("ok"):
        return ""
    if r.get("task") == "sentiment":
        return str(c.get("band") or "")
    v = c.get("final")
    return "" if v is None else str(v)


def gold_mismatch(r: dict) -> bool:
    """C终判与原标签不一致（归一后比对；无orig/C失败/情感非band标签→不算）。"""
    orig = str(r.get("orig_label") or "")
    if not orig:
        return False
    fin = gold_final_of(r)
    if not fin:
        return False
    if r.get("task") == "sentiment" and orig not in ("正面", "中性", "负面"):
        return False
    return GOLD_ALIAS.get(orig, orig) != fin


def gold_tag(r: dict) -> str:
    return f"与预标签不符（预{r.get('orig_label')}）" if gold_mismatch(r) else ""


def select_pending(by_task: dict) -> list[tuple]:
    """06汇总行=全部🔴待定+🟡人工（含gold-mismatch升级的待定）。"""
    return [(r, v) for t in ["route", "handoff", "sentiment", "spam"]
            for (r, v) in by_task.get(t, []) if v in ("待定", "⚠人工")]


def sort_key(item: tuple) -> tuple:
    r, v = item
    d = r.get("delta", -1)
    d = 999 if d == -1 else d  # 失败置顶
    return (RANK[v], -d)


def disp_val(r: dict, side: str):
    d = r.get(side, {})
    v = d.get("value" if side in ("A", "B") else "final", "")
    if r["task"] == "sentiment" and isinstance(v, (int, float)):
        if side == "C" and r["C"].get("band"):
            return f"{v}（{r['C']['band']}）"
        return v
    return "" if v is None else str(v)


def row_of(r: dict, v: str) -> list:
    ep = r.get("endpoint", {})
    return [r["id"], r.get("text", ""), r.get("orig_label", ""), r.get("source", ""),
            disp_val(r, "A"), str(r.get("A", {}).get("reason", ""))[:18],
            disp_val(r, "B"), str(r.get("B", {}).get("reason", ""))[:18],
            disp_val(r, "C"), str(r.get("C", {}).get("reason", ""))[:35],
            ("失败" if r.get("delta") == -1 else r.get("delta", "")),
            r.get("C", {}).get("confidence", ""), v,
            f"A:{ep.get('A','')} B:{ep.get('B','')} C:{ep.get('C','')}"]


def summary_lines(name: str, rows: list[tuple], gold_on: bool = False) -> list[str]:
    n = len(rows)
    greens = sum(1 for _, v in rows if v == "✓通过")
    yels = sum(1 for _, v in rows if v == "⚠人工")
    reds = sum(1 for _, v in rows if v == "待定")
    ds = [r.get("delta", -1) for r, _ in rows if isinstance(r.get("delta"), (int, float)) and r["delta"] >= 0]
    md = round(sum(ds) / len(ds), 2) if ds else "-"
    rej = sum(1 for r, _ in rows if r.get("C", {}).get("called") == "rejudge")
    skip = sum(1 for r, _ in rows if r.get("C", {}).get("called") == "skipped")
    fb = sum(1 for r, _ in rows if r.get("B", {}).get("fallback_from"))
    return [f"摘要（{name}，共{n}条）：🟢✓通过{greens} ｜ 🟡⚠人工{yels} ｜ 🔴待定{reds}（红在上，已按分差降序）",
            f"平均分差 {md}；C重判 {rej} 次 / 均值 {sum(1 for r, _ in rows if r.get('C', {}).get('called') == 'mean')} 次"
            + (f" / responses桩跳过 {skip} 次" if skip else "")
            + (f"；B路fallback到typesafe {fb} 条" if fb else ""),
            "预填规则：(一致或分差≤2)且C置信≥0.75→C action，否则待定；「我的最终」无空值，下拉可改（✓通过/⚠人工/✗删除/改标/待定）。"
            + (" gold对照开：C终判≠原标签→待定并进06（标与预标签不符）。" if gold_on else "")]


def fill_detail(ws, title: str, name: str, rows: list[tuple],
                gold_on: bool = False) -> None:
    setup_sheet(ws, title=title, last_col=len(HEADERS) + 1)
    for i, line in enumerate(summary_lines(name, rows, gold_on)):
        c = ws.cell(row=4 + i, column=2, value=line)
        c.font = Font(name=FONT_NAME, size=10, color=NEUTRAL_900)
        c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
    ws.merge_cells(start_row=4, start_column=2, end_row=4, end_column=len(HEADERS) + 1)
    ws.merge_cells(start_row=5, start_column=2, end_row=5, end_column=len(HEADERS) + 1)
    ws.merge_cells(start_row=6, start_column=2, end_row=6, end_column=len(HEADERS) + 1)
    hr = 7
    for ci, h in enumerate(HEADERS, 2):
        ws.cell(row=hr, column=ci, value=h)
    style_header_row(ws, row_num=hr, col_start=2, col_end=len(HEADERS) + 1)
    for i, (r, v) in enumerate(rows):
        rn = hr + 1 + i
        for ci, val in enumerate(row_of(r, v), 2):
            ws.cell(row=rn, column=ci, value=val)
        style_data_row(ws, row_num=rn, col_start=2, col_end=len(HEADERS) + 1, row_index=i)
        fill, font = VERDICT_STYLE[v]
        cell = ws.cell(row=rn, column=FINAL_COL)
        cell.fill, cell.font = fill, font
    for ci, w in enumerate(WIDTHS, 2):
        ws.column_dimensions[ws.cell(row=hr, column=ci).column_letter].width = w
    auto_fit_row_heights(ws, header_row=hr, data_start_row=hr + 1)
    ws.freeze_panes = ws.cell(row=hr + 1, column=4).coordinate
    dv = DataValidation(type="list", formula1='"✓通过,⚠人工,✗删除,改标,待定"', allow_blank=False)
    ws.add_data_validation(dv)
    letter = ws.cell(row=hr, column=FINAL_COL).column_letter
    if rows:
        dv.add(f"{letter}{hr + 1}:{letter}{hr + len(rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default="training/abc_out/abc_scores_smoke.jsonl")
    ap.add_argument("--out", default="training/abc_out/数据审核_v4.xlsx")
    ap.add_argument("--gold-mismatch", action="store_true",
                    help="探针gold对照：C终判与原标签不一致→待定并强制进06（标与预标签不符）；"
                         "默认关（T3 full行为不变）")
    args = ap.parse_args()

    inp = ROOT / args.inp
    if not inp.exists():  # 回退读cache
        alt = inp.parent / "abc_cache.jsonl"
        inp = alt if alt.exists() else inp
    recs = [json.loads(l) for l in open(inp, encoding="utf-8") if l.strip()]
    assert recs, f"空输入：{inp}"
    by_task: dict[str, list[tuple]] = {t: [] for t in TASK_CN}
    for r in recs:
        by_task[r["task"]].append((r, verdict_of(r, args.gold_mismatch)))
    for t in by_task:
        by_task[t].sort(key=sort_key)
    n_gold = (sum(1 for t in by_task for r, _ in by_task[t] if gold_mismatch(r))
              if args.gold_mismatch else 0)

    wb = Workbook()
    # ---- 00指南 ----
    ws = wb.active
    ws.title = "00指南"
    setup_sheet(ws, title="数据审核 v4 —— ABC三方打分审阅簿（怎么用）", last_col=8)
    prov0 = recs[0].get("provenance", {})
    guide = [
        "三方：A/B temp0.7独立打分 → C temp0.2仲裁（分差≤2取均值、>2重判；分类任务：一致取该值、不一致重判）。",
        "四任务：路由（意图唯一；smp2019_ecdt+crosswoz）/ 转人工（辱骂威胁重复催≥2次/情绪崩溃才转，投诉但冷静不转；cped+ewect）/ "
        "情感（0-10权重dsh锚点；waimai+weibo）/ 垃圾（营销引流刷屏才判，抱怨差评驳回；dmr+fbs）。",
        "红黄绿：🔴待定=需你裁定（排最上）/ 🟡⚠人工=C判需复核或低置信 / 🟢✓通过=C高置信通过；组内按分差降序，失败置顶。",
        f"本次实际提供方：A={prov0.get('a_provider')} B={prov0.get('b_effective')} C={prov0.get('c_provider')}"
        + (f"；{prov0.get('fallback_note')}" if prov0.get("fallback_note") else "")
        + (f"；responses桩={prov0.get('c_responses_stub')}" if prov0.get("c_responses_stub") else "；responses桩未启用（直调C）"),
        f"网关现实：{prov0.get('gateway_note', '')}",
        "理由压缩：A/B理由≤18字、C理由≤35字模板化截断；spark证据摘录=原文前60字（无SPARK key，见provenance.spark_excerpt）。",
        "你只改不同意的「我的最终」（下拉：✓通过/⚠人工/✗删除/改标/待定），改完保存即生效；06待审汇总=全部🔴+🟡行。",
    ]
    if args.gold_mismatch:
        guide.append(f"gold对照开：C终判与原标签不一致→待定并强制进06（标'与预标签不符'，本轮{n_gold}行）；其他预填逻辑不变。")
    for i, line in enumerate(guide):
        c = ws.cell(row=4 + i, column=2, value=("• " + line if not line.startswith("  ") else line))
        c.font = Font(name=FONT_NAME, size=10, color=NEUTRAL_900)
        c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        ws.row_dimensions[4 + i].height = 30
    ws.column_dimensions["B"].width = 150

    # ---- 01总览 ----
    ws = wb.create_sheet("01总览")
    setup_sheet(ws, title="总览——ABC冒烟结果（黑盒证据）", last_col=8)
    oh = ["任务", "条数", "AB成功", "AB成功率", "C重判", "✓通过", "⚠人工", "待定"]
    for ci, h in enumerate(oh, 2):
        ws.cell(row=4, column=ci, value=h)
    style_header_row(ws, row_num=4, col_start=2, col_end=9)
    total = [0, 0, 0, 0, 0, 0]
    for i, t in enumerate(["route", "handoff", "sentiment", "spam"]):
        rows = by_task[t]
        n = len(rows)
        ok = sum(1 for r, _ in rows if r["A"].get("ok") and r["B"].get("ok"))
        rej = sum(1 for r, _ in rows if r["C"].get("called") == "rejudge")
        g = sum(1 for _, v in rows if v == "✓通过")
        y = sum(1 for _, v in rows if v == "⚠人工")
        rd = sum(1 for _, v in rows if v == "待定")
        for ci, val in enumerate([TASK_CN[t], n, ok, (f"{100 * ok / n:.0f}%" if n else "-"),
                                  rej, g, y, rd], 2):
            ws.cell(row=5 + i, column=ci, value=val)
        style_data_row(ws, row_num=5 + i, col_start=2, col_end=9, row_index=i)
        for k, x in enumerate([n, ok, rej, g, y, rd]):
            total[k] += x
    rn = 9
    for ci, val in enumerate(["合计", total[0], total[1],
                              (f"{100 * total[1] / total[0]:.0f}%" if total[0] else "-"),
                              total[2], total[3], total[4], total[5]], 2):
        cell = ws.cell(row=rn, column=ci, value=val)
        cell.font = Font(name=FONT_NAME, size=11, color=NEUTRAL_900, bold=True)
    for ci, w in enumerate([10, 8, 10, 10, 8, 8, 8, 8], 2):
        ws.column_dimensions[ws.cell(row=4, column=ci).column_letter].width = w
    ws.cell(row=rn + 2, column=2,
            value=f"黑盒证据：cache行数={len(recs)}；sheets数=7；我的最终无空值数={sum(len(v) for v in by_task.values())}（预填率100%）。").font = Font(
        name=FONT_NAME, size=10, color=NEUTRAL_600)

    # ---- 02-05明细 ----
    for t in ["route", "handoff", "sentiment", "spam"]:
        rows = by_task[t]
        ws = wb.create_sheet(f"{TASK_SHEET[t]}_{len(rows)}")
        fill_detail(ws, f"{TASK_CN[t]}（{len(rows)}条）——{ {'route':'意图唯一','handoff':'辱骂威胁重复催≥2次/情绪崩溃才转','sentiment':'0-10权重dsh锚点','spam':'营销引流刷屏才判'}[t] }",
                    TASK_CN[t], rows, args.gold_mismatch)

    # ---- 06待审汇总 ----
    pending = select_pending(by_task)
    gold_on = args.gold_mismatch
    n_gold_in = sum(1 for r, _ in pending if gold_tag(r)) if gold_on else 0
    ws = wb.create_sheet(f"06待审汇总_{len(pending)}")
    setup_sheet(ws, title=f"待审汇总（{len(pending)}条=🔴待定+🟡人工"
                + (f"，含{n_gold_in}条与预标签不符" if gold_on else "")
                + "，先看这里）", last_col=(10 if gold_on else 9))
    ph = ["任务", "id", "文本", "A判定", "B判定", "C终判", "分差", "我的最终", "端点"] \
        + (["对照标记"] if gold_on else [])
    for ci, h in enumerate(ph, 2):
        ws.cell(row=4, column=ci, value=h)
    style_header_row(ws, row_num=4, col_start=2, col_end=(11 if gold_on else 10))
    for i, (r, v) in enumerate(pending):
        ep = r.get("endpoint", {})
        vals = [TASK_CN[r["task"]], r["id"], r.get("text", ""), disp_val(r, "A"),
                disp_val(r, "B"), disp_val(r, "C"),
                ("失败" if r.get("delta") == -1 else r.get("delta", "")), v,
                f"A:{ep.get('A','')} B:{ep.get('B','')}"] \
            + ([gold_tag(r)] if gold_on else [])
        rn = 5 + i
        for ci, val in enumerate(vals, 2):
            ws.cell(row=rn, column=ci, value=val)
        style_data_row(ws, row_num=rn, col_start=2, col_end=(11 if gold_on else 10),
                       row_index=i)
        fill, font = VERDICT_STYLE[v]
        cell = ws.cell(row=rn, column=9)
        cell.fill, cell.font = fill, font
    for ci, w in enumerate([8, 16, 60, 10, 10, 12, 7, 10, 30]
                           + ([22] if gold_on else []), 2):
        ws.column_dimensions[ws.cell(row=4, column=ci).column_letter].width = w
    auto_fit_row_heights(ws, header_row=4, data_start_row=5)
    ws.freeze_panes = "D5"
    dv = DataValidation(type="list", formula1='"✓通过,⚠人工,✗删除,改标,待定"', allow_blank=False)
    ws.add_data_validation(dv)
    if pending:
        dv.add(f"I5:I{4 + len(pending)}")

    wb.properties.creator = "jev-judge ABC"
    n_fix = fix_workbook_colors(wb)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print("saved:", out)
    print(f"evidence: cache_rows={len(recs)} sheets={len(wb.sheetnames)} "
          f"final_nonempty={sum(len(v) for v in by_task.values())} "
          f"pass={sum(1 for t in by_task for _, v in by_task[t] if v == '✓通过')}"
          + (f" gold_mismatch={n_gold}" if args.gold_mismatch else "")
          + f" color_fix={n_fix}")
    for s in wb.sheetnames:
        print(" -", s)


if __name__ == "__main__":
    main()
