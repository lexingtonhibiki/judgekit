# -*- coding: utf-8 -*-
"""judge-econ v2 去泄题审计。

用法（仓库根目录）：python training/archive/audit_leakage.py

三块检查：
  A. 标签词泄漏：每类题面出现本类标签词/禁用词的比例（目标 <10%）。
  B. 标点-标签相关性：每类含 ！？ 的条目占比，两类占比差（目标 <20 个百分点）。
  C. v1→v2 逐条 diff：按 id 词干匹配，统计 保留/改写/新增，并列出新增 id。
附带完整性检查：JSON 可解析、标签集合与 v1 一致、各类条数一致、UTF-8 无 BOM、LF 行尾。
只读，不改任何文件。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks" / "data"
V2_DIR, V1_DIR = DATA / "econ_zh", DATA / "deprecated_econ_v1"
DATASETS = ["intent_zh", "sentiment_zh", "urgency_zh", "spam_zh"]

# A 项检查用的禁用标签词（用户审查结论：这些词直接泄底）。
BANNED: dict[str, dict[str, list[str]]] = {
    "intent_zh": {
        "投诉建议": ["投诉", "举报", "差评"],
        "退款售后": ["退款", "退货", "换货"],
        "技术故障": ["报错", "闪退", "bug"],
    },
    "urgency_zh": {}, "sentiment_zh": {}, "spam_zh": {},
}
EXCLAIM, QUEST = "！", "？"


def load(path: Path) -> list[dict]:
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), f"{path.name}: 有 BOM"
    assert b"\r" not in raw, f"{path.name}: 有 CRLF"
    rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    assert all(set(r) == {"id", "text", "label"} for r in rows), f"{path.name}: 字段不是 id/text/label"
    return rows


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for ds in DATASETS:
        v1, v2 = load(V1_DIR / f"{ds}.jsonl"), load(V2_DIR / f"{ds}.jsonl")
        print(f"\n{'=' * 62}\n[{ds}]  v1 n={len(v1)}  v2 n={len(v2)}")
        lab1 = {r["label"] for r in v1}
        lab2 = {r["label"] for r in v2}
        assert lab1 == lab2, f"标签集合变了: {lab1} -> {lab2}"
        cnt1, cnt2 = {}, {}
        for r in v1:
            cnt1[r["label"]] = cnt1.get(r["label"], 0) + 1
        for r in v2:
            cnt2[r["label"]] = cnt2.get(r["label"], 0) + 1
        status = "OK" if cnt1 == cnt2 else f"类目分布变了! v1={cnt1} v2={cnt2}"
        print(f"  类目分布: {cnt2}  [{status}]")
        labels = sorted(lab2)

        # A. 标签词泄漏（有禁用词表的类按表查；其余类查类名字面是否出现在题面）
        checks = dict(BANNED.get(ds, {}))
        others = [l for l in labels if l not in checks]
        if others:
            generic = {l: [l.replace("非", "")] for l in others}  # "非紧急"查"紧急"字面
            checks.update({k: v for k, v in generic.items() if v[0] not in checks.get(k, [])})
        for label, words in checks.items():
            texts = [r["text"] for r in v2 if r["label"] == label]
            hits = [(r["id"], w) for r in v2 if r["label"] == label for w in words if w in r["text"].lower()]
            rate = len({i for i, _ in hits}) / max(1, len(texts))
            flag = "PASS" if rate < 0.10 else "FAIL"
            print(f"  A 标签词泄漏 [{label}] 命中 {len({i for i, _ in hits})}/{len(texts)} = {rate:.0%} {flag}"
                  + (f"  明细: {hits[:5]}" if hits else ""))

        # B. 标点-标签相关性（对全部数据集报告，二分类的差值对照 20pp 阈值）
        rates = {}
        for label in labels:
            texts = [r["text"] for r in v2 if r["label"] == label]
            ex = sum(1 for t in texts if EXCLAIM in t)
            qm = sum(1 for t in texts if QUEST in t)
            rates[label] = (ex / len(texts), qm / len(texts))
            print(f"  B 标点占比 [{label}] 感叹号 {ex}/{len(texts)}={ex / len(texts):.0%}"
                  f"  问号 {qm}/{len(texts)}={qm / len(texts):.0%}")
        if len(labels) == 2:
            d_ex = abs(rates[labels[0]][0] - rates[labels[1]][0])
            d_qm = abs(rates[labels[0]][1] - rates[labels[1]][1])
            print(f"  B 两组差值: 感叹号 {d_ex:.0%} ({'PASS' if d_ex < 0.20 else 'FAIL'}), "
                  f"问号 {d_qm:.0%} ({'PASS' if d_qm < 0.20 else 'FAIL'})")

        # C. v1→v2 逐条 diff（id 去掉尾缀 r 后按词干配对）
        keep = rewr = 0
        t1 = {r["id"].rstrip("r"): r["text"] for r in v1}
        t2 = {r["id"].rstrip("r"): r["text"] for r in v2}
        for stem, text in t2.items():
            if stem in t1:
                if text == t1[stem]:
                    keep += 1
                else:
                    rewr += 1
        added = [r["id"] for r in v2 if r["id"].rstrip("r") not in t1]
        removed = [i for i in t1 if i not in t2]
        print(f"  C v1→v2: 保留原文 {keep} / 改写 {rewr} / 新增 {len(added)} / 删除 {len(removed)}"
              + (f"  新增: {added}" if added else "") + (f"  删除: {removed}" if removed else ""))
    print(f"\n{'=' * 62}\n审计完成（PASS 阈值：标签词泄漏 <10%，两组感叹号占比差 <20pp）")


if __name__ == "__main__":
    main()
