# -*- coding: utf-8 -*-
"""T13离线测试：Wilson边界+富描述非空(≤40字/T7口径)+兜底不充数。

全部离线（mock provider，不联网不读key）。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from judgekit.engine import Decision  # noqa: E402
from training import eval_jev_v4 as m  # noqa: E402


class _Fail:
    def decide(self, task, x):
        raise RuntimeError("boom")


class _Fixed:
    def __init__(self, value):
        self.value = value

    def decide(self, task, x):
        return Decision(task.primitive, self.value, 0.9, "mock",
                        "typesafe", 5, 0.0)


def _row(task="spam", gold="垃圾", text="加微信链接返现", source="JD刷单"):
    return {"id": "t_1", "text": text, "task": task,
            "gold": gold, "source": source}


def test_wilson_boundaries():
    assert m.wilson(0, 0) == (0.0, 0.0)
    lo, hi = m.wilson(0, 60)
    assert lo == 0.0 and 0.0 < hi < 0.1
    lo, hi = m.wilson(60, 60)
    assert hi > 0.999 and 0.9 < lo < 1.0
    lo, hi = m.wilson(45, 60)
    assert 0.0 <= lo < 0.75 < hi <= 1.0


def test_label_desc_short_nonempty_t7():
    tasks = m.build_tasks()
    spam, handoff = tasks["spam"], tasks["handoff"]
    assert spam.labels == ["垃圾", "正常"]
    assert handoff.labels == ["转人工", "不转"]
    for t in (spam, handoff):
        for lb in t.labels:
            d = t.label_descriptions.get(lb, "")
            assert d and len(d) <= 40, (lb, d)
    s = "".join(spam.label_descriptions.values())
    assert "刷单" in s and ("零增量" in s or "单特征" in s)
    assert "举一反三" in (spam.criteria + spam.instruction)
    h = "".join(handoff.label_descriptions.values())
    assert "重复催" in h and "冷静" in h


def test_fallback_hit_not_counted():
    """provider失败→规则兜底命中（pred本会等于gold）仍记ok=False不充正确。"""
    tasks = m.build_tasks()
    rec = m.judge_row(_row(), tasks, {"typesafe": _Fail()},
                      tries=2, backoff=0)
    assert rec["ok"] is False
    assert rec["pred"] is None
    assert rec["correct"] is False


def test_fallback_miss_not_counted():
    rec = m.judge_row(_row(text="今天天气不错，无关键词"),
                      m.build_tasks(), {"typesafe": _Fail()},
                      tries=2, backoff=0)
    assert rec["ok"] is False and rec["pred"] is None
    assert rec["correct"] is False


def test_success_counted():
    rec = m.judge_row(_row(), m.build_tasks(),
                      {"typesafe": _Fixed("垃圾")}, tries=1, backoff=0)
    assert rec["ok"] is True and rec["pred"] == "垃圾"
    assert rec["correct"] is True and rec["conf"] == 0.9


def test_report_line_marks_small_sample():
    recs = [{"source": "JD刷单", "correct": True, "latency_ms": 10, "cost": 0.0},
            {"source": "CSDS", "correct": False, "latency_ms": 20, "cost": 0.0}]
    line = m.report_line(recs)
    assert "小样本" in line and "95%CI" in line and "¥/千次" in line
