# -*- coding: utf-8 -*-
"""T8单测：二值flip映射写死（每任务双向各1例）+非二值抛错不停线记warn。全离线。"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import freeze_gold as fg  # noqa: E402


def test_spam_flip_both_ways():
    assert fg.flip("spam", "正常") == "垃圾"
    assert fg.flip("spam", "垃圾") == "正常"


def test_handoff_flip_both_ways():
    assert fg.flip("handoff", "不转") == "转人工"
    assert fg.flip("handoff", "转人工") == "不转"


def test_nonbinary_raises_and_pipeline_warns_not_stops():
    with pytest.raises(fg.NonBinaryFinalError):
        fg.flip("spam", "中性")
    with pytest.raises(fg.NonBinaryFinalError):
        fg.flip("route", "xxx")
    # 流水线：坏行记warn回落C终判，好行正常flip，不停线
    gold, warn = fg.apply_decision("spam", "中性", "改标")
    assert gold == "中性" and warn is not None
    gold2, warn2 = fg.apply_decision("spam", "正常", "改标")
    assert (gold2, warn2) == ("垃圾", None)


def test_apply_decision_pass_and_drop():
    assert fg.apply_decision("spam", "垃圾", "✓通过") == ("垃圾", None)
    assert fg.apply_decision("handoff", "不转", "✓通过") == ("不转", None)
    assert fg.apply_decision("spam", "垃圾", "✗删除") == (None, None)


def test_freeze_end_to_end_counts():
    scores = {
        "a1": {"id": "a1", "text": "t", "task": "spam", "source": "S",
               "orig_label": "正常", "A": {"value": "正常"}, "B": {"value": "正常"},
               "C": {"final": "正常", "ok": True}},
        "h1": {"id": "h1", "text": "t", "task": "handoff", "source": "S",
               "orig_label": "转人工", "A": {"value": "不转"}, "B": {"value": "不转"},
               "C": {"final": "不转", "ok": True}},
        "auto1": {"id": "auto1", "text": "t", "task": "spam", "source": "S",
                  "orig_label": "垃圾", "A": {"value": "垃圾"}, "B": {"value": "垃圾"},
                  "C": {"final": "垃圾", "ok": True}},
    }
    rulings = [("a1", "spam", "正常", "改标"), ("h1", "handoff", "不转", "改标")]
    frozen, warns, dist = fg.freeze(scores, rulings)
    by_id = {r["id"]: r for r in frozen}
    assert by_id["a1"]["gold"] == "垃圾" and by_id["h1"]["gold"] == "转人工"
    assert by_id["auto1"]["gold"] == "垃圾" and by_id["auto1"]["我的最终"] == "✓通过"
    assert len(frozen) == 3 and dist["改标"] == 2
