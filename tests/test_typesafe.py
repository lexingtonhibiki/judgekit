# -*- coding: utf-8 -*-
"""typesafe 适配器离线测试：问题构造 + 响应解析（不联网）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from judgekit.engine import Task  # noqa: E402
from judgekit.providers.typesafe import TypeSafe  # noqa: E402


def _ts():
    return TypeSafe(name="ts", api_key="k")


def test_build_question_choice_map():
    t = Task.from_dict({"primitive": "route", "criteria": "选部门",
                        "labels": {"物流": "快递", "售后": "退货"}})
    q = _ts().build_question(t)
    assert q["type"] == "choice"
    assert q["criteria"] == {"物流": "快递", "售后": "退货"}
    assert "选部门" in q["instructions"]


def test_build_question_score_levels_default():
    t = Task.from_dict({"primitive": "score", "criteria": "满意度"})
    q = _ts().build_question(t)
    assert q["type"] == "score" and len(q["criteria"]) == 3


def test_build_question_noul():
    t = Task.from_dict({"primitive": "verify", "criteria": "是否紧急"})
    q = _ts().build_question(t)
    assert q["type"] == "noul" and "紧急" in q["instructions"]


def test_parse_choice_with_probabilities():
    t = Task.from_dict({"primitive": "classify", "labels": ["物流", "售后"]})
    a = {"type": "choice", "choice": "物流", "confidence": 0.99,
         "probabilities": {"物流": 1.0, "售后": 0.0}}
    d = _ts()._parse(t, a, "{}", 10, 0.0)
    assert d.ok and d.value == "物流" and d.probabilities == {"物流": 1.0, "售后": 0.0}


def test_parse_score_normalizes_by_legend_span():
    t = Task.from_dict({"primitive": "score", "levels": ["低", "中", "高"]})
    a = {"type": "score", "score": 1.97, "confidence": 0.96,
         "legend": {"0": "低", "1": "中", "2": "高"}}
    d = _ts()._parse(t, a, "{}", 10, 0.0)
    assert abs(d.value - 0.985) < 1e-6   # 1.97 / (3-1)


def test_parse_score_without_legend_uses_task_levels():
    t = Task.from_dict({"primitive": "score", "levels": ["a", "b", "c", "d"]})
    a = {"type": "score", "score": 2.0, "confidence": 0.9}
    d = _ts()._parse(t, a, "{}", 10, 0.0)
    assert abs(d.value - 2.0 / 3) < 2e-3


def test_parse_noul_verify():
    t = Task.from_dict({"primitive": "verify", "criteria": "紧急"})
    d = _ts()._parse(t, {"type": "noul", "noul": 0.95}, "{}", 10, 0.0)
    assert d.value is True and d.confidence == 0.95
    d2 = _ts()._parse(t, {"type": "noul", "noul": 0.3}, "{}", 10, 0.0)
    assert d2.value is False and d2.confidence == 0.7


def test_parse_choice_unknown_label_fails():
    t = Task.from_dict({"primitive": "classify", "labels": ["a"]})
    d = _ts()._parse(t, {"type": "choice", "choice": "zzz", "confidence": 0.9}, "{}", 10, 0.0)
    assert not d.ok and "label-not-in-candidates" in d.error


def test_parse_wrong_type_raises():
    t = Task.from_dict({"primitive": "score", "levels": ["a", "b"]})
    try:
        _ts()._parse(t, {"type": "noul", "noul": 0.5}, "{}", 10, 0.0)
        assert False
    except ValueError:
        pass
