# -*- coding: utf-8 -*-
"""openai_compat 离线测试：提示词构造 + 响应解析（不联网）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from judgekit.engine import Task  # noqa: E402
from judgekit.providers.openai_compat import build_prompt, parse_decision  # noqa: E402


def test_build_prompt_classify_includes_descriptions():
    t = Task.from_dict({"primitive": "classify", "labels": {"售后": "退货退款"}})
    _, user = build_prompt(t, {"text": "我要退货"})
    assert "售后：退货退款" in user and "我要退货" in user


def test_build_prompt_score_mentions_levels():
    t = Task.from_dict({"primitive": "score", "criteria": "满意度", "levels": ["低", "高"]})
    _, user = build_prompt(t, {"text": "x"})
    assert "0=低" in user and "1=高" in user


def test_parse_decision_classify():
    t = Task.from_dict({"primitive": "classify", "labels": ["售后", "物流"]})
    d = parse_decision(t, '{"label": "售后", "confidence": 0.9}', "n", 5, 0.0)
    assert d.value == "售后" and d.confidence == 0.9


def test_parse_decision_json_in_prose():
    t = Task.from_dict({"primitive": "verify", "criteria": "x"})
    d = parse_decision(t, '好的，结果是 {"verdict": true, "confidence": 0.8} 谢谢', "n", 5, 0.0)
    assert d.value is True


def test_parse_decision_bad_raises():
    t = Task.from_dict({"primitive": "classify", "labels": ["a"]})
    try:
        parse_decision(t, "模型拒绝输出 JSON", "n", 5, 0.0)
        assert False
    except ValueError:
        pass


def test_parse_decision_score_clamps():
    t = Task.from_dict({"primitive": "score"})
    d = parse_decision(t, '{"score": 1.7, "confidence": 0.5}', "n", 5, 0.0)
    assert d.value == 1.0
