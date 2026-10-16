# -*- coding: utf-8 -*-
"""openai_compat 离线测试：提示词构造 + 响应解析（不联网）。"""
import io
import json
import sys
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from judgekit.engine import Task, run_task  # noqa: E402
from judgekit.providers.openai_compat import OpenAICompat, build_prompt, parse_decision  # noqa: E402


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


@pytest.mark.parametrize("primitive", ["classify", "route"])
@pytest.mark.parametrize("payload", [
    {}, {"label": ""}, {"label": " \t\n"}, {"label": None},
    {"label": True}, {"label": False}, {"label": 0}, {"label": 1.5},
    {"label": []}, {"label": ["refund"]}, {"label": {"choice": "refund"}},
])
def test_parse_decision_rejects_missing_empty_and_non_string_labels(primitive, payload):
    t = Task(name="t", primitive=primitive,
             labels=["sale", "refund", "None", "True", "False", "0", "1.5"])
    with pytest.raises(ValueError, match="label-invalid"):
        parse_decision(t, json.dumps(payload), "n", 5, 0.0)


@pytest.mark.parametrize("primitive", ["classify", "route"])
@pytest.mark.parametrize("labels, label", [
    (["sale", "refund"], "sale or refund"),
    (["refund-approved", "refund-denied"], "refund"),
])
@pytest.mark.parametrize("reverse", [False, True])
def test_parse_decision_rejects_ambiguous_labels(primitive, labels, label, reverse):
    t = Task(name="t", primitive=primitive, labels=labels[::-1] if reverse else labels)
    with pytest.raises(ValueError, match="label-ambiguous"):
        parse_decision(t, json.dumps({"label": label}), "n", 5, 0.0)


@pytest.mark.parametrize("primitive", ["classify", "route"])
@pytest.mark.parametrize("labels, label, expected", [
    (["售后", "物流"], " \t售后\n", "售后"),
    (["sale", "wholesale"], "wholesale", "wholesale"),
    (["sale", "wholesale"], "sale", "sale"),
    (["sale", "sale", "refund"], "sale department", "sale"),
    (["售后", "物流"], "售后服务", "售后"),
    (["退款售后", "物流查询"], "退款", "退款售后"),
])
def test_parse_decision_preserves_unambiguous_matches(primitive, labels, label, expected):
    t = Task(name="t", primitive=primitive, labels=labels)
    raw = json.dumps({"label": label, "confidence": 0.9})
    d = parse_decision(t, raw, "n", 5, 0.01)
    assert d.ok and d.value == expected and d.primitive == primitive
    assert d.confidence == 0.9 and d.raw == raw and d.provider == "n"
    assert d.latency_ms == 5 and d.cost == 0.01


def _mock_openai_response(monkeypatch, payload):
    def respond(req, timeout=None):
        return io.BytesIO(json.dumps({"choices": [{"message": {
            "content": json.dumps(payload),
        }}]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", respond)
    return OpenAICompat("offline", "https://offline.invalid", "stub")


@pytest.mark.parametrize("primitive", ["classify", "route"])
@pytest.mark.parametrize("payload, error", [
    ({}, "label-invalid"),
    ({"label": ""}, "label-invalid"),
    ({"label": " \t\n"}, "label-invalid"),
    ({"label": ["sale"]}, "label-invalid"),
    ({"label": "sale or refund"}, "label-ambiguous"),
    ({"label": "unknown"}, "label-not-in-candidates"),
])
def test_run_task_invalid_label_uses_rules_fallback(monkeypatch, primitive, payload, error):
    p = _mock_openai_response(monkeypatch, payload)
    t = Task(name="t", primitive=primitive, labels=["sale", "refund"],
             provider="offline", fallback_rules={"refund": ["refund"]})
    d = run_task(t, {"text": "please refund"}, {"offline": p})
    assert d.ok and d.value == "refund" and d.provider == "rules-after-fail"
    assert f"provider-failed(ValueError: {error}" in d.error
    assert d.error.endswith("; fallback-ok")


@pytest.mark.parametrize("primitive", ["classify", "route"])
@pytest.mark.parametrize("fallback, text, rules, error_suffix", [
    (False, "refund", {"refund": ["refund"]}, ""),
    (True, "unmatched", {"refund": ["refund"]}, "; fallback:rules-no-hit"),
    (True, "refund", {}, ""),
])
def test_run_task_invalid_label_without_usable_fallback(
        monkeypatch, primitive, fallback, text, rules, error_suffix):
    p = _mock_openai_response(monkeypatch, {})
    t = Task(name="t", primitive=primitive, labels=["sale", "refund"],
             provider="offline", fallback_rules=rules)
    d = run_task(t, {"text": text}, {"offline": p}, fallback=fallback)
    assert not d.ok and d.value is None and d.provider == "offline"
    assert d.error == "ValueError: label-invalid: None" + error_suffix


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
