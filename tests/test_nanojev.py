# -*- coding: utf-8 -*-
"""nanojev 适配器离线测试：问题构造 + 响应解析（mock HTTP，不联网）。"""
import io
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from judgekit.engine import Task  # noqa: E402
from judgekit.providers.nanojev import NanoJev  # noqa: E402


def _ts():
    return NanoJev(name="nj", base_url="http://127.0.0.1:1")


def test_build_question_choice():
    t = Task.from_dict({"primitive": "classify", "criteria": "选部门",
                        "labels": {"物流": "快递"}})
    q = _ts()._question(t)
    assert q["type"] == "choice" and q["criteria"] == {"物流": "快递"}


def test_build_question_boolean_false_then_true():
    t = Task.from_dict({"primitive": "verify", "criteria": "紧急"})
    q = _ts()._question(t)
    assert q["type"] == "boolean" and q["criteria"][0].startswith("不成立") \
        and q["criteria"][1].startswith("成立")


def _mock(monkeypatch, answer):
    def fake_urlopen(req, timeout=None):
        class R:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps({"states": [{"id": "s1", "answers": {"d": answer}}]},
                                  ensure_ascii=False).encode("utf-8")
        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        return R()
    return fake_urlopen


def test_parse_choice(monkeypatch):
    t = Task.from_dict({"primitive": "classify", "labels": ["物流", "售后"]})
    monkeypatch.setattr("judgekit.providers.nanojev.urllib.request.urlopen",
                        _mock(None) if False else (lambda req, timeout=None: _resp(
                            {"choice": "物流", "probabilities": {"物流": 0.9, "售后": 0.1}})))
    d = _ts().decide(t, {"text": "查快递"})
    assert d.ok and d.value == "物流" and d.confidence == 0.9 and d.cost == 0.0


def test_parse_boolean(monkeypatch):
    t = Task.from_dict({"primitive": "verify", "criteria": "紧急"})
    monkeypatch.setattr("judgekit.providers.nanojev.urllib.request.urlopen",
                        lambda req, timeout=None: _resp({"p_true": 0.95}))
    d = _ts().decide(t, {"text": "x"})
    assert d.value is True and d.confidence == 0.95


def test_parse_score_normalizes(monkeypatch):
    t = Task.from_dict({"primitive": "score", "levels": ["低", "中", "高"]})
    monkeypatch.setattr("judgekit.providers.nanojev.urllib.request.urlopen",
                        lambda req, timeout=None: _resp({"score": 2.0,
                                                         "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}))
    d = _ts().decide(t, {"text": "x"})
    assert d.value == 1.0


def _resp(answer):
    class R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"states": [{"id": "s1", "answers": {"d": answer}}]},
                              ensure_ascii=False).encode("utf-8")
    return R()
