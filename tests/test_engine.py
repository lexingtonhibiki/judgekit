# -*- coding: utf-8 -*-
"""engine 单元测试：Task 加载（列表/map labels、levels）、规则兜底、run_task 调度。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from judgekit.engine import Decision, Task, rules_fallback, run_task  # noqa: E402


def test_task_from_dict_list_labels():
    t = Task.from_dict({"name": "t", "primitive": "classify", "labels": ["a", "b"]})
    assert t.labels == ["a", "b"] and t.label_descriptions == {}


def test_task_from_dict_map_labels():
    t = Task.from_dict({"name": "t", "primitive": "route",
                        "labels": {"物流": "快递", "售后": "退货"}})
    assert t.labels == ["物流", "售后"]
    assert t.label_descriptions == {"物流": "快递", "售后": "退货"}


def test_task_rejects_unknown_primitive():
    try:
        Task.from_dict({"primitive": "vibe"})
        assert False, "should raise"
    except ValueError:
        pass


def test_rules_fallback_hit_and_miss():
    t = Task.from_dict({"primitive": "classify", "labels": ["pos", "neg"],
                        "fallback_rules": {"pos": ["好"], "neg": ["差"]}})
    hit = rules_fallback(t, {"text": "太好了"})
    assert hit.ok and hit.value == "pos"
    miss = rules_fallback(t, {"text": "无关键词"})
    assert not miss.ok


def test_run_task_missing_provider_falls_back():
    t = Task.from_dict({"primitive": "classify", "labels": ["a"],
                        "provider": "ghost", "fallback_rules": {"a": ["x"]}})
    dec = run_task(t, {"text": "x"}, providers={})
    assert dec.ok and dec.value == "a" and dec.provider == "rules-after-fail"


def test_run_task_rules_provider_shortcircuits():
    from judgekit.providers import RulesProvider
    t = Task.from_dict({"primitive": "classify", "labels": ["a"],
                        "provider": "r", "fallback_rules": {"a": ["x"]}})
    dec = run_task(t, {"text": "x"}, providers={"r": RulesProvider("r")})
    assert dec.ok and dec.provider == "r"


def test_run_task_fallback_carries_original_error():
    """兜底成功时必须保留供应商失败根因（防止静默兜底产生假数据）。"""
    class Boom:
        name = "boom"
        def decide(self, task, x):
            raise RuntimeError("HTTP 429 insufficient balance")
    t = Task.from_dict({"primitive": "classify", "labels": ["a"],
                        "provider": "boom", "fallback_rules": {"a": ["x"]}})
    dec = run_task(t, {"text": "x"}, providers={"boom": Boom()})
    assert dec.ok and dec.value == "a" and dec.provider == "rules-after-fail"
    assert "RuntimeError" in dec.error and "429" in dec.error


def test_task_load_yaml(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("""
name: t
primitive: score
criteria: 拱火度
levels: [低, 中, 高]
""", encoding="utf-8")
    t = Task.load(str(p))
    assert t.levels == ["低", "中", "高"]
