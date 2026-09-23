# -*- coding: utf-8 -*-
"""T14场景锚定离线测试：四路径场景句存在断言，不调网。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import abc_score as abc  # noqa: E402
from training import eval_jev_v4 as jev  # noqa: E402


def test_jev_spam_scn_prefix():
    off = jev.build_tasks()["spam"]
    on = jev.build_tasks(scn="on")["spam"]
    assert jev.SCN_SPAM not in off.instruction  # 默认off保旧链
    assert on.instruction.startswith(jev.SCN_SPAM)  # 前置逐字句
    assert "举一反三" in on.instruction  # 与T7泛化叠加不替换
    assert on.name != off.name and "scn" in on.name  # Task名隔离旧190


def test_jev_handoff_scn_prefix():
    off = jev.build_tasks()["handoff"]
    on = jev.build_tasks(scn="on")["handoff"]
    assert jev.SCN_HANDOFF not in off.instruction
    assert on.instruction.startswith(jev.SCN_HANDOFF)
    assert on.name != off.name and "scn" in on.name


def test_abc_spam_scn_both_paths():
    # GO prompt路径
    p_off = abc.Adapter._abc_user_prompt("spam", "很好用", [], "")
    assert abc.SCN_SPAM not in p_off  # 默认off保旧链
    p_on = abc.Adapter._abc_user_prompt("spam", "很好用", [], "", "off", "off", "on")
    assert p_on.startswith(abc.SCN_SPAM)  # 前置逐字句
    assert "举一反三" in p_on  # 与T7泛化/禁令叠加不替换
    # typesafe Task路径
    from judgekit.providers.go_openai import GoChatProvider
    p = GoChatProvider(name="go-c", model="m", api_key="k-test")
    ad_off = abc.Adapter({"go-c": p}, "go-c")
    ad_on = abc.Adapter({"go-c": p}, "go-c", scn="on")
    t_off = ad_off._typesafe_task("spam", [])
    t_on = ad_on._typesafe_task("spam", [])
    assert abc.SCN_SPAM not in t_off.instruction
    assert t_on.instruction.startswith(abc.SCN_SPAM)
    assert "举一反三" in t_on.instruction
    # cache键隔离
    k_off = abc.cache_key_of("i", "a", "b", "c", "c2", 4.0, "off", 0.95)
    k_on = abc.cache_key_of("i", "a", "b", "c", "c2", 4.0, "off", 0.95,
                            "off", "off", False, "on")
    assert "|SCN=" not in k_off and "|SCN=on" in k_on and k_on != k_off


def test_abc_handoff_scn_both_paths():
    p_off = abc.Adapter._abc_user_prompt("handoff", "你好", [], "")
    assert abc.SCN_HANDOFF not in p_off
    p_on = abc.Adapter._abc_user_prompt("handoff", "你好", [], "", "off", "off", "on")
    assert p_on.startswith(abc.SCN_HANDOFF)
    assert "重复催" in p_on  # 与原rubric叠加不替换
    from judgekit.providers.go_openai import GoChatProvider
    p = GoChatProvider(name="go-c", model="m", api_key="k-test")
    ad_off = abc.Adapter({"go-c": p}, "go-c")
    ad_on = abc.Adapter({"go-c": p}, "go-c", scn="on")
    t_off = ad_off._typesafe_task("handoff", [])
    t_on = ad_on._typesafe_task("handoff", [])
    assert abc.SCN_HANDOFF not in t_off.instruction
    assert t_on.instruction.startswith(abc.SCN_HANDOFF)
    # route/sentiment不受scn影响
    r = abc.Adapter._abc_user_prompt("route", "x", ["a"], "", "off", "off", "on")
    s = abc.Adapter._abc_user_prompt("sentiment", "x", [], "", "off", "off", "on")
    assert abc.SCN_SPAM not in r and abc.SCN_HANDOFF not in r
    assert abc.SCN_SPAM not in s and abc.SCN_HANDOFF not in s
