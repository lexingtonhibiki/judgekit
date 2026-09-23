# -*- coding: utf-8 -*-
"""T11阈值回炉离线测试：--cautious off/on，不调网。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import abc_score as m  # noqa: E402

NOTE = "当证据不足、A/B分歧大或文本有刷单/升级疑点但不确凿时，判需人工复核而非硬判"


def _chat_provider(reply: dict):
    from judgekit.providers.go_openai import GoChatProvider
    calls = []

    def transport(url, headers, body):
        calls.append(json.loads(body.decode("utf-8")))
        return reply
    p = GoChatProvider(name="go-c", model="mimo-v2.6-flash",
                       api_key="k-test", transport=transport)
    return {"go-c": p}, calls


def test_cautious_off_preserves_old_prompt():
    for kind in ("route", "handoff", "sentiment", "spam"):
        p_off = m.Adapter._abc_user_prompt(kind, "文本", ["a", "b"], "")
        assert NOTE not in p_off
        p_on = m.Adapter._abc_user_prompt(kind, "文本", ["a", "b"], "", "off", "on")
        assert NOTE in p_on


def test_cautious_block_shape():
    assert m.cautious_block("spam", "off") == ""
    assert m.cautious_block("handoff", "off") == ""
    # 全任务同句，不分kind
    for kind in ("route", "handoff", "sentiment", "spam"):
        assert m.cautious_block(kind, "on") == NOTE
    assert m.CAUTIOUS_NOTE == NOTE


def test_cautious_cache_key_isolation():
    k_off = m.cache_key_of("i", "a", "b", "c", "c2", 4.0, "off", 0.95)
    assert "|CAUTIOUS=" not in k_off  # off保旧格式可比
    k_on = m.cache_key_of("i", "a", "b", "c", "c2", 4.0, "off", 0.95, "off", "on")
    assert "|CAUTIOUS=on" in k_on and k_on != k_off  # 隔离旧跑
    # 与CALIB正交
    k_both = m.cache_key_of("i", "a", "b", "c", "c2", 4.0, "off", 0.95,
                            "contrastive", "on")
    assert "|CALIB=contrastive" in k_both and "|CAUTIOUS=on" in k_both


def test_cautious_adapter_prompt_via_mock():
    ok = {"choices": [{"message": {"content":
          '{"label":"垃圾","confidence":0.9,"reason":"刷"}'}}], "usage": {}}
    provs, calls = _chat_provider(ok)
    ad = m.Adapter(provs, "go-c", transport=provs["go-c"].transport, cautious="on")
    r = ad.call("spam", "测试文本", [], 0.2)
    assert r["value"] == "垃圾"
    body_txt = calls[0]["messages"][1]["content"]
    assert NOTE in body_txt
    provs2, calls2 = _chat_provider(ok)
    ad2 = m.Adapter(provs2, "go-c", transport=provs2["go-c"].transport)  # 默认off
    ad2.call("spam", "测试文本", [], 0.2)
    assert NOTE not in calls2[0]["messages"][1]["content"]


def test_cautious_help_lists_flag():
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "training" / "abc_score.py"), "--help"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0
    assert "--cautious" in r.stdout
