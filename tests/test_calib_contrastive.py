# -*- coding: utf-8 -*-
"""T9对比few-shot离线测试：--calib off/contrastive，不调网。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import abc_score as m  # noqa: E402


def _chat_provider(reply: dict):
    from judgekit.providers.go_openai import GoChatProvider
    calls = []

    def transport(url, headers, body):
        calls.append(json.loads(body.decode("utf-8")))
        return reply
    p = GoChatProvider(name="go-c", model="mimo-v2.6-flash",
                       api_key="k-test", transport=transport)
    return {"go-c": p}, calls


def test_calib_off_preserves_old_prompt():
    p_off = m.Adapter._abc_user_prompt("spam", "很好用", [], "")
    assert "对比纠偏例" not in p_off
    h_off = m.Adapter._abc_user_prompt("handoff", "你好", [], "")
    assert "对比纠偏例" not in h_off
    # route/sentiment不受calib影响
    r = m.Adapter._abc_user_prompt("spam", "x", [], "", "contrastive")
    assert "对比纠偏例" in r


def test_calib_contrastive_examples_shape():
    assert len(m.CALIB_SPAM_EXAMPLES) <= 6 and len(m.CALIB_HANDOFF_EXAMPLES) <= 6
    assert len(m.CALIB_SPAM_EXAMPLES) == 6 and len(m.CALIB_HANDOFF_EXAMPLES) == 6
    for e in m.CALIB_SPAM_EXAMPLES:
        assert e["wrong"] == "正常" and e["gold"] == "垃圾"  # spam只取判正常→实垃圾
    for e in m.CALIB_HANDOFF_EXAMPLES:
        assert e["wrong"] == "不转" and e["gold"] == "转人工"  # handoff只取判不转→实转人工
    ps = m.calib_block("spam", "contrastive")
    ph = m.calib_block("handoff", "contrastive")
    assert ps.count("错判正常→正解垃圾") == 6
    assert ph.count("错判不转→正解转人工") == 6
    # 每例截断≤120字
    for line in (ps + ph).splitlines():
        if line.startswith("例"):
            inner = line.split("「", 1)[1].split("」", 1)[0]
            assert len(inner) <= 120, f"例超120字:{inner[:30]}"
    assert m.calib_block("spam", "off") == ""
    assert m.calib_block("route", "contrastive") == ""
    assert m.calib_block("sentiment", "contrastive") == ""


def test_calib_cache_key_isolation():
    k_off = m.cache_key_of("i", "a", "b", "c", "c2", 4.0, "off", 0.95)
    assert "|CALIB=" not in k_off  # off保旧格式可比
    k_on = m.cache_key_of("i", "a", "b", "c", "c2", 4.0, "off", 0.95, "contrastive")
    assert "|CALIB=contrastive" in k_on and k_on != k_off  # 隔离旧跑


def test_calib_adapter_prompt_via_mock():
    ok = {"choices": [{"message": {"content":
          '{"label":"垃圾","confidence":0.9,"reason":"刷"}'}}], "usage": {}}
    provs, calls = _chat_provider(ok)
    ad = m.Adapter(provs, "go-c", transport=provs["go-c"].transport, calib="contrastive")
    r = ad.call("spam", "测试文本", [], 0.95)
    assert r["value"] == "垃圾"
    body_txt = calls[0]["messages"][1]["content"]
    assert "对比纠偏例" in body_txt and "错判正常→正解垃圾" in body_txt
    provs2, calls2 = _chat_provider(ok)
    ad2 = m.Adapter(provs2, "go-c", transport=provs2["go-c"].transport)  # 默认off
    ad2.call("spam", "测试文本", [], 0.95)
    assert "对比纠偏例" not in calls2[0]["messages"][1]["content"]


def test_calib_help_marks_experimental():
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "training" / "abc_score.py"), "--help"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    assert r.returncode == 0
    assert "EXPERIMENTAL" in r.stdout  # R1围栏：help须标恶化实验勿入生产链
