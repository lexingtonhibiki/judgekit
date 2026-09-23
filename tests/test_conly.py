# -*- coding: utf-8 -*-
"""T12 C瓶颈分离离线测试：--conly冻结复用A/B原文+CONLY键隔离，不调网。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from judgekit.providers.go_openai import GoChatProvider, GoResponsesProvider  # noqa: E402
from training import abc_score as m  # noqa: E402


def _mock(kind, name, replies):
    """replies: 按调用顺序消费的响应体/异常；返回(provs, calls)。"""
    calls = []
    state = {"i": 0}

    def transport(url, headers, body):
        calls.append(json.loads(body.decode("utf-8")))
        item = replies[min(state["i"], len(replies) - 1)]
        state["i"] += 1
        if isinstance(item, Exception):
            raise item
        return item

    cls = GoResponsesProvider if kind == "responses" else GoChatProvider
    p = cls(name=name, model="mock-model", api_key="k-test", transport=transport)
    return {name: p}, calls


CHAT_LABEL = {"choices": [{"message": {"content":
    '{"label":"垃圾","confidence":0.9,"reason":"营销引流"}'}}], "usage": {}}
RESP_SCORE = {"output": [{"type": "message", "content": [
    {"type": "output_text",
     "text": '{"score": 8, "confidence": 0.8, "reason":"满意有细节"}'}]}],
    "usage": {}}
CHAT_SCORE = {"choices": [{"message": {"content":
    '{"score": 8, "confidence": 0.8, "reason":"满意有细节"}'}}], "usage": {}}


def _ab(v1="正常", v2="正常"):
    a = {"value": v1, "reason": "ra", "confidence": 0.9, "ok": True,
         "error": "", "endpoint": "ep-a", "model": "m-a", "usage": {}}
    b = {"value": v2, "reason": "rb", "confidence": 0.8, "ok": True,
         "error": "", "endpoint": "ep-b", "model": "m-b", "usage": {}}
    return {"id": "x", "text": "t", "task": "spam", "source": "s",
            "A": a, "B": b,
            "C": {"final": v1}, "cache_key": "k"}


def test_conly_cache_key_isolation():
    k_old = m.cache_key_of("i", "a", "b", "c", "c2", 4.0, "off", 0.95)
    assert "|CONLY" not in k_old  # 默认保旧格式可比
    k_new = m.cache_key_of("i", "a", "b", "c", "c2", 4.0, "off", 0.95,
                           "off", "off", True)
    assert k_new.endswith("|CONLY") and k_new != k_old  # 隔离旧跑
    k_both = m.cache_key_of("i", "a", "b", "c", "c2", 4.0, "off", 0.95,
                            "contrastive", "on", True)
    assert "|CALIB=contrastive" in k_both and "|CAUTIOUS=on" in k_both \
        and "|CONLY" in k_both  # 与CALIB/CAUTIOUS正交


def test_conly_frozen_ab_verbatim_and_independent():
    fz = _ab("正常", "垃圾")
    a, b = m.frozen_ab_pair(fz)
    assert a == fz["A"] and b == fz["B"] and a is not fz["A"]  # 逐字复用+副本
    a["value"] = "篡改"
    assert fz["A"]["value"] == "正常"  # 调用方改动不污染冻结源


def test_conly_load_frozen_validates(tmp_path):
    p = tmp_path / "fz.jsonl"
    p.write_text(json.dumps(_ab()) + "\n", encoding="utf-8")
    assert len(m.load_frozen(str(p))) == 1
    bad = _ab()
    bad["B"] = {"value": None, "ok": False}
    p.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    try:
        m.load_frozen(str(p))
    except ValueError as e:
        assert "禁重采" in str(e)
    else:
        raise AssertionError("坏冻结行应抛错而非重采")


def test_conly_agree_calls_no_c():
    fz = _ab("正常", "正常")
    a, b = m.frozen_ab_pair(fz)
    boom, boom_calls = _mock("chat", "go-boom", [RuntimeError("must-not-call")])
    provs_c, c_calls = _mock("responses", "go-c", [RESP_SCORE])
    c_ad = m.Adapter(provs_c, "go-c", transport=provs_c["go-c"].transport)
    c, c2rec, delta = m.score_c_path("spam", "t", [], a, b, c_ad, None, 4.0)
    assert c["called"] == "mean" and c["final"] == "正常"
    assert c_calls == [] and c2rec is None and delta == 0
    assert boom_calls == []  # A/B槽零调用（boom transport从未被引用）


def test_conly_disagree_rejudges_c_temp02():
    fz = _ab("正常", "垃圾")
    a, b = m.frozen_ab_pair(fz)
    provs_c, c_calls = _mock("chat", "go-c", [CHAT_LABEL])
    c_ad = m.Adapter(provs_c, "go-c", transport=provs_c["go-c"].transport)
    provs_c2, c2_calls = _mock("chat", "go-c2", [CHAT_LABEL])
    c2_ad = m.Adapter(provs_c2, "go-c2", transport=provs_c2["go-c2"].transport)
    c, c2rec, delta = m.score_c_path("spam", "原文", [], a, b, c_ad, c2_ad, 4.0)
    assert c["called"] == "rejudge" and c["final"] == "垃圾" and delta == 1
    assert c_calls[0]["temperature"] == 0.2  # C恒0.2
    assert "仲裁语境" in c_calls[0]["messages"][1]["content"]  # 原提示词
    assert c2rec is None and c2_calls == []  # 分类delta=1≤阈4：C2不调


def test_conly_c2_same_threshold_sentiment():
    a = {"value": 3.0, "reason": "ra", "confidence": 0.9, "ok": True}
    b = {"value": 8.0, "reason": "rb", "confidence": 0.8, "ok": True}
    provs_c, _ = _mock("responses", "go-c", [RESP_SCORE])
    c_ad = m.Adapter(provs_c, "go-c", transport=provs_c["go-c"].transport)
    provs_c2, c2_calls = _mock("chat", "go-c2", [CHAT_SCORE])
    c2_ad = m.Adapter(provs_c2, "go-c2", transport=provs_c2["go-c2"].transport)
    c, c2rec, delta = m.score_c_path("sentiment", "t", [], a, b, c_ad, c2_ad, 4.0)
    assert delta == 5.0 and c["called"] == "rejudge"  # 分差>2重判
    assert c2rec is not None and len(c2_calls) == 1  # 分差5>阈4：C2同阈触发


def test_conly_variance_stable_vs_drift():
    def row(i, final, src="JD刷单"):
        return {"id": i, "source": src, "C": {"final": final}}
    gold = {f"k{i}": {"gold": "垃圾"} for i in range(10)}
    fz = [row(f"k{i}", "垃圾") for i in range(10)]
    new = [row(f"k{i}", "正常" if i == 0 else "垃圾") for i in range(10)]
    t = m.conly_variance(new, fz, gold)
    assert "1/10=10.0%" in t and "稳定偏松" in t and "停调判官转源侧" in t
    assert "Jaccard=0.000" in t  # 旧mm空∩新{k0}/并1 → 0
    new2 = [row(f"k{i}", "正常" if i < 3 else "垃圾") for i in range(10)]
    t2 = m.conly_variance(new2, fz, gold)
    assert "3/10=30.0%" in t2 and "漂移" in t2 and "另立案" in t2


def test_conly_help_lists_flag():
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "training" / "abc_score.py"), "--help"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "--conly" in r.stdout
