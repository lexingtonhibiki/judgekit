# -*- coding: utf-8 -*-
"""T3 ABC-GO离线测试：GO双kind解析/usage全量/C2触发/cache键/去重/退避。

全部离线（Adapter transport注入mock，不联网不读key）。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from judgekit.providers.go_openai import GoChatProvider, GoResponsesProvider  # noqa: E402
from training.abc_score import (Adapter, apply_c2, band, cache_key_of,  # noqa: E402
                                c2_should_call, dedup_by_key, is_retryable,
                                usage_totals)


def make_providers(reply_seq, kind="chat"):
    """reply_seq: list[dict|Exception]，按调用顺序消费。"""
    state = {"i": 0}
    calls = []

    def transport(url, headers, body):
        calls.append((url, dict(headers), json.loads(body.decode("utf-8"))))
        item = reply_seq[min(state["i"], len(reply_seq) - 1)]
        state["i"] += 1
        if isinstance(item, Exception):
            raise item
        return item

    if kind == "chat":
        p = GoChatProvider(name="go-a", model="mimo-v2.6-flash",
                           api_key="k-test", transport=transport)
    else:
        p = GoResponsesProvider(name="go-c", model="gpt-5.6-luna",
                                api_key="k-test", transport=transport)
    return {"go-a": p} if kind == "chat" else {"go-c": p}, calls


CHAT_OK = {"choices": [{"message": {"content":
    '{"score": 7, "confidence": 0.8, "reason": "满意有细节"}'}}],
    "usage": {"prompt_tokens": 60, "completion_tokens": 90,
              "completion_tokens_details": {"reasoning_tokens": 47}}}
RESP_OK = {"output": [{"type": "message", "content": [
    {"type": "output_text",
     "text": '{"label": "正常", "confidence": 0.9, "reason": "普通抱怨"}'}]}],
    "usage": {"input_tokens": 56, "output_tokens": 135,
              "output_tokens_details": {"reasoning_tokens": 118}}}


def test_go_chat_sentiment_parse_and_usage():
    provs, calls = make_providers([CHAT_OK], "chat")
    ad = Adapter(provs, "go-a", transport=provs["go-a"].transport)
    r = ad.call("sentiment", "很好用", [], 0.7)
    assert r["ok"] and r["value"] == 7 and r["confidence"] == 0.8
    assert r["model"] == "mimo-v2.6-flash"
    assert "mimo-v2.6-flash" in r["endpoint"] and "chat/completions" in r["endpoint"]
    assert r["usage"]["completion_tokens_details"]["reasoning_tokens"] == 47
    assert usage_totals(r["usage"]) == (60, 90, 47)  # usage全量（含推理）
    body = calls[0][2]
    assert body["temperature"] == 0.7 and body["model"] == "mimo-v2.6-flash"


def test_go_responses_spam_parse_and_usage():
    provs, calls = make_providers([RESP_OK], "responses")
    ad = Adapter(provs, "go-c", transport=provs["go-c"].transport,
                 reasoning_effort="low")
    r = ad.call("spam", "东西坏了", [], 0.2)
    assert r["ok"] and r["value"] == "正常"
    assert "responses" in r["endpoint"] and "gpt-5.6-luna" in r["endpoint"]
    assert usage_totals(r["usage"]) == (56, 135, 118)
    body = calls[0][2]
    assert body["reasoning"] == {"effort": "low"}  # 透传降档
    assert "max_output_tokens" in body


def test_go_chat_empty_body_raises_retryable():
    empty = {"choices": [{"message": {"content": ""}}],
             "usage": {"prompt_tokens": 76, "completion_tokens": 300,
                       "completion_tokens_details": {"reasoning_tokens": 300}}}
    provs, _ = make_providers([empty], "chat")
    ad = Adapter(provs, "go-a", transport=provs["go-a"].transport)
    try:
        ad.call("sentiment", "x", [], 0.7, tries=1)
        raise AssertionError("应抛异常")
    except ValueError as e:
        assert is_retryable(str(e))  # 推理烧完预算属可重试


def test_retry_429_then_ok():
    provs, _ = make_providers(
        [RuntimeError("gateway-error: {'code': 429}"), CHAT_OK], "chat")
    ad = Adapter(provs, "go-a", transport=provs["go-a"].transport)
    import training.abc_score as m
    sleeps = []
    orig_sleep = m.time.sleep
    m.time.sleep = lambda s: sleeps.append(s)
    try:
        r = ad.call("sentiment", "很好用", [], 0.7, tries=3)
    finally:
        m.time.sleep = orig_sleep
    assert r["ok"] and r["value"] == 7
    assert sleeps and sleeps[0] == 2  # 指数退避2/4/8…


def test_fatal_403_no_retry():
    provs, _ = make_providers(
        [RuntimeError("HTTP403: forbidden")] * 3, "chat")
    ad = Adapter(provs, "go-a", transport=provs["go-a"].transport)
    import training.abc_score as m
    n = {"c": 0}
    orig = m.time.sleep
    m.time.sleep = lambda s: n.__setitem__("c", n["c"] + 1)
    try:
        try:
            ad.call("sentiment", "x", [], 0.7, tries=3)
            raise AssertionError("应抛异常")
        except RuntimeError:
            pass
    finally:
        m.time.sleep = orig
    assert n["c"] == 0  # fatal不退避


def test_cache_key_includes_models():
    k1 = cache_key_of("x_1", "go-mimo-v2.6-flash", "go-deepseek-v4.1-flash",
                      "go-gpt-5.6-luna", "go-glm-5.3-flash", 4.0, "off")
    k2 = cache_key_of("x_1", "typesafe", "glm-coding-direct",
                      "typesafe", "", 4.0, "off")
    assert k1 != k2  # 换槽即重打
    assert "go-mimo-v2.6-flash" in k1 and "go-deepseek-v4.1-flash" in k1


def test_dedup_keep_last():
    recs = [{"id": "a", "cache_key": "k1", "v": 1},
            {"id": "a", "cache_key": "k1", "v": 2},
            {"id": "b", "v": 3}]
    out = dedup_by_key(recs)
    assert len(out) == 2
    assert next(r for r in out if r.get("cache_key") == "k1")["v"] == 2


def test_c2_trigger_rules():
    assert c2_should_call("sentiment", 4.5, 4.0) is True  # 超阈才调
    assert c2_should_call("sentiment", 4.0, 4.0) is False
    assert c2_should_call("sentiment", 3.0, 4.0) is False
    assert c2_should_call("route", 1, 4.0) is False  # 分类默认不启用
    assert c2_should_call("route", 1, 1.0) is True
    assert c2_should_call("route", 0, 1.0) is False
    assert band(1.0) == "负面" and band(5.0) == "中性" and band(8.0) == "正面"


def test_apply_c2_c_ok_disagree_escalates():
    provs, _ = make_providers(
        [{"choices": [{"message": {"content":
          '{"score": 9, "confidence": 0.9, "reason": "非常满意"}'}}],
          "usage": {"prompt_tokens": 10, "completion_tokens": 20}}], "chat")
    ad = Adapter({"go-c2": provs["go-a"]}, "go-c2",
                 transport=provs["go-a"].transport)
    ad.name, ad.p = "go-glm-5.3-flash", provs["go-a"]
    a = {"value": 1.0, "reason": "差", "confidence": 0.9}
    b = {"value": 8.0, "reason": "好", "confidence": 0.9}
    c = {"final": 2.0, "band": "负面", "action": "通过", "reason": "C判",
         "confidence": 0.8, "provider": "go-gpt-5.6-luna", "called": "rejudge",
         "ok": True, "endpoint": "e", "model": "m", "usage": {}}
    c2c, c2rec = apply_c2("sentiment", "t", [], a, b, c, 7.0, ad, 4.0)
    assert c2rec is not None and c2rec["called"] == "rejudge"
    assert c2c["final"] == 2.0 and c2c["action"] == "需人工复核"  # C与C2分歧>2升级


def test_apply_c2_c_failed_c2_takes_over():
    provs, _ = make_providers(
        [{"choices": [{"message": {"content":
          '{"score": 3, "confidence": 0.7, "reason": "不满"}'}}],
          "usage": {}}], "chat")
    ad = Adapter({"go-c2": provs["go-a"]}, "go-c2",
                 transport=provs["go-a"].transport)
    ad.name = "go-glm-5.3-flash"
    a = {"value": 1.0, "reason": "差", "confidence": 0.9}
    b = {"value": 9.0, "reason": "好", "confidence": 0.9}
    c = {"final": None, "action": "需人工复核", "reason": "C失败",
         "confidence": 0.0, "provider": "x", "called": "rejudge-fail",
         "ok": False, "endpoint": "e", "model": "m", "usage": {}}
    c2c, c2rec = apply_c2("sentiment", "t", [], a, b, c, 8.0, ad, 4.0)
    assert c2c["called"] == "rejudge-c2" and c2c["final"] == 3.0
