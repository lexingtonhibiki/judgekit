# -*- coding: utf-8 -*-
"""GO 网关离线测试：请求形状断言（双端点，含 session 头）+ responses 信封解析。

全部离线（注入 mock transport，不联网）。live ping 见 test_live_ping_chat_glm，
默认跳过，加 --run-live 才跑（1 call 上限，只打最便宜的 chat 模型）。
"""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from judgekit.engine import Task  # noqa: E402
from judgekit.providers import load_providers  # noqa: E402
from judgekit.providers.go_openai import (  # noqa: E402
    GO_BASE_URL,
    GO_SESSION_HEADER,
    GoChatProvider,
    GoResponsesProvider,
    extract_responses_text,
)

VERIFY = Task.from_dict({"primitive": "verify", "criteria": "ping"})


class Recorder:
    """mock transport：录下 (url, headers, body)，按预设 JSON 回包。"""

    def __init__(self, reply: dict):
        self.reply = reply
        self.calls: list = []

    def __call__(self, url, headers, body):
        self.calls.append((url, dict(headers), json.loads(body.decode("utf-8"))))
        return self.reply


CHAT_REPLY = {"choices": [{"message": {"content":
                 '{"verdict": true, "confidence": 0.9}'}}],
              "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

RESP_REPLY = {"output": [{"type": "message", "content": [
    {"type": "output_text",
     "text": '{"verdict": false, "confidence": 0.7}'},
]}], "usage": {"input_tokens": 10, "output_tokens": 5}}


def test_chat_request_shape():
    r = Recorder(CHAT_REPLY)
    p = GoChatProvider(name="go-glm", model="glm-5.3-flash",
                       api_key="k-test", transport=r)
    d = p.decide(VERIFY, {"text": "ping"})
    assert d.ok and d.value is True
    url, headers, body = r.calls[0]
    assert url == GO_BASE_URL + "/chat/completions"
    assert headers["Authorization"] == "Bearer k-test"
    assert headers.get(GO_SESSION_HEADER)  # session 头必须带
    assert headers.get("User-Agent") and "python-urllib" not in headers["User-Agent"]
    assert body["model"] == "glm-5.3-flash"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["temperature"] == 0


def test_chat_session_stable_across_calls():
    r = Recorder(CHAT_REPLY)
    p = GoChatProvider(name="go-glm", model="glm-5.3-flash",
                       api_key="k-test", transport=r)
    p.decide(VERIFY, {"text": "a"})
    p.decide(VERIFY, {"text": "b"})
    s1 = r.calls[0][1][GO_SESSION_HEADER]
    s2 = r.calls[1][1][GO_SESSION_HEADER]
    assert s1 and s1 == s2  # run 内稳定复用


def test_responses_request_shape():
    r = Recorder(RESP_REPLY)
    p = GoResponsesProvider(name="go-luna", model="gpt-5.6-luna",
                            api_key="k-test", transport=r)
    d = p.decide(VERIFY, {"text": "ping"})
    assert d.ok and d.value is False
    url, headers, body = r.calls[0]
    assert url == GO_BASE_URL + "/responses"  # endpoint 已记录
    assert p.endpoint == GO_BASE_URL + "/responses"
    assert headers["Authorization"] == "Bearer k-test"
    assert headers.get(GO_SESSION_HEADER)
    assert headers.get("User-Agent") and "python-urllib" not in headers["User-Agent"]
    assert body["model"] == "gpt-5.6-luna"
    assert isinstance(body["input"], str) and "ping" in body["input"]


def test_responses_envelope_top_level_output_text():
    text = extract_responses_text(
        {"output_text": '{"verdict": true, "confidence": 0.5}'})
    assert '"verdict": true' in text


def test_responses_envelope_empty_raises():
    with pytest.raises(ValueError):
        extract_responses_text({"output": [{"type": "reasoning"}]})
    with pytest.raises(ValueError):
        extract_responses_text({})


def test_missing_key_raises():
    p = GoChatProvider(name="go-glm", model="glm-5.3-flash", api_key="")
    with pytest.raises(RuntimeError):
        p.decide(VERIFY, {"text": "x"})
    q = GoResponsesProvider(name="go-luna", model="gpt-5.6-luna", api_key="")
    with pytest.raises(RuntimeError):
        q.decide(VERIFY, {"text": "x"})


def test_models_yaml_go_entries_append_only():
    provs = load_providers(str(ROOT / "benchmarks" / "models.yaml"))
    # 原有 12 家含义不动
    for n in ("rules", "nanojev-local", "typesafe", "router-free-auto",
              "router-glm-5.3-flash", "router-deepseek-flash", "router-web-gemini",
              "glm-coding-direct", "deepseek-direct", "gemini-direct",
              "modelscope-deepseek", "aimlapi-jev"):
        assert n in provs, n
    assert len(provs) == 17
    # 新增 5 家（名/model/kind 原样）
    expect = {
        "go-glm-5.3-flash": (GoChatProvider, "glm-5.3-flash"),
        "go-deepseek-v4.1-flash": (GoChatProvider, "deepseek-v4.1-flash"),
        "go-mimo-v2.6-flash": (GoChatProvider, "mimo-v2.6-flash"),
        "go-gpt-5.6-luna": (GoResponsesProvider, "gpt-5.6-luna"),
        "go-muse-spark-1.3": (GoResponsesProvider, "muse-spark-1.3-contributor"),
    }
    for n, (cls, model) in expect.items():
        assert isinstance(provs[n], cls), n
        assert provs[n].model == model, n
        assert provs[n].base_url == GO_BASE_URL, n


def test_unknown_kind_raises(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("providers:\n  bad:\n    kind: nope\n"
                   "    base_url: http://x\n    model: m\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_providers(str(cfg))


def test_live_ping_chat_glm(request):
    """live ping：1 call 上限，只打最便宜的 chat 模型；默认跳过。"""
    if not request.config.getoption("run_live", default=False):
        pytest.skip("live ping 默认跳过（加 --run-live 才跑）")
    key = os.environ.get("OPENCODE_GO_KEY", "")
    if not key:
        pytest.skip("OPENCODE_GO_KEY 未设置")
    p = GoChatProvider(name="go-glm-5.3-flash", model="glm-5.3-flash",
                       api_key=key, max_tokens=64)  # 20 会顶格截断导致 flaky，64 仍便宜
    d = p.decide(VERIFY, {"text": "ping"})
    assert d.ok, d.error
