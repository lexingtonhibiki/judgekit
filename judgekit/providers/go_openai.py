# -*- coding: utf-8 -*-
"""GO 网关供应商 — https://opencode.ai/zen/go/v1（chat 三模型 + responses 双模型）。

网关事实（已验证，数值原样照用）：
  base      = https://opencode.ai/zen/go/v1
  chat      = glm-5.3-flash / deepseek-v4.1-flash / mimo-v2.6-flash
              → POST {base}/chat/completions（OpenAI 兼容 messages 信封）
  responses = gpt-5.6-luna / muse-spark-1.3-contributor
              → POST {base}/responses（responses 信封，input/text 抽取）
每次请求必须带：Authorization: Bearer $OPENCODE_GO_KEY ＋ x-opencode-session
（稳定 UUID/run 内复用）＋ 真实 User-Agent，否则 MissingSessionID。
key 只读环境变量 OPENCODE_GO_KEY，绝不入库。

传输走 curl 子进程：urllib 默认 UA（python-urllib）会被 CF 以 1010 拦截，
按任务约束必须走 curl 子进程或 urllib 之外的传输；此处选 curl 子进程。
判官提示词/决策解析复用 openai_compat（同一 JSON 契约）。
"""
from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from ..engine import Decision, Task
from .openai_compat import build_prompt, parse_decision

# ---- 网关常量（原样照用）----
GO_BASE_URL = "https://opencode.ai/zen/go/v1"
GO_API_KEY_ENV = "OPENCODE_GO_KEY"
GO_SESSION_HEADER = "x-opencode-session"
GO_USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                 "judgekit/0.2.0 (GO-gateway)")

GO_CHAT_MODELS = ("glm-5.3-flash", "deepseek-v4.1-flash", "mimo-v2.6-flash")
GO_RESPONSES_MODELS = ("gpt-5.6-luna", "muse-spark-1.3-contributor")

# transport 签名：fn(url, headers, body_bytes) -> 已解码的 JSON dict（测试注入 mock 用）
Transport = Callable[[str, dict, bytes], dict]

# run 内稳定的 session id（进程级复用；显式传入 session_id 则优先）
_SESSION_ID: str | None = None


def get_session_id(explicit: str | None = None) -> str:
    """返回本次 run 的稳定 session UUID。"""
    global _SESSION_ID
    if explicit:
        return explicit
    if _SESSION_ID is None:
        _SESSION_ID = str(uuid.uuid4())
    return _SESSION_ID


def curl_post_json(url: str, headers: dict, body: bytes, timeout: int) -> dict:
    """curl 子进程 POST JSON，返回解码后的 dict；失败抛 RuntimeError/ValueError。"""
    args = ["curl", "-sS", "--max-time", str(timeout), "-X", "POST", url]
    for k, v in headers.items():
        args += ["-H", f"{k}: {v}"]
    args += ["--data-binary", "@-"]
    try:
        p = subprocess.run(args, input=body, capture_output=True,
                           timeout=timeout + 10)
    except FileNotFoundError:
        raise RuntimeError("curl 不可用（GO 网关传输依赖 curl 子进程）")
    if p.returncode != 0:
        raise RuntimeError(f"curl exit {p.returncode}: "
                           f"{p.stderr.decode('utf-8', 'replace')[:300]}")
    try:
        return json.loads(p.stdout.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"bad-json-from-gateway: {e}; "
                         f"body={p.stdout[:200]!r}")


def extract_responses_text(data: dict) -> str:
    """从 /responses 信封抽正文。

    兼容三种形态：顶层 output_text（str）；output[].message.content[]
    里 type=output_text 的 text；content[] 里裸 text。reasoning 等条目跳过。
    无正文抛 ValueError。
    """
    if isinstance(data.get("output_text"), str) and data["output_text"]:
        return data["output_text"]
    parts: list[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for c in item.get("content") or []:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "output_text" and c.get("text"):
                    parts.append(c["text"])
                elif c.get("type") == "output_text":
                    for v in (c.get("text") or {}).values() \
                            if isinstance(c.get("text"), dict) else []:
                        if isinstance(v, str) and v:
                            parts.append(v)
                elif isinstance(c.get("text"), str) and c["text"]:
                    parts.append(c["text"])  # 部分网关的裸 text 形态
        elif item.get("type") == "output_text" and item.get("text"):
            parts.append(item["text"])
    text = "".join(parts)
    if not text:
        raise ValueError("no-text-in-responses-envelope")
    return text


def _cost(usage: dict, in_key: str, out_key: str,
          price_in: float, price_out: float,
          per_decision: float | None) -> float:
    if per_decision is not None:
        return per_decision
    return (usage.get(in_key, 0) / 1000 * price_in
            + usage.get(out_key, 0) / 1000 * price_out)


def _go_headers(api_key: str, session_id: str | None, user_agent: str) -> dict:
    if not api_key:
        raise RuntimeError(f"{GO_API_KEY_ENV} 未设置（环境变量为空）")
    return {"Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            GO_SESSION_HEADER: get_session_id(session_id),
            "User-Agent": user_agent}


@dataclass
class GoChatProvider:
    """GO 网关 chat 三模型（curl 版 OpenAI 兼容调用）。"""
    name: str
    model: str  # glm-5.3-flash | deepseek-v4.1-flash | mimo-v2.6-flash
    api_key: str = ""
    base_url: str = GO_BASE_URL
    session_id: str | None = None
    user_agent: str = GO_USER_AGENT
    timeout: int = 90
    max_tokens: int | None = None  # live ping 用小值（如 20）；bench 默认 None
    price_in_per_1k: float = 0.0
    price_out_per_1k: float = 0.0
    per_decision_cost: float | None = None
    transport: Transport | None = field(default=None, repr=False)

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def headers(self) -> dict:
        return _go_headers(self.api_key, self.session_id, self.user_agent)

    def request_body(self, task: Task, x: dict) -> bytes:
        system, user = build_prompt(task, x)
        req: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0,
        }
        if self.max_tokens is not None:
            req["max_tokens"] = self.max_tokens
        return json.dumps(req).encode("utf-8")

    def _post(self, url: str, headers: dict, body: bytes) -> dict:
        if self.transport is not None:
            return self.transport(url, headers, body)
        return curl_post_json(url, headers, body, self.timeout)

    def decide(self, task: Task, x: dict) -> Decision:
        body = self.request_body(task, x)
        headers = self.headers()
        t0 = time.monotonic()
        data = self._post(self.endpoint, headers, body)
        latency = int((time.monotonic() - t0) * 1000)
        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage") or {}
        cost = _cost(usage, "prompt_tokens", "completion_tokens",
                     self.price_in_per_1k, self.price_out_per_1k,
                     self.per_decision_cost)
        return parse_decision(task, text, self.name, latency, cost)


@dataclass
class GoResponsesProvider:
    """GO 网关 responses 双模型（responses 信封 input/text 抽取）。"""
    name: str
    model: str  # gpt-5.6-luna | muse-spark-1.3-contributor
    api_key: str = ""
    base_url: str = GO_BASE_URL
    session_id: str | None = None
    user_agent: str = GO_USER_AGENT
    timeout: int = 120
    max_output_tokens: int | None = None
    price_in_per_1k: float = 0.0
    price_out_per_1k: float = 0.0
    per_decision_cost: float | None = None
    transport: Transport | None = field(default=None, repr=False)

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/responses"

    def headers(self) -> dict:
        return _go_headers(self.api_key, self.session_id, self.user_agent)

    def request_body(self, task: Task, x: dict) -> bytes:
        system, user = build_prompt(task, x)
        req: dict[str, Any] = {"model": self.model,
                               "input": f"{system}\n\n{user}"}
        if self.max_output_tokens is not None:
            req["max_output_tokens"] = self.max_output_tokens
        return json.dumps(req).encode("utf-8")

    def _post(self, url: str, headers: dict, body: bytes) -> dict:
        if self.transport is not None:
            return self.transport(url, headers, body)
        return curl_post_json(url, headers, body, self.timeout)

    def decide(self, task: Task, x: dict) -> Decision:
        body = self.request_body(task, x)
        headers = self.headers()
        t0 = time.monotonic()
        data = self._post(self.endpoint, headers, body)
        latency = int((time.monotonic() - t0) * 1000)
        text = extract_responses_text(data)
        usage = data.get("usage") or {}
        cost = _cost(usage, "input_tokens", "output_tokens",
                     self.price_in_per_1k, self.price_out_per_1k,
                     self.per_decision_cost)
        return parse_decision(task, text, self.name, latency, cost)


__all__ = ["GO_BASE_URL", "GO_API_KEY_ENV", "GO_SESSION_HEADER", "GO_USER_AGENT",
           "GO_CHAT_MODELS", "GO_RESPONSES_MODELS",
           "get_session_id", "curl_post_json", "extract_responses_text",
           "GoChatProvider", "GoResponsesProvider"]
