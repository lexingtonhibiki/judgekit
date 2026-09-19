# -*- coding: utf-8 -*-
"""judgekit.providers — 判断供应商注册表。

统一 OpenAI 兼容 chat/completions 接口（本地路由器 / AIMLAPI / 官方 Jev 代理都走它），
外加零成本规则基线。成本估算：优先用返回的 usage，缺省按字符数粗估。
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass


def _est_tokens(text: str) -> int:
    """粗估 token：CJK 约 1 字/token，拉丁约 4 字符/token。仅用于无 usage 时的成本估算。"""
    cjk = sum(1 for ch in text if ord(ch) > 0x2E80)
    return max(1, cjk + (len(text) - cjk) // 4)


@dataclass
class OpenAICompat:
    name: str
    base_url: str
    model: str
    api_key: str = "sk-local"
    price_in_per_1k: float = 0.0     # 元/1k 输入 token
    price_out_per_1k: float = 0.0    # 元/1k 输出 token
    per_decision_cost: float | None = None  # Jev 类按次计费的供应商填这个
    timeout: int = 90

    def complete(self, system: str, user: str) -> tuple[str, dict | None, float]:
        """返回 (回复原文, usage或None, 估算成本元)。失败抛异常，由 engine 捕获。"""
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        latency = time.monotonic() - t0
        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage")
        if self.per_decision_cost is not None:
            cost = self.per_decision_cost
        elif usage:
            cost = usage.get("prompt_tokens", 0) / 1000 * self.price_in_per_1k \
                 + usage.get("completion_tokens", 0) / 1000 * self.price_out_per_1k
        else:
            cost = _est_tokens(system + user) / 1000 * self.price_in_per_1k \
                 + _est_tokens(text) / 1000 * self.price_out_per_1k
        return text, usage, cost


class RulesProvider:
    """零成本关键词基线，engine.rules_fallback 实际干活；占位以满足注册表接口。"""
    def __init__(self, name: str = "rules"):
        self.name = name


def load_providers(path: str) -> dict:
    """从 providers YAML 加载供应商注册表。"""
    import yaml
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = {}
    for name, c in (cfg.get("providers") or {}).items():
        kind = c.get("kind", "openai")
        if kind == "rules":
            out[name] = RulesProvider(name)
            continue
        key = os.environ.get(c.get("api_key_env", ""), "") or c.get("api_key_default", "sk-local")
        out[name] = OpenAICompat(
            name=name,
            base_url=c["base_url"],
            model=c["model"],
            api_key=key,
            price_in_per_1k=float(c.get("price_in_per_1k", 0.0)),
            price_out_per_1k=float(c.get("price_out_per_1k", 0.0)),
            per_decision_cost=c.get("per_decision_cost"),
            timeout=int(c.get("timeout", 90)),
        )
    return out
