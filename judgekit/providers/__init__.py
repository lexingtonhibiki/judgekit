# -*- coding: utf-8 -*-
"""judgekit.providers — 判断供应商注册表。

kind: rules | openai | typesafe，在 providers YAML 里声明；
环境变量优先于 yaml 里的 api_key_default（key 永远不进仓库）。
"""
from __future__ import annotations

import os

from .rules import RulesProvider          # noqa: F401
from .openai_compat import OpenAICompat   # noqa: F401
from .typesafe import TypeSafe            # noqa: F401
from .nanojev import NanoJev              # noqa: F401
from .go_openai import GO_BASE_URL, GoChatProvider, GoResponsesProvider  # noqa: F401


def load_providers(path: str) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out: dict = {}
    for name, c in (cfg.get("providers") or {}).items():
        kind = c.get("kind", "openai")
        key = os.environ.get(c.get("api_key_env", ""), "") or c.get("api_key_default", "")
        if kind == "rules":
            out[name] = RulesProvider(name)
        elif kind == "typesafe":
            out[name] = TypeSafe(
                name=name,
                model=c.get("model", "jev-latest"),
                api_key=key,
                base_url=c.get("base_url", "https://api.typesafe.ai/v1/systemone"),
                per_decision_cost=c.get("per_decision_cost"),
                price_in_per_1k=float(c.get("price_in_per_1k", 0.0)),
                price_out_per_1k=float(c.get("price_out_per_1k", 0.0)),
                timeout=int(c.get("timeout", 60)),
            )
        elif kind == "nanojev":
            out[name] = NanoJev(
                name=name,
                base_url=c.get("base_url", "http://127.0.0.1:8765"),
                model=c.get("model", "nanojev"),
                timeout=int(c.get("timeout", 300)),
            )
        elif kind == "openai":
            out[name] = OpenAICompat(
                name=name,
                base_url=c["base_url"],
                model=c["model"],
                api_key=key or "sk-local",
                price_in_per_1k=float(c.get("price_in_per_1k", 0.0)),
                price_out_per_1k=float(c.get("price_out_per_1k", 0.0)),
                per_decision_cost=c.get("per_decision_cost"),
                timeout=int(c.get("timeout", 90)),
            )
        elif kind == "go-chat":
            out[name] = GoChatProvider(
                name=name,
                base_url=c.get("base_url", GO_BASE_URL),
                model=c["model"],
                api_key=key,
                session_id=c.get("session_id"),
                price_in_per_1k=float(c.get("price_in_per_1k", 0.0)),
                price_out_per_1k=float(c.get("price_out_per_1k", 0.0)),
                per_decision_cost=c.get("per_decision_cost"),
                timeout=int(c.get("timeout", 90)),
            )
        elif kind == "go-responses":
            out[name] = GoResponsesProvider(
                name=name,
                base_url=c.get("base_url", GO_BASE_URL),
                model=c["model"],
                api_key=key,
                session_id=c.get("session_id"),
                price_in_per_1k=float(c.get("price_in_per_1k", 0.0)),
                price_out_per_1k=float(c.get("price_out_per_1k", 0.0)),
                per_decision_cost=c.get("per_decision_cost"),
                timeout=int(c.get("timeout", 120)),
            )
        else:
            raise ValueError(f"provider {name!r} 的 kind={kind!r} 不认识（可选 rules/openai/typesafe/nanojev/go-chat/go-responses）")
    return out


__all__ = ["load_providers", "RulesProvider", "OpenAICompat", "TypeSafe", "NanoJev",
           "GoChatProvider", "GoResponsesProvider"]
