# -*- coding: utf-8 -*-
"""原生 TypeSafe Jev 适配器 — POST /v1/systemone。

四原语到原生问题类型的映射：
  classify / route → choice（criteria = {候选: 说明}，返回全量概率分布）
  score            → score （criteria = 刻度描述列表，返回刻度索引值，此处归一化到 0-1）
  verify           → noul  （返回单值概率）

一次调用只带一个问题（bench 语义）；API 本身支持一次多问，需要批量时可直接复用
build_question()。
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field

from ..engine import Decision, Task

DEFAULT_BASE_URL = "https://api.typesafe.ai/v1/systemone"


@dataclass
class TypeSafe:
    name: str
    model: str = "jev-latest"
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    per_decision_cost: float | None = None
    price_in_per_1k: float = 0.0      # usage 计价（官方定价公布后填）
    price_out_per_1k: float = 0.0
    timeout: int = 60

    def build_question(self, task: Task) -> dict:
        ins = "；".join(s for s in (task.criteria, task.instruction) if s) or f"任务 {task.name}"
        if task.primitive in ("classify", "route"):
            crit = {lb: task.label_descriptions.get(lb, lb) for lb in task.labels}
            return {"type": "choice", "instructions": ins, "criteria": crit}
        if task.primitive == "score":
            levels = task.levels or ["完全不满足", "部分满足", "完全满足"]
            return {"type": "score", "instructions": ins, "criteria": list(levels)}
        return {"type": "noul", "instructions": ins}

    def decide(self, task: Task, x: dict) -> Decision:
        state = json.dumps(x, ensure_ascii=False)
        body = json.dumps({"state": state, "model": self.model,
                           "questions": {"d": self.build_question(task)}}).encode("utf-8")
        req = urllib.request.Request(
            self.base_url, data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        latency = int((time.monotonic() - t0) * 1000)
        usage = data.get("usage") or {}
        if self.per_decision_cost is not None:
            cost = self.per_decision_cost
        else:
            cost = (usage.get("input_tokens", 0) / 1000 * self.price_in_per_1k
                    + usage.get("output_tokens", 0) / 1000 * self.price_out_per_1k)
        raw = json.dumps(data.get("answers", {}).get("d", {}), ensure_ascii=False)
        return self._parse(task, data["answers"]["d"], raw, latency, cost)

    def _parse(self, task: Task, a: dict, raw: str, latency: int, cost: float) -> Decision:
        t = a.get("type")
        if task.primitive in ("classify", "route"):
            if t != "choice":
                raise ValueError(f"expect choice, got {t}")
            label = a.get("choice", "")
            hit = label if label in task.labels else \
                next((lb for lb in task.labels if lb in str(label)), None)
            if hit is None:
                return Decision(task.primitive, None, 0.0, raw, self.name, latency, cost,
                                ok=False, error=f"label-not-in-candidates: {label!r}")
            probs = a.get("probabilities")
            return Decision(task.primitive, hit, float(a.get("confidence", 0.5)), raw,
                            self.name, latency, cost, probabilities=probs)
        if task.primitive == "score":
            if t != "score":
                raise ValueError(f"expect score, got {t}")
            n_levels = len(a.get("legend") or task.levels or [0, 1, 2])
            span = max(1, n_levels - 1)
            value = max(0.0, min(1.0, float(a.get("score", 0.0)) / span))
            return Decision(task.primitive, round(value, 3),
                            float(a.get("confidence", 0.5)), raw, self.name, latency, cost)
        # verify → noul
        if t != "noul":
            raise ValueError(f"expect noul, got {t}")
        p = max(0.0, min(1.0, float(a.get("noul", 0.0))))
        return Decision(task.primitive, p >= 0.5, round(max(p, 1 - p), 3), raw,
                        self.name, latency, cost)
