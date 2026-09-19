# -*- coding: utf-8 -*-
"""本地 NanoJev 适配器 — 对接 NanoJev 持久服务（scripts/serve_decisions.py）。

服务：python serve_decisions.py --checkpoint-dir <dir> --precision bf16 --port 8765
端点：POST /api/evaluate  body={"states":[{"id","state","questions":{...}}]}
原语映射：classify/route→choice、score→score、verify→boolean。
限制（服务端）：≤32 states、≤96 questions、≤256 candidate paths / 请求。
注意：本地权重无 confidence 字段，Decision.confidence 以 top-1 概率近似。
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

from ..engine import Decision, Task


@dataclass
class NanoJev:
    name: str
    base_url: str = "http://127.0.0.1:8765"
    model: str = "nanojev"
    timeout: int = 300
    cost: float = 0.0  # 本地推理，边际成本为 0

    def _question(self, task: Task) -> dict:
        ins = "；".join(s for s in (task.criteria, task.instruction) if s) or f"任务 {task.name}"
        if task.primitive in ("classify", "route"):
            crit = {lb: task.label_descriptions.get(lb, lb) for lb in task.labels}
            return {"type": "choice", "instructions": ins, "criteria": crit}
        if task.primitive == "score":
            levels = task.levels or ["完全不满足", "部分满足", "完全满足"]
            return {"type": "score", "instructions": ins, "criteria": list(levels)}
        # verify → boolean（false-then-true 固定顺序，契约文档约定）
        return {"type": "boolean",
                "instructions": ins,
                "criteria": [f"不成立：{ins}", f"成立：{ins}"]}

    def decide(self, task: Task, x: dict) -> Decision:
        import time
        body = json.dumps({"states": [{"id": "s1",
                                       "state": json.dumps(x, ensure_ascii=False),
                                       "questions": {"d": self._question(task)}}]}).encode("utf-8")
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/api/evaluate", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        st = data["states"][0]
        a = st["answers"]["d"]
        probs = {str(k): float(v) for k, v in (a.get("probabilities") or {}).items()}
        raw = json.dumps(a, ensure_ascii=False)

        if task.primitive in ("classify", "route"):
            label = a.get("choice") or a.get("label") or (max(probs, key=probs.get) if probs else "")
            hit = label if label in task.labels else \
                next((lb for lb in task.labels if lb in str(label)), None)
            if hit is None:
                return Decision(task.primitive, None, 0.0, raw, self.name, 0, self.cost,
                                ok=False, error=f"label-not-in-candidates: {label!r}")
            conf = max(probs.values()) if probs else 0.5
            return Decision(task.primitive, hit, round(conf, 3), raw, self.name, 0, self.cost,
                            probabilities=probs or None)
        if task.primitive == "score":
            n_levels = len(task.levels or [0, 1, 2])
            span = max(1, n_levels - 1)
            value = a.get("score")
            if value is None and probs:  # 期望值回退：Σ i·p_i
                value = sum(i * p for i, (_, p) in enumerate(sorted(probs.items())))
            return Decision(task.primitive, round(max(0.0, min(1.0, float(value) / span)), 3),
                            max(probs.values()) if probs else 0.5, raw, self.name, 0, self.cost)
        # verify → boolean
        p_true = float(a.get("p_true", a.get("probability", probs.get("1", probs.get("true", 0.5)))))
        p_true = max(0.0, min(1.0, p_true))
        return Decision(task.primitive, p_true >= 0.5, round(max(p_true, 1 - p_true), 3),
                        raw, self.name, 0, self.cost)
