# -*- coding: utf-8 -*-
"""OpenAI 兼容供应商 — chat/completions 后端（GLM/DeepSeek/路由器/任意兼容端点）。

判官任务被翻译成「只输出一个 JSON 对象」的提示词；解析失败抛异常，
由 engine 统一走规则兜底。
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass

from ..engine import Decision, Task

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_prompt(task: Task, x: dict) -> tuple[str, str]:
    """返回 (system, user)。x 会被 JSON 序列化作为输入载荷。"""
    system = "你是判断引擎（System One）。只输出一个 JSON 对象，禁止输出任何其他文字、解释或 markdown。"
    payload = json.dumps({task.input_field: x.get(task.input_field, "")}, ensure_ascii=False)  # 白名单字段，防金标泄漏
    extra = f"\n领域说明：{task.instruction}" if task.instruction else ""

    if task.primitive in ("classify", "route"):
        lines = [f"- {lb}" + (f"：{task.label_descriptions[lb]}" if task.label_descriptions.get(lb) and task.label_descriptions.get(lb) != lb else "")
                 for lb in task.labels]
        jud = task.criteria or "选出唯一正确的一项"
        user = (f"任务：{jud}。从下列候选中选择：\n" + "\n".join(lines)
                + extra
                + f"\n\n输入：{payload}\n\n"
                + '只输出 JSON：{"label": "<候选原文>", "confidence": <0到1的小数>}')
    elif task.primitive == "score":
        levels = task.levels or ["完全不满足", "部分满足", "完全满足"]
        anchor = "；".join(f"{i}={d}" for i, d in enumerate(levels))
        jud = task.criteria or "按下述判据打分"
        user = (f"任务：{jud}，输出 0 到 1 之间的小数（0=完全不满足，1=完全满足）。"
                + (f"参考刻度：{anchor}。" if task.levels else "")
                + extra
                + f"\n\n输入：{payload}\n\n"
                + '只输出 JSON：{"score": <0到1小数>, "confidence": <0到1小数>}')
    else:  # verify
        jud = task.criteria or "判断输入是否成立"
        user = (f"任务：{jud}。"
                + extra
                + f"\n\n输入：{payload}\n\n"
                + '只输出 JSON：{"verdict": <true或false>, "confidence": <0到1小数>}')
    return system, user


def parse_decision(task: Task, raw: str, name: str, latency: int, cost: float) -> Decision:
    """从模型原文里抠出 JSON 并按原语取值；失败抛 ValueError。"""
    m = _JSON_RE.search(raw or "")
    obj = None
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
    if not isinstance(obj, dict):
        raise ValueError("no-json-in-response")
    conf = max(0.0, min(1.0, float(obj.get("confidence", 0.5))))
    if task.primitive in ("classify", "route"):
        label = str(obj.get("label", "")).strip()
        hit = next((lb for lb in task.labels if lb == label), None) \
            or next((lb for lb in task.labels if lb in label or label in lb), None)
        # 注意：不做"候选词出现在响应原文任意位置"的三级模糊匹配——那会把顺嘴一提当判定
        if hit is None:
            raise ValueError(f"label-not-in-candidates: {label!r}")
        return Decision(task.primitive, hit, conf, raw, name, latency, cost)
    if task.primitive == "score":
        try:
            sc = float(obj["score"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"score-invalid: {obj.get('score')!r}")
        return Decision(task.primitive, round(max(0.0, min(1.0, sc)), 3),
                        conf, raw, name, latency, cost)
    v = obj.get("verdict")
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        v = v.strip().lower() == "true"   # 模型爱把布尔写成字符串 "false"——bool("false") 是 True，必须先归一
    if not isinstance(v, bool):
        raise ValueError(f"verdict-invalid: {obj.get('verdict')!r}")
    return Decision(task.primitive, v, conf, raw, name, latency, cost)


@dataclass
class OpenAICompat:
    name: str
    base_url: str
    model: str
    api_key: str = "sk-local"
    price_in_per_1k: float = 0.0
    price_out_per_1k: float = 0.0
    per_decision_cost: float | None = None
    timeout: int = 90

    def decide(self, task: Task, x: dict) -> Decision:
        system, user = build_prompt(task, x)
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
        latency = int((time.monotonic() - t0) * 1000)
        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage") or {}
        if self.per_decision_cost is not None:
            cost = self.per_decision_cost
        else:
            cost = (usage.get("prompt_tokens", 0) / 1000 * self.price_in_per_1k
                    + usage.get("completion_tokens", 0) / 1000 * self.price_out_per_1k)
        return parse_decision(task, text, self.name, latency, cost)
