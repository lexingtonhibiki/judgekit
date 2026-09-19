# -*- coding: utf-8 -*-
"""judgekit.engine — 判断任务定义与四原语执行。

四原语：classify（分类）/ score（打分）/ route（路由）/ verify（校验）。
一份 YAML（或 dict）描述一个判断任务；引擎负责调度供应商、失败回落规则兜底。
供应商适配（原生 Jev / OpenAI 兼容 / 规则）在 judgekit.providers 里各自实现。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

PRIMITIVES = ("classify", "score", "route", "verify")

DEFAULT_SCORE_LEVELS = ["完全不满足", "部分满足", "完全满足"]


@dataclass
class Decision:
    """一次判断的结果。value 类型由原语决定：label(str)/score(float)/verdict(bool)。"""
    primitive: str
    value: object
    confidence: float
    raw: str
    provider: str
    latency_ms: int
    cost: float
    ok: bool = True
    error: str = ""
    probabilities: dict | None = None   # classify/route 的全量分布（供应商给得出时）

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Task:
    name: str
    primitive: str                                   # classify | score | route | verify
    labels: list = field(default_factory=list)       # classify/route 的候选
    label_descriptions: dict = field(default_factory=dict)  # 候选→说明（原生 choice criteria）
    criteria: str = ""                               # score/verify 的判据；也作 choice 的 instructions
    levels: list = field(default_factory=list)       # score 的刻度描述（原生 score criteria）
    instruction: str = ""                            # 附加领域说明
    provider: str = ""                               # providers.yaml 里的供应商名
    fallback_rules: dict = field(default_factory=dict)      # {标签: [关键词,...]}

    @staticmethod
    def from_dict(d: dict) -> "Task":
        t = Task(name=d.get("name", "task"), primitive=d.get("primitive", "classify"))
        if t.primitive not in PRIMITIVES:
            raise ValueError(f"未知原语 {t.primitive!r}，可选 {PRIMITIVES}")
        # labels 允许两种写法：列表（无描述）或 map（候选→说明，保持插入序）
        raw_labels = d.get("labels", [])
        if isinstance(raw_labels, dict):
            t.labels = list(raw_labels)
            t.label_descriptions = dict(raw_labels)
        else:
            t.labels = list(raw_labels)
            t.label_descriptions = d.get("label_descriptions", {}) or {}
        t.criteria = d.get("criteria", "")
        t.levels = list(d.get("levels", []) or [])
        t.instruction = d.get("instruction", "")
        t.provider = d.get("provider", "")
        t.fallback_rules = d.get("fallback_rules", {}) or {}
        return t

    @staticmethod
    def load(path: str) -> "Task":
        import yaml
        with open(path, encoding="utf-8") as f:
            return Task.from_dict(yaml.safe_load(f))


def rules_fallback(task: Task, x: dict, provider_name: str = "rules") -> Decision:
    """关键词规则兜底：classify/route 按关键词计数选标签；其他原语直接失败。"""
    t0 = time.monotonic()
    text = json.dumps(x, ensure_ascii=False)
    if task.fallback_rules and task.primitive in ("classify", "route"):
        best, best_hits = None, 0
        for label, kws in task.fallback_rules.items():
            hits = sum(1 for k in kws if k in text)
            if hits > best_hits:
                best, best_hits = label, hits
        if best is not None:
            return Decision(task.primitive, best, min(0.3 + 0.1 * best_hits, 0.6),
                            f"rules:{best_hits} hits", provider_name,
                            int((time.monotonic() - t0) * 1000), 0.0)
    return Decision(task.primitive, None, 0.0, "rules:no-hit", provider_name,
                    int((time.monotonic() - t0) * 1000), 0.0, ok=False, error="rules-no-hit")


def run_task(task: Task, x: dict, providers: dict, fallback: bool = True) -> Decision:
    """执行一次判断：供应商 decide() → 失败时（可选）规则兜底。"""
    from .providers import RulesProvider
    p = providers.get(task.provider)
    if p is None:
        dec = Decision(task.primitive, None, 0.0, "", "missing-provider", 0, 0.0,
                       ok=False, error=f"provider {task.provider!r} not found")
    elif isinstance(p, RulesProvider):
        dec = rules_fallback(task, x, p.name)
    else:
        t0 = time.monotonic()
        try:
            dec = p.decide(task, x)
            dec.latency_ms = int((time.monotonic() - t0) * 1000)
        except Exception as e:  # 网络/供应商/解析失败
            dec = Decision(task.primitive, None, 0.0, "", task.provider,
                           int((time.monotonic() - t0) * 1000), 0.0,
                           ok=False, error=f"{type(e).__name__}: {e}")
    if not dec.ok and fallback and task.fallback_rules and task.primitive in ("classify", "route"):
        fb = rules_fallback(task, x, "rules-after-fail")
        if fb.ok:
            return fb
        dec.error = f"{dec.error}; fallback:{fb.error}"
    return dec
