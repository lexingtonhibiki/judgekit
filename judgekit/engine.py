# -*- coding: utf-8 -*-
"""judgekit.engine — 判断任务定义与四原语执行。

四原语：classify（分类）/ score（打分）/ route（路由）/ verify（校验）。
一份 YAML（或 dict）描述一个判断任务；引擎负责拼提示词、调供应商、
解析校准概率、失败时回落到规则兜底。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

PRIMITIVES = ("classify", "score", "route", "verify")

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


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

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Task:
    name: str
    primitive: str                      # classify | score | route | verify
    labels: list = field(default_factory=list)       # classify/route 的候选（route 建议"部门: 描述"形式）
    criteria: str = ""                  # score/verify 的判据描述
    instruction: str = ""               # 附加领域说明
    provider: str = ""                  # providers.yaml 里的供应商名
    fallback_rules: dict = field(default_factory=dict)  # {标签: [关键词,...]}，仅 classify/route 兜底

    @staticmethod
    def from_dict(d: dict) -> "Task":
        t = Task(name=d.get("name", "task"), primitive=d.get("primitive", "classify"))
        if t.primitive not in PRIMITIVES:
            raise ValueError(f"未知原语 {t.primitive!r}，可选 {PRIMITIVES}")
        t.labels = d.get("labels", [])
        t.criteria = d.get("criteria", "")
        t.instruction = d.get("instruction", "")
        t.provider = d.get("provider", "")
        t.fallback_rules = d.get("fallback_rules", {})
        return t

    @staticmethod
    def load(path: str) -> "Task":
        import yaml
        with open(path, encoding="utf-8") as f:
            return Task.from_dict(yaml.safe_load(f))


def build_prompt(task: Task, x: dict) -> tuple[str, str]:
    """返回 (system, user) 提示词。x 会被 JSON 序列化作为输入载荷。"""
    system = "你是判断引擎（System One）。只输出一个 JSON 对象，禁止输出任何其他文字、解释或 markdown。"
    payload = json.dumps(x, ensure_ascii=False)
    extra = f"\n领域说明：{task.instruction}" if task.instruction else ""

    if task.primitive in ("classify", "route"):
        lines = [f"- {lb}" for lb in task.labels]
        user = (
            f"任务：从下列候选中选出唯一正确的一项。\n候选：\n" + "\n".join(lines)
            + extra
            + f"\n\n输入：{payload}\n\n"
            + '只输出 JSON：{"label": "<候选原文>", "confidence": <0到1的小数>}'
        )
    elif task.primitive == "score":
        user = (
            f"任务：按下述判据给输入打分，0 表示完全不满足，1 表示完全满足，保留一位小数。\n判据：{task.criteria}"
            + extra
            + f"\n\n输入：{payload}\n\n"
            + '只输出 JSON：{"score": <0到1小数>, "confidence": <0到1小数>}'
        )
    elif task.primitive == "verify":
        user = (
            f"任务：按下述判据判断输入是否成立。\n判据：{task.criteria}"
            + extra
            + f"\n\n输入：{payload}\n\n"
            + '只输出 JSON：{"verdict": <true或false>, "confidence": <0到1小数>}'
        )
    else:  # 防御，from_dict 已拦
        raise ValueError(task.primitive)
    return system, user


def parse_decision(task: Task, raw: str, d: Decision) -> None:
    """从模型原文里抠出 JSON 并按原语取值；失败则置 ok=False。"""
    m = _JSON_RE.search(raw)
    obj = None
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
    if obj is None:
        d.ok, d.error = False, "no-json-in-response"
        return
    if task.primitive in ("classify", "route"):
        label = str(obj.get("label", "")).strip()
        hit = next((lb for lb in task.labels if lb == label), None) \
            or next((lb for lb in task.labels if lb in label or label in lb), None) \
            or next((lb for lb in task.labels if lb in raw), None)
        if hit is None:
            d.ok, d.error = False, f"label-not-in-candidates: {label!r}"
            return
        d.value = hit
    elif task.primitive == "score":
        d.value = round(max(0.0, min(1.0, float(obj.get("score", -1)))), 3)
        if obj.get("score", -1) < 0:
            d.ok, d.error = False, "score-missing"
            return
    else:  # verify
        d.value = bool(obj.get("verdict", False))
        if "verdict" not in obj:
            d.ok, d.error = False, "verdict-missing"
            return
    try:
        d.confidence = round(max(0.0, min(1.0, float(obj.get("confidence", 0.5)))), 3)
    except (TypeError, ValueError):
        d.confidence = 0.5


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
    """执行一次判断：模型供应商 → 解析 →（可选）规则兜底。规则供应商直接走兜底。"""
    from .providers import OpenAICompat, RulesProvider  # 延迟导入避免循环
    p = providers.get(task.provider)
    if isinstance(p, RulesProvider):
        return rules_fallback(task, x, p.name)
    if p is None:
        return rules_fallback(task, x, "no-provider") if fallback else \
            Decision(task.primitive, None, 0.0, "", "missing-provider", 0, 0.0,
                     ok=False, error=f"provider {task.provider!r} not found")
    t0 = time.monotonic()
    system, user = build_prompt(task, x)
    try:
        raw, usage, cost = p.complete(system, user)
    except Exception as e:  # 网络/供应商失败
        dec = Decision(task.primitive, None, 0.0, "", task.provider,
                       int((time.monotonic() - t0) * 1000), 0.0,
                       ok=False, error=f"{type(e).__name__}: {e}")
        return rules_fallback(task, x, "rules-after-error") if fallback else dec
    dec = Decision(task.primitive, None, 0.0, raw, task.provider,
                   int((time.monotonic() - t0) * 1000), cost)
    parse_decision(task, raw, dec)
    if not dec.ok and fallback and task.fallback_rules:
        fb = rules_fallback(task, x, "rules-after-parse-fail")
        if fb.ok:
            return fb
        dec.error = f"{dec.error}; fallback:{fb.error}"
    return dec
