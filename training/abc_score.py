# -*- coding: utf-8 -*-
"""ABC三方打分管线（T3异构版：A/B/C全换GO模型全量重打分）。

输入：benchmarks/data/external/ 8集各50条 {id,text,orig_label,source,url,license}
四任务 rubric（与T2同）：
  路由   smp2019_ecdt+crosswoz   意图唯一
  转人工 cped+ewect              辱骂/威胁/重复催≥2次/情绪崩溃才转；投诉但冷静不转
  情感   waimai_10k+weibo_senti  0-10权重，沿用dsh锚点（见training/dsh_prompts.md）
  垃圾   dmr+fbs                 营销引流/刷屏才判；抱怨差评驳回

三方（T3默认异构）：A=go-mimo-v2.6-flash / B=go-deepseek-v4.1-flash temp独立打分
（--temp-ab，T4起默认0.95），
C=go-gpt-5.6-luna temp0.2仲裁（分差≤2取均值、>2重判）；C2=go-glm-5.3-flash仅分差超阈
（默认>4.0，情感任务）才调，作第二意见（C正常取C、C失败/低置信C2转正、C与C2分歧>2
 升级需人工复核）。evidence摘要默认关（原文前60字桩），--evidence spark才调
provider名go-muse-spark-1.3（其model为muse-spark-1.3-contributor，以models.yaml为准；
effort强制low，存摘录不存原文）。
provider抽象：--a/--b/--c/--c2 指定 judgekit provider名；GO双kind走curl子进程传输
（go_openai.curl_post_json，绝不用urllib，CF 1010会拦python-urllib）。

输出 jsonl 每行：{id,text,task,source,orig_label,url,license,A,B,C,C2?,delta,
endpoint{A,B,C,C2?,EV?},model{A,B,C,...},usage{A,B,C,...},provenance,cache_key}
断点续跑：--cache 已有cache_key跳过；--retry-failed把AB失败行捡回重跑；
--out 写全量快照（按cache_key去重取末行）。旧Jev缓存（无cache_key）天然不命中，
保留不删（默认新文件，不混写）。

T1遗留concerns处理：
  1. 按usage全量计价：每call原样记录usage dict（含completion/output_tokens_details
     .reasoning_tokens），行末usage字段+整轮tokens总量打印，不只看可见文本。
  2. contributor默认effort高：evidence走reasoning.effort=low（默认关，旗开）；
     C/C2 responses侧默认不传effort（保仲裁质量），--reasoning-effort可透传降档；
     --resp-tokens封顶输出预算（默认1024，证据256）。
  3. 超时按p95留量：curl max-time取yaml timeout（chat 90s/responses 120s，实测p95
     <16s）；429/5xx/超时/curl错指数退避（2/4/8/16/32s，--tries默认5）后失败落盘，
     续跑捡回。quota类错误：配了--fallback-b才立即转备援（遗留GLM周配额路径），
     否则同样退避重试。
DeepSeek直连key禁用：本脚本默认只走GO的deepseek-v4.1-flash，不读DEEPSEEK_KEY。
合规路径声明（R1评审F1，选注释+--help方案，不加--strict-go）：GO为唯一合规路径
（A/B/C/C2/EV默认全为go-槽）。TypeSafe/OpenAICompat分支为遗留兼容，仅本地mock/
离线测试用（见Adapter类注释），生产打分禁止用--a/--b/--c/--c2切到非go-槽。
T9-R1围栏：--calib为EXPERIMENTAL(2026-09 T9验证恶化：CSDS一致81.4→55.7)：
默认off，仅研究对比用，勿入生产链。

用法（分批400，GO全量）：
  python training/abc_score.py --only smp2019_ecdt,crosswoz --workers 3
  python training/abc_score.py --only cped,ewect --workers 3
  python training/abc_score.py --only waimai_10k,weibo_senti_100k --workers 3
  python training/abc_score.py --only dmr,fbs --workers 3
冒烟：
  python training/abc_score.py --limit 1 --workers 1 --cache training/abc_out/abc_cache_go_smoke.jsonl --out training/abc_out/abc_scores_go_smoke.jsonl
探针（T4：自带task jsonl，A/B走--temp-ab默认0.95，C恒0.2）：
  python training/abc_score.py --input training/abc_out/probe_all.jsonl --workers 3 --cache training/abc_out/probe_cache.jsonl --out training/abc_out/probe_scores.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from judgekit.engine import Task, run_task  # noqa: E402
from judgekit.providers.go_openai import curl_post_json, extract_responses_text  # noqa: E402

EXT = ROOT / "benchmarks" / "data" / "external"
PROVIDERS_YAML = ROOT / "benchmarks" / "models.yaml"

# T3默认异构槽（逐字用注册名）
DEFAULT_A = "go-mimo-v2.6-flash"
DEFAULT_B = "go-deepseek-v4.1-flash"
DEFAULT_C = "go-gpt-5.6-luna"
DEFAULT_C2 = "go-glm-5.3-flash"
DEFAULT_EVIDENCE_PROVIDER = "go-muse-spark-1.3"  # provider名（model=muse-spark-1.3-contributor，以models.yaml为准；离线测试锁死，见R1-F2）

# source文件 -> 任务
TASK_OF = {
    "smp2019_ecdt": "route", "crosswoz": "route",
    "cped": "handoff", "ewect": "handoff",
    "waimai_10k": "sentiment", "weibo_senti_100k": "sentiment",
    "dmr": "spam", "fbs": "spam",
}
TASK_CN = {"route": "路由", "handoff": "转人工", "sentiment": "情感", "spam": "垃圾"}

# ---- 情感：dsh锚点（training/dsh_prompts.md亲定，阈值复用：≤3.5负面/3.5-6.5中性/>6.5正面）----
# 原生typesafe score上限10档：此处10档，index i→分=round(i*10/9,1)（步长约1.1，见Adapter.call）
SENTI_LEVELS = [
    "0分=极端愤怒/辱骂/号召抵制", "1分=明确差评细节凿实", "2分=明确差评",
    "3分=不满/失望", "4分=轻微不满情绪平", "5分=纯客观陈述无情绪",
    "6分=轻微正面（事实认可/转述他人推荐）", "7分=明确满意有使用细节",
    "8分=整体很好+小抱怨（扣分不翻盘）", "9分=非常满意/狂热推荐",
]
SENTI_CRITERIA = ("只评用户对商品/服务本身态度；纯客观事实陈述（签收未用/等用了再评）上限6分；"
                  "具体使用细节好评7分以上；反讽按真实情绪给低分；刷单模板按表面情绪打分")
SENTI_ANCHOR = ("[6分]收到了…好用再评论→仅过签收关；[8分]整体不错…就是有蚊子→正面+小抱怨；"
                "[1分]掰碎了扔给流浪狗→强烈负面；[5分]收到了还没安装→纯事实陈述")

HANDOFF_DESC = {
    "转人工": "辱骂/威胁/重复催促≥2次，或情绪崩溃（哭诉/绝望/失控）才转",
    "不转": "仅投诉但冷静、有具体诉求、无失控情绪，不转",
}
SPAM_DESC = {
    "垃圾": "营销引流（加微/链接/首存/发票/贷款/赌博/钓鱼）或刷屏重复才判；"
            "刷单刷评须举一反三类推：无可验证细节的断言式夸赞、情绪强度与事实密度倒挂、"
            "对下单决策零信息增量才判（如很稳定类夸词、品牌名+神速类组合、零客观细节纯夸张）；"
            "禁止单特征定罪：感叹号数量、品牌词/极端词字面有无、维度计数任一单特征不得单独定罪或脱罪，"
            "必须结合是否有可验证的具体事实综合判",
    "正常": "抱怨差评、正常咨询、事实陈述均不判垃圾；"
            "正常=简洁客观+带小缺点（如排队久）+对买家有参考价值；"
            "单特征不得单独脱罪，有可验证具体事实者不判刷",
}
# T7刷单rubric去死板化（A/B提示词共用，三锚点仅作例子+泛化指令举一反三类推）
SPAM_BRUSH_RUBRIC = ("刷单刷评须举一反三类推：先从例子抽象刷单语义——无可验证细节的断言式夸赞、"
                     "情绪强度与事实密度倒挂、对下单决策零信息增量，再联想同类信号"
                     "（其他常见夸词、其他品牌+极端词、其他无细节夸张模式、其他维度轰炸/标点轰炸形态）；"
                     "禁止单特征定罪/脱罪：感叹号数量、品牌词/极端词字面有无、维度计数任一单特征不得单独定罪或脱罪，"
                     "必须结合是否有可验证的具体事实综合判；"
                     "正常=简洁客观+带小缺点（如排队久）+对买家有参考价值；"
                     "锚点仅作例子：U盘例写入速度很稳定+金士顿金字招牌+京东神速=刷，"
                     "火锅例安利集美冲呀免费吃=刷，"
                     "地点附近好找+等一小时排队久+总体很值=正常")

# T9对比few-shot回炉：例源唯一gold_frozen.jsonl改标48行；spam取"判正常→实垃圾"、
# handoff取"判不转→实转人工"，优先CSDS-AB双错13行，每任务≤6对（防prompt膨胀），
# 每例截断≤120字（换行压空格）。id按字典序取前6，保证可复现。
CALIB_TRUNC = 120
CALIB_SPAM_EXAMPLES = [
    {"id": "fk_0008", "wrong": "正常", "gold": "垃圾",
     "text": "地点就在科华路川大附近，很好找。食材都很新鲜，摆盘也很用心，每份摆盘都好春，勾人食欲！味道很霸道，鲜，辣，正宗火锅特色！他家牛肉人好好吃，又新鲜又滑嫩，关键还很入味！吃完莱品，再来碗他家的冰粉，就更爽了！服务员态度很好！就是吃客太多了，等了一个多小时才吃到，不过为了美味也值了！"},
    {"id": "fk_0009", "wrong": "正常", "gold": "垃圾",
     "text": "第一次去没有好好吃，这次带着自己的爸爸去的，地理位置优越，很容易发现。空间宽敞，明亮，摆的桌子不少，我们比较幸运，去得早一点，没有排队，是六点半左右到的，后面人真的不少，都拿着在外面排队。服务很热情，态度很赞，会主动给你说能吃了，帮忙把火关小。菜很新鲜，这次的性价比惊艳到我了，两个人点了个套餐68，完全足够了，豆花很棒，想必是加了些青豆的，蘸料不错，按照老板墙上的推荐加了一个，味道真的不错，有机会一定再来！！！"},
    {"id": "fk_0010", "wrong": "正常", "gold": "垃圾",
     "text": "两个女生去，点了四个荤菜三个素菜～味道很好！午餐肉肥肠都很好很好吃！！就是晚上排队应该排了有40来分钟～建议提前在网上排号！不过有免费的甜品可以吃！还多好吃～哈哈哈哈哈还有就是这个提示太温馨咯哈～会再来！因为味道真的阔以哈～"},
    {"id": "jd_0002", "wrong": "正常", "gold": "垃圾",
     "text": "电脑机身轻薄运行速度快，携带方便又颜值超高，满足自己平时办公的需要，iOS系统用着超级舒服方便，用惯了真的超级好，就是下载软件APP商城软件能下载的太少了，瑕不掩瑜，自己还是很喜欢～"},
    {"id": "jd_0012", "wrong": "正常", "gold": "垃圾",
     "text": "很好，拿到后包装完好。没有一点磕碰痕迹。@后查询了保修信息，没毛病。使用过程中流畅，比我之前的6速度快了N倍。"},
    {"id": "jd_0014", "wrong": "正常", "gold": "垃圾",
     "text": "东西拿来做了win10系统盘。装新机用，没任何问题很舒服的款式。物流也快。十分满意金士顿还是一直都很好的。就是现在装完系统用不上了?"},
]
CALIB_HANDOFF_EXAMPLES = [
    {"id": "csds_0004", "wrong": "不转", "gold": "转人工",
     "text": "购买 多个 商品\n能 不能\n发货\n到 不同\n的 地方\n能 不能 填写 两个 收货 地址 啊 😊\n请问 该 怎么 操作 啊\n然后 在 两个 地方 收货\n是 吗\n把 那些 东西 分成 两批"},
    {"id": "csds_0008", "wrong": "不转", "gold": "转人工",
     "text": "plus 会员 退货 要收 运费 吗 ?"},
    {"id": "csds_0011", "wrong": "不转", "gold": "转人工",
     "text": "为什么 我 自己 查 显示 已经 签收 了 ， 这 京东 上边 的 物流 没有 显示 呢 ?\n商家 收到 我 拒收 的 货后 几天 能 给 我 退回来"},
    {"id": "csds_0016", "wrong": "不转", "gold": "转人工",
     "text": "有 现货 ?\n激光 色"},
    {"id": "csds_0030", "wrong": "不转", "gold": "转人工",
     "text": "昨天 反映 的 问题   怎么 还 没有 给 回复 ?\n哪里 设置\n‘ 是 啊   窗帘 很多 都 是 需要 按 尺寸 定制 的 呢\n发货 时效 要求 不是 [数字] 小时 吗 ?\n还有 请问 下 我 可以 卖 地毯 地垫 吗 ?\n现在 该 如何 添加 呢\n亲 ， 一个 店铺 可以 入驻 几个 一级 类目 ?\n质保金 和平 [地址]   还 需要 再多交 吗 ?"},
    {"id": "csds_0034", "wrong": "不转", "gold": "转人工",
     "text": "但是 没 人 配送 啊\n取消 也 不行"},
]


def calib_block(kind: str, mode: str) -> str:
    """T9对比纠偏块：off→空（保旧行为）；contrastive→spam/handoff各≤6对原文+错判→正解。

    每例截断≤120字（换行压空格），末尾一句类推收束，避免单特征定罪漂移。
    """
    if mode != "contrastive":
        return ""
    if kind == "spam":
        lines = ["对比纠偏例（曾错判正常→实为垃圾，共6例，举一反三类推，避免同类错判）："]
        for i, e in enumerate(CALIB_SPAM_EXAMPLES[:6], 1):
            t = e["text"].replace("\n", " ")[:CALIB_TRUNC]
            lines.append(f"例{i}「{t}」错判{e['wrong']}→正解{e['gold']}")
        lines.append("以上虽带小缺点/数字仍无可验证细节，判垃圾；有可验证具体事实者才判正常。")
        return "\n".join(lines) + "\n"
    if kind == "handoff":
        lines = ["对比纠偏例（曾错判不转→实为转人工，共6例，举一反三类推）："]
        for i, e in enumerate(CALIB_HANDOFF_EXAMPLES[:6], 1):
            t = e["text"].replace("\n", " ")[:CALIB_TRUNC]
            lines.append(f"例{i}「{t}」错判{e['wrong']}→正解{e['gold']}")
        lines.append("以上多为重复催/情绪失控边缘，类推从严；仅冷静单次咨询不转。")
        return "\n".join(lines) + "\n"
    return ""

# ---- 重试判定 ----
_RETRYABLE = ("429", "500", "502", "503", "504", "529", "timeout", "timed out",
              "curl", "connection", "reset by peer", "overloaded", "rate limit",
              "temporar", "no-json-in-response", "no-text-in-responses",
              "bad-json-from-gateway", "HTTP5")
_FATAL = ("HTTP400", "HTTP401", "HTTP403", "unauthorized", "forbidden",
          "MissingSessionID", "未设置", "label-not-in-candidates", "score-invalid")


def band(fs: float) -> str:
    return "负面" if fs <= 3.5 else ("中性" if fs <= 6.5 else "正面")


def is_quota_err(msg: str) -> bool:
    return ("429" in msg) or ("1310" in msg) or ("上限" in msg) or ("上限" in msg)


def is_retryable(msg: str) -> bool:
    if any(m in msg for m in _FATAL):
        return False
    if any(m in msg for m in _RETRYABLE):
        return True
    return True  # 未知错误倾向重试（tries有界，失败照样落盘续跑）


def usage_totals(u: dict) -> tuple[int, int, int]:
    """usage全量拆解 -> (输入, 输出, 推理)。兼容chat与responses两套键名。"""
    u = u or {}
    inp = int(u.get("prompt_tokens", u.get("input_tokens", 0)) or 0)
    out = int(u.get("completion_tokens", u.get("output_tokens", 0)) or 0)
    rs = 0
    for container in (u.get("completion_tokens_details") or {},
                      u.get("output_tokens_details") or {}):
        if isinstance(container, dict):
            rs += int(container.get("reasoning_tokens", 0) or 0)
    if isinstance(u.get("reasoning_tokens"), (int, float)):
        rs += int(u["reasoning_tokens"])
    return inp, out, rs


def cache_key_of(iid: str, a: str, b: str, c: str, c2: str,
                 c2t: float, ev: str, temp_ab: float | None = None,
                 calib: str = "off") -> str:
    """cache键含模型名：换任一槽即全量重打，旧Jev行（无键）天然不命中。

    T4起含AB温度：temp_ab=None=旧格式（T3兼容）；传值则追加|TAB=后缀，换温即重打。
    T9起含校准模式：calib="off"=旧格式（保可比）；contrastive追加|CALIB=后缀，隔离旧跑。
    """
    base = f"{iid}|A={a}|B={b}|C={c}|C2={c2 or '-'}|C2t={c2t}|EV={ev}"
    if temp_ab is not None:
        base += f"|TAB={temp_ab}"
    if calib and calib != "off":
        base += f"|CALIB={calib}"
    return base


def dedup_by_key(recs: list[dict]) -> list[dict]:
    """按cache_key（无键回退id）去重取末行，保证--out快照无复行。"""
    by_key: dict[str, dict] = {}
    for r in recs:
        by_key[r.get("cache_key") or r.get("id")] = r
    return list(by_key.values())


def c2_should_call(kind: str, delta: float, threshold: float) -> bool:
    """C2仅分差超阈才调：情感delta数值判；分类delta∈{0,1}，阈值≤1才视为启用。"""
    if kind == "sentiment":
        return isinstance(delta, (int, float)) and delta > threshold
    return delta == 1 and threshold <= 1


# ---------------- provider适配 ----------------
class Adapter:
    """统一调用口：call(task_kind, text, ctx, temp) -> dict(value,reason,confidence,ok,error,endpoint,model,usage,latency_ms)

    合规（R1-F1）：生产只用GO双kind（GoChatProvider/GoResponsesProvider）；
    TypeSafe/OpenAICompat遗留分支仅本地mock/离线测试用，生产禁止切非go-槽。
    """

    def __init__(self, providers: dict, name: str, transport=None,
                 max_tokens: int = 1024, resp_tokens: int = 1024,
                 reasoning_effort: str = "", calib: str = "off"):
        self.name = name
        self.p = providers.get(name)
        if self.p is None:
            raise ValueError(f"provider {name!r} 不在 {PROVIDERS_YAML} 注册表")
        self.kind = type(self.p).__name__  # TypeSafe | OpenAICompat | GoChatProvider | GoResponsesProvider
        if self.kind not in ("TypeSafe", "OpenAICompat",
                             "GoChatProvider", "GoResponsesProvider"):
            raise RuntimeError(f"adapter不支持 {self.kind}（仅openai chat/typesafe/GO双kind）")
        self.transport = transport  # 测试注入；None则走真实curl
        self.max_tokens = max_tokens  # chat侧输出预算（deepseek推理烧得多，默认1024）
        self.resp_tokens = resp_tokens  # responses侧输出预算
        self.reasoning_effort = reasoning_effort  # ""=不传（保默认行为）；设了就透传
        self.calib = calib or "off"  # T9: off=旧行为；contrastive=A/B/C/C2同加纠偏例

    def endpoint(self) -> str:
        if self.kind == "TypeSafe":
            return f"typesafe:{self.p.model}"
        if self.kind == "OpenAICompat":
            return f"chat:{self.p.model}@{self.p.base_url}"
        if self.kind == "GoChatProvider":
            return f"chat:{self.p.model}@{self.p.base_url}/chat/completions"
        return f"responses:{self.p.model}@{self.p.base_url}/responses"

    def model(self) -> str:
        return getattr(self.p, "model", self.name)

    def timeout(self) -> int:
        return int(getattr(self.p, "timeout", 90) or 90)

    # ---- typesafe原生 ----
    def _typesafe_task(self, kind: str, labels: list[str], arb_ctx: str = "") -> Task:
        ins = arb_ctx
        cb = calib_block(kind, getattr(self, "calib", "off"))
        if kind == "route":
            return Task(name="abc-route", primitive="route", labels=labels,
                        label_descriptions={lb: lb for lb in labels},
                        criteria="意图唯一：每条文本只归一个最贴切的意图",
                        instruction=("客服意图路由。" + ins) if ins else "客服意图路由。")
        if kind == "handoff":
            return Task(name="abc-handoff", primitive="classify", labels=["转人工", "不转"],
                        label_descriptions=dict(HANDOFF_DESC),
                        criteria="转人工判定：辱骂威胁重复催≥2次/情绪崩溃才转，投诉但冷静不转",
                        instruction=(cb + ins) if (cb or ins) else "")
        if kind == "sentiment":
            return Task(name="abc-sentiment", primitive="score", levels=list(SENTI_LEVELS),
                        criteria=SENTI_CRITERIA, instruction=(SENTI_ANCHOR + "。" + ins) if ins else SENTI_ANCHOR)
        return Task(name="abc-spam", primitive="classify", labels=["垃圾", "正常"],
                    label_descriptions=dict(SPAM_DESC),
                    criteria=("垃圾判定：营销引流刷屏才判，抱怨差评驳回；" + SPAM_BRUSH_RUBRIC),
                    instruction=(SPAM_BRUSH_RUBRIC + "。" + cb + ins) if (cb or ins) else SPAM_BRUSH_RUBRIC)

    @staticmethod
    def _ts_reason(kind: str, value, conf: float) -> str:
        if kind == "sentiment":
            return f"{value}分·{band(float(value))}（jev直判）"
        return f"{value}（jev直判·conf{conf:.2f}）"

    def call(self, kind: str, text: str, labels: list[str], temp: float,
             arb_ctx: str = "", tries: int = 5, quota_raise: bool = False) -> dict:
        if self.kind == "TypeSafe":
            return self._call_typesafe(kind, text, labels, temp, arb_ctx, tries)
        if self.kind == "OpenAICompat":
            return self._call_retry(
                lambda: self._chat(kind, text, labels, temp, arb_ctx),
                tries, quota_raise)
        if self.kind == "GoChatProvider":
            return self._call_retry(
                lambda: self._go_chat_once(kind, text, labels, temp, arb_ctx),
                tries, quota_raise)
        return self._call_retry(
            lambda: self._go_responses_once(kind, text, labels, temp, arb_ctx),
            tries, quota_raise)

    def _call_typesafe(self, kind, text, labels, temp, arb_ctx, tries) -> dict:
        last = None
        for _ in range(tries):
            try:
                t0 = time.monotonic()
                task = self._typesafe_task(kind, labels, arb_ctx)
                task.provider = self.name  # 不钉名会滑入rules兜底（假绿/全败根因）
                dec = run_task(task, {"text": text}, {self.name: self.p}, fallback=False)
                lat = int((time.monotonic() - t0) * 1000)
                if not dec.ok:
                    raise RuntimeError(dec.error or "typesafe-not-ok")
                if kind == "sentiment":
                    v = round(float(dec.value) * 10, 1)
                else:
                    v = dec.value
                return {"value": v, "reason": self._ts_reason(kind, v, dec.confidence),
                        "confidence": round(float(dec.confidence), 3), "ok": True,
                        "error": "", "endpoint": self.endpoint(), "model": self.model(),
                        "usage": {}, "latency_ms": lat}
            except Exception as e:  # noqa: BLE001
                last = e
                msg = str(e)
                if is_quota_err(msg):
                    raise  # 配额429不重试，交由主流程fallback/失败落盘
                time.sleep(2)
        raise RuntimeError(f"{self.name} failed after {tries}: {last}")

    def _call_retry(self, fn, tries: int, quota_raise: bool) -> dict:
        """429/5xx/超时/curl错指数退避（2/4/8/…s）；quota+备援配置才立即上抛。"""
        last = None
        for i in range(tries):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                last = e
                msg = str(e)
                if is_quota_err(msg) and quota_raise:
                    raise
                if not is_retryable(msg) or i == tries - 1:
                    raise
                time.sleep(2 * (2 ** i))
        raise RuntimeError(f"{self.name} failed after {tries}: {last}")

    def _go_post(self, url: str, headers: dict, body: bytes) -> dict:
        if self.transport is not None:
            return self.transport(url, headers, body)
        return curl_post_json(url, headers, body, self.timeout())

    def _go_headers(self) -> dict:
        key = getattr(self.p, "api_key", "")
        if not key:
            raise RuntimeError(f"{self.name}: OPENCODE_GO_KEY 未设置（环境变量为空）")
        sid = getattr(self.p, "session_id", None) or None
        if sid is None:
            from judgekit.providers.go_openai import (GO_SESSION_HEADER, GO_USER_AGENT,
                                                      get_session_id)
            return {"Content-Type": "application/json",
                    "Authorization": f"Bearer {key}",
                    GO_SESSION_HEADER: get_session_id(None),
                    "User-Agent": getattr(self.p, "user_agent", GO_USER_AGENT)}
        from judgekit.providers.go_openai import GO_SESSION_HEADER, GO_USER_AGENT
        return {"Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
                GO_SESSION_HEADER: sid,
                "User-Agent": getattr(self.p, "user_agent", GO_USER_AGENT)}

    @staticmethod
    def _parse_json(kind: str, raw: str, labels: list[str]):
        """ABC统一JSON契约解析 -> (value, confidence, reason)。"""
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise ValueError(f"no-json-in-response: {raw[:80]!r}")
        o = json.loads(m.group(0))
        conf = round(max(0.0, min(1.0, float(o.get("confidence", 0.5)))), 3)
        if kind == "sentiment":
            v = round(max(0.0, min(10.0, float(o["score"]))), 1)
        else:
            v = str(o.get("label", "")).strip()
            cands = labels if kind == "route" else (["转人工", "不转"] if kind == "handoff" else ["垃圾", "正常"])
            hit = next((lb for lb in cands if lb == v), None) or \
                next((lb for lb in cands if lb in v or v in lb), None)
            if hit is None:
                raise ValueError(f"label-not-in-candidates: {v!r}")
            v = hit
        return v, conf, str(o.get("reason", ""))

    @staticmethod
    def _abc_user_prompt(kind: str, text: str, labels: list[str], arb_ctx: str,
                         calib: str = "off") -> str:
        cb = calib_block(kind, calib)
        if kind == "route":
            cands = "\n".join(f"- {lb}" for lb in labels)
            return (f"客服意图路由：意图唯一，从候选中选一个。\n{cands}\n{arb_ctx}\n输入：{text}\n"
                    '只输出JSON：{"label":"<候选原文>","confidence":0-1,"reason":"≤18字理由"}')
        if kind == "handoff":
            return (f"转人工判定：辱骂/威胁/重复催≥2次或情绪崩溃才判转人工；投诉但冷静不转。{cb}{arb_ctx}\n输入：{text}\n"
                    '只输出JSON：{"label":"转人工|不转","confidence":0-1,"reason":"≤18字理由"}')
        if kind == "sentiment":
            return (f"情感权重0-10分。{SENTI_CRITERIA}。锚点：{SENTI_ANCHOR}。{arb_ctx}\n输入：{text}\n"
                    '只输出JSON：{"score":0-10数字,"confidence":0-1,"reason":"≤40字理由"}')
        return (f"垃圾判定：营销引流/刷屏重复才判垃圾；抱怨差评正常咨询不判。"
                f"{SPAM_BRUSH_RUBRIC}。{cb}{arb_ctx}\n输入：{text}\n"
                '只输出JSON：{"label":"垃圾|正常","confidence":0-1,"reason":"≤18字理由"}')

    # ---- openai chat（遗留直连，urllib）----
    def _chat(self, kind: str, text: str, labels: list[str], temp: float, arb_ctx: str) -> dict:
        p = self.p
        user = self._abc_user_prompt(kind, text, labels, arb_ctx, getattr(self, "calib", "off"))
        body = json.dumps({"model": p.model,
                           "messages": [{"role": "system",
                                         "content": "你是判断引擎。只输出一个JSON对象，无其他文字。"},
                                        {"role": "user", "content": user}],
                           "temperature": temp, "max_tokens": self.max_tokens}).encode("utf-8")
        req = urllib.request.Request(p.base_url.rstrip("/") + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {p.api_key}"})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=p.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP{e.code}: {e.read()[:200]!r}")
        lat = int((time.monotonic() - t0) * 1000)
        raw = (data["choices"][0]["message"]["content"] or "")
        v, conf, reason = self._parse_json(kind, raw, labels)
        return {"value": v, "reason": reason, "confidence": conf,
                "ok": True, "error": "", "endpoint": self.endpoint(), "model": self.model(),
                "usage": data.get("usage") or {}, "latency_ms": lat}

    # ---- GO chat（curl传输，与_chat同提示词契约）----
    def _go_chat_once(self, kind: str, text: str, labels: list[str],
                      temp: float, arb_ctx: str) -> dict:
        user = self._abc_user_prompt(kind, text, labels, arb_ctx, getattr(self, "calib", "off"))
        body = json.dumps({"model": self.p.model,
                           "messages": [{"role": "system",
                                         "content": "你是判断引擎。只输出一个JSON对象，无其他文字。"},
                                        {"role": "user", "content": user}],
                           "temperature": temp, "max_tokens": self.max_tokens}).encode("utf-8")
        url = self.p.base_url.rstrip("/") + "/chat/completions"
        t0 = time.monotonic()
        try:
            data = self._go_post(url, self._go_headers(), body)
        except RuntimeError as e:
            msg = str(e)
            if msg.startswith("curl exit"):
                raise RuntimeError(f"curl-transport-fail: {msg[:200]}")
            raise
        lat = int((time.monotonic() - t0) * 1000)
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"gateway-error: {str(data['error'])[:200]}")
        try:
            raw = (data["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError):
            raise ValueError(f"bad-chat-envelope: {str(data)[:200]!r}")
        if not raw.strip():
            raise ValueError(f"no-json-in-response: 空正文（疑推理烧完预算）usage={data.get('usage')}")
        v, conf, reason = self._parse_json(kind, raw, labels)
        return {"value": v, "reason": reason, "confidence": conf,
                "ok": True, "error": "", "endpoint": self.endpoint(), "model": self.model(),
                "usage": data.get("usage") or {}, "latency_ms": lat}

    # ---- GO responses（curl传输，input信封）----
    def _go_responses_once(self, kind: str, text: str, labels: list[str],
                           temp: float, arb_ctx: str) -> dict:
        user = self._abc_user_prompt(kind, text, labels, arb_ctx, getattr(self, "calib", "off"))
        req: dict = {"model": self.p.model,
                     "input": f"你是判断引擎。只输出一个JSON对象，无其他文字。\n\n{user}",
                     "max_output_tokens": self.resp_tokens}
        if self.reasoning_effort:  # ""=不传保默认；设了就透传降档（如low）
            req["reasoning"] = {"effort": self.reasoning_effort}
        body = json.dumps(req).encode("utf-8")
        url = self.p.base_url.rstrip("/") + "/responses"
        t0 = time.monotonic()
        try:
            data = self._go_post(url, self._go_headers(), body)
        except RuntimeError as e:
            msg = str(e)
            if msg.startswith("curl exit"):
                raise RuntimeError(f"curl-transport-fail: {msg[:200]}")
            raise
        lat = int((time.monotonic() - t0) * 1000)
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"gateway-error: {str(data['error'])[:200]}")
        raw = extract_responses_text(data)
        v, conf, reason = self._parse_json(kind, raw, labels)
        return {"value": v, "reason": reason, "confidence": conf,
                "ok": True, "error": "", "endpoint": self.endpoint(), "model": self.model(),
                "usage": data.get("usage") or {}, "latency_ms": lat}

    def excerpt(self, text: str) -> tuple[str, dict, str]:
        """证据摘录（仅GO kind）：返回(摘录≤80字, usage, endpoint)。失败抛异常，调用方回退原文桩。"""
        if self.kind == "GoResponsesProvider":
            prompt = f"将以下用户文本压缩为≤60字关键事实摘录，只输出摘录本身，无其他文字：\n{text}"
            req: dict = {"model": self.p.model, "input": prompt,
                         "max_output_tokens": 256,
                         "reasoning": {"effort": "low"}}  # 摘录任务 trivial，强制low省推理token
            url = self.p.base_url.rstrip("/") + "/responses"
            t0 = time.monotonic()
            data = self._go_post(url, self._go_headers(), json.dumps(req).encode("utf-8"))
            lat = int((time.monotonic() - t0) * 1000)
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(f"gateway-error: {str(data['error'])[:200]}")
            return (extract_responses_text(data).strip()[:80],
                    data.get("usage") or {}, self.endpoint())
        if self.kind == "GoChatProvider":
            body = json.dumps({"model": self.p.model,
                               "messages": [{"role": "system", "content": "只输出摘录本身，无其他文字。"},
                                            {"role": "user", "content":
                                             f"将以下文本压缩为≤60字关键事实摘录：\n{text}"}],
                               "temperature": 0, "max_tokens": 256}).encode("utf-8")
            url = self.p.base_url.rstrip("/") + "/chat/completions"
            data = self._go_post(url, self._go_headers(), body)
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(f"gateway-error: {str(data['error'])[:200]}")
            raw = (data["choices"][0]["message"]["content"] or "").strip()
            if not raw:
                raise ValueError("no-json-in-response: 摘录空正文")
            return raw[:80], data.get("usage") or {}, self.endpoint()
        raise RuntimeError(f"excerpt仅支持GO kind（当前{self.kind}）")


def load_items(limit: int, only: str = "") -> tuple[list[dict], dict[str, list[str]]]:
    items, route_labels = [], {}
    want = {s.strip() for s in only.split(",") if s.strip()} if only else set()
    for fname, task in TASK_OF.items():
        if want and fname not in want:
            continue
        rows = [json.loads(l) for l in open(EXT / f"{fname}.jsonl", encoding="utf-8") if l.strip()]
        if task == "route":  # 路由候选=全集去重orig_label（意图唯一全覆盖）
            route_labels[fname] = sorted({r["orig_label"] for r in rows})
        for r in rows[:limit] if limit else rows:
            items.append({"id": r["id"], "text": r["text"], "task": task,
                          "source": r.get("source", fname), "src_file": fname,
                          "orig_label": r.get("orig_label", ""),
                          "url": r.get("url", ""), "license": r.get("license", "")})
    return items, route_labels


def load_input(path: str, limit: int = 0) -> tuple[list[dict], dict[str, list[str]]]:
    """T4探针入口：自带task的jsonl（{id,text,task[,source,orig_label,url,license]}）。"""
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    items = []
    for r in rows[:limit] if limit else rows:
        task = r.get("task", "")
        if task not in TASK_CN:
            raise ValueError(f"未知task: {task!r}（行{r.get('id')}）")
        if not (r.get("id") and r.get("text")):
            raise ValueError(f"行缺id/text: {str(r)[:80]!r}")
        items.append({"id": r["id"], "text": r["text"], "task": task,
                      "source": r.get("source", "probe"), "src_file": "probe",
                      "orig_label": r.get("orig_label", ""),
                      "url": r.get("url", ""), "license": r.get("license", "")})
    return items, {}


def arbitrate(kind: str, text: str, a: dict, b: dict, c_ad: Adapter,
              c_responses: str) -> tuple[dict, float]:
    """C仲裁：分类一致/情感分差≤2→均值或取该值（不调C）；否则C重判。返回(C, delta)。"""
    if kind == "sentiment":
        delta = round(abs(float(a["value"]) - float(b["value"])), 1)
        if delta <= 2:
            fin = round((float(a["value"]) + float(b["value"])) / 2, 1)
            conf = round((a["confidence"] + b["confidence"]) / 2, 3)
            action = "需人工复核" if delta > 4 else "通过"
            return ({"final": fin, "band": band(fin), "action": action,
                     "reason": f"分差{delta}≤2取均值", "confidence": conf,
                     "provider": c_ad.name, "called": "mean", "ok": True,
                     "endpoint": c_ad.endpoint(), "model": c_ad.model(),
                     "usage": {}}, delta)
        if c_responses:  # responses桩：跳过重判
            return ({"final": None, "band": "", "action": "需人工复核",
                     "reason": "responses桩跳过", "confidence": 0.0,
                     "provider": c_ad.name, "called": "skipped", "ok": False,
                     "endpoint": f"responses:{c_responses}:skipped(未接网关)",
                     "model": c_responses, "usage": {}}, delta)
        ctx = f"仲裁语境：A判{a['value']}分（{a['reason']}），B判{b['value']}分（{b['reason']}），分差{delta}>2，请独立重读原文判定，不要折中。"
        c = c_ad.call(kind, text, [], 0.2, ctx)
        fin = float(c["value"])
        action = "需人工复核" if abs(fin - float(a["value"])) > 4 and abs(fin - float(b["value"])) > 4 else "通过"
        return ({"final": fin, "band": band(fin), "action": action, "reason": c["reason"][:50],
                 "confidence": c["confidence"], "provider": c_ad.name,
                 "called": "rejudge", "ok": True, "endpoint": c["endpoint"],
                 "model": c.get("model", ""), "usage": c.get("usage", {}),
                 "latency_ms": c.get("latency_ms", 0)}, delta)
    # 分类三任务：一致→取该值；不一致→C重判
    agree = a["value"] == b["value"]
    delta = 0 if agree else 1
    if agree:
        conf = round((a["confidence"] + b["confidence"]) / 2, 3)
        return ({"final": a["value"], "action": "通过", "reason": f"AB一致取{a['value']}",
                 "confidence": conf, "provider": c_ad.name, "called": "mean",
                 "ok": True, "endpoint": c_ad.endpoint(), "model": c_ad.model(),
                 "usage": {}}, delta)
    if c_responses:
        return ({"final": None, "action": "需人工复核", "reason": "responses桩跳过",
                 "confidence": 0.0, "provider": c_ad.name, "called": "skipped",
                 "ok": False, "endpoint": f"responses:{c_responses}:skipped(未接网关)",
                 "model": c_responses, "usage": {}}, delta)
    ctx = f"仲裁语境：A判{a['value']}（{a['reason']}），B判{b['value']}（{b['reason']}），请独立重判。"
    # 分类不一致的重判由调用方执行（需route全集labels，此处只处理一致/跳过路径）
    return ({"final": None, "action": "需人工复核", "reason": "未仲裁",
             "confidence": 0.0, "provider": c_ad.name, "called": "none",
             "ok": False, "endpoint": c_ad.endpoint(), "model": c_ad.model(),
             "usage": {}}, delta)


def apply_c2(kind: str, text: str, labels: list[str], a: dict, b: dict,
             c: dict, delta: float, c2_ad, threshold: float) -> tuple[dict, dict | None]:
    """C2第二意见（仅分差超阈）：C正常取C（C2分歧>2升级人工）；C失败/低置信C2转正。"""
    if c2_ad is None or not c2_should_call(kind, delta, threshold):
        return c, None
    ctx = (f"二次仲裁语境：A判{a['value']}（{a['reason']}），B判{b['value']}（{b['reason']}），"
           f"C判{c.get('final')}（{c.get('reason')}），分差{delta}超阈{threshold}，请独立重读原文判定，不要折中。")
    try:
        r = c2_ad.call(kind, text, labels if kind != "sentiment" else [], 0.2, ctx)
    except Exception as e:  # noqa: BLE001
        return c, {"final": None, "band": "", "action": "需人工复核",
                   "reason": f"C2失败:{str(e)[:60]}", "confidence": 0.0,
                   "provider": c2_ad.name, "called": "rejudge-fail", "ok": False,
                   "endpoint": c2_ad.endpoint(), "model": c2_ad.model(), "usage": {}}
    ok_c = bool(c.get("ok")) and (c.get("confidence") or 0) >= 0.6
    if kind == "sentiment":
        v = float(r["value"])
        c2rec = {"final": v, "band": band(v), "action": "通过",
                 "reason": r["reason"][:50], "confidence": r["confidence"],
                 "provider": c2_ad.name, "called": "rejudge", "ok": True,
                 "endpoint": r["endpoint"], "model": r.get("model", ""),
                 "usage": r.get("usage", {}), "latency_ms": r.get("latency_ms", 0)}
        if ok_c:
            if abs(v - float(c["final"])) > 2:
                c = dict(c, action="需人工复核",
                         reason=(str(c.get("reason", "")) + "|C2分歧")[:50])
            return c, c2rec
        act = "通过" if r["confidence"] >= 0.6 else "需人工复核"
        return {"final": v, "band": band(v), "action": act,
                "reason": r["reason"][:50], "confidence": r["confidence"],
                "provider": c2_ad.name, "called": "rejudge-c2", "ok": True,
                "endpoint": r["endpoint"], "model": r.get("model", ""),
                "usage": r.get("usage", {}),
                "latency_ms": r.get("latency_ms", 0)}, c2rec
    v = str(r["value"])
    c2rec = {"final": v, "action": "通过", "reason": r["reason"][:50],
             "confidence": r["confidence"], "provider": c2_ad.name,
             "called": "rejudge", "ok": True, "endpoint": r["endpoint"],
             "model": r.get("model", ""), "usage": r.get("usage", {}),
             "latency_ms": r.get("latency_ms", 0)}
    if ok_c:
        if v != c.get("final"):
            c = dict(c, action="需人工复核",
                     reason=(str(c.get("reason", "")) + "|C2分歧")[:50])
        return c, c2rec
    act = "通过" if r["confidence"] >= 0.6 else "需人工复核"
    return {"final": v, "action": act, "reason": r["reason"][:50],
            "confidence": r["confidence"], "provider": c2_ad.name,
            "called": "rejudge-c2", "ok": True, "endpoint": r["endpoint"],
            "model": r.get("model", ""), "usage": r.get("usage", {}),
            "latency_ms": r.get("latency_ms", 0)}, c2rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="",
                    help="仅跑指定source文件（逗号分隔，如smp2019_ecdt,crosswoz），分批用")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--sleep", type=float, default=0,
                    help="每行完成后暂停秒数（降速保429配额）")
    ap.add_argument("--tries", type=int, default=5)
    ap.add_argument("--a", default=DEFAULT_A,
                    help="A路provider（合规只用go-槽；遗留非go槽仅本地测试）")
    ap.add_argument("--b", default=DEFAULT_B,
                    help="B路provider（合规只用go-槽，deepseek直连禁用；遗留非go槽仅本地测试）")
    ap.add_argument("--c", default=DEFAULT_C,
                    help="C路provider（合规只用go-槽；遗留非go槽仅本地测试）")
    ap.add_argument("--c2", default=DEFAULT_C2,
                    help="C2第二意见模型；空串禁用（合规只用go-槽）")
    ap.add_argument("--c2-threshold", type=float, default=4.0,
                    help="C2触发阈值（情感分差>阈值才调；分类需阈值≤1才启用）")
    ap.add_argument("--temp-ab", type=float, default=0.95,
                    help="A/B独立打分温度（T4起默认0.95；C恒0.2不受此影响）")
    ap.add_argument("--input", default="",
                    help="T4探针入口：自带task的jsonl路径（设了则忽略--only/EXT八集）")
    ap.add_argument("--evidence", default="off", choices=("off", "spark"),
                    help="off=原文前60字桩；spark=调evidence摘要模型（默认关，旗开）")
    ap.add_argument("--evidence-provider", default=DEFAULT_EVIDENCE_PROVIDER,
                    help="evidence摘要provider名（合规只用go-槽；须为models.yaml注册名）")
    ap.add_argument("--fallback-b", default="",
                    help="B路quota时备援provider（遗留GLM周配额路径）；空=退避重试+失败落盘")
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="chat侧输出预算（含推理token，deepseek实测 trivial prompt 烧300）")
    ap.add_argument("--resp-tokens", type=int, default=1024,
                    help="responses侧输出预算")
    ap.add_argument("--reasoning-effort", default="",
                    help="responses侧reasoning.effort透传（low/medium/high）；空=不传保默认")
    ap.add_argument("--providers", default=str(PROVIDERS_YAML))
    ap.add_argument("--out", default="training/abc_out/abc_scores_go.jsonl")
    ap.add_argument("--cache", default="training/abc_out/abc_cache_go.jsonl")
    ap.add_argument("--retry-failed", action="store_true",
                    help="把缓存中AB失败行捡回重跑（成功行仍跳过）")
    ap.add_argument("--calib", default="off", choices=("off", "contrastive"),
                    help="T9对比few-shot：off=旧行为（默认，可比）；contrastive=spam/handoff各≤6对错判→正解例（源gold_frozen改标48，AB双错优先，每例≤120字）。"
                    "EXPERIMENTAL(2026-09 T9验证恶化：CSDS一致81.4→55.7)：默认off，仅研究对比用，勿入生产链")
    ap.add_argument("--c-responses", default="",
                    help="responses模型桩（遗留：只记endpoint、不硬调）")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.calib == "contrastive":
        print("⚠ WARNING: --calib contrastive为已知恶化实验（T9: CSDS一致81.4→55.7），仅研究对比用，勿入生产链！",
              file=sys.stderr, flush=True)

    from judgekit.providers import load_providers
    # .env 本地加载（key永不进仓库；环境变量已存在则不覆盖）
    env_p = ROOT / ".env"
    if env_p.exists():
        for line in open(env_p, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    providers = load_providers(args.providers)
    a_ad = Adapter(providers, args.a, max_tokens=args.max_tokens,
                   resp_tokens=args.resp_tokens, reasoning_effort=args.reasoning_effort,
                   calib=args.calib)
    b_ad = Adapter(providers, args.b, max_tokens=args.max_tokens,
                   resp_tokens=args.resp_tokens, reasoning_effort=args.reasoning_effort,
                   calib=args.calib)
    c_ad = Adapter(providers, args.c, max_tokens=args.max_tokens,
                   resp_tokens=args.resp_tokens, reasoning_effort=args.reasoning_effort,
                   calib=args.calib)
    c2_ad = (Adapter(providers, args.c2, max_tokens=args.max_tokens,
                     resp_tokens=args.resp_tokens,
                     reasoning_effort=args.reasoning_effort,
                     calib=args.calib)
             if args.c2 else None)
    ev_ad = (Adapter(providers, args.evidence_provider)
             if args.evidence == "spark" else None)
    fb_ad = Adapter(providers, args.fallback_b, calib=args.calib) if args.fallback_b else None
    quota_raise = fb_ad is not None

    items, route_labels = (load_input(args.input, args.limit) if args.input
                             else load_items(args.limit, args.only))
    out_p, cache_p = Path(args.out), Path(args.cache)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    failed: set[str] = set()
    if cache_p.exists():
        for l in open(cache_p, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            k = r.get("cache_key")
            if not k:
                continue  # 旧Jev行无键：保留不删，天然不命中
            (done if (r.get("A", {}).get("ok") and r.get("B", {}).get("ok"))
             else failed).add(k)
    cur_key = lambda iid: cache_key_of(iid, args.a, args.b, args.c,
                                       args.c2, args.c2_threshold, args.evidence,
                                       args.temp_ab, args.calib)
    skip = done if args.retry_failed else (done | failed)
    todo = [r for r in items if cur_key(r["id"]) not in skip]
    print(f"items={len(items)} cached_ok={len(done)} cached_fail={len(failed)} "
          f"todo={len(todo)} A={args.a} B={args.b} C={args.c} C2={args.c2 or '禁用'} "
          f"C2t={args.c2_threshold} evidence={args.evidence} retry_failed={args.retry_failed} "
          f"calib={args.calib}")

    lock = threading.Lock()
    stats = {"ok": 0, "fail": 0, "c_rejudge": 0, "c2": 0, "ev": 0, "n": 0,
             "tok_in": 0, "tok_out": 0, "tok_rs": 0}
    gw = ("GO网关curl直调（chat+responses，session头+UA）；DEEPSEEK直连key禁用；"
          "usage全量计价（含reasoning）；超时取yaml（chat90s/resp120s，p95留量）+"
          "429/5xx退避续跑；spark证据默认桩，原文前60字")

    def add_usage(u: dict) -> None:
        i, o, r = usage_totals(u)
        stats["tok_in"] += i
        stats["tok_out"] += o
        stats["tok_rs"] += r

    def fail_side(ad: Adapter, e: Exception) -> dict:
        return {"value": None, "reason": "", "confidence": 0.0, "ok": False,
                "error": f"{type(e).__name__}: {str(e)[:160]}",
                "endpoint": ad.endpoint(), "model": ad.model(), "usage": {}}

    def score_one(r: dict) -> dict:
        kind, text = r["task"], r["text"]
        labels = route_labels.get(r["src_file"], []) if kind == "route" else []
        key = cur_key(r["id"])
        t0 = time.time()
        try:
            try:
                a = a_ad.call(kind, text, labels, args.temp_ab, tries=args.tries,
                              quota_raise=quota_raise)
            except Exception as e:  # noqa: BLE001
                a = fail_side(a_ad, e)
            try:
                b = b_ad.call(kind, text, labels, args.temp_ab, tries=args.tries,
                              quota_raise=quota_raise)
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if is_quota_err(msg) and fb_ad is not None:
                    try:
                        b = fb_ad.call(kind, text, labels, args.temp_ab,
                                       tries=args.tries)
                        b["fallback_from"] = args.b
                    except Exception as e2:  # noqa: BLE001
                        b = fail_side(fb_ad, e2)
                        b["error"] = f"B-fallback-fail {b['error']}"
                else:
                    b = fail_side(b_ad, e)
            if not (a.get("ok") and b.get("ok")):
                c = {"final": None, "band": "", "action": "需人工复核",
                     "reason": "AB有失败", "confidence": 0.0,
                     "provider": args.c, "called": "none", "ok": False,
                     "endpoint": c_ad.endpoint(), "model": c_ad.model(), "usage": {}}
                delta = -1
                c2rec = None
            elif kind == "sentiment":
                try:
                    c, delta = arbitrate(kind, text, a, b, c_ad, args.c_responses)
                except Exception as e:  # noqa: BLE001
                    c = {"final": None, "band": "", "action": "需人工复核",
                         "reason": f"C失败:{str(e)[:60]}", "confidence": 0.0,
                         "provider": args.c, "called": "rejudge-fail", "ok": False,
                         "endpoint": c_ad.endpoint(), "model": c_ad.model(), "usage": {}}
                    _aa, _bb = float(a["value"]), float(b["value"])
                    delta = round(abs(_aa - _bb), 1)
                c, c2rec = apply_c2(kind, text, labels, a, b, c, delta,
                                    c2_ad, args.c2_threshold)
            elif a["value"] == b["value"]:
                c, delta = arbitrate(kind, text, a, b, c_ad, args.c_responses)
                c, c2rec = apply_c2(kind, text, labels, a, b, c, delta,
                                    c2_ad, args.c2_threshold)
            else:
                if args.c_responses:
                    c, delta = arbitrate(kind, text, a, b, c_ad, args.c_responses)
                    c, c2rec = apply_c2(kind, text, labels, a, b, c, delta,
                                        c2_ad, args.c2_threshold)
                else:
                    ctx = (f"仲裁语境：A判{a['value']}（{a['reason']}），"
                           f"B判{b['value']}（{b['reason']}），请独立重判。")
                    try:
                        cc = c_ad.call(kind, text, labels, 0.2, ctx, tries=args.tries)
                        act = "通过" if cc["confidence"] >= 0.6 else "需人工复核"
                        c = {"final": cc["value"], "action": act,
                             "reason": cc["reason"][:50],
                             "confidence": cc["confidence"], "provider": args.c,
                             "called": "rejudge", "ok": True, "endpoint": cc["endpoint"],
                             "model": cc.get("model", ""), "usage": cc.get("usage", {}),
                             "latency_ms": cc.get("latency_ms", 0)}
                    except Exception as e:  # noqa: BLE001
                        c = {"final": None, "action": "需人工复核",
                             "reason": f"C失败:{str(e)[:60]}",
                             "confidence": 0.0, "provider": args.c,
                             "called": "rejudge-fail", "ok": False,
                             "endpoint": c_ad.endpoint(), "model": c_ad.model(),
                             "usage": {}}
                    delta = 1
                    c, c2rec = apply_c2(kind, text, labels, a, b, c, delta,
                                        c2_ad, args.c2_threshold)
            # evidence摘要（默认桩；spark旗开，存摘录不存原文）
            ev_usage: dict = {}
            ev_ep = ""
            if ev_ad is not None:
                try:
                    excerpt, ev_usage, ev_ep = ev_ad.excerpt(text)
                except Exception:  # noqa: BLE001
                    excerpt, ev_usage, ev_ep = text[:60], {}, ev_ad.endpoint() + "(fallback-stub)"
            else:
                excerpt = text[:60]
            ep = {"A": a.get("endpoint", ""), "B": b.get("endpoint", ""),
                  "C": c.get("endpoint", "")}
            md = {"A": a.get("model", ""), "B": b.get("model", ""),
                  "C": c.get("model", "")}
            us = {"A": a.get("usage", {}), "B": b.get("usage", {}),
                  "C": c.get("usage", {})}
            if c2rec is not None:
                ep["C2"] = c2rec.get("endpoint", "")
                md["C2"] = c2rec.get("model", "")
                us["C2"] = c2rec.get("usage", {})
            if ev_ep:
                ep["EV"] = ev_ep
                md["EV"] = ev_ad.model()
                us["EV"] = ev_usage
            rec = {"id": r["id"], "text": text, "task": kind, "source": r["source"],
                   "orig_label": r["orig_label"], "url": r["url"], "license": r["license"],
                   "A": a, "B": b, "C": c, "delta": delta,
                   **({"C2": c2rec} if c2rec is not None else {}),
                   "endpoint": ep, "model": md, "usage": us,
                    "provenance": {"a_provider": args.a, "b_provider": args.b,
                                   "b_effective": ("typesafe(fallback)" if b.get("fallback_from") else args.b),
                                   "c_provider": args.c, "c_responses_stub": args.c_responses or None,
                                   "c2_provider": args.c2 or None,
                                   "c2_threshold": args.c2_threshold,
                                   "c2_called": (c2rec or {}).get("called", "none"),
                                   "evidence_mode": args.evidence,
                                   "reasoning_effort": args.reasoning_effort or None,
                                   "max_tokens_chat": args.max_tokens,
                                   "max_tokens_resp": args.resp_tokens,
                                    "temp_AB": args.temp_ab, "temp_C": 0.2,
                                   "calib": args.calib,
                                   "gateway_note": gw,
                                  "fallback_note": b.get("fallback_from", ""),
                                  "spark_excerpt": excerpt,
                                  "elapsed_s": round(time.time() - t0, 1)},
                   "cache_key": key}
        except Exception as e:  # noqa: BLE001 单轮修复上限内先保证cache落盘
            rec = {"id": r["id"], "text": r["text"], "task": r["task"],
                   "source": r["source"], "orig_label": r["orig_label"],
                   "url": r["url"], "license": r["license"],
                   "A": {"value": None, "ok": False,
                         "error": f"row-fatal {type(e).__name__}: {str(e)[:120]}",
                         "endpoint": a_ad.endpoint(), "model": a_ad.model(),
                         "reason": "", "confidence": 0.0, "usage": {}},
                   "B": {"value": None, "ok": False, "error": "row-fatal",
                         "endpoint": b_ad.endpoint(), "model": b_ad.model(),
                         "reason": "", "confidence": 0.0, "usage": {}},
                   "C": {"final": None, "ok": False, "error": "row-fatal",
                         "provider": args.c, "called": "none", "action": "需人工复核",
                         "reason": "行级异常", "confidence": 0.0,
                         "endpoint": c_ad.endpoint(), "model": c_ad.model(), "usage": {}},
                   "delta": -1,
                   "endpoint": {"A": a_ad.endpoint(), "B": b_ad.endpoint(),
                                "C": c_ad.endpoint()},
                   "model": {"A": a_ad.model(), "B": b_ad.model(), "C": c_ad.model()},
                   "usage": {},
                   "provenance": {"a_provider": args.a, "b_provider": args.b,
                                  "c_provider": args.c, "gateway_note": gw,
                                  "spark_excerpt": r["text"][:60],
                                  "elapsed_s": round(time.time() - t0, 1)},
                   "cache_key": key}
        with lock:
            for u in rec.get("usage", {}).values():
                add_usage(u if isinstance(u, dict) else {})
            if rec.get("C", {}).get("called") in ("rejudge", "rejudge-c2", "rejudge-fail"):
                stats["c_rejudge"] += 1
            if rec.get("C2") is not None:
                stats["c2"] += 1
            if ev_ad is not None:
                stats["ev"] += 1
            stats["ok" if (rec["A"].get("ok") and rec["B"].get("ok")) else "fail"] += 1
            stats["n"] += 1
            with open(cache_p, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if stats["n"] % 25 == 0:
                print(f"  ... {stats['n']}/{len(todo)} ok={stats['ok']} "
                      f"fail={stats['fail']} C2={stats['c2']} "
                      f"tok(in/out/rs)={stats['tok_in']}/{stats['tok_out']}/{stats['tok_rs']}",
                      flush=True)
        if args.sleep:
            time.sleep(args.sleep)
        return rec

    recs: list[dict] = []
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            recs = list(ex.map(score_one, todo))
    else:
        recs = [score_one(r) for r in todo]
    # --out 全量快照（cache追加+out按cache_key去重取末行；旧Jev行无键按id去重保留） 
    all_recs = []
    if cache_p.exists():
        for l in open(cache_p, encoding="utf-8"):
            if l.strip():
                all_recs.append(json.loads(l))
    all_recs = dedup_by_key(all_recs)
    with open(out_p, "w", encoding="utf-8") as f:
        for r in all_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_ok = sum(1 for r in all_recs if r["A"].get("ok") and r["B"].get("ok"))
    print(f"DONE todo={len(todo)} cache_rows={len(all_recs)} AB_ok={n_ok} "
          f"AB_fail={len(all_recs) - n_ok} C_rejudge={stats['c_rejudge']} "
          f"C2={stats['c2']} EV={stats['ev']} "
          f"tok_in={stats['tok_in']} tok_out={stats['tok_out']} "
          f"tok_reasoning={stats['tok_rs']} -> {out_p.name}")


if __name__ == "__main__":
    main()
