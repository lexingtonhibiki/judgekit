# judgekit 判官工具箱

[![CI](https://github.com/lexingtonhibiki/judgekit/actions/workflows/ci.yml/badge.svg)](https://github.com/lexingtonhibiki/judgekit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

> **Runtime judgment engine for System One (judge) models** — YAML 定义判断任务，
> `classify / score / route / verify` 四原语执行，供应商无关，成本透明。
> 原生支持 TypeSafe Jev decisions API，兼容任意 OpenAI-compatible 端点。
>
> English TL;DR: judgekit is a provider-agnostic runtime layer that turns judge models
> (TypeSafe Jev natively, or any OpenAI-compatible LLM) into cheap, typed decisions —
> classification, scoring, routing, verification — with calibrated probabilities,
> rule-based fallbacks, and per-decision cost accounting. Ship it inside your app,
> not just your test suite.

## 这不是又一个评测框架

| 项目 | 回答的问题 | 时机 |
|---|---|---|
| [JudgeBench](https://github.com/ScalerLab/JudgeBench) (ICLR'25) / [JudgeLM](https://github.com/baaivision/JudgeLM) | 判官模型判断得**准不准**（难题对抗对） | 学术基准 |
| [DeepEval](https://github.com/confident-ai/deepeval) / [promptfoo](https://github.com/promptfoo/promptfoo) | LLM 输出在**测试时**过不过关 | 开发期/CI |
| [semantic-router](https://github.com/aurelio-labs/semantic-router) | 请求路由给哪个模型（向量相似度） | 运行时，但仅路由 |
| **judgekit** | 生产请求的**每一个判断**要花多少钱、多可靠 | **运行时判断层** |

一句话：**别人测判官聪不聪明，我们让判断以可控成本常驻生产。**

## 实测（judge-econ mini 基准，n=24，详见 benchmarks/）

| 供应商 | 后端 | acc | 延迟/次 | 成本 |
|---|---|---|---|---|
| **typesafe**（Jev `jev-1.13.0`，原生 decisions API） | choice→全量概率分布 | **95.8%**（intent 11/12 · sentiment 12/12） | **~1.0s** | **¥0.109 / 千次**（官方 $0.042/MTok 输入，输出免费） |
| rules（关键词基线） | 规则 | 87.5% | ~0ms | ¥0 |
| LLM 对照组（GLM / DeepSeek / 免费链） | OpenAI 兼容 | 补测中（本地网关渠道波动） | ~3s | 视渠道 |

两个先说清楚的点：

1. **这是中文场景下判官模型的第一批公开评测数字之一**——awesome-jev-zh 收录方明确写着"中文场景至今没有公开评测，这是目前最缺的一块"，judge-econ 就是要补这块；mini 基准全部开源，欢迎 PR 扩充。
2. **原子任务设计有实证背书**：社区钓鱼评测显示，让判官直接回答复合问题 62.6% vs 拆成原子信号+代码组合 95.0%——judgekit 的 Task（单原语、单判断、置信度门控、兜底显式）就是这个结论的工程化。

![cost-accuracy pareto](docs/pareto.png)

## 结构（monorepo）

```
judgekit/                 核心框架（本仓库同名包）
  engine.py               Task/Decision + 四原语调度 + 规则兜底
  providers/
    typesafe.py           原生 TypeSafe Jev（/v1/systemone：choice/score/noul 全量概率）
    openai_compat.py      任意 OpenAI 兼容端点（提示词自动生成 + JSON 解析）
    rules.py              关键词规则（零成本基线/兜底）
  cli.py                  judgekit run task.yaml --input x.jsonl
  examples/               triage.yaml（派单路由）、score_comment.yaml（刻度打分）
benchmarks/               judge-econ：判官 cost-accuracy 排行榜（每元准确率 Pareto）
recipes/                  官方配方（全部建在 judgekit 之上，双模式：本地 0 成本 / 判官增强）
  learn_next/             学习导航：下一个最该学的知识点（知识图谱 + 目标可配置）
  resume_lens/            简历×岗位双向匹配（意愿问卷前置 + JD 红旗核验）
tests/                    22 个单元测试（离线，CI 必跑）
```

## 快速开始

```bash
git clone https://github.com/lexingtonhibiki/judgekit && cd judgekit
pip install -e .              # 唯一硬依赖 pyyaml；judgekit 命令同时可用
cp .env.example .env          # 填入 TYPESAFE_API_KEY（可选，不填走零成本规则兜底）

# ① 一份 YAML，跑一个派单判断（无 key 时自动走规则兜底，0 成本）
judgekit run judgekit/examples/triage.yaml --input benchmarks/data/intent_zh.jsonl --limit 3

# ② 原生 Jev：export TYPESAFE_API_KEY=... 后同一份 YAML 直接进原生 decisions API
#    classify/route → choice（全量概率分布）；score → 刻度打分；verify → noul 单值概率
judgekit run judgekit/examples/triage.yaml --input benchmarks/data/intent_zh.jsonl --limit 3

# ③ 排行榜：规则基线 → 报告（全程 0 成本）
python benchmarks/run_bench.py --models rules --limit 6
python benchmarks/report.py   # → benchmarks/results/report.md + docs/pareto.png

# ④ 任意 OpenAI 兼容渠道（GLM/DeepSeek/OpenRouter/自建网关…改 benchmarks/models.yaml）
python benchmarks/run_bench.py --models rules,typesafe --limit 4

# ⑤ 配方（默认 0 成本启发式；--model typesafe 开判官精排）
python recipes/learn_next/next.py --model typesafe --top 3
python recipes/resume_lens/match.py --model typesafe
```

## 同一份任务定义，任意后端

```yaml
name: 工单派单
primitive: route
criteria: 判断该求助内容应流转到哪个部门
labels:                       # map 写法：候选→说明
  退款售后: 退货、退款、换货、发票、赔偿
  物流查询: 快递、运单、发货进度、签收
provider: typesafe            # 换成任意 openai 兼容供应商名即切换后端，YAML 零改动
fallback_rules:               # 供应商失败/无 key 时的零成本兜底
  退款售后: [退款, 退货, 换货, 发票]
```

- **typesafe 后端**：choice 原生 criteria（候选→说明）、score 原生刻度、verify→noul，
  返回全量概率分布与 confidence
- **openai 后端**：同一份 YAML 自动翻译成「只输出 JSON」的提示词（含刻度锚点）
- **rules 后端**：关键词兜底，永远免费

## 供应商无关（bring your own providers）

`benchmarks/models.yaml` 声明供应商：`kind: typesafe | openai | rules`。
key 一律走环境变量（`.env.example` 模板），**永远不进仓库**。没有国外渠道完全成立：
本地网关/GLM/DeepSeek/任何兼容端点即可跑通全链路，Jev 只是其中一家。

## 配方四原则

1. 判官给概率，人做决定（涉人场景强制意愿/知情前置）
2. 不做员工逐人标签；不做企业侧自动淘汰
3. 每个配方必须带零成本本地模式——判官是增强，不是依赖
4. 成本透明：每次运行打印本次判官花费

## 开发

```bash
pip install -e .[dev]
pytest               # 22 tests, 离线可跑
```

CI（GitHub Actions）在 Ubuntu/Windows × Python 3.10/3.12 上跑测试 + 零成本烟测。

## Roadmap

- [x] judgekit 四原语引擎 + 原生 TypeSafe Jev 适配器 + OpenAI 兼容 + 规则兜底
- [x] judge-econ 迷你基准 + Pareto 报告（Jev 实测 95.8% @ ¥0.109/千次）
- [x] learn_next / resume_lens 配方（双模式）
- [x] 测试套件 + CI
- [ ] LLM 对照组全量补测（免费链/GLM/DeepSeek 渠道恢复后）
- [ ] 完整数据集（ag_news / sst2 / 客服意图全量）
- [ ] smart-triage 配方（12345 政务热线派单）
- [ ] PyPI 发布 / 英文文档 / GIF

## 贡献

配方欢迎 PR（放 `recipes/`，遵守四原则）；基准数据欢迎扩充（放 `benchmarks/data/`，带 `.rules.yaml` 基线）；新供应商适配欢迎 PR（`judgekit/providers/`，实现 `decide(task, x) -> Decision` 即可）。

## License

MIT
