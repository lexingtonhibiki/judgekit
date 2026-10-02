# judgekit 判官工具箱

[![CI](https://github.com/lexingtonhibiki/judgekit/actions/workflows/ci.yml/badge.svg)](https://github.com/lexingtonhibiki/judgekit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

[English](README.md) | **简体中文**

**用一份 YAML 做分类、派单、打分或校验，得到统一的决策 JSON 与费用记录。**

从免费的本地关键词规则开始；需要模型时，同一份任务定义可接 TypeSafe Jev 原生 API、
任意 OpenAI 兼容端点或本地 NanoJev。适合把小而明确的判断接进脚本、工单流程和应用。

## 30 秒试用：无 key、无 API 花费

```bash
git clone https://github.com/lexingtonhibiki/judgekit
cd judgekit
python -m pip install -e .
python -m judgekit demo
```

安装后演示可在任意目录运行，只使用本地规则，不读取供应商注册表、不调用 API。
stdout 输出四行 JSON：三条命中，以及一条明确的 `rules-no-hit`。物流行包含：

```json
{"primitive": "route", "value": "shipping", "provider": "rules", "cost": 0.0, "ok": true}
```

样例用于展示行为，不是准确率评测；规则置信度是启发式。
四个预期结果全部出现时，演示退出 0，包含故意展示的 no-hit。
可运行中文演示，或给现有 YAML 传入自己的文本：

```bash
python -m judgekit demo --lang zh
python -m judgekit judge judgekit/examples/triage.en.yaml "my parcel has not arrived"
python -m judgekit judge judgekit/examples/triage.yaml "快递三天了还没到，催单"
```

`judge` 遇到未命中输入时退出 1。自定义 YAML 路径相对于当前目录；内置演示无需路径或源码目录。

[安装与 Windows 指引](docs/usage.zh-CN.md#安装) · [Python API / 供应商 / 退出码](docs/usage.zh-CN.md) · [评测证据](docs/evaluation.md)

## 一份任务定义，多个后端

```yaml
name: ticket-routing
primitive: route
criteria: which department should handle this ticket
labels:
  refund: returns, refunds, exchanges, invoices
  shipping: parcels, tracking, delivery
provider: rules
fallback_rules:
  refund: [refund, return, invoice]
  shipping: [parcel, tracking, delivery]
```

| 原语 | 输出 | 用例 |
|---|---|---|
| `classify` | 标签 | 情感、垃圾内容识别 |
| `route` | 路由标签 | 工单派单 |
| `score` | 0–1 分数 | 候选知识点或岗位匹配评分 |
| `verify` | 布尔值 | 按明确定义的判据校验 |

- 规则后端本地运行，支持 `classify` / `route`；`score` / `verify` 需要模型。
- 原生 Jev / NanoJev 可提供标签概率分布；OpenAI 兼容后端解析并验证 JSON 输出。
- `run_task` 仅把声明的 `input_field`（默认 `text`）传给供应商，金标和额外元数据留在本地。
- 模型失败时，分类和路由任务可走显式关键词兜底；实际后端与失败原因会写进决策。
- 费用按你配置的价格计算。未配置价格、订阅边际费用或本地 API 费用为零，都不等于总成本为零。
- 已计费但响应解析失败的请求可能漏记费用，总支出应核对供应商账单。
- 置信度的统计校准取决于后端和你的数据；低置信度不会自动触发规则兜底。

```python
from judgekit import Task, run_task

task = Task.load("judgekit/examples/triage.en.yaml")
decision = run_task(task, {"text": "my parcel has not arrived"}, {})
print(decision.value, decision.provider, decision.cost)  # shipping rules 0.0
```

## 接进你的工作流

```bash
python -m judgekit run judgekit/examples/triage.yaml --input benchmarks/data/econ_zh/intent_zh.jsonl --limit 3 --out decisions.jsonl
```

每行输入一个 JSON 对象，每行输出一个决策。`--input -` 支持 stdin 管道；
`--fail-under 80` 可在成功执行率低于 80% 时退出 2。
它检查 `ok` 的比例，不检查分类准确率或远端模型是否成功。

供应商配置见 [benchmarks/models.yaml](benchmarks/models.yaml)。`--provider typesafe` 与
`--providers benchmarks/models.yaml` 一起覆盖 YAML 的规则后端。
CLI 读取环境变量，**不会自动加载 `.env`**；接远端前先配置 key、端点和价格。
完整例子见 [使用指南](docs/usage.zh-CN.md#更换供应商)。

## 两个可直接运行的 recipe

| recipe | 作用 | 免费本地试用 |
|---|---|---|
| learn_next | 结合先修关系、掌握缺口与考试权重，排序下一步学什么 | `python recipes/learn_next/next.py` |
| resume_lens | 结合简历、岗位与意愿问卷，给求职者匹配参考 | `python recipes/resume_lens/match.py` |

两者默认走本地启发式；`--model` 才启用模型评分。人的决定由人作出，
简历 recipe 保留意愿问卷，不用于企业自动淘汰。

## 评测：先看数据版本

judge-econ 关注同一任务上的准确率、耗时与配置费用。小样本与阈值选择结果均保留局限。

| 口径 | 可引用事实 | 边界 |
|---|---|---|
| **当前 econ_zh v2** | 规则基线 **107/130 = 82.3%**，四个中文任务 | 去除 v1 的标签词/标点捷径；2026-10-02 离线复跑 |
| 历史 v0.1 mini 集 | Jev **127/130 = 97.7%**，历史规则 **119/130 = 91.5%** | 2026-09-20 的旧数据成绩，不是当前 v2 模型跑分 |
| 外部 gold_spam120 | argmax **72/120 = 60.0%**；τ=0.10 **82/120 = 68.3%** | 120 条人工逐条直标；阈值在同一集合选择，**in-sample，无留出** |

当前零成本复现：

```bash
python benchmarks/run_bench.py --models rules --limit 0 --datasets intent_zh,sentiment_zh,spam_zh,urgency_zh
```

详细结果、历史榜单、失败数与复现边界见 [评测证据](docs/evaluation.md)。
外部研究的协议、提示词、负结果与局限见 [v4 规范报告](docs/jev-v4-report.md)；
概率阈值解释见 [中文研究文章](docs/release-post-v0.3.md)。
当前 v2 规则复跑不调用远端模型，也不验证历史模型成绩。

## 参与开发

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

CI 覆盖 Ubuntu / Windows × Python 3.10 / 3.12。
欢迎补文档、离线示例、供应商 mock 测试与有来源/许可的评测数据。
[贡献说明](CONTRIBUTING.zh-CN.md) · [反馈问题](https://github.com/lexingtonhibiki/judgekit/issues)

## 下一步

- 对当前 v2 数据做新模型对照，并保留原始逐条结果与失败记录。
- 给概率阈值准备独立留出集，验证召回率与误报取舍。
- 扩充有发布许可、标签清楚的真实场景数据和可运行 recipe。
- 核对发布包与演示材料后准备正式版本。

## 许可

[MIT](LICENSE)
