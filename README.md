# judgekit 判官工具箱

> **Runtime judgment engine for System One (judge) models** — YAML 定义判断任务，
> `classify / score / route / verify` 四原语执行，供应商无关，成本透明。
>
> English TL;DR: judgekit is a provider-agnostic runtime layer that turns judge models
> (TypeSafe Jev or any OpenAI-compatible LLM) into cheap, typed decisions —
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

三个差异化，一句话：**别人测判官聪不聪明，我们测判官划不划算，并把判断做成常驻生产的可插拔层。**

## 结构（monorepo）

```
judgekit/                 核心框架（本仓库同名包）
  engine.py               四原语 + 校准概率解析 + 规则兜底
  providers.py            供应商注册表（OpenAI 兼容统一接口；按 token / 按次计费）
  cli.py                  python -m judgekit run task.yaml --input x.jsonl
  examples/               triage.yaml（派单路由）、score_comment.yaml（拱火度打分）
benchmarks/               judge-econ：判官 cost-accuracy 排行榜（每元准确率 Pareto）
  data/                   迷你中文基准（意图分类 40 例 / 情感 30 例）+ 关键词规则基线
  run_bench.py / report.py  跑分 → report.md + pareto.csv + pareto.png
recipes/                  官方配方（全部建在 judgekit 之上，双模式：本地 0 成本 / 判官增强）
  learn_next/             学习导航：下一个最该学的知识点（知识图谱 + 目标可配置）
  resume_lens/            简历×岗位双向匹配（意愿问卷前置 + JD 红旗核验）
```

## 快速开始

```bash
pip install pyyaml            # 唯一硬依赖；matplotlib 可选（出 Pareto 图）

# ① 框架：一份 YAML，跑一个派单判断（0 成本规则兜底）
python -m judgekit run judgekit/examples/triage.yaml --input benchmarks/data/intent_zh.jsonl --limit 3

# ② 排行榜：规则基线 → 报告（全程 0 成本）
python benchmarks/run_bench.py --models rules --limit 6
python benchmarks/report.py   # → benchmarks/results/report.md + pareto.png

# ③ 接入任意 OpenAI 兼容渠道跑真模型（每样本 1 次调用，--limit 控额度）
python benchmarks/run_bench.py --models rules,router-free-auto --limit 12

# ④ 配方（默认 0 成本启发式；--model 开判官精排）
python recipes/learn_next/next.py --model <provider名> --top 3
python recipes/resume_lens/match.py --model <provider名>
```

## 供应商无关（bring your own providers）

`benchmarks/models.yaml` 声明供应商，全部走 OpenAI 兼容 `/chat/completions`：

- **本地路由 / 自建网关**（本项目开发时用 `http://127.0.0.1:3456/v1`，换成你的 base_url 即可）
- **任何兼容端点**：GLM / DeepSeek / Qwen / 硅基流动 / OpenRouter / AIMLAPI…
- **TypeSafe Jev**：`model: typesafe/jev`，按次计费填 `per_decision_cost`（见 models.yaml 内示例）
- **`rules` 关键词基线**：永远免费的对照锚点，也是判官不可用时的兜底

> 没有 Claude/GPT/Grok 的 key 也完全成立：方法论的贡献是**统一的成本-可靠度测量框架**，
> 而不是绑定某个模型。你手上有哪家就测哪家，排行榜是你自己基础设施的镜子。

## 配方四原则

1. 判官给概率，人做决定（涉人场景强制意愿/知情前置）
2. 不做员工逐人标签；不做企业侧自动淘汰
3. 每个配方必须带零成本本地模式——判官是增强，不是依赖
4. 成本透明：每次运行打印本次判官花费

## Roadmap

- [x] judgekit 四原语引擎 + 规则兜底 + 成本核算
- [x] judge-econ 迷你基准 + Pareto 报告
- [x] learn_next / resume_lens 配方（双模式）
- [ ] Jev（AIMLAPI）接入 + 首张真实 Pareto 图
- [ ] 完整数据集（ag_news / sst2 / 客服意图全量）
- [ ] smart-triage 配方（12345 政务热线派单）
- [ ] 英文文档 / GIF / PyPI 发布

## 贡献

配方欢迎 PR（放 `recipes/`，遵守四原则）；基准数据欢迎扩充（放 `benchmarks/data/`，带 `.rules.yaml` 基线）。

## License

MIT
