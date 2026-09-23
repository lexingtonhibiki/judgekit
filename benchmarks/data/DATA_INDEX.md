# benchmarks/data 数据集导览

> 给人工审查者的导览文件。评测跑分入口：`python benchmarks/run_bench.py --models <model> --datasets <名称>`，
> 数据集名解析顺序：先找 `data/{名称}.jsonl`（旧布局兼容），找不到则递归搜 `data/**/{名称}.jsonl`；
> 同名时**优先正式目录**，其次 `deprecated_*`，最后 `drafts/`（后两者均非评测口径，仅作兜底）。
> `rules.yaml` / `task.yaml` 一律与对应 `.jsonl` 同目录解析。

## econ_zh/ —— judge-econ 公开榜单（v2 当前版，去泄题重写）

2026-09 人工审查发现 v1 泄题：intent 标签词直接出现在题面（投诉类含"投诉"、退款类含"退款"、故障类含"报错"），
urgency 标点即答案（带"！/？"基本是"紧急"）。v2 逐条去泄题重写，审计脚本 `training/audit_leakage.py`，
规则基线从 91.5%（含泄题水分）回落到 **82.3%**（健康水位）。

| 数据集 | 条数 | 任务 | 标签分布 | v2 状态 |
|---|---|---|---|---|
| `intent_zh` | 40 | route：电商求助→部门 | 退款售后8 / 物流查询9 / 投诉建议8 / 技术故障8 / 咨询其他7 | 去泄题重写：保留原文17、改写23（含2条反向陷阱 i017r/i030r） |
| `sentiment_zh` | 30 | classify：评论情感 | 正面15 / 负面15 | 原文全部保留，均衡检查通过 |
| `urgency_zh` | 30 | classify：紧急程度 | 紧急15 / 非紧急15 | 标点-标签解耦：紧急60%无"！"、非紧急46.7%无"！"（差13pp）；保留18、改写12 |
| `spam_zh` | 30 | classify：垃圾内容 | 垃圾15 / 正常15 | 垃圾侧真实特征保留；正常侧补3条"提到广告/优惠券但不是广告"陷阱（p018r/p023r/p024r） |

id 规则：改写条 = 原 id + `r` 后缀（如 i033r）；新增条用新 id（本轮无）。

## deprecated_econ_v1/ —— 历史快照（v1，含泄题，冻结不再评测）

v1 四件套原样封存（git mv，逐字节不变），仅供 diff 与审计对照：

- `intent_zh.jsonl` + `.rules.yaml`（40 条，标签词泄漏版）
- `sentiment_zh.jsonl` + `.rules.yaml`（30 条）
- `urgency_zh.jsonl` + `.rules.yaml`（30 条，标点泄漏版）
- `spam_zh.jsonl` + `.rules.yaml`（30 条）

**不要**把它们加回评测——榜单口径以 `econ_zh/` v2 为准。

## dev_v1/ —— 开发调参集第一代（jsonl + task.yaml + rules.yaml）

| 数据集 | 条数 | 用途 |
|---|---|---|
| `dev_route_zh` | 24 | 中文路由调参 |
| `cmd_risk_zh` | 24 | 命令风险分级调参 |
| `err_triage` | 24 | 报错分诊调参 |
| `commit_type` | 25 | commit 类型分类调参 |

## dev_v2/ —— 开发调参集第二代（jsonl + task.yaml，8 个）

`dev2_route_zh` / `dev2_route_en` / `dev2_gate_en` / `dev2_num_calc` / `dev2_num_slo` /
`dev2_commit_zh` / `dev2_code_review` / `dev2_triage_mixed`（各 24-25 条）。
第二代去掉了 rules.yaml（规则基线不覆盖这批，走模型侧评测）。

## 领地边界（并行代理约定）

`data/real_raw/`、`training/real_*` 由数据采集代理维护；本目录结构与 `econ_zh/`、
`deprecated_econ_v1/`、`dev_v1/`、`dev_v2/` 由评测数据代理维护。
