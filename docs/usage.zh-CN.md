# 使用指南

[首页](../README.zh-CN.md) · [English](usage.md)

## 安装

Python 3.10 及以上。从源码安装；当前快速开始不依赖 PyPI 上的同名包。

```bash
git clone https://github.com/lexingtonhibiki/judgekit
cd judgekit
python -m venv .venv
```

macOS / Linux：

```bash
source .venv/bin/activate
python -m pip install -e .
```

Windows PowerShell（无需激活虚拟环境）：

```powershell
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m judgekit demo
```

后续示例的 `python` 指已安装 judgekit 的解释器。唯一必需依赖是 PyYAML。

## 内置离线演示

```bash
python -m judgekit demo
python -m judgekit demo --lang zh
```

安装后可在任意目录运行，任务 YAML 从包资源读取。仅使用关键词规则：
不读取供应商注册表、不需要 API key、不下载模型、不发起网络请求。
stdout 输出四行 JSON，stderr 说明演示口径。前三条命中，第四条返回
`ok=false`、`value=null`、`error="rules-no-hit"`。
退出 0 表示选定样例得到全部预期结果，包含故意展示的 no-hit，并非 100% 准确率声明。
演示出现非预期行为时退出 1。

## 单条与批量

```bash
python -m judgekit judge judgekit/examples/triage.yaml "快递三天了还没到，催单"
python -m judgekit run judgekit/examples/triage.yaml --input benchmarks/data/econ_zh/intent_zh.jsonl --limit 3 --out decisions.jsonl
```

stdout 输出 JSON；批量摘要写 stderr。`--out` 每次重写文件。
输出若与已打开的输入是同一个文件，会被拒绝，包括硬链接、符号链接，以及重定向给 stdin 的文件。
其他已存在的输出文件仍会覆盖。
`provider` 字段说明本次实际使用的后端；`rules-after-fail` 表示远端失败后规则兜底，
此时 `error` 保留失败原因。`cost` 使用供应商配置的价格计算；未配置价格的零值不代表免费。
费用在响应成功解析后计算。已计费但解析失败的请求，包括随后走规则兜底的情况，仍可能记为零费用；总支出应核对供应商账单。

| 情况 | 退出码 |
|---|---:|
| 单条成功判断 | 0 |
| 内置演示产生全部预期结果，含故意展示的 no-hit | 0 |
| 单条未命中或判断失败；批量空输入 | 1 |
| 参数错误；批量成功执行率低于 `--fail-under` | 2 |

批量默认退出 0，即使部分行失败。对工作流使用成功执行率门禁：

```bash
python -m judgekit run judgekit/examples/triage.yaml --input tickets.jsonl --fail-under 80
```

输入每行一个 JSON 对象，如 `{"id": "t1", "text": "想退款"}`。
每个非空行产出一条记录。非法 JSON 产出 `bad-json`；合法 JSON 但非对象产出 `bad-input`。
二者同时保留在 stdout 与 `--out` 中，计入 `--limit`，在 `--fail-under` 中计为失败；空行忽略。
`--fail-under` 接受 0–100。
`--input -` 从 stdin 读；macOS/Linux 可用 `cat tickets.jsonl | ...`，
PowerShell 可用 `Get-Content -Encoding utf8 tickets.jsonl | ...`。
`--fail-under` 检查 `ok=true` 的比例，**不检查分类准确率，也不要求判断来自远端模型**。
上线门禁还应检查实际 provider 与你自己的金标/业务规则。

## 四个原语

| 原语 | 返回值 | 规则后端 |
|---|---|---|
| `classify` | 标签字符串 | 关键词命中 |
| `route` | 部门/路由标签字符串 | 关键词命中 |
| `score` | 0–1 分数 | 无规则兜底 |
| `verify` | 布尔值 | 无规则兜底 |

定义见 [英文派单 YAML](../judgekit/examples/triage.en.yaml)、
[中文派单 YAML](../judgekit/examples/triage.yaml) 与
[打分 YAML](../judgekit/examples/score_comment.yaml)。
规则是区分大小写的子串计数，平票按 YAML 中先出现的规则选；无命中返回失败。
规则置信度是启发值；OpenAI 兼容后端的置信度由模型自报；可用时原生后端提供完整标签分布。
这些值的统计校准都需要在你的数据上另行验证。

## 更换供应商

`--provider` 覆盖任务 YAML 的 provider；`--providers` 加载注册表。
供应商种类支持 `typesafe`、`openai`、`rules`、`nanojev`、`go-chat`、`go-responses`；
现成配置在 [benchmarks/models.yaml](../benchmarks/models.yaml)。

CLI **不会自动加载 `.env`**。`.env.example` 是变量名参考；调用前在当前终端导出 key，
或由你自己的应用加载环境变量。不要把 key 写进 YAML、命令历史或提交。
例如变量已存在于环境后：

```bash
python -m judgekit run judgekit/examples/triage.yaml --providers benchmarks/models.yaml --provider typesafe --input benchmarks/data/econ_zh/intent_zh.jsonl --limit 3
```

此命令会调用远端 API。务必核对 `models.yaml` 的模型 ID、端点、价格与账户额度。
规则兜底只处理 `classify` / `route` 的失败，不按低置信度自动触发。

## Python API 与输入边界

```python
from judgekit import Task, run_task

task = Task.load("judgekit/examples/triage.en.yaml")
decision = run_task(task, {"text": "my parcel has not arrived"}, {})
print(decision.value, decision.provider, decision.cost)
```

`run_task` 向供应商只传任务的 `input_field`（默认 `text`）；同一行的金标和其他元数据留在本地。
自定义 `input_field: body` 后，批量输入改用 `body` 字段；单条 `judge` 会自动放入该字段。
打分用的结构化对象也放在字段内部，如 `{"text": {"course": "math", "mastery": 0.2}}`。

## 两个本地 recipe

```bash
python recipes/learn_next/next.py
python recipes/resume_lens/match.py
```

随仓库的示例数据会给出“下一步学什么”或求职者侧岗位匹配参考，无 key、无 API 花费。
加 `--model typesafe` 才调用 judge；默认注册表是 `benchmarks/models.yaml`。
`learn_next --top 3` 仅限制送入 judge 的候选数量，不限制最终显示条数。
简历 recipe 先收集意愿问卷，用于求职者参考，不用于企业自动淘汰。
