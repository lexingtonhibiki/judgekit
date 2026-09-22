# training/archive — 冻结归档索引（Task 2）

本目录收容已完工的一次性训练脚本与中间产物，**冻结存档、可读不可改**。
复活某脚本时：按“新路径”列的命令从仓库根运行；输出默认仍写回本目录。

## 归档内容（what / why frozen）

| 文件 | 说明 | 为何冻结 |
|---|---|---|
| `build_dev_records.py` → 读/写本目录 `dev_task_specs.json`、`dev_pool.jsonl`、`dev_labels.jsonl`、`dev_records_split.jsonl` | dev v1 训练记录构建（101/16/14/15 切分） | dev v1 混训已交付 `mixed_records_split.jsonl`，上游重跑才需动 |
| `build_mixed_records.py` → 读本目录 `zh_v2_train.jsonl`、`dev_records_split.jsonl`、`dev2_pool.jsonl`、`dev2_labels.jsonl`，写 `mixed_records_split.jsonl` | 三域混训 mix06（638 条）构建 | mix06 已定稿，再训=新实验而非重跑旧脚本 |
| `build_v2_dataset.py` → 输入仍读平铺 `training/zhft_records.jsonl` 等（见下），输出默认本目录 `zh_v2_train*.jsonl`、`zh_v2_heldout.jsonl`、`zh_v2_report.json` | v2 数据工程（去重/去污染/hard 标记/留出 100） | v2 数据集已交付，重跑只会复算相同数字 |
| `split_zh_v2.py` | zh_v2 四段重切（seed 17） | 切分结果已入库 `zh_v2_train_split.jsonl` |
| `label_dev_pool.py` / `label_dev2_pool.py` | Jev 教师软标签标注（dev 146 / dev2 140） | 标签已入库 `dev_labels.jsonl` / `dev2_labels.jsonl`，重跑≈烧钱复算 |
| `pretrain_mini.py` / `soup_backbone.py` / `eval_zhft_heldout.py` / `eval_pc.py` / `fit_temperature.py` | 0.1B 三臂预训练、骨干汤、留出评测、pc 条件化评测、温度拟合 | mini 实验结论已定（见 `docs/mini-recipe.md`），checkpoint 在 `D:/Models/NanoJev-zh/` |
| `analyze_bench.py` / `check_dev2.py` / `audit_leakage.py` / `check_real.py` | 靶场分析、dev2 自检、泄题审计、真实池自检 | 审计结论已定稿，重跑=复读 |
| `run_deepseek_score.py` + `score_cache.jsonl`（含 `.bak_garbageC`） | DeepSeek A/B/C 三方情感打分管线及 414 条断点缓存 | 138 条已全量命中、结果已填回工作簿（重跑零 API） |
| `dev_pool.jsonl`、`dev_labels.jsonl`、`dev_records_split.jsonl`、`dev_task_specs.json`、`dev2_pool.jsonl`、`dev2_labels.jsonl` | dev v1/v2 训练池与标签 | 同上 |
| `zh_v2_train.jsonl`、`zh_v2_train_split.jsonl`、`zh_v2_heldout.jsonl`、`zh_v2_report.json` | v2 数据集与报告 | 同上 |
| `mixed_records_split.jsonl`、`mixed_records_pc.jsonl`、`mini_exp_results.json`、`proposal_absurd_v3.jsonl` | 混训记录、pc 记录、五臂快照、离谱草案 | 实验交付物，备查 |
| `label_dev.log`、`label_dev2.log` | 标注过程日志 | 备查 |
| `kill_serve.ps1` | 本地 serve 进程清理 | 工具脚本，随手存档 |

## 故意留在 `training/` 平铺的（ACTIVE / live，不归档）

| 文件 | 原因 |
|---|---|
| `abc_score.py`、`build_v4_xlsx.py`、`abc_out/` | ACTIVE：Task 3 直接使用，禁动 |
| `dsh_prompts.md` | `abc_score.py` 注释引用的 live 情感锚点（`training/dsh_prompts.md`），移动会破坏 ACTIVE 文件引用 |
| `real_pools/` | `benchmarks/data/external/fetch_sample.py`、`benchmarks/real_raw/_build_pools.py` 引用 `training/real_pools/clean/…`（禁动文件），保持不动 |
| `build_review_xlsx.py` | ACTIVE 的 `build_v4_xlsx.py` 文档注明“样式沿用”它，保持不动（v4 的旧 flat 引用依然有效） |
| `gen_zh_data.py`、`build_dataset.py`、`prep_smoke.py`、`run_zh_pipeline.sh`、`pipeline.log`、`samples_raw.jsonl`、`smoke_*.jsonl`、`zh_records.jsonl`、`zhft_*` | round-1 中文管线主干（已提交历史），非本次死脚本 |

## 本次 stale 路径修复记录（只改归档脚本内部 + 用法行）

- `parents[1]`→`parents[2]`（`build_dev_records`、`build_mixed_records`、`eval_pc`、`label_dev_pool`、`label_dev2_pool`、`run_deepseek_score`、`analyze_bench`、`audit_leakage`；`check_dev2` 加一层 `dirname`；`check_real` 的 `PDIR` 指回 `training/real_pools`）：否则 `ROOT` 会错指 `training/archive/`，连 `judgekit` 导入都会断。
- 数据引用一律指向 `training/archive/<原名>`（`ARCH` 变量）；`build_v2_dataset.py` 的**输入**仍读平铺 `training/`（`zhft_records.jsonl` 等主干文件未动），只有三个输出默认改到本目录。
- `split_zh_v2.py`、`eval_zhft_heldout.py` 的写死绝对路径改为 `__file__` 推导（换机器可跑）。
- 用法行统一为 `python training/archive/<脚本>.py`。
- 未动项：`docs/*`、`benchmarks/data/*`、`PROJECT_STATE.md` 中的历史 flat 路径是过程记录原文，保持原样，以本索引为准。

验证：`python -m pytest -q` 全绿；归档脚本 `py_compile` 全过；引用存在性逐项核对（见 Task 2 报告）。
