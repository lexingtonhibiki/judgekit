# 判官模型训练前沿调研 → v2 训练方案（效率拉满版）

> 2026-09-20 ｜ 基于四轮文献调研，映射到本机场景（NanoJev 0.6B / 8GB 卡 / Jev+GLM 教师 / 中文四任务）

## 一、前沿发现 → 对我们的启示

| # | 前沿工作 | 核心结论 | 对我们的映射 |
|---|---|---|---|
| 1 | **Skywork-Reward "Bag of Tricks"**（arXiv 2024，RewardBench 榜首系） | 收益几乎全部来自**数据侧**：合成+过滤、去重、去污染、任务类型均衡；V2 用 405B 标注管线扩数据 | 我们最大的杠杆也是数据：**去重 + 对 judge-econ 测试集做污染检查 + 难度分层**，目前完全没做 |
| 2 | **On-Policy Distillation**（Thinking Machines 2025.10；HF 跨家族 recipe；survey 2026.6） | 从 **student** 采样、teacher 逐 token 打分，比静态 off-policy SFT 的样本效率高一个量级 | 判别式头没有 rollout，但原理可迁移：**用 student 的不确定/错误来选数据**（主动学习）——本地推理免费，teacher 预算只花在信息量大的样本上 |
| 3 | **Qwen3 Technical Report**（2025.5） | 小模型（0.6B–8B）**不做自己的 SFT+RL**，直接从大模型 logits 做 KL 蒸馏（~800B token，两阶段） | 我们的路（Jev→0.6B 软标签蒸馏）方向正确；"两阶段"启发：先广域中文判断蒸馏，再任务精修 |
| 4 | **Distill Not Only Data but Also Rewards**（OpenReview） | 用 teacher 多数投票标签训练小 RM 替代 teacher judge | 我们 zhft 400 条正是此配方 ✓；但教师信号可以更便宜——Jev 原生输出完整分布，**一次调用=k次自一致性** |
| 5 | **ArmoRM 多目标多头 + MoE 门控**（RLHFlow） | 共享底座+每属性独立头+门控校准组合；BT+多目标让 7B 打 70B | 四任务共享一个头的设置可升级为**每任务族独立头**或任务条件化门控（中期项） |
| 6 | **Label Smoothing 保校准**（Huang et al., ICML 2025） | SFT 期间 label smoothing 是保校准的功能性机制（LLM 大词表下需修正用法） | 直击 smoke1 的"自信乱猜"残留；配合 Brier 损失（NanoJev 原生支持） |
| 7 | **后处理温度缩放** | 校准 split 上拟合 temperature 是零训练成本的校准修复 | **NanoJev 训练器原生带 calibration split 和 temperature_fitted 字段——我们一直没用**（summary 显示 False），白捡的校准修复 |
| 8 | **Fireworks "Turn Your LLM into a Calibrated Classifier for $2"**（2025.12） | 小模型 LoRA+校准的工程化便宜路线 | 验证我们的成本模型：这条路线就是几美元级 |

## 二、v2 训练方案（按杠杆排序）

**v1 的问题**：104 条训练（太少）、无去重/去污染、未用 calibration split 拟合温度、问题编码不一致（已发现）。

### 第 1 杠杆：数据工程（Skywork 结论：最大头）
1. **合并数据源**：zhft 400（Jev 教师、富描述）+ GLM 自一致性 2000（额度重置后）+ judge-econ 130 中的 train 段
2. **去重**：文本 minhash/精确去重（跨源重复率高——GLM 生成与 zhft 生成可能撞句）
3. **去污染（必须做）**：删除与 judge-econ 130 测试段 n-gram 重合的样本，否则验收数字虚高
4. **难度分层保留**：教师间分歧大的样本单独标记（hard subset），它们对校准最有价值

### 第 2 杠杆：主动学习选样（On-Policy 的判别式改编）
- 当前 student（smoke1）本地推理免费 → 先让 student 给候选池打分
- **只把 teacher 预算花在：student 低置信、student 与规则基线分歧、教师间分歧的样本上**
- 预计 teacher 调用量省 3–5×，同样预算下有效数据量翻倍以上

### 第 3 杠杆：损失与校准（白捡的）
- 损失：soft-label **Brier**（NanoJev 原生 `--loss brier`）为主；CE 对照
- `--objective gold_distribution`（已验证）+ 训练后**拟合 temperature**（calibration split）
- 软标签天然含 label smoothing 效应 ✓（q 来自教师分布而非 one-hot）

### 第 4 杠杆：训练配置（已验证的参数）
- 热启动 `--init-checkpoint`（smoke1 或 base）+ `--steps 150–300 --batch-questions 8 --microbatch-questions 1~2 --max-microbatch-tokens 4096 --gradient-checkpointing`
- 2070 实测：25 步 37 分钟、峰值 11.9GB（WDDM 兜底）——150 步约 2–3.5h，可过夜
- 中期：每任务族独立头（ArmoRM 式）或任务前缀条件化

### 验收协议（不变，冻结测试集）
judge-econ 130（**先做污染检查**）+ zhft 留出 100 + 3 轮零漂移 + 门控捕获率 + ECE。

## 三、执行顺序建议

1. 数据工程脚本（去重/去污染/合并）——半天，零 GPU
2. 主动学习选样管线（student 预筛 → Jev/GLM 只标高价值样本）——半天
3. v2 训练（150–300 步，过夜）+ 温度拟合 + 全套验收
4. 若 0.6B 天花板已到（zhft 留出集 >90% 后停滞）：换 Qwen3.5-4B 底座重跑同管线（Wukong GGUF 证明该家族可用，训练用 HF 全精度底座）

## 来源

Skywork-Reward (arXiv 2410.18487) · Skywork-Reward-V2 (2025.7) · ArmoRM (RLHFlow) · BT+Multi-Objective (arXiv 2507) · Thinking Machines On-Policy Distillation (2025.10) · HF On-Policy Distillation (2025.10) · Survey of On-Policy Distillation (alphaxiv 2026.6) · Qwen3 Technical Report (arXiv 2505.09388) · Huang et al. Label Smoothing & Calibration (ICML 2025) · Fireworks Calibrated Classifier ($2) · Dropbox R1 Re-Distillation · Raschka Hard Distillation (2026.3) · NanoJev RLCD_EXPERIMENT/TYPESAFE_CONTRACT
