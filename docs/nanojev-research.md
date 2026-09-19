# NanoJev 研究报告（TianyuCodings/NanoJev）

> 调研日期 2026-09-20 ｜ 结论：**值得接入 judgekit 作为首个本地判官 provider**（0.6B 可本地推理），
> 但其训练域是游戏/几何判断，中文客服分类泛化未验证——接入后第一件事就是跑 judge-econ 基准。

## 1. 它是什么

Qwen3-0.6B 底座 + 决策头的"nano 复刻 Jev"：state + question 进，**完整概率分布**出，
零输出 token 解码（一次 forward 直接读概率，无自回归生成）。HF 权重：`C-Tianyu/NanoJev`（含数据集 repo）。
实测吞吐（官方持久服务口径）：6 states · 18 questions · 44 candidate paths · **1 次 backbone forward**。

三种问题类型与 TypeSafe 官方对齐：

| 类型 | 实现机制 | 规模 |
|---|---|---|
| Choice | 共享标量头 + set attention（候选集合注意力） | 2–255 动态候选 |
| Boolean（≈noul） | 单路径 sigmoid 出 p_true | — |
| Score | 有序 level 各自编码，输出分布 + 概率加权期望 | 2–10 级 |

## 2. "逆向"的真相：独立研究候选，不是官方配方恢复

官方（TypeSafe）只公开了 RLCD = *Reinforcement Learning for Calibrated Decisions* 的名字和
calibrated-decisions 理念，**未公开可复现的 reward/优化器/训练实现**。NanoJev 的做法：

- **数据**：模拟器产生"观察事件"数据集——每个问题的真实事件概率 q 由模拟器给出（不是蒸馏 LLM 标签）
- **直接损失**：observed CE 与 vector Brier（两者总体最优都恢复 q）
- **RLCD 式采样目标**：M≥2 次有放回采样 A_i~p，配对 proper-reward
  `R = (2/M)Σ1[A_i=Y] − Σ_k c_k(c_k−1)/(M(M−1))`，detached baseline，**期望梯度 = ‖p−q‖² 的梯度**
  （proper scoring 理论推论，float64 穷举梯度校验最大误差 1.39e-16）
- **对照实验的诚实度是亮点**：correctness-only REINFORCE 作为反面对照（test L2 0.017 崩掉），
  CPU 消融 + Qwen3-0.6B 全参微调（100 步，BF16，backbone 2e-5/head 2e-4）都做了；
  并明确声明"evidence does not establish the sampled method is better than direct controls"

## 3. 成绩（他们自己测的，协议可复核）

| 场景 | NanoJev | Jev API | 未调 Qwen3-0.6B |
|---|---|---|---|
| 40 图导航 4×4 test / 6×6 OOD（T=1 采样） | 95% / 90% | 100% / 95% | 35% / 15% |
| 50×50 迷宫（局部判断+代码规划） | 244 步 36 碰撞到达 | 2738 步 1044 碰撞到达 | — |
| Snake 12×12（活到 horizon） | 27 食物 | 30 | 25（被困） |
| 局部安全判断 test / 50×50 OOD | 77.84% / 76.56% | — | — |

评估纪律值得信任：地图分离的数据切分、冻结游戏队列、真实模型执行、独立轨迹回放校验。

## 4. 与 TypeSafe 契约的兼容性（docs/TYPESAFE_CONTRACT.md，兼容性审计表）

已对齐：state/question 隔离（无跨问题 attention）、ID 无关性、Choice 名称+描述语义输入、
Score 有序索引语义、任务组合哲学（原子判断+代码组合）。

**接入 judgekit 需要的适配**（作者自己在审计表里列了）：
- `noul` vs `boolean` 命名映射 + 响应字段适配（本地无 noul 校验）
- 本地推理**缺 confidence 和 Score legend**（judgekit Decision.confidence 要另造或置缺省）
- state 的 dict/list 序列化是非规范 JSON（单引号/True/None/插入序）——输入契约需固定版本
- 结构化（object/array/null）描述本地推理还不支持，需非空字符串

## 5. 对 judgekit 的三个可行动项（按价值排序）

1. **NanoJev provider**（`judgekit/providers/nanojev.py`）：本地 HTTP 服务（作者有持久服务形态，
   load once 复用）→ 实现 `decide(task, x) -> Decision`，映射 boolean→verify / choice→classify /
   score→score。0.6B CPU/低显存可跑 = **零成本本地判官**，judgekit 的"无 key 也能跑"故事闭环
2. **用 judge-econ 评 NanoJev**：它的训练域是游戏几何判断，中文客服分类从未被测过——
   跑我们的 130 条基准即是全网首个 NanoJev 中文泛化数据点；结果好可提交给 NanoJev 作者
   （他们重视独立评测，awesome-jev-zh 也缺），结果差也是诚实的边界发现
3. **与作者联动**：仓库研究纪律极好（控制实验/公开协议/自我 caveat），是高质量社区伙伴；
   issue 里提"provider 互操作 + 中文基准交叉评测"两全其美

## 5.5 本仓库实测（judge-econ，2026-09-20）

已接入 judgekit（`judgekit/providers/nanojev.py`，零成本 provider），130 条中文基准实测：

| dataset | NanoJev 本地 | Jev API | 规则基线 |
|---|---|---|---|
| intent_zh | 80.0% | 97.5% | 95.0% |
| sentiment_zh | **53.3%** | 100% | 73.3% |
| spam_zh | **46.7%** | 93.3% | 100% |
| urgency_zh | 60.0% | 100% | 96.7% |
| **总体** | **61.5%（80/130）** | 97.7% | 91.5% |

延迟 **~220ms/次**（Jev 的 1/4、GLM 的 1/18）、¥0 成本、确定性推理、零兜底混入。
结论（诚实口径）：**英文游戏域 → 中文泛化确实失败**（情感/垃圾接近掷硬币），概率头机制完整迁移
（分布完整、无解码、~220ms），但需要目标域微调才能上生产——这正好量化了 §6 的预测，
也是"本地判官要先在私有域 LoRA"论点的实证。

## 6. 风险与局限

- 局部安全判断 77.84% 的绝对值不高——demo 里靠"代码规划过滤+模型只做平票裁决"的分工撑起系统，
  单独当通用判官用会失望；**它的正确用法正是 judgekit 的原子任务哲学**
- 中文能力完全未验证（训练数据全是英文游戏环境）
- 权重/推理栈是 PyTorch 全参（0.6B 也需要 ~2GB 级资源），Windows CPU 推理延迟需实测
