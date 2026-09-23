# Jev v4 规范报告（公开版）

> 口径声明：本报告数字只抄各任务已验证值（task-13~18 报告），不新算。
> 金标以 `gold_frozen190`（spam120/handoff70）与 `gold_spam120`（120直标）为准。
> `benchmarks/models.yaml` 与旧榜单数字（`docs/report.md`、`README` 榜单行）一字未碰。
> pytest 基线：124 passed + 1 skipped（T18）。
> Key 自备声明：复现需自备 `TYPESAFE_API_KEY`（`.env`/环境变量）；缺 key 即停线
> `NEEDS_CONTEXT`，零调用不硬跑。ABC 槽位 key（DeepSeek/GLM 按量计费）不经确认不批量调用。

## 1 背景与口径史（v1 泄题 → v2 → v4）

- v1（含泄题，冻结于 `benchmarks/data/deprecated_econ_v1/`，不再评测）：人工审查发现 intent
  标签词直接出现在题面（投诉类含"投诉"、退款类含"退款"、故障类含"报错"），urgency
  标点即答案（带"！/？"基本是"紧急"）。v1-on-v1 复现 119/130 证明原数字含泄题成分。
- v2（当前公开榜单，`benchmarks/data/econ_zh/`，130 条）：逐条去泄题重写（intent 改 23 /
  urgency 改 12 / spam 改 3 / sentiment 保留；紧急题一半平静语气、非紧急一半夸张语气）。
  审计（`training/archive/audit_leakage.py`）：标签词泄漏 0%、标点差 13pp 反转 PASS。
  规则基线 91.5% → 82.3%（挤出泄题水分的健康回落）。
- v4（本报告口径）：探针 T6 扩采（150 新采 + 旧 40 去重 = 190 行：FakeReview60/JD60/CSDS70，
  即 spam120/handoff70）+ 异构 ABC 打分（A/B 同 temp 0.95，C 恒 0.2）+ 用户终审冻结
  `gold_frozen190`。Jev 原生 typesafe 逐条一判跑榜单续集（T13~T18 七轮）。

## 2 数据集（8×50 + 探针 + 许可表）

### 2.1 外部小闭环 8×50（`benchmarks/data/external/`，seed=42，每集 50 条，共 400 行）

采样：urllib 直连 + UA，hf-mirror 302 签名直跟，decode 全部 utf-8-sig，重试 4 次，未调用
任何付费 API。归一化：NFKC + opencc 繁→简 + 去首尾空白 + 空白折叠 + 小写；去重键 =
归一化 sha256。污染基：`training/real_pools/clean/real_sentiment.jsonl` +
`benchmarks/data/econ_zh/*.jsonl`，共 268 条（归一化 hash）；命中即弃并补采（内部污染基，未随库发布）。
详见 `benchmarks/data/external/external_SAMPLING_REPORT.md`（随提交入库）。

| 数据集 | 条数 | 候选池 | 去重弃数 | 污染命中 | 许可（原文照抄） |
|---|---|---|---|---|---|
| smp2019_ecdt | 50 | 2579 | 0 | 0 | SMP 竞赛公开数据，研究使用（原版权归赛事组织方；本采样仅研究用途，注明来源） |
| crosswoz | 50 | 42346 | 4 | 0 | Apache-2.0（HF GEM/CrossWOZ 元数据声明；另需引用 Zhu et al. TACL 2020） |
| ewect | 50 | 34766 | 0 | 0 | 竞赛数据研究使用（原版权归赛事/平台方；本采样仅研究用途，注明来源） |
| cped | 50 | 94187 | 0 | 0 | Apache-2.0（仓库 LICENSE） |
| waimai_10k | 50 | 11987 | 0 | 0 | 未声明（原作者/来源不详，见 SophonPlus 说明）；仅研究使用 |
| weibo_senti_100k | 50 | 119988 | 0 | 0 | 未声明（原作者/来源不详，见 SophonPlus 说明；原数据来自 CSDN 转载）；仅研究使用 |
| dmr | 50 | 50 | 0 | 0 | 研究使用（遵守豆瓣隐私政策，user_id 已加密；引用 ICPR'22 GAIM 论文） |
| fbs | 50 | 14074 | 4 | 0 | 研究使用（需注明 source-link 并引用 CCS'20 Lies in the Air；已做匿名化预处理版本） |

已知缺口（照抄采样报告备注，不隐瞒）：DMR 主库仅 README 无数据文件，改取同作者
DMR-Dataset-1/reviews.json（仅 44 条：22 正常/22 刷评），另从同域 HF
tracywong117/spam-douban-movie-review 补 6 条凑 25/25，补采行 url 单独标注且
Fake=1→spam 语义待人工复核。FBS 纯 spam（14 类，本批全 spam 标签），需配 ham
（ham 从 dmr.jsonl 中 orig_label=正常 的条目补）。crosswoz 标签词泄漏 42 条
（如 crosswoz_0002/0003/0004 命中"景点/酒店/出租"），使用时注意。waimai_10k /
weibo_senti_100k 许可未声明，仅研究使用。

### 2.2 探针（T6，`benchmarks/data/external/probe_*.jsonl` 随提交入库）

- `probe_spam_t6.jsonl` 100 行 + `probe_handoff_t6.jsonl` 50 行 = 探针 150（T6 新采部分）。
- `probe_handoff_t6_70.jsonl` 70 行（CSDS 派生终版：转人工预 9 / 不转预 41 + 旧 20）。
- `probe_t6_190.jsonl` 190 行（FakeReview60/JD60/CSDS70，spam120/handoff70；去重 190/190）。
- `probe_t6_fetch_report.json`（seed=43，污染基 268；t6raw 先取 FK60/JD60/CSDS70，
  再去重旧 40（exact1 + CSDS-DID3）后定额新 FK50/JD50/CSDS50；旧 40 orig 经 xlsx
  我的最终校正 fk_0003/jd_0008→正常，其余不变；reatiny FETCH_FAIL 沿用 T4 不补）。
- `probe_t6_gates.json`（三源需人工率 0/60、0/60、0/70 全 PASS；CSDS 一致 57/70=81.4% PASS）。

## 3 金标（65 终审 + 120 直标口径与冲突覆盖）

### 3.1 `gold_frozen190.jsonl`（190 行：spam120/handoff70；JD60/FK60/CSDS70）

冻结声明照抄 `training/abc_out/FREEZE_v4.md`：gold 口径以
`training/abc_out/数据审核_v4_full.xlsx`（06 待审汇总_65"我的最终"列）为准。

- gold190 组成：125 自动通过（gold=C 终判，我的最终=✓通过）+ 65 终审
 （改标 48 取反 + ✓通过 17 认同 + 删除 0）；id 去重 190/190，空 gold 0；warns=0。
- gold 分布：spam 垃圾 61 / 正常 59；handoff 不转 56 / 转人工 14；
  我的最终 ✓通过 142 / 改标 48。
- 改标口径（原样执行）：改标=C 取反（spam 正常↔垃圾，handoff 不转↔转人工；
  刷单/刷评归垃圾侧仅归一比对用，gold 仍为二值 C 空间）。
- 48 取反分布（按源）：JD刷单 16 / FakeReview 18 / CSDS 14。
- 三源 verdict（T8 段照抄）：JD 翻转 26.7% / A66.7% B66.7% C73.3% 回炉；
  FK 翻转 30.0% / A66.7% B75.0% C70.0% 回炉；
  CSDS 翻转 20.0% / A80.0% B81.4% C80.0% 回炉候选 13 行复核
  （AB 双错行；翻转率未达 25% 门，只列候选复核，不判回炉 rubric）。

### 3.2 `gold_spam120.jsonl`（120 行用户直标，最高 gold：FK60+JD60，直标垃圾 71/正常 49，零空）

- 与 `gold_frozen` spam 行冲突 14 行，口径 120 直标赢，旧文件不回写（冻结保持）。
- 冲突 id：fk_0001，fk_0005，fk_0006，fk_0007，fk_0023，fk_0049，fk_0053，
  jd_0010，jd_0011，jd_0038，jd_0039，jd_0046，jd_0048，jd_0060。
- 方向：12 行旧正常→新垃圾（用户从严），2 行旧垃圾→新正常（jd_0011、fk_0023，用户从宽）。
- 冲突行 T17 命中 3/14（pred 正常 11 / 垃圾 3，模型欠判垃圾）。
- 口径注记：T16 的 51 行（06 待审 spam 子集，亲审改标 34=flip / 通过 17=认同）是
  C 终判 vs 预标签难例口径；120 直标是用户逐条亲标最高口径，14 冲突一律 120 赢
 （含 T16 子集内行以新 gold 为准，旧 51 口径被覆盖）。

## 4 提示词附录（三版逐字 + spam 单任务版）

以下逐字抄 `training/eval_jev_v4.py`（SCN/SCNv2）与 ABC rubric，不改一字。
`eval_spam_only.py` / `eval_spam120.py` / `eval_spam_probs.py` 的 SPAM_PROMPT
与 `jev.SCNV2_SPAM` 为同一对象（import 复用，禁复制防漂移）。

### 4.1 裸版（T13，Jev 两 Task instruction + criteria + 富描述）

- spam instruction（裸）：`举一反三类推，禁单特征定罪脱罪；只看text字段独立判定。`
- handoff instruction（裸）：`只看text字段独立判定。`
- `SPAM_CRITERIA`：`垃圾判定：营销引流/刷屏才判，抱怨差评驳回；刷单须举一反三类推——无可验证细节的断言式夸赞、情绪强度与事实密度倒挂、对下单决策零信息增量，联想同类夸词/品牌极端词/无细节夸张/维度标点轰炸形态；禁单特征定罪或脱罪，须结合可验证具体事实综合判`
- `HANDOFF_CRITERIA`：`转人工判定：辱骂/威胁/重复催≥2次或情绪崩溃才转；投诉但冷静、有具体诉求、无失控情绪不转`
- spam 富描述：垃圾=`营销引流/刷屏/刷单，或无细节断言夸赞情绪倒挂零增量`；
  正常=`抱怨咨询事实陈述不判；单特征不定罪看可验证事实`。
- handoff 富描述：转人工=`辱骂威胁重复催≥2次或情绪崩溃才转`；
  不转=`投诉冷静有诉求无失控情绪则不转`。
- ABC A/B 两 rubric 同语义（T7 泛化 + 单特征禁令），默认链可复现。

### 4.2 SCN 版（T14，Jev 两 Task instruction 前置场景句；ABC A/B 两 rubric 前置同句）

- `SCN_SPAM`：`场景：购物评价区。读者默认为潜在买家。朋友间私下夸赞式语句出现在这里，即视为刷单/刷评嫌疑。`
- `SCN_HANDOFF`：`场景：用户对客服说话。读者为客服或分诊系统。生活情绪独白（如路况、天气抱怨）无客服指向，不算转人工。`
- 用法：两 Task instruction = 场景句 + 裸 instruction（叠加不替换）；
  Task 名追加 `-scn`，落盘 `jev_v4_scn_scores.jsonl` 隔离旧 190；
  ABC cache 键追加 `|SCN=on`（off 保旧格式）。T15 起 SCN 作废替换（代码保留仅隔离）。

### 4.3 SCNv2 用户版（T15，两 Task instruction 整体替换逐字句 + labels 压短呼应）

- `SCNV2_SPAM`：`场景：购物评价区。你的身份是潜在买家。垃圾判定：对于你而言，纯粹情绪称赞（刷好评返现）/谩骂（恶意攻击）等不提供真实消费反馈的均为垃圾；只看text字段独立判定`
- `SCNV2_HANDOFF`：`场景：用户对客服说话。你的身份为客服或分诊系统。转人工判定：需要人工介入操作的, 如修改订单,多次发消息催促的(>=2), 非规范化信息回复的；text字段换行表示多段对话.只看text字段独立判定。`
- labels 压短呼应（不加新约束）：spam 垃圾=`无消费反馈的称赞/谩骂/引流其余不判` /
  正常=`其余`；handoff 转人工=`需人工介入：改单/连催/非规范回复` / 不转=`其余`。
- 用法：两 Task instruction 整体 == 上两句逐字；Task 名追加 `-scnv2`，
  落盘 `jev_v4_scnv2_scores.jsonl`；ABC cache 键追加 `|SCNv2`；
  ABC A/B 两 rubric 同句替换，与旧泛化串并存、位置放最前。

### 4.4 spam 单任务版（T16/T17/T18，与 SCNv2 spam 同一对象，禁改字）

- 全文：`场景：购物评价区。你的身份是潜在买家。垃圾判定：对于你而言，纯粹情绪称赞（刷好评返现）/谩骂（恶意攻击）等不提供真实消费反馈的均为垃圾；只看text字段独立判定`
- Task 单例 classify，labels=[垃圾, 正常]，criteria/labels/兜底与 SCNv2-spam 侧一致；
  handoff 链不跑不碰。T16 与 SCNv2-spam 子集同 51 id 预测逐行一致（0 翻转），
  证明单例复现 SCNv2-spam 路径精确成立。

## 5 结果（可引用成绩置顶；其余为探索口径附录，不引用）

> 口径收缩声明（Task 20，文档-only，不重算）：唯一可引用成绩 = 120 直标双数
> （`gold_spam120`，用户逐条亲标最高口径）。190 系五数为探索口径
> （混合 gold：125 自动 + 派生，已被 120 直标取代，不引用、不对标）。
> CI 均为 Wilson 95%，小样本如实标；数字只抄 task-13~18 已验证值。

### 5.1 可引用成绩（120 直标双数，置顶）

| # | run | n | acc | 95% CI | ok 率 | 分源 |
|---|---|---|---|---|---|---|
| T17 | 直标 120（gold=120 直标口径） | 120 | 72/120=60.0% | [51.1%, 68.3%] | 120/120 | JD31/60=51.7%[39.3,63.8] / FK41/60=68.3%[55.8,78.7]；垃圾召回 24/71=33.8%，正常 48/49=98.0% |
| T18 | 阈值 τ=0.10（同 120，p_spam≥τ 判垃圾） | 120 | 82/120=68.3% | —（in-sample，无留出） | 120/120（p_spam 零缺分布） | recall 59.2% / spec 81.6% / prec 82.4% / F1 0.689 / Youden 0.408 / flips_vs_argmax 25（TP42/TN40/FP9/FN29）；Δ vs 60.0% = +8.3pt |

补充（照抄，不作新断言）：T18 本轮 argmax 为 73/120=60.8%（pred 垃圾 26/正常 94），
与 T17 的 72/120=60.0% 差 1 行 jd_0059（T17 判正常错 → 本轮 p_spam=0.50 判垃圾对，
重跑边界抖动非口径差）。acc 最高档为 τ=0.05 的 69.2%，按 Youden 优先规则不取。

### 5.2 探索口径附录（混合 gold，已被取代，不引用）

> 本附录五数均为探索口径（`gold_frozen190`：125 自动通过 + 65 终审混合，
> 含 CSDS 派生 70，已被 120 直标取代）。不引用、不对标、不作显著性断言。

| # | run | n | acc | 95% CI | ok 率 | 分源 | 口径注 |
|---|---|---|---|---|---|---|---|
| T13 | 裸 190 | 190 | 138/190=72.6% | [65.9%, 78.5%] | 190/190 | JD40/60=66.7%[54.1,77.3] / FK42/60=70.0%[57.5,80.1] / CSDS56/70=80.0%[69.2,87.7] | 探索口径：混合 gold 190（125 自动 + 65 终审），裸提示词 |
| T14-R1 | 复跑 off（同窗噪音基线） | 190 | 135/190=71.1% | — | 188/190（csds_0001/0002 DNS 失败记 ok=False） | JD39/60=65.0% / FK42/60=70.0% / CSDS54/70=77.1% | 探索口径：混合 gold 190，同窗 off 重跑噪音基线 |
| T14 | 场景 SCN | 190 | 141/190=74.2% | [67.6%, 79.9%] | 190/190 | JD42/60=70.0%[57.5,80.1] / FK43/60=71.7%[59.2,81.5] / CSDS56/70=80.0%[69.2,87.7] | 探索口径：混合 gold 190，场景句 SCN 版 |
| T15 | 用户版 SCNv2 | 190 | 131/190=68.9% | [62.0%, 75.1%] | 190/190 | JD39/60=65.0%[52.4,75.8] / FK41/60=68.3%[55.8,78.7] / CSDS51/70=72.9%[61.5,81.9] | 探索口径：混合 gold 190，用户版 SCNv2 |
| T16 | 硬子集 spam-only（06 难例 51） | 51 | 20/51=39.2% | [27.0%, 52.9%] | 51/51 | JD12/29=41.4%[25.5,59.3] / FK8/22=36.4%[19.7,57.0]；与 SCNv2-spam 子集同 id 0 翻转 | 探索口径：06 待审 spam 难例子集（C 终判 vs 预标签口径），与全量不直接对标 |

补充（照抄，不作新断言）：T16 的 39.2% 系难例富集（gold 垃圾 41/正常 10，
pred 垃圾 12/正常 39，偏向判正常），与全量口径不直接对标。
全部 CI 重叠，不作显著性断言。

## 6 参数影响（temp 0.95 / 场景 +3 / 阈值 +8.3pt / in-sample 注）

- temp：ABC A/B 全量 `--temp-ab 0.95`（cache 键 `|TAB=0.95` 与旧链隔离），C 恒 0.2，
  C2（GLM）阈值 4.0。AB 同温重跑漂移 26/190（TAB0.95 非确定），同 AB 下 C 差异仅 3 行，
  故 T11 净恶化多为 AB 漂移（见 §7）。
- 场景：SCN vs T13 基线净 +3 行（JD+2/FK+1/CSDS±0，翻转 9 行：正常→垃圾 9，垃圾→正常 0，
  handoff 0；miss→ok 6、ok→miss 3）；以同窗 rerun-off 为基准净 +6 行。
  噪音地板 ≈1/190=0.5%（off-vs-off 真实模型抖动仅 1 行 jd_0017，conf 恒 0.09 超低置信区；
  另 2 行系基建 DNS 失败）。两种基准下净效应（+3/+6）均超噪音地板。
  conf 分布位移：总体均值 0.804→0.780（−0.024），分源 JD−0.021/FK−0.055/CSDS−0.003：
  场景效应体现为标签方向翻转（低置信边界行正常→垃圾），非置信度整体抬升。
- 用户版：SCNv2 68.9% < SCN 74.2%（vs 基线 −7、vs SCN −10、vs off-rerun −4 行；
  翻转：不转→转人工 17，垃圾→正常 6；handoff 侧"需人工介入：改单/连催/非规范回复"
  过度收紧为主要拖累 CSDS 56→51，spam 侧第二人称放宽与 T14 收紧反向）。
- 阈值：τ=0.10（Youden=0.408 最大，F1=0.689 次之，并列取小 τ），acc 68.3%，
  Δ vs 60.0% = +8.3pt（vs 本轮 argmax 60.8% = +7.5pt）。
  in-sample 注：阈值在同 120 上选优，无留出验证，+8.3pt 不宜作泛化断言；
  低 τ 换召回付代价（τ=0.10 正常误杀 9 行，τ=0.05 误杀 16 行）。

## 7 负结果（三轮）

- few-shot 回炉（T9，`--calib contrastive`，纠偏例 12 对：spam 判正常→实垃圾 6 对
  FK3+JD3、handoff 判不转→实转人工 6 对 CSDS，每例 ≤120 字；同槽位同温重跑 190）：
  主指标翻转率 JD 26.7→30.0 / FK 30.0→28.3 / CSDS 20.0→44.3（仅 FK 微降 1.7pt），
  次指标 CSDS 一致 81.4%（pre 口径）→55.7 大跌。双指标均未达，不合入 rubric。
  方向明确过纠：单向 6 对 + "类推从严"收束语把边界推过头（CSDS 转人工召回 12/14 修好
  但 29/56 不转误杀；JD 垃圾召回 +8 但正常误杀 +10）。已标 EXPERIMENTAL，默认 off。
- 阈值回炉（T11，`--cautious on`：C/C2 提示词追加存疑→人工句，全任务生效；
  同槽位同温重跑 190）：待定率 0/190→0/190（17 个 rejudge 全 Through，无一转人工），
  翻转率 JD 26.7→31.7（+5.0pt 恶化）/ FK 30.0→35.0（+5.0pt 恶化）/ CSDS 20.0→20.0（持平）。
  双指标均未达，围栏不合入；默认 off。测量混杂 AB 温噪（见 §6）。
- C 重打（T12，`--conly`：A/B 冻结 190 行逐字复用零重采，只重打 C luna temp0.2 +
  C2 glm 同阈，零失败）：C 翻转 4/190=2.1%（全 rejudge 行 temp0.2 温噪；一致行 mean162
  零翻转符合预期），mismatch 重叠 Jaccard 0.923（48∩52/并52；JD0.941/FK0.857/CSDS1.000）。
  flip 率 JD28.3/FK35.0/CSDS20.0（+1/+3/+0 行，无一改善），错分 92.3% 偏松侧
  （C 判正常/不转）。结论：稳定偏松（2.1% ≤ 10% 阈值），瓶颈在 A/B 召回或任务难度，
  停调判官转源侧。

## 8 局限与复现命令

局限（照抄各任务 concerns，不淡化）：①三源 n=60/60/70 均为小样本，全部 CI 重叠，
不作显著性断言。②T16 39.2% 为难例富集口径，与全量不直接对标；T17 60.0% 与 T16
口径不同不直接对标。③阈值 +8.3pt 为 in-sample，无留出验证。④off-rerun 有 2 行基建
DNS 失败（非模型行为）。⑤handoff 零翻转解释：CSDS 显式客服指向词命中仅 17/70=24.3%
（订单 13/售后 3/退款 2/客服 1），加第二人称/请求标记并集 30/70=42.9% 仍为少数，
"文本自带客服上下文"解释无数据支撑，记待查。⑥外部集局限见 §2.1 缺口段
（crosswoz 标签词泄漏 42 条、DMR 补采 6 条语义待复核、FBS 纯 spam 需配 ham、
未声明许可集仅研究使用）。⑦latency 波动（T13 ~1.5s、T14 ~1.7~2.9s 含冷启动、
T15 ~0.9s）属正常波动；成本按 models.yaml 官方价 usage 实算（¥/千次 ≈0.14~0.20）。
⑧Jev 72.6% 与 C luna 74.7% 数值接近、CI 重叠，是否等价、判官是否到顶，待源侧实验支撑。

复现命令（key 自备，缺 key 停线；`abc_out/` 产物按惯例 gitignored 不入库）：

```bash
python -m pytest -q                                   # 124 passed + 1 skipped
python training/eval_jev_v4.py                        # T13 裸 190（ok190）
python training/eval_jev_v4.py --scn off --out training/abc_out/jev_v4_off_rerun.jsonl  # R1 同窗基线
python training/eval_jev_v4.py --scn on                # T14 SCN（ok190）
python training/eval_jev_v4.py --scn scnv2             # T15 SCNv2（ok190）
python training/eval_spam_only.py                     # T16 spam 51（ok51）
python training/eval_spam120.py                       # T17 spam 120（ok120，落 gold_spam120.jsonl）
python training/eval_spam_probs.py                    # T18 概率阈值（落 spam120_probs.jsonl，推荐 τ=0.10）
```

数据提交（tracked，共约 20 文件；xlsx/cache/ledger/`__pycache__` 不进）：

```text
benchmarks/data/external/cped.jsonl
benchmarks/data/external/crosswoz.jsonl
benchmarks/data/external/dmr.jsonl
benchmarks/data/external/ewect.jsonl
benchmarks/data/external/fbs.jsonl
benchmarks/data/external/smp2019_ecdt.jsonl
benchmarks/data/external/waimai_10k.jsonl
benchmarks/data/external/weibo_senti_100k.jsonl
benchmarks/data/external/external_SAMPLING_REPORT.md
benchmarks/data/external/fetch_sample.py
benchmarks/data/external/probe_spam_t6.jsonl          # 100（探针 150 之半）
benchmarks/data/external/probe_handoff_t6.jsonl       # 50（探针 150 之半）
benchmarks/data/external/probe_handoff_t6_70.jsonl    # 70（handoff 终版）
benchmarks/data/external/probe_t6_190.jsonl           # 190（gold 源）
benchmarks/data/external/probe_t6_fetch_report.json
benchmarks/data/external/probe_t6_gates.json
benchmarks/data/external/gold_spam120.jsonl           # 120 直标（垃圾 71/正常 49）
benchmarks/data/external/gold_frozen190.jsonl         # 190（spam120/handoff70）
docs/jev-v4-report.md                                 # 本报告
```

未碰清单：`benchmarks/models.yaml`、`docs/report.md`、`README.md` /
`README.zh-CN.md` 榜单行、ABC 默认链（`--scn` 默认 off，`--calib`/`--cautious` 默认 off）。
