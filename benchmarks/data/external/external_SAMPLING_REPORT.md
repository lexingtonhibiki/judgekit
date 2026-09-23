# external 小闭环采样报告 (8x50)

- 生成时间: 2026-09-22 19:06 中国标准时间; 采样 seed=42, 每集目标 50 条; 归一化: NFKC+opencc 繁→简+去首尾空白+空白折叠+小写; 去重键=归一化 sha256.
- 污染基: training/real_pools/clean/real_sentiment.jsonl + benchmarks/data/econ_zh/*.jsonl, 共 268 条(归一化 hash); 命中即弃并补采.
- 下载方式: urllib 直连 + UA, hf-mirror 302 签名直跟, decode 全部 utf-8-sig, 重试 4 次. 未调用任何付费 API.

| 数据集 | 实际条数 | 候选池 | 去重弃数 | 污染命中 | 许可 |
|---|---|---|---|---|---|
| smp2019_ecdt(SMP2019-ECDT) | 50 | 2579 | 0 | 0 | SMP 竞赛公开数据, 研究使用(原版权归赛事组织方; 本采样仅研究用途, 注明来源) |
| crosswoz(CrossWOZ) | 50 | 42346 | 4 | 0 | Apache-2.0(HF GEM/CrossWOZ 元数据声明; 另需引用 Zhu et al. TACL 2020) |
| ewect(EWECT) | 50 | 34766 | 0 | 0 | 竞赛数据研究使用(原版权归赛事/平台方; 本采样仅研究用途, 注明来源) |
| cped(CPED) | 50 | 94187 | 0 | 0 | Apache-2.0(仓库 LICENSE) |
| waimai_10k(waimai_10k) | 50 | 11987 | 0 | 0 | 未声明(原作者/来源不详, 见 SophonPlus 说明); 仅研究使用 |
| weibo_senti_100k(weibo_senti_100k) | 50 | 119988 | 0 | 0 | 未声明(原作者/来源不详, 见 SophonPlus 说明; 原数据来自 CSDN 转载); 仅研究使用 |
| dmr(DMR) | 50 | 50 | 0 | 0 | 研究使用(遵守豆瓣隐私政策, user_id 已加密; 引用 ICPR'22 GAIM 论文) |
| fbs(FBS) | 50 | 14074 | 4 | 0 | 研究使用(需注明 source-link 并引用 CCS'20 Lies in the Air; 已做匿名化预处理版本) |

## smp2019_ecdt (SMP2019-ECDT)
- 下载: https://github.com/hml-ubt/SMP2017-2019-ECDT-data (SMP2019_data/train.json)
- 许可声明: SMP 竞赛公开数据, 研究使用(原版权归赛事组织方; 本采样仅研究用途, 注明来源)
- 标签分布: {"website:OPEN": 2, "cinemas:QUERY": 2, "novel:QUERY": 2, "stock:RISERATE_QUERY": 2, "tvchannel:PLAY": 4, "map:ROUTE": 2, "map:POSITION": 2, "stock:QUERY": 3, "email:REPLY": 2, "flight:QUERY": 2, "match:QUERY": 1, "poetry:QUERY": 2, "bus:QUERY": 1, "radio:LAUNCH": 2, "translation:TRANSLATION": 1, "cookbook:QUERY": 3, "email:FORWARD": 1, "contacts:CREATE": 1, "poetry:DEFAULT": 1, "app:QUERY": 2, "email:CREATE": 2, "telephone:DIAL": 1, "train:QUERY": 1, "email:REPLAY_ALL": 1, "telephone:QUERY": 1, "message:SENDCONTACTS": 1, "message:SEND": 1, "health:QUERY": 1, "music:SEARCH": 1, "music:PLAY": 1, "epg:QUERY": 1}
- 标签词泄漏: 0 条(通过)
- 标点分布(含！!？? 占比/类): {"website:OPEN": 0.0, "cinemas:QUERY": 0.0, "novel:QUERY": 0.0, "stock:RISERATE_QUERY": 0.0, "tvchannel:PLAY": 0.0, "map:ROUTE": 0.0, "map:POSITION": 0.0, "stock:QUERY": 0.0, "email:REPLY": 0.0, "flight:QUERY": 0.0, "match:QUERY": 0.0, "poetry:QUERY": 0.5, "bus:QUERY": 0.0, "radio:LAUNCH": 0.0, "translation:TRANSLATION": 0.0, "cookbook:QUERY": 0.667, "email:FORWARD": 0.0, "contacts:CREATE": 0.0, "poetry:DEFAULT": 1.0, "app:QUERY": 0.0, "email:CREATE": 0.0, "telephone:DIAL": 0.0, "train:QUERY": 0.0, "email:REPLAY_ALL": 0.0, "telephone:QUERY": 0.0, "message:SENDCONTACTS": 0.0, "message:SEND": 0.0, "health:QUERY": 1.0, "music:SEARCH": 0.0, "music:PLAY": 0.0, "epg:QUERY": 0.0}
- 长度: mean=9.2 min=4 max=21; 异常(<5 或 >500 字): 3 条 ['smp2019_ecdt_0001', 'smp2019_ecdt_0022', 'smp2019_ecdt_0030']
- minhash 近重复(>0.85): 0 对 (无)

## crosswoz (CrossWOZ)
- 下载: https://hf-mirror.com/datasets/GEM/CrossWOZ
- 许可声明: Apache-2.0(HF GEM/CrossWOZ 元数据声明; 另需引用 Zhu et al. TACL 2020)
- 标签分布: {"景点+餐馆": 2, "景点": 4, "景点+酒店": 10, "出租": 3, "地铁": 7, "General": 4, "酒店+餐馆": 6, "出租+地铁": 2, "酒店": 7, "餐馆": 5}
- 标签词泄漏: 42 条, 例: [["crosswoz_0002", ["景点"]], ["crosswoz_0003", ["酒店"]], ["crosswoz_0004", ["出租"]]]
- 标点分布(含！!？? 占比/类): {"景点+餐馆": 1.0, "景点": 1.0, "景点+酒店": 0.6, "出租": 1.0, "地铁": 0.714, "General": 0.5, "酒店+餐馆": 0.5, "出租+地铁": 0.0, "酒店": 0.571, "餐馆": 0.6}
- 长度: mean=37.8 min=6 max=69; 异常(<5 或 >500 字): 0 条 []
- minhash 近重复(>0.85): 0 对 (无)

## ewect (EWECT)
- 下载: https://hf-mirror.com/datasets/hecongqing/EWECT_weibo_senti (镜像; 原 SMP2020-EWECT 微博情绪竞赛数据)
- 许可声明: 竞赛数据研究使用(原版权归赛事/平台方; 本采样仅研究用途, 注明来源)
- 标签分布: {"neutral": 4, "sad": 8, "happy": 10, "angry": 7, "surprise": 10, "fear": 11}
- 标签词泄漏: 0 条(通过)
- 标点分布(含！!？? 占比/类): {"neutral": 0.0, "sad": 0.125, "happy": 0.3, "angry": 0.429, "surprise": 0.6, "fear": 0.364}
- 长度: mean=43.4 min=14 max=141; 异常(<5 或 >500 字): 0 条 []
- minhash 近重复(>0.85): 0 对 (无)

## cped (CPED)
- 下载: https://github.com/scutcyr/CPED (data/CPED/train_split.csv)
- 许可声明: Apache-2.0(仓库 LICENSE)
- 标签分布: {"depress": 3, "disgust": 9, "neutral": 5, "worried": 4, "negative-other": 3, "grateful": 1, "relaxed": 4, "happy": 6, "anger": 4, "positive-other": 6, "fear": 2, "astonished": 2, "sadness": 1}
- 标签词泄漏: 0 条(通过)
- 标点分布(含！!？? 占比/类): {"depress": 0.0, "disgust": 0.0, "neutral": 0.0, "worried": 0.0, "negative-other": 0.0, "grateful": 0.0, "relaxed": 0.0, "happy": 0.0, "anger": 0.0, "positive-other": 0.0, "fear": 0.0, "astonished": 0.0, "sadness": 0.0}
- 长度: mean=7.9 min=1 max=19; 异常(<5 或 >500 字): 15 条 ['cped_0001', 'cped_0003', 'cped_0008', 'cped_0010', 'cped_0017']
- minhash 近重复(>0.85): 0 对 (无)

## waimai_10k (waimai_10k)
- 下载: https://hf-mirror.com/datasets/dirtycomputer/waimai_10k (整理自 SophonPlus/ChineseNlpCorpus)
- 许可声明: 未声明(原作者/来源不详, 见 SophonPlus 说明); 仅研究使用
- 标签分布: {"负面": 25, "正面": 25}
- 标签词泄漏: 0 条(通过)
- 标点分布(含！!？? 占比/类): {"负面": 0.24, "正面": 0.28}
- 长度: mean=20.0 min=5 max=72; 异常(<5 或 >500 字): 0 条 []
- minhash 近重复(>0.85): 0 对 (无)

## weibo_senti_100k (weibo_senti_100k)
- 下载: https://hf-mirror.com/datasets/dirtycomputer/weibo_senti_100k (整理自 SophonPlus/ChineseNlpCorpus)
- 许可声明: 未声明(原作者/来源不详, 见 SophonPlus 说明; 原数据来自 CSDN 转载); 仅研究使用
- 标签分布: {"负面": 25, "正面": 25}
- 标签词泄漏: 0 条(通过)
- 标点分布(含！!？? 占比/类): {"负面": 0.52, "正面": 0.56}
- 长度: mean=68.9 min=9 max=166; 异常(<5 或 >500 字): 0 条 []
- minhash 近重复(>0.85): 0 对 (无)

## dmr (DMR)
- 下载: https://github.com/yiyepianzhounc/DMR-Dataset-1 (reviews.json; 主库 DMR-Dataset 仅 README 无数据文件, 取同作者 -1 库)
- 许可声明: 研究使用(遵守豆瓣隐私政策, user_id 已加密; 引用 ICPR'22 GAIM 论文)
- 标签分布: {"正常": 25, "刷评spam": 25}
- 标签词泄漏: 0 条(通过)
- 标点分布(含！!？? 占比/类): {"正常": 0.36, "刷评spam": 0.2}
- 长度: mean=65.6 min=13 max=346; 异常(<5 或 >500 字): 0 条 []
- minhash 近重复(>0.85): 0 对 (无)

## fbs (FBS)
- 下载: https://github.com/Cypher-Z/FBS_SMS_Dataset
- 许可声明: 研究使用(需注明 source-link 并引用 CCS'20 Lies in the Air; 已做匿名化预处理版本)
- 标签分布: {"spam:IL:Gambling": 4, "spam:IL:Fake_ID_and_invoice": 6, "spam:AD:Retail": 3, "spam:FR:Phishing(Other)": 1, "spam:FR:Financial": 3, "spam:IL:Political_propaganda": 2, "spam:AD:Other": 4, "spam:AD:Loan": 3, "spam:AD:Network_service": 6, "spam:FR:Phishing(Bank)": 4, "spam:Other": 4, "spam:IL:Escort_service": 3, "spam:AD:Real_estate": 4, "spam:FR:Other": 3}
- 标签词泄漏: 0 条(通过)
- 标点分布(含！!？? 占比/类): {"spam:IL:Gambling": 0.0, "spam:IL:Fake_ID_and_invoice": 0.0, "spam:AD:Retail": 0.0, "spam:FR:Phishing(Other)": 0.0, "spam:FR:Financial": 0.0, "spam:IL:Political_propaganda": 0.0, "spam:AD:Other": 0.0, "spam:AD:Loan": 0.0, "spam:AD:Network_service": 0.0, "spam:FR:Phishing(Bank)": 0.0, "spam:Other": 0.0, "spam:IL:Escort_service": 0.0, "spam:AD:Real_estate": 0.0, "spam:FR:Other": 0.0}
- 长度: mean=87.9 min=24 max=284; 异常(<5 或 >500 字): 0 条 []
- minhash 近重复(>0.85): 0 对 (无)

## 备注与缺口
- SMP2019-ECDT: 主库直取成功, 未启用 SMP2017 替代.
- DMR: 主库 yiyepianzhounc/DMR-Dataset 仅含 README 无数据文件, 改取同作者 DMR-Dataset-1/reviews.json(仅 44 条: 22 正常/22 刷评); 另从同域 HF tracywong117/spam-douban-movie-review 补 6 条凑 25/25, 补采行 url 单独标注且 Fake=1→spam 语义待人工复核.
- FBS: 纯 spam(14 类, 本批全 spam 标签); 需配 ham——ham 从 dmr.jsonl 中 orig_label=正常 的条目补(见 dmr_*.id).
- EWECT: GitHub smp2020ewect 系仓库多为代码/复现无直接数据文件, 改走 HF 镜像 hecongqing/EWECT_weibo_senti(usual_train+test_labeled+eval_labeled, 有标注版); 转人工映射(angry/sad 等→转人工)留待 ABC 打分阶段定义, 本步保留原始情绪标签.
- weibo_senti_100k: GitHub SophonPlus 仅百度网盘(不可直连), 改走 HF 镜像 dirtycomputer 版.
- CPED: 取 train_split.csv 的 Utterance, orig_label=Emotion(DA 列未收录, 转人工映射留待后步).
- 缺口: 无(8 集均满 50 条).
