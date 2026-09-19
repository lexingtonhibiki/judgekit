# judge-econ 结果

协议：temperature=0，每样本 1 次判断调用，无 few-shot；CI 为 Wilson 95%。

| provider | dataset | n | accuracy (95% CI) | ¥/1000次 | 平均延迟ms | err |
|---|---|---|---|---|---|---|
| deepseek-direct | intent_zh | 40 | 97.5% [87%–100%] | 0.0000 | 1196 | 0 |
| typesafe | intent_zh | 40 | 97.5% [87%–100%] | 0.1183 | 887 | 0 |
| glm-coding-direct | intent_zh | 40 | 95.0% [83%–99%] | 0.0000 | 5008 | 0 |
| rules | intent_zh | 40 | 95.0% [83%–99%] | 0.0000 | 0 | 0 |
| deepseek-direct | sentiment_zh | 30 | 100.0% [89%–100%] | 0.0000 | 846 | 0 |
| glm-coding-direct | sentiment_zh | 30 | 100.0% [89%–100%] | 0.0000 | 2420 | 0 |
| typesafe | sentiment_zh | 30 | 100.0% [89%–100%] | 0.0992 | 885 | 0 |
| rules | sentiment_zh | 30 | 73.3% [56%–86%] | 0.0000 | 0 | 8 |
| rules | spam_zh | 30 | 100.0% [89%–100%] | 0.0000 | 0 | 0 |
| glm-coding-direct | spam_zh | 30 | 96.7% [83%–99%] | 0.0000 | 4768 | 0 |
| deepseek-direct | spam_zh | 30 | 93.3% [79%–98%] | 0.0000 | 1117 | 0 |
| typesafe | spam_zh | 30 | 93.3% [79%–98%] | 0.0982 | 919 | 0 |
| glm-coding-direct | urgency_zh | 30 | 100.0% [89%–100%] | 0.0000 | 3680 | 0 |
| typesafe | urgency_zh | 30 | 100.0% [89%–100%] | 0.1004 | 881 | 0 |
| rules | urgency_zh | 30 | 96.7% [83%–99%] | 0.0000 | 0 | 1 |
| deepseek-direct | urgency_zh | 30 | 93.3% [79%–98%] | 0.0000 | 1414 | 0 |

## Cost-Accuracy Pareto（x=¥/1000次判断, y=accuracy）

```
 96.2% | ● deepseek-direct (¥0.0/k)
 97.7% | ● glm-coding-direct (¥0.0/k)
 91.5% | ● rules (¥0.0/k)
 97.7% | ● typesafe (¥0.1051/k)
```

图：results/pareto.png
