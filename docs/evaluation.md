# Evaluation evidence

Dataset versions are part of the result. A benchmark command using the current
`econ_zh` files does **not** reproduce the historical v0.1 leaderboard.
All figures below include unsuccessful decisions in the accuracy denominator.

## Current mini sets: econ_zh v2

Offline rule baseline, rerun on 2026-10-02:

| Dataset | Samples | Correct | Accuracy | Wilson 95% CI | Unsuccessful decisions |
|---|---:|---:|---:|---:|---:|
| intent_zh | 40 | 33 | 82.5% | [68.0%, 91.3%] | 6 |
| sentiment_zh | 30 | 22 | 73.3% | [55.6%, 85.8%] | 8 |
| spam_zh | 30 | 30 | 100.0% | [88.6%, 100.0%] | 0 |
| urgency_zh | 30 | 22 | 73.3% | [55.6%, 85.8%] | 8 |
| **Total** | **130** | **107** | **82.3%** | **[74.8%, 87.9%]** | **22** |

```bash
python benchmarks/run_bench.py --models rules --limit 0 --datasets intent_zh,sentiment_zh,spam_zh,urgency_zh
```

This command is offline, costs zero, and writes JSONL to `benchmarks/results/`.
Unsuccessful decisions here are keyword no-hits. See the
[dataset index](../benchmarks/data/DATA_INDEX.md) for the v2 rewrite and original
label/keyword leakage. These are small in-house sets, not a production estimate.
The historical model figures below must not be presented as v2 measurements.

## Historical mini sets: v0.1, 2026-09-20

The original README reported these results on the old 130-item mini sets:

| Provider | Correct / total | Accuracy | Reported mean latency | Reported cost / 1,000 |
|---|---:|---:|---:|---:|
| TypeSafe Jev (`jev-1.13.0`) | 127/130 | 97.7% | ~890 ms | ¥0.105 |
| GLM-5.3-flash (coding-plan endpoint) | 127/130 | 97.7% | ~4.0 s | ¥0 subscription marginal cost |
| DeepSeek-flash | 125/130 | 96.2% | ~1.1 s | Usage based |
| Keyword rules | 119/130 | 91.5% | ~0 ms | ¥0 |
| NanoJev 0.6B (local, zero-shot Chinese transfer) | 80/130 | 61.5% | ~220 ms | ¥0 API spend |

These are historical observations from the original README, retained for context;
we did not rerun remote models during the 2026-10-02 maintenance pass.
Subscription marginal cost and local API spend exclude subscription fees,
hardware, and electricity. Historical prices and model IDs are not current offers.
The v1 data is frozen in `benchmarks/data/deprecated_econ_v1/`; v2 intentionally
removes shortcuts that made the original rule baseline easier.

The original confidence analysis found three errors below 0.7 confidence and
9.2% of decisions below that threshold on this same small set. That observation
does not establish calibration on other data, or the accuracy of a rule fallback.
`temperature=0` and repeated identical outputs are not a general determinism
guarantee. The engine's automatic fallback is triggered by **failed decisions**;
it has no built-in confidence-threshold policy. Apply any threshold in your own
application and validate it on held-out examples.

## External spam study: gold_spam120

The 120 public-source items were labeled individually by a human (71 spam,
49 normal). The published results are:

- Historical argmax run: **72/120 = 60.0%**, Wilson 95% CI [51.1%, 68.3%].
- Threshold run at **τ=0.10: 82/120 = 68.3%**. The threshold was chosen by Youden's
  statistic on the same 120 items; this is an **in-sample result with no holdout**.
- The threshold run's own argmax was 73/120 = 60.8%, one item different from the
  historical argmax run. The within-run increase is 7.5 percentage points.

No claim of generalization follows from this threshold search. The full
[v4 report](jev-v4-report.md) documents prompts, sampling, negative results and
limitations. The [Chinese write-up](release-post-v0.3.md) explains the probability
experiment. The older 190-item mixed-gold runs remain exploratory, not headline
results.

## Run a new model comparison

Configure providers and token prices in `benchmarks/models.yaml`, export the
named API-key environment variables, then use a separate output directory or
`--tag` for each run. Remote calls consume the selected provider's quota/budget.

```bash
python benchmarks/run_bench.py --models rules,typesafe --limit 0 --datasets intent_zh,sentiment_zh,spam_zh,urgency_zh --tag v2
python benchmarks/report.py
```

The first command evaluates **v2**, not v1. The report requires the optional
plotting dependency (`python -m pip install -e ".[plot]"`). Retain the dataset
revision, model ID, provider configuration, raw per-item outputs, failures and
run date when reporting a result. A benchmark subprocess exit of zero only means
the runner finished; inspect the `ok`, `error`, and `provider` fields before
claiming that a remote model was evaluated successfully.
