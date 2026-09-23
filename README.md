# judgekit

[![CI](https://github.com/lexingtonhibiki/judgekit/actions/workflows/ci.yml/badge.svg)](https://github.com/lexingtonhibiki/judgekit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

**English** | [简体中文](README.zh-CN.md)

> **Runtime judgment engine for System One (judge) models.**
> Define a judgment task once in YAML, execute it natively on TypeSafe Jev's
> decisions API or translate it to any OpenAI-compatible LLM, get typed decisions
> with calibrated probabilities — and measure what every judgment costs.

| Project | Answers the question | When |
|---|---|---|
| [JudgeBench](https://github.com/ScalerLab/JudgeBench) (ICLR'25) / [JudgeLM](https://github.com/baaivision/JudgeLM) | Are judge models **accurate** on hard adversarial pairs? | Academic benchmarks |
| [DeepEval](https://github.com/confident-ai/deepeval) / [promptfoo](https://github.com/promptfoo/promptfoo) | Does the LLM output **pass tests**? | Dev-time / CI |
| [semantic-router](https://github.com/aurelio-labs/semantic-router) | Which model should serve this request? | Runtime, routing only |
| **judgekit** | What does **every production judgment** cost, and can I trust it? | **Runtime judgment layer** |

One line: **others measure whether judges are smart — judgekit makes judgment a
dependable, cost-accounted resident of your production stack.**

## Benchmarks (judge-econ)

judge-econ measures **cost-accuracy**, not brilliance: accuracy, latency, and
¥-per-1000-decisions across providers on the same tasks. To our knowledge these
are the first public Chinese-scenario judge evaluation numbers
([awesome-jev-zh](https://github.com/yzfly/awesome-jev-zh) explicitly lists the
lack of zh-scenario public evaluations as a gap).

### Datasets (hand-built mini sets, v0.1)

| dataset | n | task | examples |
|---|---|---|---|
| `intent_zh` | 40 | route: e-commerce ticket → department | 退货 / 物流 / 投诉 / 故障 / 咨询 |
| `sentiment_zh` | 30 | classify: review polarity | 正面 / 负面 |
| `spam_zh` | 30 | classify: comment spam | 垃圾(广告导流/灌水) / 正常 |
| `urgency_zh` | 30 | classify: support urgency | 紧急(安全/资损) / 非紧急 |

Total 130 samples, all included in this repo with keyword-rule baselines
(`*.rules.yaml`). Public large-scale sets (ag_news, sst2) are on the roadmap —
PRs welcome.

### Protocol

- One judgment call per sample; `temperature=0`; no few-shot examples.
- Accuracy reported with **Wilson 95% CI** (small-sample honest).
- Cost = provider list price at run time (Jev: $0.042/MTok input, output free; cost/1k = total cost / total decisions).
- Every run is reproducible: `python benchmarks/run_bench.py --models rules,typesafe --limit 0`.

### Results (2026-09-20, n=130)

| provider | backend | accuracy (95% CI) | latency/decision | cost / 1000 decisions |
|---|---|---|---|---|
| **typesafe** (Jev `jev-1.13.0`, native decisions API) | choice → full probability distribution | **97.7% [93.4–99.2%]** (127/130; **3 independent runs, 0 judgment flips**) | **~890 ms** | **¥0.105** |
| glm-5.3-flash (Zhipu coding-plan endpoint) | OpenAI-compatible | 97.7% (127/130; shares 2 of Jev's 3 errors) | ~4.0 s | ¥0 (subscription) |
| deepseek-flash (official API) | OpenAI-compatible | 96.2% (125/130; **missed 2 urgent tickets**) | ~1.1 s | usage-based |
| rules (keyword baseline) | — | 91.5% [85.5–95.2%] (119/130) | ~0 ms | ¥0 |
| nanojev-local (NanoJev 0.6B, [open replica](https://github.com/TianyuCodings/NanoJev), runs on this GPU) | local decisions API | 61.5% (80/130; zero-shot zh transfer from game-domain training) | **~220 ms** | ¥0 |

Reading: Jev matches the much larger GLM-5.3-flash at **4.5× lower latency** and
3 runs with zero decision drift (temperature=0 is fully deterministic here).
The keyword baseline stays competitive — traditional baselines are not dead.
Error overlap: Jev and GLM share the same 2 genuinely-borderline misjudgments;
DeepSeek additionally missed 2 urgent tickets (u009/u011), the error type that
matters most for support routing.

![cost-accuracy pareto](docs/pareto.png)

### Finding: Jev's calibrated probabilities are actually calibrated

All 3 misjudgments fell **below 0.7 confidence** (0.63 / 0.42 / 0.37), while
low-confidence (<0.7) outputs made up only 9.2% of all decisions:

> **A confidence gate at 0.7 catches 100% of errors at the price of handing
> ~9% of decisions to a free rule fallback.** This is exactly the
> "atomic task + confidence gating + explicit fallback" pattern judgekit's Task
> abstraction enforces — consistent with community findings that atomized
> judgments beat monolithic ones (62.6% → 95.0% on phishing-detection evals).

Misjudgment details ([errors.json](docs/errors.json)): 1 ambiguous
label (warranty-policy inquiry), 1 soft-ad undetected, 1 negative review flagged
as spam — all genuinely borderline, all low-confidence.

### External candidate set v4 (frozen)

 Quotable result only (120 directly-labeled spam, `gold_spam120`):
 - argmax 60.0% (72/120, 95% CI [51.1%, 68.3%]) / τ=0.10 68.3% in-sample (82/120, Youden-optimal, no holdout).
 - Probability-threshold write-up: [docs/release-post-v0.3.md](docs/release-post-v0.3.md) (zh) — argmax discards the distribution's separation signal; τ recalibration is a zero-retrain lever.
 - Other exploratory calibers (190-series mixed gold, superseded) see appendix in `docs/jev-v4-report.md` §5.2 — not quotable.

 The headline judge-econ numbers above are unchanged.

## Quick start

```bash
git clone https://github.com/lexingtonhibiki/judgekit && cd judgekit
pip install -e .              # only hard dep is pyyaml; installs `judgekit` CLI
cp .env.example .env          # optional: TYPESAFE_API_KEY (works keyless via rule fallback)

# ⓪ 10-second trial — no data file, no key (rule fallback, 0 cost); exit 0 = judged, exit 1 = no-hit/failed (shell gate)
judgekit judge judgekit/examples/triage.yaml "我的订单三天了还没发货，再不处理就投诉了"
# ① One YAML, one routing decision (rule fallback when no key → 0 cost)
judgekit run judgekit/examples/triage.yaml --input benchmarks/data/econ_zh/intent_zh.jsonl --limit 3
# ⓪b Workflow pipes & CI gate: read stdin, exit 2 when ok-rate < 80%
cat tickets.jsonl | judgekit run judgekit/examples/triage.yaml --input - --fail-under 80

# ② Same YAML natively on Jev (choice/score/noul, full probability distribution)
export TYPESAFE_API_KEY=...
judgekit run judgekit/examples/triage.yaml --providers benchmarks/models.yaml --provider typesafe \
  --input benchmarks/data/econ_zh/intent_zh.jsonl --limit 3

# ③ Full benchmark + Pareto report
python benchmarks/run_bench.py --models rules,typesafe --limit 0 \
  --datasets intent_zh,sentiment_zh,spam_zh,urgency_zh    # n=130 — the headline numbers' exact recipe
python benchmarks/report.py   # → benchmarks/results/ (copy artifacts to docs/ when publishing)

# ④ Recipes (heuristic mode is free; --model enables judge rescoring)
python recipes/learn_next/next.py --model typesafe --top 3
python recipes/resume_lens/match.py --model typesafe
```

## One task definition, any backend

```yaml
name: ticket-routing
primitive: route
criteria: which department should handle this ticket
labels:                       # map form: option → description
  refund: returns, refunds, exchanges, invoices
  logistics: shipping, tracking, delivery
provider: typesafe            # swap to any openai-compatible provider name — YAML unchanged
fallback_rules:               # zero-cost fallback when provider fails / no key
  refund: [refund, return, invoice]
```

- **typesafe backend**: native choice criteria, scored levels (normalized to
  0–1), noul → verify; returns full probability distribution + confidence
- **openai backend**: same YAML auto-translated to a JSON-only prompt (with
  level anchors); response parsed and label-validated
- **rules backend**: keyword matching on the declared `input_field` (default `text`),
  always free; fallback applies to classify/route only — score/verify fail straight
- **one input field per task**: rules matching and LLM prompts only see the declared field,
  so gold labels / metadata in your jsonl never leak into prompts or inflate hit rates

## Providers (bring your own)

Declared in `benchmarks/models.yaml`: `kind: typesafe | openai | rules`.
Keys come from env vars only (`.env.example` template) — never committed.
No Claude/GPT/Grok keys? Fine: any OpenAI-compatible endpoint (local gateways,
GLM, DeepSeek, OpenRouter…) runs the full pipeline; Jev is one provider among many.

## Recipe principles

1. Judges give probabilities, humans make decisions (willingness survey mandatory
   for people-facing recipes)
2. No per-employee labeling; no employer-side auto-screening
3. Every recipe ships a zero-cost local mode — the judge is an enhancement,
   never a dependency
4. Cost transparency: every run prints its judge spend

## Development

```bash
pip install -e .[dev]
pytest               # 124 offline unit tests (1 live-ping skipped by default)
```

CI runs tests + zero-cost smoke on Ubuntu/Windows × Python 3.10/3.12.

## Roadmap

- [x] Four-primitive engine + native TypeSafe adapter + OpenAI-compat + rules
- [x] judge-econ mini benchmark + Pareto report (Jev 97.7% @ ¥0.105/1k, n=130)
- [x] Confidence-gating study (0.7 gate → 100% error capture @ 9% escalation)
- [x] learn_next / resume_lens recipes (dual-mode)
- [x] Test suite + CI
- [ ] LLM-judge对照组 full run (GLM / DeepSeek / free chain)
- [ ] Public large-scale datasets (ag_news / sst2 / full support-intent sets)
- [ ] smart-triage recipe (12345 government hotline routing)
- [ ] PyPI release / GIF demo

## Contributing

Recipes welcome (under `recipes/`, follow the four principles); benchmark data
welcome (under `benchmarks/data/`, with a `.rules.yaml` baseline); new providers
welcome (`judgekit/providers/`, implement `decide(task, x) -> Decision`).

## License

MIT
