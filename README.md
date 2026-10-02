# judgekit

[![CI](https://github.com/lexingtonhibiki/judgekit/actions/workflows/ci.yml/badge.svg)](https://github.com/lexingtonhibiki/judgekit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

**English** | [简体中文](README.zh-CN.md)

**Classify, route, score, or verify with one YAML task. Get a consistent decision JSON and a cost record.**

Start with free local keyword rules. When you need a model, the same task can use
TypeSafe Jev's native API, an OpenAI-compatible endpoint, or local NanoJev.
Use it for small, clearly defined decisions in scripts, support workflows, and applications.

## Try it in 30 seconds: no key, no API spend

```bash
git clone https://github.com/lexingtonhibiki/judgekit
cd judgekit
python -m pip install -e .
python -m judgekit demo
```

The demo works from any directory after installation, uses only local rules, and never
loads a provider registry or makes an API call. stdout contains four JSON records:
three matches and one visible `rules-no-hit`. The shipping record includes:

```json
{"primitive": "route", "value": "shipping", "provider": "rules", "cost": 0.0, "ok": true}
```

These are selected examples, not an accuracy benchmark. Rule confidence is heuristic.
The demo exits 0 when all four expected outcomes occur, including the deliberate no-hit.
Try the Chinese demo, or pass your own text to the existing YAML:

```bash
python -m judgekit demo --lang zh
python -m judgekit judge judgekit/examples/triage.en.yaml "my parcel has not arrived"
python -m judgekit judge judgekit/examples/triage.yaml "快递三天了还没到，催单"
```

`judge` exits 1 for an unmatched input. Custom YAML paths are relative to your current directory;
the built-in demo needs no path or checkout.

[Installation and Windows instructions](docs/usage.md#installation) · [Python API / providers / exit codes](docs/usage.md) · [Evaluation evidence](docs/evaluation.md)

## One task definition, multiple backends

```yaml
name: ticket-routing
primitive: route
criteria: which department should handle this ticket
labels:
  refund: returns, refunds, exchanges, invoices
  shipping: parcels, tracking, delivery
provider: rules
fallback_rules:
  refund: [refund, return, invoice]
  shipping: [parcel, tracking, delivery]
```

| Primitive | Output | Example |
|---|---|---|
| `classify` | Label | Sentiment or spam classification |
| `route` | Routing label | Support ticket routing |
| `score` | 0–1 score | Ranking learning topics or job matches |
| `verify` | Boolean | Checking a clearly defined criterion |

- Rules run locally and support `classify` / `route`; `score` / `verify` require a model.
- Native Jev / NanoJev can provide label probability distributions; OpenAI-compatible backends parse and validate JSON responses.
- `run_task` passes only the declared `input_field` (default `text`) to providers. Gold labels and extra metadata stay local.
- Failed model decisions can use explicit keyword fallbacks for classification and routing. Decisions record the actual backend and failure reason.
- Costs use the prices you configure. An unconfigured price, zero subscription marginal cost, or zero local API spend does not mean zero total cost.
- Billed requests that fail response parsing can be undercounted; check provider bills for total spend.
- Statistical confidence calibration depends on the backend and your data. Low confidence does not automatically trigger a rule fallback.

```python
from judgekit import Task, run_task

task = Task.load("judgekit/examples/triage.en.yaml")
decision = run_task(task, {"text": "my parcel has not arrived"}, {})
print(decision.value, decision.provider, decision.cost)  # shipping rules 0.0
```

## Use it in a workflow

```bash
python -m judgekit run judgekit/examples/triage.yaml --input benchmarks/data/econ_zh/intent_zh.jsonl --limit 3 --out decisions.jsonl
```

Each input line is a JSON object; each output line is a decision.
`--input -` reads stdin. `--fail-under 80` exits with 2 if the successful execution
rate falls below 80%. It checks the proportion of `ok` results, not classification
accuracy or whether a remote model succeeded.

Provider configuration lives in [benchmarks/models.yaml](benchmarks/models.yaml).
Use `--provider typesafe` with `--providers benchmarks/models.yaml` to override the
YAML's rule backend. The CLI reads environment variables and **does not load `.env`
automatically**. Configure keys, endpoints, and prices before using a remote model.
See the [usage guide](docs/usage.md#changing-providers) for a complete example.

## Two runnable recipes

| Recipe | Purpose | Free local trial |
|---|---|---|
| learn_next | Rank what to study next using prerequisites, mastery gaps, and exam weights | `python recipes/learn_next/next.py` |
| resume_lens | Give a job seeker matching suggestions using a resume, jobs, and a preference survey | `python recipes/resume_lens/match.py` |

Both default to local heuristics; `--model` enables model scoring.
People make the decisions. The resume recipe keeps the preference survey and
is intended for job seekers rather than employer screening.

## Evaluation: check the dataset version

judge-econ compares accuracy, latency, and configured cost on the same tasks.
Small-sample limitations and threshold-selection caveats remain explicit.

| Dataset / run | Reported observation | Scope |
|---|---|---|
| **Current econ_zh v2** | Rule baseline **107/130 = 82.3%**, four Chinese tasks | Removes v1 label/keyword and punctuation shortcuts; offline rerun on 2026-10-02 |
| Historical v0.1 mini sets | Jev **127/130 = 97.7%**; historical rules **119/130 = 91.5%** | Results from 2026-09-20 on old data, not model measurements on current v2 |
| External gold_spam120 | Argmax **72/120 = 60.0%**; τ=0.10 **82/120 = 68.3%** | 120 individually human-labeled items; threshold selected on the same set, **in-sample, no holdout** |

Run the current baseline at zero cost:

```bash
python benchmarks/run_bench.py --models rules --limit 0 --datasets intent_zh,sentiment_zh,spam_zh,urgency_zh
```

The [evaluation evidence](docs/evaluation.md) includes detailed results, historical
figures, failures, and reproduction limits. The [v4 report](docs/jev-v4-report.md)
documents the external study's protocol, prompts, negative results, and limitations;
the [Chinese research write-up](docs/release-post-v0.3.md) explains the threshold experiment.
The current v2 rule rerun does not call remote models or verify historical model results.

## Development and contributions

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

CI covers Ubuntu / Windows × Python 3.10 / 3.12.
Contributions to documentation, offline examples, provider mock tests, and evaluation
data with clear sources and licenses are welcome.
[Contribution guide](CONTRIBUTING.md) · [Report an issue](https://github.com/lexingtonhibiki/judgekit/issues)

## Next steps

- Run fresh model comparisons on current v2 data, retaining per-item outputs and failures.
- Validate probability thresholds on an independent holdout set, including recall and false positives.
- Add real-world datasets with publication rights and clear labels, and runnable recipes.
- Verify release packaging and demo materials before preparing a formal release.

## License

[MIT](LICENSE)
