# User Guide

[Home](../README.md) · [简体中文](usage.zh-CN.md)

## Installation

Python 3.10 or later. Install from source; the current quickstart does not depend on the package of the same name on PyPI.

```bash
git clone https://github.com/lexingtonhibiki/judgekit
cd judgekit
python -m venv .venv
```

macOS / Linux:

```bash
source .venv/bin/activate
python -m pip install -e .
```

Windows PowerShell (no virtual environment activation required):

```powershell
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m judgekit judge judgekit/examples/triage.en.yaml "my parcel has not arrived"
```

In subsequent examples, `python` refers to the interpreter in which judgekit is installed. The only required dependency is PyYAML.

## Single-item and batch processing

```bash
python -m judgekit judge judgekit/examples/triage.yaml "快递三天了还没到，催单"
python -m judgekit run judgekit/examples/triage.yaml --input benchmarks/data/econ_zh/intent_zh.jsonl --limit 3 --out decisions.jsonl
```

stdout emits JSON; batch summaries are written to stderr. `--out` overwrites the file each time.
The `provider` field identifies the backend actually used for the invocation; `rules-after-fail` indicates a rules fallback after a remote failure, in which case `error` retains the failure reason. `cost` is calculated using the provider's configured prices; a zero value for an unconfigured price does not mean free.

| Situation | Exit code |
|---|---:|
| Successful single-item decision | 0 |
| No match or decision failure for a single item; empty batch input | 1 |
| Argument error; batch successful execution rate below `--fail-under` | 2 |

Batch runs exit with 0 by default, even if some lines fail. Use a successful execution rate gate for workflows:

```bash
python -m judgekit run judgekit/examples/triage.yaml --input tickets.jsonl --fail-under 80
```

Each input line is a JSON object, such as `{"id": "t1", "text": "想退款"}`.
`--input -` reads from stdin; on macOS/Linux, use `cat tickets.jsonl | ...`; in PowerShell, use `Get-Content -Encoding utf8 tickets.jsonl | ...`.
`--fail-under` checks the proportion of `ok=true`; **it does not check classification accuracy or require the decision to come from a remote model**.
Deployment gates should also check the actual provider and your own gold labels/business rules.

## Four primitives

| Primitive | Return value | Rules backend |
|---|---|---|
| `classify` | Label string | Keyword match |
| `route` | Department/routing label string | Keyword match |
| `score` | 0–1 score | No rules fallback |
| `verify` | Boolean | No rules fallback |

See the [English triage YAML](../judgekit/examples/triage.en.yaml), [Chinese triage YAML](../judgekit/examples/triage.yaml), and [scoring YAML](../judgekit/examples/score_comment.yaml) for definitions.
Rules use case-sensitive substring counts; ties are resolved by selecting the rule that appears first in the YAML; if there is no match, the operation returns a failure.
Rule confidence values are heuristic; confidence values from OpenAI-compatible backends are self-reported by the model; native backends provide the full label distribution when available. The statistical calibration of these values must be validated separately on your data.

## Changing providers

`--provider` overrides the provider in the task YAML; `--providers` loads the registry.
Supported provider types are `typesafe`, `openai`, `rules`, `nanojev`, `go-chat`, and `go-responses`; ready-made configurations are in `benchmarks/models.yaml`.

The CLI **does not automatically load `.env`**. `.env.example` is a reference for variable names; before making a call, export the key in the current terminal, or have your own application load environment variables. Do not put keys in YAML, command history, or commits.
For example, once the variable exists in the environment:

```bash
python -m judgekit run judgekit/examples/triage.yaml --providers benchmarks/models.yaml --provider typesafe --input benchmarks/data/econ_zh/intent_zh.jsonl --limit 3
```

This command makes a remote API call. Always verify the model IDs, endpoints, prices, and account quota in `models.yaml`.
The rules fallback only handles failures for `classify` / `route`; it is not triggered automatically based on low confidence.

## Python API and input boundaries

```python
from judgekit import Task, run_task

task = Task.load("judgekit/examples/triage.en.yaml")
decision = run_task(task, {"text": "my parcel has not arrived"}, {})
print(decision.value, decision.provider, decision.cost)
```

`run_task` passes only the task's `input_field` (default: `text`) to the provider; gold labels and other metadata in the same row remain local.
After customizing `input_field: body`, batch input uses the `body` field instead; single-item `judge` automatically populates that field.
Structured objects for scoring are also nested inside the field, such as `{"text": {"course": "math", "mastery": 0.2}}`.

## Two local recipes

```bash
python recipes/learn_next/next.py
python recipes/resume_lens/match.py
```

The sample data included with the repository provides either a “what to learn next” recommendation or a candidate-side job matching reference, with no key and no API cost.
Use `--model typesafe` (or another configured provider name) to enable model calls; the default registry is `benchmarks/models.yaml`.
`learn_next --top 3` limits only the number of candidates sent to the judge, not the number of results ultimately displayed.
The resume recipe first collects an intent questionnaire for candidate-side reference; it is not intended for automatic candidate elimination by companies.
