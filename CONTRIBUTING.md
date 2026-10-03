# Contributing

[English](CONTRIBUTING.md) | [简体中文](CONTRIBUTING.zh-CN.md)

Small, complete changes are easiest to review. You can submit a PR directly to fix documentation, add offline examples, or add provider support; for major features, open an issue first describing the intended users, inputs, expected outputs, and validation method.

## Local checks

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m judgekit demo
python -m judgekit demo --lang zh
python benchmarks/run_bench.py --models rules --limit 4
```

Default tests do not require a key and should not access remote APIs. Checks that depend on local research artifacts will explicitly skip; CI uses Ubuntu / Windows and Python 3.10 / 3.12.
Behavior changes should add an offline regression check that captures the original issue; when changing text only, verify commands, links, and consistency between both languages.

For CLI or packaging changes, also build and check the installed wheel outside the checkout:

```bash
python -m pip wheel --no-deps --wheel-dir dist .
python tools/check_wheel.py
```

The checker uses a temporary virtual environment, reuses the PyYAML installed above, and installs
the wheel with `--no-index --no-deps`. It runs both languages through the console script and
`python -m judgekit`, with network connections and child processes blocked during each demo.
CI runs this check for every supported OS/Python combination. Package installation can download
build dependencies; the demos and default tests make no paid API calls.

## Contribution types

- **Documentation and translation**: Preserve commands, paths, numbers, dataset versions, and limitations; update both English and Chinese versions, and do not present historical results as current measurements.
- **Examples and recipes**: Provide a runnable local mode with zero API spend. People make the final decisions; job-search tools should retain a preference survey and serve job seekers rather than automated employer screening.
- **Providers**: Implement `decide(task, x) -> Decision` and integrate it into the registry; submit mock HTTP tests covering failure behavior, the actual provider, and cost calculation.
- **Evaluation data**: State the source, license, sampling method, label definitions, data version, and count; separate training/threshold-selection sets from the test set. Do not include original text in the public repository if it lacks a publication license or contains personal information.

## Results and costs

Reports should state the model ID, date, data revision, prompt, denominator, number of failures, and whether rule fallback occurred.
Accuracy on small samples should include a confidence interval; a threshold selected on the same dataset must be labeled in-sample.
Subscription marginal cost, API spend, hardware, and electricity are not the same metric.
See [evaluation evidence](docs/evaluation.md) for current and historical measurement definitions.

## Submission boundaries

Do not commit `.env`, keys, private résumés, personal data, raw model response caches, internal state, or complete run logs.
Only stage files that have been explicitly reviewed; a shared working directory may contain other contributors' uncommitted research artifacts.
The PR description should explain the issue, the changed behavior, the checks actually run, and the scope that remains unverified.
