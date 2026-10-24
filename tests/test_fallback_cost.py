"""Preserve a provider's reported cost when rules recover a failed decision."""
import io
import json

import pytest

from judgekit.engine import Task, run_task
from judgekit.providers import TypeSafe


@pytest.mark.parametrize("primitive", ["classify", "route"])
@pytest.mark.parametrize("pricing,expected", [
    ({"per_decision_cost": 0.125}, 0.125),
    ({"price_in_per_1k": 0.2, "price_out_per_1k": 0.4}, 0.04),
])
@pytest.mark.parametrize("mode", ["hit", "miss", "disabled", "provider-success"])
def test_reported_provider_cost_survives_fallback(monkeypatch, primitive, pricing, expected, mode):
    task = Task.from_dict({"primitive": primitive, "labels": ["a", "b"],
                           "provider": "native", "fallback_rules": {"a": ["match"]}})
    payload = {"answers": {"d": {"type": "choice", "choice":
                                "b" if mode == "provider-success" else "unknown"}},
               "usage": {"input_tokens": 100, "output_tokens": 50}}
    calls = []

    def fake_urlopen(request, **kwargs):
        calls.append(request)
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = TypeSafe("native", **pricing)
    result = run_task(task, {"text": "none" if mode == "miss" else "match"},
                      {"native": provider}, fallback=mode != "disabled")

    assert len(calls) == 1
    assert result.cost == pytest.approx(expected)
    assert result.to_dict()["cost"] == pytest.approx(expected)
    assert result.ok == (mode in ("hit", "provider-success"))
    assert result.provider == ("rules-after-fail" if mode == "hit" else "native")
    if mode == "hit":
        assert result.value == "a"
        assert "label-not-in-candidates" in result.error and "fallback-ok" in result.error
    elif mode == "provider-success":
        assert result.value == "b" and result.error == ""


@pytest.mark.parametrize("primitive", ["classify", "route"])
def test_benchmark_keeps_recovered_cost(monkeypatch, primitive):
    from benchmarks.run_bench import bench_one

    task = Task.from_dict({"primitive": primitive, "labels": ["a"],
                           "provider": "native", "fallback_rules": {"a": ["match"]}})
    payload = {"answers": {"d": {"type": "choice", "choice": "unknown"}}}
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k:
                        io.BytesIO(json.dumps(payload).encode()))
    row = bench_one(task, {"native": TypeSafe("native", per_decision_cost=0.125)},
                    {"id": "sample", "label": "a", "text": "match"})
    assert row["correct"] and row["provider"] == "rules-after-fail"
    assert row["cost"] == 0.125


@pytest.mark.parametrize("provider", ["rules", "missing", "raised"])
def test_fallback_does_not_invent_unreported_cost(provider):
    class Raised:
        def decide(self, task, x):
            raise RuntimeError("no reported cost")

    task = Task.from_dict({"primitive": "classify", "labels": ["a"],
                           "provider": provider, "fallback_rules": {"a": ["match"]}})
    result = run_task(task, {"text": "match"}, {"raised": Raised()})
    assert result.ok and result.cost == 0
