"""A tagged report must not mix versions or misidentify providers."""
import csv
import json
import sys

import pytest

from benchmarks import report


def test_report_selects_one_tag_and_uses_it_for_errors(tmp_path, monkeypatch):
    row = {"id": "one", "gold": "spam", "pred": "spam", "correct": True,
           "confidence": 0.9, "cost": 0.0, "latency_ms": 1, "ok": True}
    bad = {**row, "pred": "normal", "correct": False}
    for name, rec in [("rules__spam_zh.jsonl", bad),
                      ("rules__spam_zh__v2.jsonl", row),
                      ("rules__spam_zh__r2.jsonl", bad)]:
        (tmp_path / name).write_text(json.dumps(rec) + "\n", encoding="utf-8")
    monkeypatch.setattr(report, "RES", tmp_path)
    monkeypatch.setattr(sys, "argv", ["report.py", "--results-dir", str(tmp_path), "--tag", "v2"])
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    report.main()
    with (tmp_path / "pareto.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["provider"] == "rules"
    assert float(rows[0]["accuracy"]) == 1.0
    assert "| rules | spam_zh |" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "errors.json").read_text(encoding="utf-8")) == []


def test_default_report_excludes_every_tag(tmp_path, monkeypatch):
    row = {"id": "one", "gold": "spam", "pred": "spam", "correct": True,
           "confidence": 0.9, "cost": 0.0, "latency_ms": 1, "ok": True}
    (tmp_path / "rules__spam_zh.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    tagged = {**row, "pred": "normal", "correct": False}
    (tmp_path / "rules__spam_zh__v2.jsonl").write_text(json.dumps(tagged) + "\n", encoding="utf-8")
    monkeypatch.setattr(report, "RES", tmp_path)
    monkeypatch.setattr(sys, "argv", ["report.py", "--results-dir", str(tmp_path)])
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    report.main()
    with (tmp_path / "pareto.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["provider"] == "rules"
    assert float(rows[0]["accuracy"]) == 1.0
    assert json.loads((tmp_path / "errors.json").read_text(encoding="utf-8")) == []


def test_missing_tag_fails_without_writing_report(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["report.py", "--results-dir", str(tmp_path), "--tag", "missing"])
    with pytest.raises(SystemExit) as error:
        report.main()
    assert error.value.code == 2
    assert not (tmp_path / "report.md").exists()


def test_report_discloses_successful_rule_fallback(tmp_path, monkeypatch, capsys):
    from benchmarks.run_bench import bench_one
    from judgekit.engine import Task

    class Unavailable:
        def decide(self, task, row):
            raise RuntimeError("offline fixture failure")

    task = Task.from_dict({"primitive": "classify", "labels": ["refund"],
                           "provider": "model-under-test",
                           "fallback_rules": {"refund": ["refund"]}})
    row = bench_one(task, {"model-under-test": Unavailable()},
                    {"id": "one", "text": "refund please", "label": "refund"})
    assert row["ok"] and row["correct"] and row["provider"] == "rules-after-fail"
    (tmp_path / "model-under-test__routing.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["report.py", "--results-dir", str(tmp_path)])
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    report.main()
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "rule_fallbacks" in text
    assert "| model-under-test | routing | 1 | 100.0%" in text
    assert "| 0 | 1 |" in text  # The final decision succeeded; one fallback occurred.
    assert "model-under-test [rules fallback: 1]" in text
    assert "model-under-test [rules fallback: 1]" in capsys.readouterr().out
    with (tmp_path / "pareto.csv").open(encoding="utf-8", newline="") as stream:
        summary = list(csv.DictReader(stream))
    assert summary == [{"provider": "model-under-test", "accuracy": "1.0",
                        "cost_per_1k": "0.0", "rule_fallbacks": "1"}]


def test_fallback_counts_aggregate_without_changing_accuracy_or_errors(tmp_path, monkeypatch):
    row = {"id": "one", "gold": "spam", "pred": "spam", "correct": True,
           "confidence": 0.9, "cost": 0.0, "latency_ms": 1, "ok": True,
           "provider": "model"}
    fallback = {**row, "provider": "rules-after-fail"}
    wrong_fallback = {**fallback, "pred": "normal", "correct": False}
    failed = {**row, "pred": None, "correct": False, "ok": False}
    for name, records in [
        ("model__first.jsonl", [fallback, wrong_fallback, failed]),
        ("model__second.jsonl", [row]),
        ("rules__first.jsonl", [{**row, "provider": "rules"}]),
        ("legacy__first.jsonl", [{k: v for k, v in
                                  {**row, "pred": "normal", "correct": False}.items()
                                  if k != "provider"}]),
        ("model__first__other-tag.jsonl", [fallback]),
    ]:
        (tmp_path / name).write_text(
            "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["report.py", "--results-dir", str(tmp_path)])
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    report.main()
    with (tmp_path / "pareto.csv").open(encoding="utf-8", newline="") as stream:
        summary = {r["provider"]: r for r in csv.DictReader(stream)}
    assert summary["model"]["accuracy"] == "0.5"
    assert summary["model"]["rule_fallbacks"] == "2"
    assert summary["rules"]["rule_fallbacks"] == "0"
    assert summary["legacy"]["rule_fallbacks"] == "0"
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "| model | first | 3 | 33.3%" in text
    assert "| 1 | 2 |" in text  # One final failure, two successful rule fallbacks.
    assert "model [rules fallback: 2]" in text
    assert "rules [rules fallback:" not in text
    errors = json.loads((tmp_path / "errors.json").read_text(encoding="utf-8"))
    assert len(errors) == 3
    assert [r["provider"] for r in errors if r["file"] == "model__first"] == [
        "rules-after-fail", "model"]
    assert [r["provider"] for r in errors if r["file"] == "legacy__first"] == [None]


def test_png_labels_disclose_rule_fallback(tmp_path, monkeypatch):
    from types import ModuleType

    row = {"id": "one", "gold": "spam", "pred": "spam", "correct": True,
           "confidence": 0.4, "cost": 0.0, "latency_ms": 1, "ok": True,
           "provider": "rules-after-fail"}
    (tmp_path / "model__spam.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    matplotlib = ModuleType("matplotlib")
    pyplot = ModuleType("matplotlib.pyplot")
    calls = {}
    for name in ("use", "figure", "scatter", "annotate", "xscale", "xlabel", "ylabel",
                 "title", "grid", "tight_layout", "savefig"):
        def record(*args, _name=name, **kwargs):
            calls[_name] = args
        setattr(matplotlib if name == "use" else pyplot, name, record)
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    monkeypatch.setattr(sys, "argv", ["report.py", "--results-dir", str(tmp_path)])
    report.main()
    assert calls["annotate"][0] == "model [rules fallback: 1]"
    assert "end-to-end" in calls["ylabel"][0] and "fallback" in calls["ylabel"][0]
    assert calls["savefig"][0] == tmp_path / "pareto.png"
