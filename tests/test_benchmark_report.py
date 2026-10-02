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
