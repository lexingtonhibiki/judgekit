"""Offline CLI regressions: preserve source data and every attempted row."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


TASK = Path(__file__).resolve().parents[1] / "judgekit/examples/triage.en.yaml"
VALID = json.dumps({"id": "last", "text": "please refund this order"}) + "\n"


def run_batch(data, *args, stdin=None):
    return subprocess.run(
        [sys.executable, "-m", "judgekit", "run", str(TASK), "--input", str(data), *args],
        input=stdin, capture_output=True, text=True, encoding="utf-8", timeout=30,
    )


@pytest.mark.parametrize("invalid", ["[]", "null", '"a string"', "42"])
def test_non_object_input_is_an_event_and_batch_continues(tmp_path, invalid):
    data, out = tmp_path / "input.jsonl", tmp_path / "out.jsonl"
    data.write_text(invalid + "\n" + VALID, encoding="utf-8")
    result = run_batch(data, "--out", str(out))
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(records) == 2
    assert records[0]["ok"] is False and records[0]["error"].startswith("bad-input:")
    assert records[1]["id"] == "last" and records[1]["value"] == "refund"
    assert out.read_text(encoding="utf-8") == result.stdout


def test_bad_json_is_saved_with_successes_and_gate_counts_it(tmp_path):
    data, out = tmp_path / "input.jsonl", tmp_path / "out.jsonl"
    data.write_text("not-json\n" + VALID, encoding="utf-8")
    result = run_batch(data, "--out", str(out), "--fail-under", "60")
    assert result.returncode == 2, result.stderr
    assert out.read_text(encoding="utf-8") == result.stdout
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(records) == 2 and records[0]["error"].startswith("bad-json:")
    assert records[1]["value"] == "refund"
    assert "50.0%" in result.stderr


def test_bad_json_counts_toward_limit(tmp_path):
    data = tmp_path / "input.jsonl"
    data.write_text("not-json\n" + VALID, encoding="utf-8")
    result = run_batch(data, "--limit", "1", "--fail-under", "100")
    assert result.returncode == 2, result.stderr
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(records) == 1 and records[0]["error"].startswith("bad-json:")
    assert "1 decisions" in result.stderr


@pytest.mark.parametrize("alias", ["same", "hardlink", "symlink"])
def test_input_output_aliases_are_rejected_without_truncation(tmp_path, alias):
    data = tmp_path / "input.jsonl"
    data.write_text(VALID, encoding="utf-8")
    out = data if alias == "same" else tmp_path / "alias.jsonl"
    if alias != "same":
        try:
            if alias == "hardlink":
                os.link(data, out)
            else:
                out.symlink_to(data)
        except OSError as exc:
            pytest.skip(f"{alias} unavailable on this filesystem: {exc}")
    before = data.read_bytes()
    result = run_batch(data, "--out", str(out))
    assert data.read_bytes() == before, "CLI must not truncate its own input"
    assert result.returncode == 2, result.stderr
    assert result.stdout == "" and "--out" in result.stderr


def test_redirected_stdin_is_not_truncated_by_out(tmp_path):
    data = tmp_path / "input.jsonl"
    data.write_text(VALID, encoding="utf-8")
    before = data.read_bytes()
    with data.open(encoding="utf-8") as stream:
        result = subprocess.run(
            [sys.executable, "-m", "judgekit", "run", str(TASK), "--input", "-", "--out", str(data)],
            stdin=stream, capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
    assert data.read_bytes() == before
    assert result.returncode == 2 and result.stdout == ""


def test_unwritable_output_is_an_argument_error(tmp_path):
    data = tmp_path / "input.jsonl"
    data.write_text(VALID, encoding="utf-8")
    result = run_batch(data, "--out", str(tmp_path / "missing" / "out.jsonl"))
    assert result.returncode == 2 and "Traceback" not in result.stderr
    assert data.read_text(encoding="utf-8") == VALID


@pytest.mark.parametrize("threshold", ["-1", "101", "nan"])
def test_invalid_success_gate_is_rejected(tmp_path, threshold):
    data = tmp_path / "input.jsonl"
    data.write_text(VALID, encoding="utf-8")
    result = run_batch(data, "--fail-under", threshold)
    assert result.returncode == 2 and result.stdout == ""
