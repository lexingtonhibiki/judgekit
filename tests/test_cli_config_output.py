"""Batch output must not destroy the configuration it was invoked with."""
import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("config_name", ["task.yaml", "providers.yaml"])
@pytest.mark.parametrize("alias", ["same", "hardlink", "symlink"])
@pytest.mark.parametrize("stdin", [False, True])
def test_output_cannot_replace_configuration(tmp_path, config_name, alias, stdin):
    task = tmp_path / "task.yaml"
    task.write_text("primitive: classify\nlabels: [refund]\nprovider: local\nfallback_rules: {refund: [refund]}\n", encoding="utf-8")
    providers = tmp_path / "providers.yaml"
    providers.write_text("providers: {local: {kind: rules}}\n", encoding="utf-8")
    data = tmp_path / "input.jsonl"
    data.write_text('{"text":"refund"}\n', encoding="utf-8")
    source = tmp_path / config_name
    output = source if alias == "same" else tmp_path / "output.jsonl"
    if alias != "same":
        try:
            if alias == "hardlink":
                os.link(source, output)
            else:
                output.symlink_to(source)
        except OSError as exc:
            pytest.skip(f"{alias} unavailable: {exc}")
    before = {p: p.read_bytes() for p in (task, providers, data)}
    result = subprocess.run(
        [sys.executable, "-m", "judgekit", "run", str(task), "--providers", str(providers),
         "--input", "-" if stdin else str(data), "--out", str(output)],
        input=data.read_text(encoding="utf-8") if stdin else None,
        text=True, capture_output=True, encoding="utf-8", timeout=30,
    )
    assert {p: p.read_bytes() for p in before} == before
    assert result.returncode == 2, result.stderr
    assert result.stdout == ""
    assert "--out" in result.stderr and "Traceback" not in result.stderr


def test_distinct_existing_output_can_still_be_replaced(tmp_path):
    task = tmp_path / "task.yaml"
    task.write_text("primitive: classify\nlabels: [refund]\nfallback_rules: {refund: [refund]}\n", encoding="utf-8")
    output = tmp_path / "output.jsonl"
    output.write_text("old result\n", encoding="utf-8")
    before = task.read_bytes()
    result = subprocess.run(
        [sys.executable, "-m", "judgekit", "run", str(task), "--input", "-", "--out", str(output)],
        input='{"text":"refund"}\n', text=True, capture_output=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert task.read_bytes() == before
    assert output.read_text(encoding="utf-8") == result.stdout
    assert json.loads(result.stdout)["value"] == "refund"
