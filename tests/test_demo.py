"""The packaged demo must be useful even with network access forbidden."""
import json
import socket
import subprocess
import sys

import pytest

from judgekit import cli


@pytest.mark.parametrize("lang,values", [
    ("en", ["refund", "shipping", "technical", None]),
    ("zh", ["退款售后", "物流查询", "技术故障", None]),
])
def test_demo_is_offline_and_does_not_need_checkout(tmp_path, monkeypatch, capsys, lang, values):
    def forbidden(*args, **kwargs):
        pytest.fail("demo must not use network, external processes, or provider config")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr("judgekit.providers.load_providers", forbidden)
    monkeypatch.setattr(sys, "argv", ["judgekit", "demo", "--lang", lang])
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 0
    output = capsys.readouterr()
    records = [json.loads(line) for line in output.out.splitlines()]
    assert [row["value"] for row in records] == values
    assert [row["ok"] for row in records] == [True, True, True, False]
    assert all(row["provider"] == "rules" and row["cost"] == 0 for row in records)
    assert records[-1]["error"] == "rules-no-hit" and "no-hit" in output.err


def test_demo_reports_unexpected_sample_behavior(monkeypatch, capsys):
    from judgekit.engine import Decision

    monkeypatch.setattr(
        "judgekit.engine.run_task",
        lambda *args, **kwargs: Decision("route", "wrong", 0.4, "", "rules", 0, 0),
    )
    assert cli._cmd_demo("en") == 1
    assert "unexpected result" in capsys.readouterr().err
