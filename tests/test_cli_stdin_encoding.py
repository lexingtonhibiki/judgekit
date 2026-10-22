"""Byte-stream parity between JSONL files and standard input, without providers."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

TASK = Path(__file__).resolve().parents[1] / "judgekit/examples/triage.en.yaml"
VALID = b'{"id":1,"text":"please refund this order"}\n'
BOM = b"\xef\xbb\xbf"


@pytest.mark.parametrize("payload,expected_code,expected_records", [
    (VALID, 0, 1),
    (BOM + VALID, 0, 1),
    (BOM + VALID + VALID, 0, 2),
    (b"", 1, 0),
    (BOM, 1, 0),
    (BOM + VALID + BOM + VALID, 2, 2),
    (b'{"id":1,"text":"refund ' + BOM + b'"}\n', 0, 1),
])
def test_file_and_stdin_utf8_parity(tmp_path, payload, expected_code, expected_records):
    data = tmp_path / "input.jsonl"
    data.write_bytes(payload)
    results = []
    for source in (str(data), "-"):
        result = subprocess.run(
            [sys.executable, "-m", "judgekit", "run", str(TASK),
             "--input", source, "--fail-under", "100"],
            input=payload if source == "-" else None,
            capture_output=True, timeout=30,
        )
        assert result.returncode == expected_code, result.stderr.decode("utf-8")
        records = [json.loads(line) for line in result.stdout.decode("utf-8").splitlines()]
        assert len(records) == expected_records
        # Timing is not part of the input decoding contract.
        for record in records:
            record.pop("latency_ms", None)
        results.append(records)
    assert results[0] == results[1]
    if BOM in payload[len(BOM):] and b'"refund ' in payload:
        assert results[0][0]["input"]["text"] == "refund \ufeff"
    if payload == BOM + VALID + BOM + VALID:
        assert results[0][1]["error"].startswith("bad-json:")
