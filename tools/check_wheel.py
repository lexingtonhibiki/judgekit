"""Check the installed wheel, reusing dev-environment PyYAML without downloading it."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wheel", nargs="?", type=Path, help="wheel path; default: the single judgekit wheel in dist/")
    args = ap.parse_args()
    wheels = [args.wheel] if args.wheel else sorted(Path("dist").glob("judgekit-*.whl"))
    if len(wheels) != 1 or not wheels[0].is_file():
        ap.error("provide one built judgekit wheel, or build it into dist/ first")
    wheel = wheels[0].resolve()

    with tempfile.TemporaryDirectory(prefix="judgekit-wheel-") as directory:
        outside = Path(directory).resolve()
        assert outside.parent == Path(tempfile.gettempdir()).resolve()
        environment = outside / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
        bin_dir = environment / ("Scripts" if os.name == "nt" else "bin")
        python = bin_dir / ("python.exe" if os.name == "nt" else "python")
        console = bin_dir / ("judgekit.exe" if os.name == "nt" else "judgekit")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["PYTHONNOUSERSITE"] = "1"
        install = subprocess.run(
            [str(python), "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall", str(wheel)],
            cwd=outside, env=env, capture_output=True, text=True, encoding="utf-8",
        )
        if install.returncode:
            sys.exit(install.stdout + install.stderr)

        # Loaded at interpreter startup for both the console script and -m entry.
        (outside / "sitecustomize.py").write_text(
            "import socket, subprocess, sys\n"
            "def forbidden(*args, **kwargs):\n"
            "    raise RuntimeError('wheel demo attempted a network call or child process')\n"
            "socket.socket.connect = forbidden\n"
            "socket.socket.connect_ex = forbidden\n"
            "socket.create_connection = forbidden\n"
            "subprocess.Popen = forbidden\n"
            "print('wheel-smoke: offline guard active', file=sys.stderr)\n",
            encoding="utf-8",
        )
        env["PYTHONPATH"] = str(outside)
        probe = subprocess.run(
            [str(python), "-c", "import json, judgekit; print(json.dumps(judgekit.__file__))"],
            cwd=outside, env=env, capture_output=True, text=True, encoding="utf-8", check=True,
        )
        loaded = Path(json.loads(probe.stdout)).resolve()
        assert loaded.is_relative_to(environment), f"loaded source checkout instead of wheel: {loaded}"
        for command in ([str(python), "-m", "judgekit"], [str(console)]):
            for lang, values in (
                ("en", ["refund", "shipping", "technical", None]),
                ("zh", ["退款售后", "物流查询", "技术故障", None]),
            ):
                result = subprocess.run(
                    [*command, "demo", "--lang", lang], cwd=outside, env=env,
                    capture_output=True, text=True, encoding="utf-8",
                )
                assert result.returncode == 0, result.stderr
                assert "wheel-smoke: offline guard active" in result.stderr
                rows = [json.loads(line) for line in result.stdout.splitlines()]
                assert [row["value"] for row in rows] == values
                assert [row["ok"] for row in rows] == [True, True, True, False]
                assert all(row["provider"] == "rules" and row["cost"] == 0 for row in rows)
                assert rows[-1]["error"] == "rules-no-hit"
        print(f"wheel smoke passed: {wheel.name}; installed package, en/zh, module/console, offline guard")


if __name__ == "__main__":
    main()
