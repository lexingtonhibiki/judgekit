# -*- coding: utf-8 -*-
"""judgekit 命令行：python -m judgekit run <task.yaml> --input <in.jsonl> [--out out.jsonl]"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="judgekit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="对 jsonl 每行执行一次判断")
    p.add_argument("task", help="任务 YAML 路径")
    p.add_argument("--providers", default=None, help="providers.yaml 路径（缺省用任务内联 fallback）")
    p.add_argument("--input", required=True, help="输入 jsonl，每行一个 JSON 对象")
    p.add_argument("--out", default=None, help="输出 jsonl 路径（缺省只打印）")
    p.add_argument("--limit", type=int, default=0, help="最多处理 N 行（0=全部）")
    args = ap.parse_args()

    from .engine import Task, run_task
    task = Task.load(args.task)
    providers = {}
    if args.providers:
        from .providers import load_providers
        providers = load_providers(args.providers)

    done = 0
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            x = {k: v for k, v in row.items() if k != "id"}
            dec = run_task(task, x, providers)
            rec = {"id": row.get("id", done), "input": x, **dec.to_dict()}
            print(json.dumps(rec, ensure_ascii=False))
            if args.out:
                with open(args.out, "a", encoding="utf-8") as w:
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done += 1
            if args.limit and done >= args.limit:
                break
    print(f"--- {done} decisions, task={task.name}, primitive={task.primitive}", file=sys.stderr)


if __name__ == "__main__":
    main()
