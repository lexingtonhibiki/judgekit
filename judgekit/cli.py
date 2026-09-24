# -*- coding: utf-8 -*-
"""judgekit 命令行。

  judgekit judge <task.yaml> <文本...>          单条快速判断（退出码：ok=0 / 失败=1，可做 shell 门）
  judgekit run <task.yaml> --input in.jsonl     批量判断（--input - 读 stdin 管道）
  judgekit run ... --fail-under 80              ok 率低于 80% 退出码 2（CI 质量门禁）
"""
from __future__ import annotations

import argparse
import json
import sys


def _load_task_and_providers(args):
    from .engine import Task
    task = Task.load(args.task)
    if getattr(args, "provider", None):        # CLI 显式覆盖任务 YAML 里钉死的 provider
        task.provider = args.provider
    providers = {}
    if getattr(args, "providers", None):
        from .providers import load_providers
        providers = load_providers(args.providers)
        if providers and task.provider not in providers and task.provider not in ("", "rules"):
            print(f"⚠ provider {task.provider!r} 不在注册表 → 全部走规则兜底（假绿风险）", file=sys.stderr)
        elif providers and task.provider in ("", "rules"):
            print(f"⚠ task 钉在 rules：要测 LLM 请加 --provider {'/'.join(list(providers)[:3])} 覆盖", file=sys.stderr)
    return task, providers


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")  # Windows非中文locale下中文警告防转义
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")  # Windows GBK 管道防崩
    ap = argparse.ArgumentParser(prog="judgekit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="对 jsonl 每行执行一次判断")
    p.add_argument("task", help="任务 YAML 路径")
    p.add_argument("--providers", default=None, help="providers.yaml 路径（缺省用任务内联 fallback）")
    p.add_argument("--provider", default=None, help="覆盖任务 YAML 里钉死的 provider 名")
    p.add_argument("--input", required=True, help="输入 jsonl；'-' 读 stdin（管道）")
    p.add_argument("--out", default=None, help="输出 jsonl 路径（每次 run 重写）")
    p.add_argument("--limit", type=int, default=0, help="最多处理 N 行（0=全部）")
    p.add_argument("--fail-under", type=float, default=None, metavar="PCT",
                   help="ok 率（%%）低于此值时退出码 2，CI 门禁用")

    j = sub.add_parser("judge", help="单条文本快速判断（试用/调试/shell 门）")
    j.add_argument("task", help="任务 YAML 路径")
    j.add_argument("--providers", default=None, help="providers.yaml 路径")
    j.add_argument("--provider", default=None, help="覆盖任务 YAML 里钉死的 provider 名")
    j.add_argument("text", nargs="+", help="待判断文本（多词自动拼接）")

    args = ap.parse_args()
    if args.cmd == "run" and args.limit < 0:
        ap.error("--limit 需 ≥ 0")
    task, providers = _load_task_and_providers(args)

    from .engine import run_task

    if args.cmd == "judge":
        dec = run_task(task, {"text": " ".join(args.text)}, providers)
        print(json.dumps(dec.to_dict(), ensure_ascii=False))
        sys.exit(0 if dec.ok else 1)

    done = ok = 0
    try:
        fh = sys.stdin if args.input == "-" else open(args.input, encoding="utf-8-sig")
    except OSError as e:
        ap.error(f"输入打不开: {e}")
    out_fh = open(args.out, "w", encoding="utf-8") if args.out else None  # 每次 run 重写，防跨 run 追加混数据
    try:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                # 坏行不炸批次：记为 error 事件继续（借鉴 openai/evals 事件化容错）
                print(json.dumps({"id": done, "input": {"_raw": line[:200]},
                                  "primitive": task.primitive, "value": None, "ok": False,
                                  "error": f"bad-json: {e}"}, ensure_ascii=False))
                done += 1
                continue
            x = {k: v for k, v in row.items() if k != "id"}
            dec = run_task(task, x, providers)
            ok += dec.ok
            rec = {"id": row.get("id", done), "input": x, **dec.to_dict()}
            print(json.dumps(rec, ensure_ascii=False))
            if out_fh:
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done += 1
            if args.limit and done >= args.limit:
                break
    finally:
        if fh is not sys.stdin:
            fh.close()
        if out_fh:
            out_fh.close()
    if done == 0:
        print("--- 0 decisions：空输入按失败处理（exit 1）", file=sys.stderr)
        sys.exit(1)
    print(f"--- {done} decisions, task={task.name}, primitive={task.primitive}", file=sys.stderr)
    if args.fail_under is not None and done:
        rate = 100.0 * ok / done
        if rate < args.fail_under:
            print(f"fail-under: ok 率 {rate:.1f}% < {args.fail_under}%", file=sys.stderr)
            sys.exit(2)


if __name__ == "__main__":
    main()
