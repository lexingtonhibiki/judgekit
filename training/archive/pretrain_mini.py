# -*- coding: utf-8 -*-
"""MiniMind2-104M 底座预训练（从零 / 继续预训练两用）—— NanoJev 判官底座实验。

流水线：jsonl 中文语料 -> 流式 tokenize（uint16 .bin 磁盘缓存，vocab 6400 < 65536）
-> 512 定长块打包 + 块级乱序 -> AdamW(warmup 2% + cosine 到 10%) + autocast 预训练
-> 吞吐探针定总步数 + 硬墙钟停机 -> save_pretrained + tokenizer（HF 格式，可被
AutoModel.from_pretrained / train_pipeline_decisions.py 直接加载）。

语料构建方式（本实验实测口径）：
  - 来源 HF `jingyaogong/minimind_dataset`（走 hf-mirror 镜像下载），取
    pretrain_t2t_mini_part.jsonl（67.8 万行，每行一个 {"text": ...} 中文文档）。
  - 脚本流式读取 jsonl 的 text 字段，文档间补 eos，token 总量由 --token-cap
    截断（实验用 96M token，RTX 2070 上 tokenize 约 7 分钟）。
  - 两线对比时用 --reuse-bin 复用同一 .bin，保证 token 流逐字节同源。

用法（从零预训练，75 分钟时间盒：探针 300s + 训练硬墙 3300s）：
  HF_HUB_OFFLINE=1 python pretrain_mini.py --corpus corpus.jsonl \
      --tok-dir D:/Models/minimind2-mini --out D:/Models/miniexp/fromscratch \
      --bin tokens_u16.bin --csv pretrain_log.csv --dtype fp16 \
      --reuse-bin --probe-seconds 300 --train-seconds 3300

用法（继续预训练 / DAPT）：同上，但把 --out 换成新目录并加
  --init pretrained --lr 1e-4（加载现成权重 + 小 lr 防灾难遗忘）；
A/B 对比实验中两线唯一差异 = 初始化 + lr，其余逐字节同源（--reuse-bin 同一 token 流）。

显存策略：fp32 主权重 + AdamW 状态 + micro_batch x 512 autocast 前后向；
探针阶段实测显存，超过 3GiB 预算自动降 micro_batch / 加梯度累积；
训练循环内 OOM 自动跳步（>20 次才 raise）。

!! Turing（RTX 2070，sm_75）无 bf16 tensor core：bf16 autocast 走模拟路径，
!! 实测慢约 6 倍。一律用 --dtype fp16（默认即 fp16，配 GradScaler）；
!! "auto" 会先跑 7 步 bf16/fp16 基准择优，仅在非 Turing 卡上建议使用。
!! serve/推理侧不受此限制（bf16 推理可用），只是预训练循环禁 bf16。
"""
import argparse
import json
import math
import os
import time
from array import array

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True,
                   help="jsonl 语料（每行一个文档，优先取 text 字段）")
    p.add_argument("--tok-dir", required=True,
                   help="tokenizer 与 config 来源目录（也是继续预训练的权重来源）")
    p.add_argument("--out", required=True, help="save_pretrained 输出目录")
    p.add_argument("--bin", default="tokens_u16.bin")
    p.add_argument("--csv", default="pretrain_log.csv")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4,
                   help="从零预训练 3e-4；继续预训练建议 1e-4")
    p.add_argument("--init", default="scratch", choices=["scratch", "pretrained"],
                   help="scratch=随机初始化（从零）；pretrained=加载 tok-dir 现成权重（继续预训练/DAPT）")
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--warmup-frac", type=float, default=0.02)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--probe-seconds", type=int, default=300,
                   help="吞吐探针时长（权重丢弃后重建），顺带测显存决定降批")
    p.add_argument("--train-seconds", type=int, default=3300,
                   help="正式训练硬墙钟；到点即停并保存")
    p.add_argument("--token-cap", type=int, default=96_000_000)
    p.add_argument("--tokenize-timebox", type=int, default=720)
    p.add_argument("--dtype", default="fp16", choices=["auto", "bf16", "fp16", "fp32"],
                   help="默认 fp16：Turing 无 bf16 tensor core，bf16 模拟慢 ~6x")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="梯度检查点：省激活显存，换 ~33%% 计算开销")
    p.add_argument("--max-steps", type=int, default=0, help=">0 时覆盖探针估算")
    p.add_argument("--skip-probe", action="store_true",
                   help="跳过探针（必须配 --max-steps，否则 lr 计划表失真）")
    p.add_argument("--reuse-bin", action="store_true", help="复用已有 .bin 跳过 tokenize")
    return p.parse_args()


# ---------------------------------------------------------------- tokenize

def tokenize_corpus(args):
    """流式 tokenize：每行 jsonl 一个文档（text 字段），文档间补 eos；返回统计 dict。"""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tok_dir, local_files_only=True)
    eos = tok.eos_token_id
    assert eos is not None and 0 <= eos < 65536, f"eos id 越界: {eos}"
    acc = array("H")
    n_docs, bytes_read = 0, 0
    t0 = time.time()
    chunk = []

    def flush(texts):
        nonlocal n_docs
        if not texts:
            return
        enc = tok(texts, add_special_tokens=False)["input_ids"]
        for ids in enc:
            acc.extend(ids)
            acc.append(eos)
        n_docs += len(texts)

    with open(args.corpus, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            bytes_read += len(line.encode("utf-8", errors="ignore"))
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # 截断的行尾等脏数据
            text = r.get("text") if isinstance(r, dict) else None
            if not isinstance(text, str) or not text:
                if isinstance(r, dict):
                    for v in r.values():
                        if isinstance(v, str) and v:
                            text = v
                            break
            if not text:
                continue
            chunk.append(text)
            if len(chunk) >= 2000:
                flush(chunk)
                chunk = []
                if len(acc) >= args.token_cap or time.time() - t0 > args.tokenize_timebox:
                    break
    flush(chunk)
    dt = time.time() - t0
    np.frombuffer(acc, dtype=np.uint16).tofile(args.bin)
    stopped_by = ("token_cap" if len(acc) >= args.token_cap
                  else "timebox" if dt > args.tokenize_timebox else "eof")
    return {"docs": n_docs, "tokens": len(acc), "corpus_bytes": bytes_read,
            "seconds": round(dt, 1), "tok_per_s": int(len(acc) / max(dt, 1e-9)),
            "stopped_by": stopped_by, "bin": args.bin}


# ---------------------------------------------------------------- train

def lr_at(step, total, args):
    warmup = max(1, int(total * args.warmup_frac))
    if step <= warmup:
        return args.lr * step / warmup
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    cos = 0.5 * (1.0 + math.cos(math.pi * prog))
    return args.lr * (args.min_lr_frac + (1.0 - args.min_lr_frac) * cos)


def build_model(args, device, from_pretrained: bool):
    """from_pretrained=False 从零随机初始化（seed 17）；True 加载 tok-dir 现成权重。"""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        if from_pretrained:
            model = AutoModelForCausalLM.from_pretrained(
                args.tok_dir, local_files_only=True, dtype=torch.float32)
        else:
            cfg = AutoConfig.from_pretrained(args.tok_dir, local_files_only=True)
            cfg.dtype = "float32"  # 覆盖底座 config 里的 float16，保证随机初始化为 fp32 主权重
            torch.manual_seed(17)
            try:
                model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
            except TypeError:  # 旧版 transformers 用 torch_dtype
                model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32)
    except TypeError:  # 旧版 transformers 用 torch_dtype
        if from_pretrained:
            model = AutoModelForCausalLM.from_pretrained(
                args.tok_dir, local_files_only=True, torch_dtype=torch.float32)
    assert next(model.parameters()).dtype == torch.float32
    model.to(device)
    model.train()
    model.config.use_cache = False
    if getattr(args, "grad_checkpoint", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("gradient checkpointing enabled (use_reentrant=False)", flush=True)
    return model


def make_batch_iter(tokens_mm, seq_len, micro_batch, rng):
    import torch

    n_blocks = len(tokens_mm) // seq_len
    assert n_blocks >= micro_batch, f"语料块数不足: {n_blocks}"
    blocks = tokens_mm[: n_blocks * seq_len].reshape(n_blocks, seq_len)
    while True:
        perm = rng.permutation(n_blocks)
        for i in range(0, n_blocks - micro_batch + 1, micro_batch):
            idx = perm[i: i + micro_batch]
            np_batch = np.asarray(blocks[idx]).astype(np.int64)
            yield torch.from_numpy(np_batch).pin_memory().cuda(non_blocking=True)


def run_steps(model, opt, scaler, gen, *, n_steps, hard_seconds, accum, clip,
              csv_path, log_every, tokens_per_step, autocast_dtype):
    """统一训练循环；返回 (steps, tokens, last_loss, elapsed, oom_skips)。"""
    import torch

    steps, tokens, oom_skips = 0, 0, 0
    last_loss = float("nan")
    t0 = time.time()
    csv_f = open(csv_path, "a", encoding="utf-8") if csv_path else None
    if csv_f and csv_f.tell() == 0:
        csv_f.write("step,wall_s,tokens,loss\n")
    while steps < n_steps:
        if time.time() - t0 > hard_seconds:
            print(f"[hard-stop] 达到 {hard_seconds}s 硬墙钟，停止", flush=True)
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(steps + 1, n_steps, _ARGS_HACK)
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        try:
            for _ in range(accum):
                x = next(gen)
                with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                    out = model(x, labels=x)
                    loss = out.loss
                if scaler is not None:
                    scaler.scale(loss / accum).backward()
                else:
                    (loss / accum).backward()
                loss_sum += loss.detach()
        except torch.cuda.OutOfMemoryError:
            oom_skips += 1
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if oom_skips > 20:
                raise
            continue
        if scaler is not None:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        steps += 1
        tokens += tokens_per_step
        last_loss = float(loss_sum) / accum
        if csv_f and steps % log_every == 0:
            el = time.time() - t0
            csv_f.write(f"{steps},{el:.1f},{tokens},{last_loss:.4f}\n")
            csv_f.flush()
            print(f"step {steps} loss {last_loss:.4f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"tok/s {tokens / el:.0f} mem {torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
                  flush=True)
    if csv_f:
        el = time.time() - t0
        csv_f.write(f"{steps},{el:.1f},{tokens},{last_loss:.4f}\n")
        csv_f.close()
    return steps, tokens, last_loss, time.time() - t0, oom_skips


_ARGS_HACK = None  # lr_at 需要 args；模块级引用由 main 注入


def quick_dtype_bench(model, opt, gen, args, autocast_dtype):
    """7 步快速基准（前 2 步预热丢弃），返回平均每步秒数；失败返回 None。仅 --dtype auto 时使用。"""
    import torch

    scaler = torch.amp.GradScaler("cuda", enabled=(autocast_dtype == torch.float16))
    times = []
    for i in range(7):
        opt.zero_grad(set_to_none=True)
        t0 = time.time()
        try:
            for _ in range(args.accum):
                x = next(gen)
                with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                    out = model(x, labels=x)
                    loss = out.loss / args.accum
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            if scaler is not None:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return None
        except RuntimeError as e:
            print(f"dtype {autocast_dtype} 失败: {e}", flush=True)
            return None
        if i >= 2:
            times.append(time.time() - t0)
    return sum(times) / len(times)


def main():
    global _ARGS_HACK
    args = parse_args()
    _ARGS_HACK = args
    import torch

    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    rng = np.random.default_rng(17)

    if args.reuse_bin and os.path.exists(args.bin):
        stats = {"reused_bin": args.bin, "tokens": os.path.getsize(args.bin) // 2}
    else:
        print("== 阶段 1: tokenize 语料 ==", flush=True)
        stats = tokenize_corpus(args)
        print("tokenize:", json.dumps(stats, ensure_ascii=False), flush=True)

    tokens_mm = np.memmap(args.bin, dtype=np.uint16, mode="r")
    n_blocks = len(tokens_mm) // args.seq_len
    print(f"token 流 {len(tokens_mm)} ({len(tokens_mm) / 1e6:.1f}M) -> {n_blocks} 个 {args.seq_len} 块",
          flush=True)

    pretrained_init = args.init == "pretrained"
    print(f"== 阶段 2: 构建模型（{'加载现成权重' if pretrained_init else '随机初始化'}）==", flush=True)
    model = build_model(args, device, from_pretrained=pretrained_init)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.1f}M (fp32)", flush=True)

    # ---- dtype 择优（默认 fp16 跳过；仅 --dtype auto 时跑基准）----
    if args.dtype == "auto":
        gen = make_batch_iter(tokens_mm, args.seq_len, args.micro_batch, rng)
        opt = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=args.weight_decay)
        t_bf = quick_dtype_bench(model, opt, gen, args, torch.bfloat16)
        t_fp = quick_dtype_bench(model, opt, gen, args, torch.float16)
        print(f"bench: bf16 {t_bf} s/step, fp16 {t_fp} s/step", flush=True)
        if t_bf is not None and (t_fp is None or t_bf <= t_fp * 1.15):
            choice = torch.bfloat16
        elif t_fp is not None:
            choice = torch.float16
        else:
            choice = None  # fp32
        del opt
    elif args.dtype == "bf16":
        choice = torch.bfloat16
    elif args.dtype == "fp16":
        choice = torch.float16
    else:
        choice = None
    print(f"选定 autocast dtype: {choice}", flush=True)

    # ---- 吞吐探针（丢弃权重）----
    gen = make_batch_iter(tokens_mm, args.seq_len, args.micro_batch, rng)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(choice == torch.float16))
    if not args.skip_probe:
        print(f"== 阶段 3: {args.probe_seconds}s 吞吐探针（丢弃）==", flush=True)
        p_steps, p_tokens, _, p_el, _ = run_steps(
            model, opt, scaler, gen, n_steps=10**9, hard_seconds=args.probe_seconds,
            accum=args.accum, clip=args.clip, csv_path=None, log_every=50,
            tokens_per_step=args.micro_batch * args.accum * args.seq_len,
            autocast_dtype=choice)
        s_per_step = p_el / max(1, p_steps)
        mem_gib = torch.cuda.max_memory_allocated() / 2 ** 30
        print(f"探针: {p_steps} 步 / {p_el:.1f}s = {s_per_step:.3f} s/step, "
              f"{p_tokens / p_el:.0f} tok/s, 峰值显存 {mem_gib:.2f} GiB", flush=True)
        if mem_gib > 3.0 and args.micro_batch > 1:
            new_mb = max(1, args.micro_batch // 2)
            print(f"显存超 3GiB 预算，micro_batch {args.micro_batch} -> {new_mb}（accum 翻倍）", flush=True)
            args.micro_batch = new_mb
            args.accum *= 2
            gen = make_batch_iter(tokens_mm, args.seq_len, args.micro_batch, rng)
        max_steps = args.max_steps if args.max_steps > 0 else max(
            50, int(args.train_seconds / s_per_step * 0.97))
        # 重建模型（探针权重丢弃；DAPT 时重新加载现成权重）
        del model, opt
        torch.cuda.empty_cache()
        model = build_model(args, device, from_pretrained=pretrained_init)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=(choice == torch.float16))
        gen = make_batch_iter(tokens_mm, args.seq_len, args.micro_batch, rng)
    else:
        max_steps = args.max_steps if args.max_steps > 0 else 10 ** 9

    print(f"== 阶段 4: 正式预训练 {max_steps} 步（硬上限 {args.train_seconds}s）==", flush=True)
    steps, tokens, last_loss, el, oom_skips = run_steps(
        model, opt, scaler, gen, n_steps=max_steps, hard_seconds=args.train_seconds,
        accum=args.accum, clip=args.clip, csv_path=args.csv, log_every=50,
        tokens_per_step=args.micro_batch * args.accum * args.seq_len,
        autocast_dtype=choice)

    print("== 阶段 5: 保存 ==", flush=True)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(args.tok_dir, local_files_only=True).save_pretrained(args.out)
    summary = {
        "params_M": round(n_params / 1e6, 1), "steps": steps, "tokens": tokens,
        "final_loss": round(last_loss, 4), "elapsed_s": round(el, 1),
        "tok_per_s": int(tokens / max(el, 1e-9)), "oom_skips": oom_skips,
        "micro_batch": args.micro_batch, "accum": args.accum,
        "autocast": str(choice), "max_steps_planned": max_steps,
        "tokenize": stats, "out": args.out,
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(args.csv)), "pretrain_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
