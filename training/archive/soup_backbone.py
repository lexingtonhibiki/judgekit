# -*- coding: utf-8 -*-
"""骨干汤（backbone soup）：平均 mini01 与 preB 的 DecisionModel 骨干权重（fp32 平均 -> fp16），
判头 key 一律不平均（两次独立 SFT 的判头神经元不对齐），输出 HF llama 格式底座目录，
可直接被 train_pipeline_decisions.py --model 加载并训练全新判头。

用法：python soup_backbone.py
"""
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

CKPT_A = r"D:/Models/NanoJev-zh/mini01/best.safetensors"
CKPT_B = r"D:/Models/NanoJev-zh/preB/best.safetensors"
BASE = r"D:/Models/minimind2-mini"
OUT = r"D:/Models/miniexp/soup_backbone"
PREFIX = "backbone."


def main():
    sa = load_file(CKPT_A)
    sb = load_file(CKPT_B)
    ka = {k for k in sa if k.startswith(PREFIX)}
    kb = {k for k in sb if k.startswith(PREFIX)}
    assert ka == kb, f"两 checkpoint 骨干 key 不一致: only_a={ka-kb} only_b={kb-ka}"
    head_keys = [k for k in sa if not k.startswith(PREFIX)]
    print(f"backbone keys {len(ka)} | head keys skipped {len(head_keys)}: "
          f"{sorted(head_keys)}")

    avg = {}
    for k in sorted(ka):
        fa, fb = sa[k].float(), sb[k].float()
        assert fa.shape == fb.shape, f"shape mismatch {k}: {fa.shape} vs {fb.shape}"
        avg["model." + k[len(PREFIX):]] = ((fa + fb) / 2.0).half()  # backbone.X -> model.X

    # 对照底座命名验证覆盖率
    with safe_open(os.path.join(BASE, "model.safetensors"), framework="pt",
                   device="cpu") as f:
        base_keys = set(f.keys())
    matched = base_keys & set(avg)
    print(f"base keys {len(base_keys)} | soup->base 覆盖 {len(matched)}/{len(base_keys)}"
          f" | soup 多出 {sorted(set(avg) - base_keys)}")
    assert len(matched) == len(base_keys) == len(avg), "key 覆盖不完整，中止"

    os.makedirs(OUT, exist_ok=True)
    save_file(avg, os.path.join(OUT, "model.safetensors"), metadata={"format": "pt"})
    for fname in ["config.json", "generation_config.json", "tokenizer.json",
                  "tokenizer_config.json", "special_tokens_map.json",
                  "chat_template.jinja"]:
        src = os.path.join(BASE, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUT, fname))

    # CPU 加载自检：能被 AutoModelForCausalLM 读回，且抽查张量与平均值一致
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(OUT, local_files_only=True)
    n_params = sum(p.numel() for p in model.parameters())
    sd = model.state_dict()
    max_diff = max(float((sd["model." + k] - avg["model." + k].float()).abs().max())
                   for k in ["embed_tokens.weight", "norm.weight",
                             "layers.0.self_attn.q_proj.weight"])
    print(f"self-check OK: params {n_params / 1e6:.1f}M, spot max|diff| {max_diff:.3e}")
    print("SUMMARY " + json.dumps({
        "out": OUT, "backbone_keys_averaged": len(ka), "head_keys_skipped": len(head_keys),
        "base_key_coverage": f"{len(matched)}/{len(base_keys)}",
        "params_M": round(n_params / 1e6, 1), "spot_max_absdiff": max_diff,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
