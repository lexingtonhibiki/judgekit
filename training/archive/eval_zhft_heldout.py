# -*- coding: utf-8 -*-
"""Validation B: zh_v2_heldout.jsonl (100 records) via DecisionPredictor on a
trained checkpoint dir. Argmax agreement (same convention as round-1 67.8%) +
10-bin ECE (correctness vs top confidence). Writes per-record evidence JSONL.

用法：
  python training/archive/eval_zhft_heldout.py --checkpoint D:/Models/NanoJev-zh/mini01 [--out evidence.jsonl]
默认（无参数）= 评测 mini01 基线，行为与参数化前逐字节一致。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "C:/Users/14970/AppData/Local/Temp/NanoJev/scripts")
from predict_toy_decisions import DecisionPredictor  # noqa: E402

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--checkpoint", default="D:/Models/NanoJev-zh/mini01",
                 help="SFT checkpoint dir (default: mini01 基线)")
_ap.add_argument("--out", default=None,
                 help="evidence JSONL 输出路径（默认 <checkpoint>/zhft_heldout_eval.jsonl）")
ARGS = _ap.parse_args()

CKPT = ARGS.checkpoint
ROOT = Path(__file__).resolve().parents[2]
HELD = ROOT / "training" / "archive" / "zh_v2_heldout.jsonl"
DETAIL = Path(ARGS.out) if ARGS.out else Path(CKPT) / "zhft_heldout_eval.jsonl"

records = [json.loads(l) for l in HELD.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"heldout records: {len(records)}", flush=True)

eng = DecisionPredictor(CKPT, precision="bf16")

def argmax_label(d):
    """First-max-wins over dict insertion order (candidate/criteria order)."""
    best, best_p = None, -1.0
    for k, v in d.items():
        if v > best_p:
            best, best_p = k, v
    return best

rows, agree, total = [], 0, 0
fam_tot, fam_agree = {}, {}
confidences, corrects = [], []
for i in range(0, len(records), 20):
    batch = records[i:i + 20]
    payload = {"states": [{k: r[k] for k in ("id", "state", "questions")} for r in batch]}
    out = eng.predict(payload, batch_questions=20)
    by_id = {s["id"]: s for s in out["states"]}
    for r in batch:
        st = by_id[r["id"]]
        for qname, gold_dist in r["gold_probs"].items():
            pred_probs = st["answers"][qname]["probabilities"]
            pred = argmax_label(pred_probs)
            ref = argmax_label(gold_dist)
            ok = int(pred == ref)
            total += 1
            agree += ok
            fam = r["family_id"]
            fam_tot[fam] = fam_tot.get(fam, 0) + 1
            fam_agree[fam] = fam_agree.get(fam, 0) + ok
            conf = max(pred_probs.values())
            confidences.append(conf)
            corrects.append(ok)
            rows.append({"id": r["id"], "family_id": fam, "qid": qname, "pred": pred,
                         "ref": ref, "correct": ok, "top_conf": conf,
                         "pred_probs": pred_probs})

# ECE, 10 equal-width bins on [0,1]
bins = [[] for _ in range(10)]
for c, ok in zip(confidences, corrects):
    bins[min(int(c * 10), 9)].append((c, ok))
ece = 0.0
bin_report = []
for b, items in enumerate(bins):
    if not items:
        bin_report.append({"bin": b, "n": 0})
        continue
    acc = sum(o for _, o in items) / len(items)
    conf = sum(c for c, _ in items) / len(items)
    ece += len(items) / total * abs(acc - conf)
    bin_report.append({"bin": b, "n": len(items), "conf": round(conf, 4), "acc": round(acc, 4)})

print(f"\nTOTAL agreement: {agree}/{total} = {agree/total:.4f}")
for fam in sorted(fam_tot):
    print(f"  {fam:14s} {fam_agree[fam]}/{fam_tot[fam]} = {fam_agree[fam]/fam_tot[fam]:.4f}")
print(f"ECE (10 bins): {ece:.4f}")
print("bins:", json.dumps(bin_report))

with DETAIL.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("evidence written:", DETAIL)
