# -*- coding: utf-8 -*-
"""Stratified-by-family, grouped-by-state_id split of zh_v2_train.jsonl.
train 70% / dev 12% / calibration 11% / rest test. Seed 17. Writes a NEW file."""
import json
import random
import collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCH = ROOT / "training" / "archive"
SRC = ARCH / "zh_v2_train.jsonl"
DST = ARCH / "zh_v2_train_split.jsonl"

records = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
assert len({r["state_id"] for r in records}) == len(records), "state_id not unique"
assert len({r["id"] for r in records}) == len(records), "id not unique"

rng = random.Random(17)
by_family = collections.defaultdict(list)
for r in records:
    by_family[r["family_id"]].append(r)

for fam, rows in by_family.items():
    rng.shuffle(rows)
    n = len(rows)
    n_train = round(n * 0.70)
    n_dev = round(n * 0.12)
    n_calib = round(n * 0.11)
    assignment = (["train"] * n_train + ["dev"] * n_dev + ["calibration"] * n_calib
                  + ["test"] * (n - n_train - n_dev - n_calib))
    for row, sp in zip(rows, assignment):
        row["split"] = sp

counts = collections.Counter(r["split"] for r in records)
cross = collections.Counter((r["family_id"], r["split"]) for r in records)
with DST.open("w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("total:", len(records))
print("by split:", dict(counts))
print("by family x split:")
for (fam, sp), c in sorted(cross.items()):
    print(f"  {fam:14s} {sp:12s} {c}")
print("wrote:", DST)
