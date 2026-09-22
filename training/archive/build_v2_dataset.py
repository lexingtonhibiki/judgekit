# -*- coding: utf-8 -*-
"""v2 数据工程（PROJECT_STATE.md 接续动作第 1 条）。

合并三源 → minhash 去重 → 对 judge-econ 130 测试段去污染 → 教师分歧 hard 标记 → zhft 留出 100。

三源：
  A zhft_records.jsonl   400 条 Jev 教师软标签（富描述 criteria）
  B zh_records.jsonl     GLM 自一致性合成（gen_zh_data.py 产物；可为空 = 槽位）
  C smoke_train.jsonl    judge-econ 130 的 train 段 52 条（交接方案指名合并）；
                         它与冻结测试集的重叠是设计内记忆成分，标记 eval_overlap 保留

口径：
  - 去污染基准 = judge-econ 全部 130 条（benchmarks/data/*.jsonl，冻结测试集）
  - zhft / GLM 与 130 重合（精确或 Jaccard>=0.8）→ 删除（意外污染）
  - 去重按 family 分桶（同一文本在不同任务族下不算重复）；污染的精确匹配跨族也计
  - 编码统一为富描述 criteria（round-1 教训：裸标签与富描述不一致）；类空间不足的补齐
  - hard subset：教师分布 top_prob < 阈值（zhft 0.9；GLM k=3 平滑票分布全票≈0.875/0.636、
    一票分歧≈0.625/0.455，故用 0.55 分离全票与分歧）
  - zhft 留出 100：每族 25 条，seed=17，优先取 round-1 未见过的 id（--exclude-ids）

输出（全部为新文件，不修改任何现有文件）：
  training/archive/zh_v2_train.jsonl / training/archive/zh_v2_heldout.jsonl / training/archive/zh_v2_report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "training"
ARCH = ROOT / "training" / "archive"
BENCH = ROOT / "benchmarks" / "data"
BENCH_FILES = {"zh_intent": "intent_zh", "zh_sentiment": "sentiment_zh",
               "zh_spam": "spam_zh", "zh_urgency": "urgency_zh"}

P = (1 << 61) - 1
# 32 band × 4 row：0.8 相似度处漏检率 ~5e-8（16×8 配置实测 0.7 处即有 ~35% 漏检，弃用）
N_PERM, N_BAND, N_ROW = 128, 32, 4


def norm_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).lower()
    return re.sub(r"\s+", "", t)


def shingles(t: str, k: int = 3) -> frozenset:
    n = norm_text(t)
    if len(n) < k:
        return frozenset([n]) if n else frozenset()
    return frozenset(n[i:i + k] for i in range(len(n) - k + 1))


_AB = []
_rng0 = random.Random(17)
for _ in range(N_PERM):
    _AB.append((_rng0.randrange(1, P), _rng0.randrange(0, P)))


def minhash(sh: frozenset) -> list:
    if not sh:
        return [0] * N_PERM
    xs = [int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")
          for s in sh]
    return [min((a * x + b) % P for x in xs) for a, b in _AB]


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class LSH:
    """minhash banding 索引：同桶返回已注册 key 的候选集。"""

    def __init__(self):
        self.buckets = defaultdict(set)

    def add(self, key, sig):
        for b in range(N_BAND):
            self.buckets[(b, tuple(sig[b * N_ROW:(b + 1) * N_ROW]))].add(key)

    def candidates(self, sig):
        out = set()
        for b in range(N_BAND):
            out |= self.buckets.get((b, tuple(sig[b * N_ROW:(b + 1) * N_ROW])), set())
        return out


def read_jsonl(p: Path) -> list:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def prep(rec: dict, source: str) -> dict:
    text = json.loads(rec["state"])["text"]
    sh = shingles(text)
    return {"rec": rec, "source": source, "id": rec["id"], "family": rec["family_id"],
            "text": text, "norm": norm_text(text), "sh": sh, "sig": minhash(sh)}


def canonical_specs(zhft_recs: list) -> dict:
    """各 family 出现最多的 (qid, instructions, criteria) 作为富描述规范。"""
    votes = defaultdict(Counter)
    for r in zhft_recs:
        for qid, q in r["questions"].items():
            key = json.dumps({"qid": qid, "instructions": q["instructions"],
                              "criteria": q["criteria"]}, ensure_ascii=False, sort_keys=True)
            votes[r["family_id"]][key] += 1
    return {fam: json.loads(c.most_common(1)[0][0]) for fam, c in votes.items()}


def unify_encoding(rec: dict, fam_spec: dict) -> bool:
    """裸标签 criteria → 富描述（fam_spec = 该 family 的规范，由调用方解析）；类空间不足则补齐 gold_probs（新增类记 0.0）。

    返回是否发生了改写。
    """
    s = fam_spec
    old_qid = next(iter(rec["questions"]))
    q = rec["questions"][old_qid]
    changed = (q["criteria"] != s["criteria"]
               or q["instructions"] != s["instructions"])
    rec["questions"] = {s["qid"]: {"type": q.get("type", "choice"),
                                   "instructions": s["instructions"],
                                   "criteria": dict(s["criteria"])}}
    gold_probs = rec["gold_probs"][old_qid]
    full = {lb: float(gold_probs.get(lb, 0.0)) for lb in s["criteria"]}
    total = sum(full.values())
    rec["gold_probs"] = {s["qid"]: {lb: v / total for lb, v in full.items()}}
    if "gold" in rec:
        rec["gold"] = {s["qid"]: rec["gold"][old_qid]}
    return changed


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zhft", default=str(TRAIN / "zhft_records.jsonl"))
    ap.add_argument("--glm", default=str(TRAIN / "zh_records.jsonl"))
    ap.add_argument("--jecon-train", default=str(TRAIN / "smoke_train.jsonl"))
    ap.add_argument("--out-train", default=str(ARCH / "zh_v2_train.jsonl"))
    ap.add_argument("--out-heldout", default=str(ARCH / "zh_v2_heldout.jsonl"))
    ap.add_argument("--out-report", default=str(ARCH / "zh_v2_report.json"))
    ap.add_argument("--dup-threshold", type=float, default=0.8)
    ap.add_argument("--contam-threshold", type=float, default=0.8)
    ap.add_argument("--hard-threshold-zhft", type=float, default=0.9)
    ap.add_argument("--hard-threshold-glm", type=float, default=0.55)
    ap.add_argument("--heldout-per-family", type=int, default=25)
    ap.add_argument("--exclude-ids", default="",
                    help="round-1 已训练 id 清单（.jsonl 取 id 字段 / 纯文本每行一个），留出集优先避开")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    # 1. 载入三源（列表顺序 = 去重保留优先级：bench 金标 > Jev 教师 > GLM 合成）
    sources = [("judge_econ_train", Path(args.jecon_train)),
               ("zhft", Path(args.zhft)),
               ("glm_synth", Path(args.glm))]
    pool, input_counts = [], Counter()
    for source, path in sources:
        for rec in read_jsonl(path):
            pool.append(prep(rec, source))
            input_counts[source] += 1

    # 2. 编码统一为富描述（以 zhft 为主体规范）
    spec = canonical_specs(read_jsonl(Path(args.zhft)))
    unified = sum(1 for r in pool if unify_encoding(r["rec"], spec[r["family"]]))
    for r in pool:  # 自检：类空间与分布
        qid = next(iter(r["rec"]["questions"]))
        probs = r["rec"]["gold_probs"][qid]
        assert set(probs) == set(r["rec"]["questions"][qid]["criteria"]), r["id"]
        assert abs(sum(probs.values()) - 1.0) <= 1e-6, r["id"]

    # 3. 去污染：对 judge-econ 130（冻结测试集）
    bench = []
    for fam, ds in BENCH_FILES.items():
        for o in read_jsonl(BENCH / f"{ds}.jsonl"):
            sh = shingles(o["text"])
            bench.append({"id": o["id"], "family": fam, "norm": norm_text(o["text"]),
                          "sh": sh, "sig": minhash(sh)})
    bench_exact = {b["norm"]: b for b in bench}
    bench_lsh = LSH()
    for i, b in enumerate(bench):
        bench_lsh.add(i, b["sig"])

    removed_contam, by_design = Counter(), 0
    kept = []
    for r in pool:
        hit = None
        if r["norm"] in bench_exact:
            hit = ("exact", bench_exact[r["norm"]]["id"])
        else:
            for j in bench_lsh.candidates(r["sig"]):
                b = bench[j]
                if b["family"] == r["family"] and jaccard(r["sh"], b["sh"]) >= args.contam_threshold:
                    hit = ("near", b["id"])
                    break
        if hit and r["source"] == "judge_econ_train":
            r["eval_overlap"] = True  # 设计内记忆成分：方案指名合并，保留并标记
            by_design += 1
            kept.append(r)
        elif hit:
            removed_contam[r["source"]] += 1
            r["contam_of"] = f"{hit[0]}:{hit[1]}"
        else:
            r["eval_overlap"] = False
            kept.append(r)
    if by_design != input_counts["judge_econ_train"]:
        print(f"WARN: judge_econ_train 设计内重合 {by_design}/{input_counts['judge_econ_train']}，"
              f"存在未精确命中（近重复阈值内已计入）", file=sys.stderr)

    # 4. 去重：精确（同族同文）+ minhash 近似（LSH 候选 → Jaccard 校验）
    exact_seen, dropped_exact, dropped_near = {}, Counter(), Counter()
    lsh, final = LSH(), []
    for r in kept:
        ek = (r["family"], r["norm"])
        if ek in exact_seen:
            dropped_exact[r["source"]] += 1
            continue
        dup_of = None
        for j in lsh.candidates(r["sig"]):
            o = final[j]
            if o["family"] == r["family"] and jaccard(r["sh"], o["sh"]) >= args.dup_threshold:
                dup_of = o["id"]
                break
        if dup_of:
            dropped_near[r["source"]] += 1
            continue
        exact_seen[ek] = r
        lsh.add(len(final), r["sig"])
        final.append(r)

    # 5. hard 标记（教师分歧）
    t_map = {"zhft": args.hard_threshold_zhft,
             "glm_synth": args.hard_threshold_glm,
             "judge_econ_train": None}
    hard_counts = Counter()
    for r in final:
        t = t_map[r["source"]]
        r["hard"] = False
        if t is not None:
            qid = next(iter(r["rec"]["questions"]))
            top = max(r["rec"]["gold_probs"][qid].values())
            if top < t:
                r["hard"] = True
                r["hard_reason"] = f"teacher_top_prob={top:.3f}<{t}"
        if r["hard"]:
            hard_counts[r["source"]] += 1

    # 6. zhft 留出（每族 N 条，优先 round-1 未见 id）
    exclude = set()
    if args.exclude_ids:
        p = Path(args.exclude_ids)
        if p.suffix == ".jsonl":
            exclude = {o["id"] for o in read_jsonl(p)}
        else:
            exclude = {x.strip() for x in p.open(encoding="utf-8") if x.strip()}
    rng = random.Random(args.seed)
    by_fam = defaultdict(list)
    for r in final:
        if r["source"] == "zhft":
            by_fam[r["family"]].append(r)
    heldout_ids = set()
    for fam in sorted(by_fam):
        rows = by_fam[fam]
        pref = [r for r in rows if r["id"] not in exclude]
        rest = [r for r in rows if r["id"] in exclude]
        rng.shuffle(pref)
        rng.shuffle(rest)
        heldout_ids |= {r["id"] for r in (pref + rest)[:args.heldout_per_family]}
    for r in final:
        r["split"] = "heldout" if r["id"] in heldout_ids else "train"

    # 7. 落盘
    def out_rec(r):
        rec = dict(r["rec"])
        rec["split"] = r["split"]
        rec["source"] = r["source"]
        rec["eval_overlap"] = r["eval_overlap"]
        rec["hard"] = r["hard"]
        if r["hard"]:
            rec["hard_reason"] = r["hard_reason"]
        return rec

    def write_jsonl(path, rows):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    train_rows = [out_rec(r) for r in final if r["split"] == "train"]
    held_rows = [out_rec(r) for r in final if r["split"] == "heldout"]
    write_jsonl(Path(args.out_train), train_rows)
    write_jsonl(Path(args.out_heldout), held_rows)

    total_in = sum(input_counts.values())
    total_dup = sum(dropped_exact.values()) + sum(dropped_near.values())
    total_contam = sum(removed_contam.values())
    train_n, held_n = len(train_rows), len(held_rows)
    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "inputs": dict(input_counts),
        "encoding_unified_records": unified,
        "decontamination": {
            "benchmark": "judge-econ 130（冻结测试集）",
            "bench_size": len(bench),
            "thresholds": {"exact": "normalized text", "near": f"jaccard>={args.contam_threshold}"},
            "accidental_removed": dict(removed_contam),
            "by_design_kept": by_design,
            "by_design_note": "smoke_train 即 judge-econ 130 的 train 段，与冻结测试集重叠为方案设计内成分",
        },
        "dedup": {
            "thresholds": {"near": f"jaccard>={args.dup_threshold}"},
            "exact_dropped": dict(dropped_exact),
            "near_dropped": dict(dropped_near),
            "dup_rate": round(total_dup / total_in, 4) if total_in else 0.0,
        },
        "post": {
            "kept_total": len(final),
            "train": train_n,
            "heldout": held_n,
            "train_by_source": dict(Counter(r["source"] for r in train_rows)),
            "train_by_family": dict(Counter(r["family_id"] for r in train_rows)),
            "strict_if_jecon_also_removed": len(final) - by_design,
        },
        "hard_subset": {"thresholds": {"zhft": args.hard_threshold_zhft,
                                       "glm_synth": args.hard_threshold_glm},
                        "counts": dict(hard_counts),
                        "total": sum(hard_counts.values())},
        "glm_slot": "空——gen_zh_data.py 未运行（GLM 配额 08:19 重置）；跑完后重执行本脚本即并入",
        "notes": [],
    }
    if args.exclude_ids:
        report["notes"].append(f"留出集优先避开 round-1 已见 id（{len(exclude)} 条清单：{args.exclude_ids}）")
    else:
        report["notes"].append("未提供 round-1 已见清单：留出集为纯分层随机，最坏 52/400 中部分曾被 smoke1 训练")
    with open(args.out_report, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("== v2 数据工程验收 ==")
    print(f"输入: zhft={input_counts['zhft']} glm_synth={input_counts['glm_synth']} "
          f"judge_econ_train={input_counts['judge_econ_train']} (共{total_in})")
    print(f"去污染(对 judge-econ {len(bench)} 冻结测试段): 意外污染删除 {total_contam} "
          f"(zhft={removed_contam['zhft']} glm={removed_contam['glm_synth']}) | "
          f"设计内保留 {by_design} (smoke_train, 已标 eval_overlap)")
    print(f"去重: 精确={sum(dropped_exact.values())} 近似={sum(dropped_near.values())} "
          f"重复率={report['dedup']['dup_rate']:.1%}")
    print(f"去污染后样本数: 交付口径={len(final)} (train={train_n} + zhft留出={held_n}) | "
          f"严格口径(C段也移除)={report['post']['strict_if_jecon_also_removed']}")
    print(f"hard subset(教师分歧): {report['hard_subset']['total']} (阈值 zhft<{args.hard_threshold_zhft})")
    print(f"编码统一(裸标签→富描述/类空间补齐): {unified} 条")
    print(f"GLM 槽位: {report['glm_slot']}")
    print(f"输出: {args.out_train} / {args.out_heldout} / {args.out_report}")


if __name__ == "__main__":
    main()
