# -*- coding: utf-8 -*-
"""T4 探针采样：垃圾三件套×10（FakeReview/JD刷单/reatiny）+ CSDS×20对话。

来源（brief指定，唯一需求源）：
  FakeReview  https://github.com/FakeReview/Data_set（Dp-CDC：text+metadata，label 1=fake）
  JD刷单      https://doi.org/10.57760/sciencedb.j00133.00268（ScienceDB，CC0 anonymous下载）
  reatiny     https://github.com/reatiny/chinesespam + HF reatiny/chinese-spam-10000（gated→走GitHub仓）
  CSDS        https://github.com/xiaolinAndy/CSDS（数据走作者GDrive直链；Baidu盘为镜像说明）

污染/去重逻辑仿 benchmarks/data/external/fetch_sample.py：
  归一化+exact键（此处按brief字面用 md5，fetch_sample用sha256，均为归一化hash）+
  minhash近重复（>0.85，fetch_sample的L0审计同款，brief称simhash族）+
  污染基=real_sentiment138+econ_zh130（=268条）命中即弃并补采。
  norm/interleave/select/minhash_sig/l0_audit/load_contam_base 直接复用 fetch_sample（importlib）。

输出（全进 training/abc_out/，gitignore已覆盖，不入库）：
  probe_spam.jsonl      20行 task=spam（fakereview×10 + jd×10）
  probe_csds_raw.jsonl  20行 CSDS原始对话（derive_csds.py再派生）
  probe_fetch_report.json 采样报告（各源 counts/dup/contam/urls/licenses/失败证据）

用法：
  python training/fetch_probe.py [--out-dir training/abc_out] [--seed 42]
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_FETCH = ROOT / "benchmarks" / "data" / "external" / "fetch_sample.py"
_spec = importlib.util.spec_from_file_location("fetch_sample", _FETCH)
fetch_sample = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_sample)

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) data-sampling"}
SEED = 42

FAKE_TEXT_URL = "https://raw.githubusercontent.com/FakeReview/Data_set/main/Dp-CDC_text.csv"
FAKE_META_URL = "https://raw.githubusercontent.com/FakeReview/Data_set/main/Dp-CDC_metadata.csv"
FAKE_PAGE = "https://github.com/FakeReview/Data_set"
FAKE_LIC = "研究使用(点评平台评论，user/shop已md5；注明来源，见仓README)"
JD_DOI = "https://doi.org/10.57760/sciencedb.j00133.00268"
JD_API = ("https://www.scidb.cn/api/sdb-filetree-service/getAllUrl"
          "?dataSetId=cb24f7bec8bb42a2b90f8fc0a084b86f&type=download&version=V1&global=en")
JD_LIC = "CC0-1.0（库元数据license字段声明；PUBLIC匿名可下）"
REATINY_GH = "https://github.com/reatiny/chinesespam"
REATINY_HF = "https://huggingface.co/datasets/reatiny/chinese-spam-10000"
REATINY_LIC = "未取到（HF gated + GitHub空仓，无许可可录）"
CSDS_VAL_URL = "https://drive.google.com/uc?export=download&id=1JLW5iRUjdFz1BUGypyGHAm6vEkMI20sS"
CSDS_TRAIN_URL = "https://drive.google.com/uc?export=download&id=1-xwmTxDAZXlt3YhPBgQWETzw_yM5Xui2"
CSDS_PAGE = "https://github.com/xiaolinAndy/CSDS"
CSDS_LIC = "研究使用(原仓仅代码；数据走作者GDrive直链，Baidu盘为镜像：pan.baidu.com/s/1KKKNuQO5af3JQuun1G3JDg 提取码5dii；注明来源+引用EMNLP21)"


def fetch(url: str, timeout: int = 120, retries: int = 4) -> bytes:
    last = None
    for i in range(retries):
        try:
            safe = urllib.parse.quote(url, safe=":/?&=%")
            req = urllib.request.Request(safe, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"FETCH FAIL {url}: {last!r}")


def md5key(s: str) -> str:
    return hashlib.md5(fetch_sample.norm(s).encode("utf-8")).hexdigest()


def pick_balanced(groups: dict, contam: set[str], per_lab: int,
                  seed: int = SEED) -> tuple[list, dict]:
    """每类定额（分层5/5），类内md5去重+污染过滤+补采。"""
    picked, stats = [], {"dup": 0, "near": 0, "contam": 0}
    for gi, lab in enumerate(sorted(groups)):
        sub, dup, near, hit = select_md5(list(groups[lab]), contam, per_lab,
                                         seed + gi)
        stats["dup"] += dup
        stats["near"] += near
        stats["contam"] += hit
        picked += [(t, lab, x) for (t, _lab, x) in sub]
    rnd = random.Random(seed)
    rnd.shuffle(picked)
    return picked, stats


def select_md5(pool, contam: set[str], n: int, seed: int = SEED):
    """pool: [(text, orig_label, extra)]。md5 exact + minhash近重复 + 污染基过滤。"""
    rnd = random.Random(seed)
    idx = list(range(len(pool)))
    rnd.shuffle(idx)
    seen, picked = set(), []
    dup = near = hit = 0
    sims = []
    for i in idx:
        t, lab, _ = pool[i]
        h = md5key(t)
        if h in seen:
            dup += 1
            continue
        if h in contam:
            hit += 1
            continue
        sig = fetch_sample.minhash_sig(t)
        is_near = False
        for psig in sims:
            s = sum(1 for a, b in zip(sig, psig) if a == b) / len(sig)
            if s > 0.85:
                is_near = True
                break
        if is_near:
            near += 1
            continue
        seen.add(h)
        sims.append(sig)
        picked.append(pool[i])
        if len(picked) >= n:
            break
    return picked, dup, near, hit


# ---------------- FakeReview ----------------
def pool_fakereview() -> tuple[dict, list]:
    text_raw = fetch(FAKE_TEXT_URL).decode("utf-8-sig")
    meta_raw = fetch(FAKE_META_URL).decode("utf-8-sig")
    texts = {}
    for row in csv.DictReader(io.StringIO(text_raw)):
        t = (row.get("text") or "").strip()
        if t:
            texts[(row.get("id") or "").strip()] = t
    groups: dict[str, list] = {"刷评spam": [], "正常": []}
    for row in csv.DictReader(io.StringIO(meta_raw)):
        rid = (row.get("id") or "").strip()
        t = texts.get(rid)
        if not t:
            continue
        lab = "刷评spam" if (row.get("label") or "").strip() == "1" else "正常"
        groups[lab].append((t, lab, f"Dp-CDC:{rid}"))
    return groups, [FAKE_TEXT_URL, FAKE_META_URL]


# ---------------- JD刷单 ----------------
def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("gbk", errors="replace")


def pool_jd() -> tuple[dict, list]:
    listing = fetch(JD_API, timeout=60).decode("utf-8", errors="replace").splitlines()
    lines = [l.strip() for l in listing if l.strip().startswith("http")]
    main = [u for u in lines if "京东虚假评论数据集" in u]
    if not main:
        raise RuntimeError(f"JD文件清单无主集：{listing[:3]!r}")
    groups: dict[str, list] = {"刷单spam": [], "正常": []}
    for u in main:
        raw = fetch(u, timeout=180)
        for line in _decode(raw).splitlines():
            parts = line.split("\t")
            if len(parts) < 2 or not parts[1].strip():
                continue
            # 约定（report留证）：1=虚假/刷单（短促模板夸），0=真实（细节实）——与FakeReview 1=fake同向
            lab = "刷单spam" if parts[0].strip() == "1" else "正常"
            fname = urllib.parse.unquote(u.split("path=")[1].split("&")[0]).split("/")[-1]
            groups[lab].append((parts[1].strip(), lab, f"JD:{fname}"))
    return groups, [JD_DOI]


# ---------------- reatiny（真实尝试，失败留证） ----------------
def attempt_reatiny() -> dict:
    ev: dict = {"rows": 0, "steps": []}
    try:
        tree = json.loads(fetch(
            "https://api.github.com/repos/reatiny/chinesespam/git/trees/main?recursive=1",
            timeout=30).decode("utf-8"))
        blobs = [x["path"] for x in tree.get("tree", []) if x["type"] == "blob"]
        ev["steps"].append(f"github-tree blobs={blobs}")
        data = [b for b in blobs if b.lower().endswith((".csv", ".txt", ".json", ".parquet"))]
        if not data:
            ev["status"] = "FETCH_FAIL: GitHub仓仅3文件（.gitignore/LICENSE/README），无数据文件"
        else:
            ev["status"] = f"GitHub仓有数据文件：{data}"
            return ev
    except Exception as e:  # noqa: BLE001
        ev["steps"].append(f"github-tree error: {e!r}")
    try:
        probe = fetch(
            "https://hf-mirror.com/datasets/reatiny/chinese-spam-10000/resolve/main/README.md",
            timeout=30, retries=1)
        ev["steps"].append(f"hf-mirror bytes={len(probe)}")
        if not probe:
            ev["status"] = "FETCH_FAIL: HF gated（mirror回空，需登录accept）+ GitHub空仓"
    except Exception as e:  # noqa: BLE001
        ev["steps"].append(f"hf-mirror error: {str(e)[:100]}")
        ev["status"] = "FETCH_FAIL: HF gated（mirror不可达/需登录）+ GitHub空仓"
    return ev


# ---------------- CSDS ----------------
def fetch_csds(n: int = 20, seed: int = SEED) -> tuple[list[dict], dict]:
    try:
        raw = fetch(CSDS_VAL_URL, timeout=300)
        src = "GDrive-val.json"
    except RuntimeError:
        raw = fetch(CSDS_TRAIN_URL, timeout=600)
        src = "GDrive-train.json"
    dlgs = json.loads(raw.decode("utf-8"))
    rnd = random.Random(seed)
    idx = list(range(len(dlgs)))
    rnd.shuffle(idx)
    return [dlgs[i] for i in idx[:n]], {"pool": len(dlgs), "src": src}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="training/abc_out")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-fake", type=int, default=10)
    ap.add_argument("--n-jd", type=int, default=10)
    ap.add_argument("--n-csds", type=int, default=20)
    args = ap.parse_args()

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    contam, contam_n = fetch_sample.load_contam_base()
    print(f"contam base: {contam_n} texts (real_sentiment138+econ_zh130)")
    rep: dict = {"seed": args.seed, "contam_n": contam_n, "sources": {}}

    # FakeReview
    groups, dl = pool_fakereview()
    pool_size = sum(len(v) for v in groups.values())
    picked, st = pick_balanced(groups, contam, args.n_fake // 2, args.seed)
    rows = [{"id": f"fk_{i + 1:04d}", "text": t, "task": "spam", "orig_label": lab,
             "source": "FakeReview", "url": FAKE_PAGE, "license": FAKE_LIC,
             "extra": x} for i, (t, lab, x) in enumerate(picked)]
    rep["sources"]["FakeReview"] = {"n": len(rows), "pool": pool_size,
                                    "dup": st["dup"],
                                    "near": st["near"], "contam": st["contam"],
                                    "dl": dl,
                                    "label_dist": {k: sum(1 for r in rows if r["orig_label"] == k)
                                                   for k in groups}}
    print(f"[FakeReview] pool={pool_size} picked={len(rows)} dup={st['dup']} "
          f"near={st['near']} contam={st['contam']}")

    # JD
    groups, dl = pool_jd()
    pool_size = sum(len(v) for v in groups.values())
    picked, st = pick_balanced(groups, contam, args.n_jd // 2, args.seed)
    jrows = [{"id": f"jd_{i + 1:04d}", "text": t, "task": "spam", "orig_label": lab,
              "source": "JD刷单", "url": JD_DOI, "license": JD_LIC,
              "extra": x} for i, (t, lab, x) in enumerate(picked)]
    rep["sources"]["JD刷单"] = {"n": len(jrows), "pool": pool_size,
                                "dup": st["dup"],
                                "near": st["near"], "contam": st["contam"],
                                "dl": dl,
                                "label_dist": {k: sum(1 for r in jrows if r["orig_label"] == k)
                                               for k in groups}}
    print(f"[JD刷单] pool={pool_size} picked={len(jrows)} dup={st['dup']} "
          f"near={st['near']} contam={st['contam']}")
    rows += jrows

    with open(out / "probe_spam.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # reatiny
    rev = attempt_reatiny()
    rep["sources"]["reatiny"] = {"n": 0, "pool": 0, "dl": [REATINY_GH, REATINY_HF],
                                 "license": REATINY_LIC, **rev}
    print(f"[reatiny] {rev.get('status')}")

    # CSDS
    dlgs, cinfo = fetch_csds(args.n_csds, args.seed)
    rep["sources"]["CSDS"] = {"n": len(dlgs), "pool": cinfo["pool"], "src": cinfo["src"],
                              "dl": [CSDS_VAL_URL], "page": CSDS_PAGE,
                              "license": CSDS_LIC,
                              "ids": [d.get("DialogueID") for d in dlgs]}
    print(f"[CSDS] pool={cinfo['pool']} picked={len(dlgs)} src={cinfo['src']}")
    with open(out / "probe_csds_raw.jsonl", "w", encoding="utf-8") as f:
        for d in dlgs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    with open(out / "probe_fetch_report.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"saved: {out / 'probe_spam.jsonl'} ({len(rows)}行) + "
          f"{out / 'probe_csds_raw.jsonl'} ({len(dlgs)}对话)")


if __name__ == "__main__":
    main()
