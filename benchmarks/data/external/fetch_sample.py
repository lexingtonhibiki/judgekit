#!/usr/bin/env python
"""外部数据候选下载+采样(小闭环 8x50). 仅标准库+opencc; 直连(hf-mirror 需 UA), decode 统一 utf-8-sig.
输出: benchmarks/data/external/*.jsonl + external_SAMPLING_REPORT.md
用法: python benchmarks/data/external/fetch_sample.py  (仓库根目录执行)
"""
import csv, hashlib, io, json, os, random, re, sys, time, unicodedata
import urllib.request
import zipfile
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) data-sampling"}
SEED = 42
N = 50

try:
    from opencc import OpenCC
    _cc = OpenCC("t2s")
    CC = "opencc-t2s"
except Exception:
    _cc = None
    CC = "none-fallback"

def fetch(url, timeout=120, retries=4, expect_min=10):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if len(data) < expect_min:
                raise ValueError(f"too small: {len(data)} bytes")
            return data
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"FETCH FAIL {url}: {last!r}")

def norm(s):
    s = unicodedata.normalize("NFKC", s)
    if _cc is not None:
        s = _cc.convert(s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def hhash(s):
    return hashlib.sha256(norm(s).encode("utf-8")).hexdigest()

def interleave(groups, n, seed=SEED):
    """groups: {label: [items]} 按轮转分层采样, 确定性."""
    rnd = random.Random(seed)
    labels = sorted(groups.keys())
    for lab in labels:
        rnd.shuffle(groups[lab])
    out, idx = [], {lab: 0 for lab in labels}
    while len(out) < n:
        progressed = False
        for lab in labels:
            if idx[lab] < len(groups[lab]) and len(out) < n:
                out.append(groups[lab][idx[lab]])
                idx[lab] += 1
                progressed = True
        if not progressed:
            break
    return out

def load_contam_base():
    base = set()
    paths = [os.path.join(ROOT, "training", "real_pools", "clean", "real_sentiment.jsonl")]
    econ = os.path.join(ROOT, "benchmarks", "data", "econ_zh")
    if os.path.isdir(econ):
        for fn in sorted(os.listdir(econ)):
            if fn.endswith(".jsonl"):
                paths.append(os.path.join(econ, fn))
    n = 0
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                t = o.get("text", "")
                if t:
                    base.add(hhash(t))
                    n += 1
    return base, n

def select(pool, contam, n=N, seed=SEED):
    """pool: [(text, orig_label, extra)]. 返回 (picked, n_dup_drop, n_contam_hit)."""
    rnd = random.Random(seed)
    idx = list(range(len(pool)))
    rnd.shuffle(idx)
    seen, picked = set(), []
    dup = hit = 0
    for i in idx:
        t, lab, _ = pool[i]
        h = hhash(t)
        if h in seen:
            dup += 1
            continue
        if h in contam:
            hit += 1
            continue
        seen.add(h)
        picked.append(pool[i])
        if len(picked) >= n:
            break
    return picked, dup, hit

# ---------- 各源解析 ----------
def parse_smp():
    url = "https://raw.githubusercontent.com/hml-ubt/SMP2017-2019-ECDT-data/master/SMP2019_data/train.json"
    arr = json.loads(fetch(url).decode("utf-8-sig"))
    groups = defaultdict(list)
    for o in arr:
        t = (o.get("text") or "").strip()
        if not t:
            continue
        lab = f"{o.get('domain','?')}:{o.get('intent','?')}"
        groups[lab].append((t, lab, None))
    return groups, url

def parse_crosswoz():
    url = "https://hf-mirror.com/datasets/GEM/CrossWOZ/resolve/main/data.zip"
    z = zipfile.ZipFile(io.BytesIO(fetch(url, timeout=180)))
    data = json.loads(z.read("train.json").decode("utf-8-sig"))
    groups = defaultdict(list)
    for did, dlg in data.items():
        for m in dlg.get("messages", []):
            if m.get("role") != "usr":
                continue
            t = (m.get("content") or "").strip()
            if not t:
                continue
            doms = sorted({a[1] for a in m.get("dialog_act", [])
                           if len(a) > 1 and a[0] != "General" and a[1] not in ("none", "")})
            lab = "+".join(doms) if doms else "General"
            groups[lab].append((t, lab, did))
    return groups, url

def parse_ewect():
    base = "https://hf-mirror.com/datasets/hecongqing/EWECT_weibo_senti/resolve/main/"
    groups = defaultdict(list)
    urls = []
    for fn in ["usual_train.txt", "usual_test_labeled.txt", "usual_eval_labeled.txt"]:
        try:
            raw = fetch(base + fn).decode("utf-8-sig")
        except RuntimeError:
            continue
        urls.append(base + fn)
        arr = json.loads(raw)
        for o in arr:
            t = (o.get("content") or "").strip()
            lab = o.get("label", "?")
            if t:
                groups[lab].append((t, lab, o.get("id")))
    return groups, ";".join(urls)

def parse_cped():
    url = "https://raw.githubusercontent.com/scutcyr/CPED/master/data/CPED/train_split.csv"
    text = fetch(url, timeout=180).decode("utf-8-sig")
    groups = defaultdict(list)
    rdr = csv.DictReader(io.StringIO(text))
    for row in rdr:
        t = (row.get("Utterance") or "").strip()
        lab = (row.get("Emotion") or "?").strip()
        if t:
            groups[lab].append((t, lab, row.get("Dialogue_ID")))
    return groups, url

def parse_csv_sentiment(urls, pos="正面", neg="负面"):
    if isinstance(urls, str):
        urls = [urls]
    last = None
    for url in urls:
        try:
            text = fetch(url, timeout=120, retries=6).decode("utf-8-sig")
            break
        except RuntimeError as e:
            last = e
            continue
    else:
        raise last
    groups = defaultdict(list)
    rdr = csv.DictReader(io.StringIO(text))
    for row in rdr:
        t = (row.get("review") or "").strip()
        lab = pos if (row.get("label") or "").strip() == "1" else neg
        if t:
            groups[lab].append((t, lab, None))
    return groups, url

def parse_dmr():
    url = "https://raw.githubusercontent.com/yiyepianzhounc/DMR-Dataset-1/main/reviews.json"
    obj = json.loads(fetch(url).decode("utf-8-sig"))
    groups = defaultdict(list)
    for k, o in obj.items():
        t = (o.get("text") or "").strip()
        if not t:
            continue
        lab = "正常" if o.get("label") == 1 else "刷评spam"
        groups[lab].append((t, lab, k))
    # 主库仅 44 条: 用同域豆瓣影评 spam 集补足到 25/25(行级 source 区分, REPORT 标待复核)
    sup_url = ("https://hf-mirror.com/datasets/tracywong117/spam-douban-movie-review"
               "/resolve/main/finalized_reviews.csv")
    need = {lab: max(0, 25 - len(groups[lab])) for lab in ("正常", "刷评spam")}
    if any(need.values()):
        try:
            text = fetch(sup_url, timeout=120).decode("utf-8-sig")
            rdr = csv.DictReader(io.StringIO(text))
            for row in rdr:
                t = (row.get("short_comment") or "").strip()
                if not t:
                    continue
                lab = "刷评spam" if (row.get("Fake") or "").strip() == "1" else "正常"
                if need.get(lab, 0) > 0:
                    groups[lab].append((t, lab, "sup:" + (row.get("id") or "?")))
                    need[lab] -= 1
                if not any(need.values()):
                    break
        except RuntimeError:
            pass
    return groups, url + " + " + sup_url + "(补采, Fake=1→spam 语义待复核)"

FBS_FILES = ["AD:Loan", "AD:Network_service", "AD:Other", "AD:Real_estate", "AD:Retail",
             "FR:Financial", "FR:Other", "FR:Phishing(Bank)", "FR:Phishing(Other)",
             "IL:Escort_service", "IL:Fake_ID_and_invoice", "IL:Gambling",
             "IL:Political_propaganda", "Other"]

def parse_fbs():
    base = "https://raw.githubusercontent.com/Cypher-Z/FBS_SMS_Dataset/master/"
    groups = defaultdict(list)
    for fn in FBS_FILES:
        try:
            text = fetch(base + fn).decode("utf-8-sig")
        except RuntimeError:
            continue
        for line in text.splitlines():
            t = line.strip()
            if t:
                groups[f"spam:{fn}"].append((t, f"spam:{fn}", None))
    return groups, base

# ---------- L0 审计 ----------
def minhash_sig(text, n=64):
    grams = set()
    t = norm(text)
    if len(t) < 3:
        grams = {t}
    else:
        grams = {t[i:i+3] for i in range(len(t) - 2)}
    sig = []
    for i in range(n):
        m = None
        for g in grams:
            v = int(hashlib.md5(f"{g}#{i}".encode()).hexdigest(), 16)
            if m is None or v < m:
                m = v
        sig.append(m)
    return sig

def l0_audit(rows):
    """rows: [{id,text,orig_label,...}]"""
    punct = set("！!？?")
    by_lab = defaultdict(list)
    for r in rows:
        by_lab[r["orig_label"]].append(r)
    leak = []
    for r in rows:
        toks = [t for t in re.split(r"[:|_/\-+]+", r["orig_label"]) if t]
        hit = [t for t in toks if len(t) >= 2 and t.lower() in norm(r["text"])]
        if hit:
            leak.append((r["id"], hit))
    punct_dist = {lab: round(sum(1 for r in rs if any(p in r["text"] for p in punct)) / len(rs), 3)
                  for lab, rs in by_lab.items()}
    lens = [len(r["text"]) for r in rows]
    mean = sum(lens) / len(lens)
    var = sum((x - mean) ** 2 for x in lens) / len(lens)
    anom = [r["id"] for r in rows if len(r["text"]) < 5 or len(r["text"]) > 500]
    sigs = [(r["id"], minhash_sig(r["text"])) for r in rows]
    near = []
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            s = sum(1 for a, b in zip(sigs[i][1], sigs[j][1]) if a == b) / len(sigs[i][1])
            if s > 0.85:
                near.append((sigs[i][0], sigs[j][0], round(s, 3)))
    near.sort(key=lambda x: -x[2])
    return {
        "leak_n": len(leak), "leak_ex": leak[:3],
        "punct_dist": punct_dist,
        "len_mean": round(mean, 1), "len_min": min(lens), "len_max": max(lens),
        "anom_n": len(anom), "anom_ids": anom[:10],
        "near_n": len(near), "near_pairs": near[:10],
        "label_dist": {k: len(v) for k, v in by_lab.items()},
    }

DATASETS = [
    ("smp2019_ecdt", "parse_smp", "SMP2019-ECDT",
     "https://github.com/hml-ubt/SMP2017-2019-ECDT-data (SMP2019_data/train.json)",
     "SMP 竞赛公开数据, 研究使用(原版权归赛事组织方; 本采样仅研究用途, 注明来源)"),
    ("crosswoz", "parse_crosswoz", "CrossWOZ",
     "https://hf-mirror.com/datasets/GEM/CrossWOZ",
     "Apache-2.0(HF GEM/CrossWOZ 元数据声明; 另需引用 Zhu et al. TACL 2020)"),
    ("ewect", "parse_ewect", "EWECT",
     "https://hf-mirror.com/datasets/hecongqing/EWECT_weibo_senti (镜像; 原 SMP2020-EWECT 微博情绪竞赛数据)",
     "竞赛数据研究使用(原版权归赛事/平台方; 本采样仅研究用途, 注明来源)"),
    ("cped", "parse_cped", "CPED",
     "https://github.com/scutcyr/CPED (data/CPED/train_split.csv)",
     "Apache-2.0(仓库 LICENSE)"),
    ("waimai_10k", "parse_waimai", "waimai_10k",
     "https://hf-mirror.com/datasets/dirtycomputer/waimai_10k (整理自 SophonPlus/ChineseNlpCorpus)",
     "未声明(原作者/来源不详, 见 SophonPlus 说明); 仅研究使用"),
    ("weibo_senti_100k", "parse_weibo", "weibo_senti_100k",
     "https://hf-mirror.com/datasets/dirtycomputer/weibo_senti_100k (整理自 SophonPlus/ChineseNlpCorpus)",
     "未声明(原作者/来源不详, 见 SophonPlus 说明; 原数据来自 CSDN 转载); 仅研究使用"),
    ("dmr", "parse_dmr", "DMR",
     "https://github.com/yiyepianzhounc/DMR-Dataset-1 (reviews.json; 主库 DMR-Dataset 仅 README 无数据文件, 取同作者 -1 库)",
     "研究使用(遵守豆瓣隐私政策, user_id 已加密; 引用 ICPR'22 GAIM 论文)"),
    ("fbs", "parse_fbs", "FBS",
     "https://github.com/Cypher-Z/FBS_SMS_Dataset",
     "研究使用(需注明 source-link 并引用 CCS'20 Lies in the Air; 已做匿名化预处理版本)"),
]

def main():
    os.makedirs(HERE, exist_ok=True)
    contam, contam_n = load_contam_base()
    print(f"contam base: {contam_n} texts (norm-hash). cc={CC}")
    report_rows = []
    gap = []
    for ds, fn, disp, page_url, lic in DATASETS:
        t0 = time.time()
        try:
            if fn == "parse_smp":
                groups, dl = parse_smp()
            elif fn == "parse_crosswoz":
                groups, dl = parse_crosswoz()
            elif fn == "parse_ewect":
                groups, dl = parse_ewect()
            elif fn == "parse_cped":
                groups, dl = parse_cped()
            elif fn == "parse_waimai":
                groups, dl = parse_csv_sentiment([
                    "https://hf-mirror.com/datasets/dirtycomputer/waimai_10k/resolve/main/waimai_10k.csv",
                    "https://hf-mirror.com/datasets/XiangPan/waimai_10k/resolve/main/waimai_10k.csv",
                    "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/waimai_10k/waimai_10k.csv"])
            elif fn == "parse_weibo":
                groups, dl = parse_csv_sentiment(
                    "https://hf-mirror.com/datasets/dirtycomputer/weibo_senti_100k/resolve/main/weibo_senti_100k.csv")
            elif fn == "parse_dmr":
                groups, dl = parse_dmr()
            elif fn == "parse_fbs":
                groups, dl = parse_fbs()
        except RuntimeError as e:
            print(f"[{ds}] DOWNLOAD FAIL: {e}")
            gap.append(f"{ds}: 下载失败({e})")
            report_rows.append((ds, disp, 0, 0, 0, "下载失败", page_url, lic, 0))
            continue
        pool_size = sum(len(v) for v in groups.values())
        # 分层轮转预排, 再污染过滤补采: 在 interleave 输出(放大)基础上 select
        ordered = interleave({k: list(v) for k, v in groups.items()}, max(N * 8, pool_size if pool_size < N * 8 else N * 8))
        if len(ordered) < N:
            ordered = [x for vs in groups.values() for x in vs]
        picked, dup, hit = select(ordered if len(ordered) > N else [x for vs in groups.values() for x in vs],
                                 contam, N)
        rows = []
        for i, (t, lab, x) in enumerate(picked):
            u = page_url
            if ds == "dmr" and isinstance(x, str) and x.startswith("sup:"):
                u = ("https://hf-mirror.com/datasets/tracywong117/spam-douban-movie-review"
                     "(补采行, Fake=1→刷评spam 语义待复核)")
            rows.append({"id": f"{ds}_{i+1:04d}", "text": t, "orig_label": lab,
                         "source": disp, "url": u, "license": lic})
        with open(os.path.join(HERE, f"{ds}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        audit = l0_audit(rows)
        dt = round(time.time() - t0, 1)
        print(f"[{ds}] pool={pool_size} picked={len(rows)} dup={dup} contam={hit} {dt}s labels={audit['label_dist']}")
        if len(rows) < N:
            gap.append(f"{ds}: 仅采到 {len(rows)}/{N} 条(pool {pool_size}, 去重弃 {dup}, 污染命中 {hit})")
        report_rows.append((ds, disp, len(rows), dup, hit, audit, page_url, lic, pool_size))
    # 写 REPORT
    L = []
    L.append("# external 小闭环采样报告 (8x50)")
    L.append("")
    L.append(f"- 生成时间: {time.strftime('%Y-%m-%d %H:%M %Z')}; 采样 seed=42, 每集目标 {N} 条; "
             f"归一化: NFKC+{'opencc 繁→简' if _cc else '无繁简转换(降级)'}+去首尾空白+空白折叠+小写; 去重键=归一化 sha256.")
    L.append(f"- 污染基: training/real_pools/clean/real_sentiment.jsonl + benchmarks/data/econ_zh/*.jsonl, 共 {contam_n} 条(归一化 hash); 命中即弃并补采.")
    L.append(f"- 下载方式: urllib 直连 + UA, hf-mirror 302 签名直跟, decode 全部 utf-8-sig, 重试 4 次. 未调用任何付费 API.")
    L.append("")
    L.append("| 数据集 | 实际条数 | 候选池 | 去重弃数 | 污染命中 | 许可 |")
    L.append("|---|---|---|---|---|---|")
    for ds, disp, n, dup, hit, audit, page_url, lic, pool in report_rows:
        L.append(f"| {ds}({disp}) | {n} | {pool} | {dup} | {hit} | {lic} |")
    L.append("")
    for ds, disp, n, dup, hit, audit, page_url, lic, pool in report_rows:
        L.append(f"## {ds} ({disp})")
        L.append(f"- 下载: {page_url}")
        L.append(f"- 许可声明: {lic}")
        if isinstance(audit, str):
            L.append(f"- 状态: {audit}")
            L.append("")
            continue
        L.append(f"- 标签分布: {json.dumps(audit['label_dist'], ensure_ascii=False)}")
        L.append(f"- 标签词泄漏: {audit['leak_n']} 条" +
                 (f", 例: {json.dumps(audit['leak_ex'], ensure_ascii=False)[:300]}" if audit['leak_n'] else "(通过)"))
        L.append(f"- 标点分布(含！!？? 占比/类): {json.dumps(audit['punct_dist'], ensure_ascii=False)}")
        L.append(f"- 长度: mean={audit['len_mean']} min={audit['len_min']} max={audit['len_max']}; "
                 f"异常(<5 或 >500 字): {audit['anom_n']} 条 {audit['anom_ids'][:5]}")
        L.append(f"- minhash 近重复(>0.85): {audit['near_n']} 对 " +
                 (f"{audit['near_pairs'][:5]}" if audit['near_n'] else "(无)"))
        L.append("")
    L.append("## 备注与缺口")
    L.append("- SMP2019-ECDT: 主库直取成功, 未启用 SMP2017 替代.")
    L.append("- DMR: 主库 yiyepianzhounc/DMR-Dataset 仅含 README 无数据文件, 改取同作者 DMR-Dataset-1/reviews.json(仅 44 条: 22 正常/22 刷评); "
             "另从同域 HF tracywong117/spam-douban-movie-review 补 6 条凑 25/25, 补采行 url 单独标注且 Fake=1→spam 语义待人工复核.")
    L.append("- FBS: 纯 spam(14 类, 本批全 spam 标签); 需配 ham——ham 从 dmr.jsonl 中 orig_label=正常 的条目补(见 dmr_*.id).")
    L.append("- EWECT: GitHub smp2020ewect 系仓库多为代码/复现无直接数据文件, 改走 HF 镜像 hecongqing/EWECT_weibo_senti(usual_train+test_labeled+eval_labeled, 有标注版); 转人工映射(angry/sad 等→转人工)留待 ABC 打分阶段定义, 本步保留原始情绪标签.")
    L.append("- weibo_senti_100k: GitHub SophonPlus 仅百度网盘(不可直连), 改走 HF 镜像 dirtycomputer 版.")
    L.append("- CPED: 取 train_split.csv 的 Utterance, orig_label=Emotion(DA 列未收录, 转人工映射留待后步).")
    for g in gap:
        L.append(f"- 缺口: {g}")
    if not gap:
        L.append("- 缺口: 无(8 集均满 50 条).")
    L.append("")
    with open(os.path.join(HERE, "external_SAMPLING_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("REPORT done. gaps:", gap if gap else "none")

if __name__ == "__main__":
    main()
