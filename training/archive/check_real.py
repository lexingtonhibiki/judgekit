# -*- coding: utf-8 -*-
"""自检 training/real_pools/ 下五个真实语料候选池。
输出：每池条数(对照下限目标)/类目分布/来源分布/池内与跨池重复率/needs_review 比例/长度合规/schema 校验，并每池抽样 5 条。
用法: python training/archive/check_real.py
（real_pools/ 仍在 training/ 平铺，本脚本经 archive 目录用 __file__ 的上上级定位它。）
"""
import json, re, hashlib, collections, os, sys

sys.stdout.reconfigure(encoding='utf-8')
PDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'real_pools')
POOLS = ['real_sentiment', 'real_spam', 'real_route', 'real_urgency', 'real_absurd']
TARGETS = {'real_sentiment': 120, 'real_spam': 60, 'real_route': 80, 'real_urgency': 60, 'real_absurd': 40}
KEYS = ['id', 'target_dataset', 'text', 'proposed_label', 'source', 'source_url', 'gold_source', 'needs_review']


def nh(t):
    return hashlib.md5(re.sub(r'\s+', '', t).lower().encode()).hexdigest()


all_h = collections.Counter()
total_rows = 0
for name in POOLS:
    path = os.path.join(PDIR, name + '.jsonl')
    if not os.path.exists(path):
        print(f'== {name}: 文件缺失 ==')
        continue
    rows = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
    total_rows += len(rows)
    lab = collections.Counter(r['proposed_label'] for r in rows)
    src = collections.Counter(r['source'].split('(')[0] for r in rows)
    nr = sum(1 for r in rows if r['needs_review'])
    bad_schema = [r['id'] for r in rows if any(k not in r for k in KEYS)]
    bad_id = [r['id'] for r in rows if not r['id'].startswith('real_')]
    bad_len = [r['id'] for r in rows if not (4 <= len(r['text']) <= 300)]
    hs = [nh(r['text']) for r in rows]
    dup_in = len(hs) - len(set(hs))
    seen_before = sum(1 for h in hs if all_h[h] > 0)
    for h in hs:
        all_h[h] += 1
    print(f'== {name} ==  条数 {len(rows)} (下限 {TARGETS[name]}, {"达标" if len(rows) >= TARGETS[name] else "未达标"})')
    print(f'  类目分布: {dict(lab.most_common())}')
    print(f'  来源分布: {dict(src.most_common())}')
    print(f'  needs_review: {nr}/{len(rows)} ({nr*100//max(len(rows),1)}%)  池内重复: {dup_in}  与其他池重复: {seen_before}')
    print(f'  schema缺字段: {len(bad_schema)}  id前缀错: {len(bad_id)}  长度越界: {len(bad_len)}')
    for r in rows[:5]:
        print(f'    样例 {r["id"]} [{r["proposed_label"]}] {r["text"][:52]}')
    print()

uniq = sum(1 for v in all_h.values() if v == 1)
print(f'== 汇总 ==  总条数 {total_rows}  全局唯一文本 {uniq}  跨池重复 {total_rows - uniq}')
