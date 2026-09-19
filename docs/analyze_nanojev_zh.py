# -*- coding: utf-8 -*-
"""NanoJev 中文低准根因分析：置信度分层、校准行为、级联实验、分布坍缩。全零成本。"""
import json
from pathlib import Path

RES = Path('E:/Projects/MyGitHub/jev-judge-projects/benchmarks/results')
DS = ['intent_zh', 'sentiment_zh', 'spam_zh', 'urgency_zh']


def load(prov, ds):
    return [json.loads(l) for l in open(RES / f'{prov}__{ds}.jsonl', encoding='utf-8') if l.strip()]


recs = [r for ds in DS for r in load('nanojev-local', ds)]
rules = {r['id'] + '@' + ds: r
         for ds in DS for r in load('rules', ds)}

print('=== 1) 置信度 × 正确性（NanoJev 130 条，OOD 行为）===')
bins = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.8), (0.8, 1.01)]
for lo, hi in bins:
    g = [r for r in recs if lo <= r['confidence'] < hi]
    if g:
        acc = sum(r['correct'] for r in g) / len(g)
        print(f'  conf [{lo:.1f},{hi:.1f}): n={len(g):3d} acc={acc:.1%}')

print('\n=== 2) 预测分布坍缩检查（各数据集 pred 直方图，top-1 占比）===')
for ds in DS:
    g = load('nanojev-local', ds)
    from collections import Counter
    c = Counter(r['pred'] for r in g)
    gold = Counter(r['gold'] for r in g)
    top_pred, n_top = c.most_common(1)[0]
    print(f'  {ds:14s} 预测最多类={top_pred}({n_top}/{len(g)}) | gold 分布={dict(gold)} | 预测熵类数={len(c)}')

print('\n=== 3) 级联实验：NanoJev(conf>=τ) 自己答，否则交给 rules ===')
print('   τ     升级率   级联准确率')
best = None
for i in range(0, 21):
    tau = i / 20
    esc = [r for r in recs if r['confidence'] < tau]
    keep = [r for r in recs if r['confidence'] >= tau]
    if not keep:
        continue
    acc = (sum(r['correct'] for r in keep)
           + sum(rules[r['id'] + '@' + ('intent_zh' if r['id'].startswith("i") else
                                        'sentiment_zh' if r['id'].startswith("s") else
                                        'spam_zh' if r['id'].startswith("p") else 'urgency_zh')]['correct']
                 for r in esc)) / len(recs)
    rate = len(esc) / len(recs)
    tag = ''
    if best is None or acc > best[1]:
        best = (tau, acc, rate)
    print(f'  {tau:.2f}  {rate:6.1%}   {acc:.1%}')
print(f'\n  最优: τ={best[0]:.2f} -> acc={best[1]:.1%} (升级率 {best[2]:.1%})')
print(f'  对照: 纯 NanoJev 61.5% | 纯 rules 91.5% | Jev API 97.7%')

print('\n=== 4) Jev vs NanoJev 同题对照：Jev 错的 3 条 NanoJev 对了几条 ===')
jev_wrong = {r['id'] for ds in DS for r in load('typesafe', ds) if not r['correct']}
nj = {r['id']: r for ds in DS for r in load('nanojev-local', ds)}
for i in sorted(jev_wrong):
    print(f"  {i}: NanoJev {'对' if nj[i]['correct'] else '错'} (conf {nj[i]['confidence']:.2f})")
