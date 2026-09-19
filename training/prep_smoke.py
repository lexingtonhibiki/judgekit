# -*- coding: utf-8 -*-
"""把 judge-econ 130 条切成 train/test 并转成 NanoJev 训练记录（hard gold）。"""
import json
import random
from pathlib import Path

import yaml

ROOT = Path('E:/Projects/MyGitHub/jev-judge-projects')
RES = ROOT / 'benchmarks' / 'results'
OUT = ROOT / 'training'
DS = ['intent_zh', 'sentiment_zh', 'spam_zh', 'urgency_zh']
META = {
    'intent_zh': ('zh_intent', 'route', '判断该电商求助内容应流转到哪个部门'),
    'sentiment_zh': ('zh_sentiment', 'polarity', '判断这条商品评论的情感倾向'),
    'spam_zh': ('zh_spam', 'spam', '判断这条评论是正常内容还是垃圾广告'),
    'urgency_zh': ('zh_urgency', 'urgent', '判断这条客服消息的紧急程度'),
}

# 原始文本（bench 结果没存 text，按 id 从数据集找回）
ORIG, GOLD_ALL = {}, {}
for ds in DS:
    for l in open(ROOT / 'benchmarks' / 'data' / f'{ds}.jsonl', encoding='utf-8'):
        if l.strip():
            o = json.loads(l)
            ORIG[o['id']] = o['text']
            GOLD_ALL.setdefault(ds, {})[o['id']] = o['label']

rng = random.Random(17)
splits = {'train': [], 'dev': [], 'calibration': [], 'test': [], 'ood': []}
for ds in DS:
    family, qid, instructions = META[ds]
    labels = sorted(set(GOLD_ALL[ds].values()))
    criteria = {lb: lb for lb in labels}
    rows = [json.loads(l) for l in open(RES / f'typesafe__{ds}.jsonl', encoding='utf-8') if l.strip()]
    rows = [r for r in rows if r['ok']]
    rng.shuffle(rows)
    n = len(rows)
    cuts = [('train', n * 4 // 10), ('dev', n * 2 // 10), ('calibration', n // 10),
            ('test', n // 10), ('ood', n - (n * 4 // 10 + n * 2 // 10 + n // 10 + n // 10))]
    idx = 0
    for split, cnt in cuts:
        for r in rows[idx:idx + cnt]:
            text = ORIG[r['id']]
            splits[split].append({
                'id': f'smoke_{r["id"]}', 'state_id': f'smoke_{r["id"]}',
                'family_id': family, 'split': split,
                'state': json.dumps({'text': text}, ensure_ascii=False),
                'questions': {qid: {'type': 'choice', 'instructions': instructions,
                                     'criteria': criteria}},
                'gold': {qid: r['gold']},
                'gold_probs': {qid: {lb: (1.0 if lb == r['gold'] else 0.0) for lb in labels}},
                'gold_probs_kind': {qid: 'deterministic_truth'},
            })
        idx += cnt

for split, rows in splits.items():
    with open(OUT / f'smoke_{split}.jsonl', 'w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
with open(OUT / 'smoke_all.jsonl', 'w', encoding='utf-8', newline='\n') as f:
    for rows in splits.values():
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
print({k: len(v) for k, v in splits.items()})
