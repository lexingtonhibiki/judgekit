# -*- coding: utf-8 -*-
"""T16离线测试：06汇总spam集合51行+提示词逐字单例+gold亲审口径。全离线，不联网不读key。"""
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import eval_spam_only as m  # noqa: E402
from training import eval_jev_v4 as jev  # noqa: E402
from training import freeze_gold as fg  # noqa: E402

EXP_SPAM = "场景：购物评价区。你的身份是潜在买家。垃圾判定：对于你而言，纯粹情绪称赞（刷好评返现）/谩骂（恶意攻击）等不提供真实消费反馈的均为垃圾；只看text字段独立判定"

XLSX_PATH = ROOT / "training" / "abc_out" / "数据审核_v4_full.xlsx"
needs_local_review = pytest.mark.skipif(
    not XLSX_PATH.exists(),
    reason="依赖本机终审产物 training/abc_out/数据审核_v4_full.xlsx（gitignored），无则跳过")


@needs_local_review
def test_spam_id_set_51_fk22_jd29():
    ids = m.load_spam_ids()
    assert len(ids) == 51
    assert len(set(ids)) == 51  # 去重51
    c = Counter(i.split("_")[0] for i in ids)
    assert c["fk"] == 22 and c["jd"] == 29
    gold_by_id = {json.loads(l)["id"]: json.loads(l)
                  for l in open(ROOT / "training" / "abc_out" / "gold_frozen.jsonl",
                                encoding="utf-8") if l.strip()}
    for i in ids:
        assert i in gold_by_id
        assert gold_by_id[i]["task"] == "spam"
    # handoff 14行（转人工）未混入
    assert not any(i.startswith("csds_") for i in ids)


def test_prompt_verbatim_singleton():
    assert m.SPAM_PROMPT == EXP_SPAM
    assert jev.SCNV2_SPAM == EXP_SPAM  # 与SCNv2-spam句同一口径
    assert "你的身份是潜在买家" in m.SPAM_PROMPT  # 第二人称身份版
    t = m.build_spam_task()
    assert t.primitive == "classify"
    assert t.labels == ["垃圾", "正常"]
    assert t.instruction == EXP_SPAM  # 整体逐字，非前置叠加
    assert t.provider == "typesafe"
    assert t.input_field == "text"


@needs_local_review
def test_gold_frozen_flip_51():
    """gold口径：改标=C取反/✓通过=C认同（亲审口径，不重算只断言）。"""
    ids = set(m.load_spam_ids())
    assert len(ids) == 51
    n_flip = n_pass = 0
    with open(ROOT / "training" / "abc_out" / "gold_frozen.jsonl",
              encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o["id"] not in ids:
                continue
            assert o["gold"] in ("垃圾", "正常")
            if o["我的最终"] == "改标":
                assert o["gold"] == fg.flip("spam", o["C终判"])
                n_flip += 1
            elif o["我的最终"] == "✓通过":
                assert o["gold"] == o["C终判"]
                n_pass += 1
            else:
                pytest.fail(f"未知裁决:{o['id']}={o['我的最终']!r}")
    assert n_flip + n_pass == 51 and n_flip == 34 and n_pass == 17
