# -*- coding: utf-8 -*-
"""T4探针离线测试：--temp-ab/C恒0.2/--input装载/去重污染/CSDS派生/验收门。

全部离线（mock transport + 合成数据，不联网不读key）。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import abc_score as m  # noqa: E402
from training import derive_csds as dc  # noqa: E402
from training import fetch_probe as fp  # noqa: E402


def _chat_provider(reply: dict):
    from judgekit.providers.go_openai import GoChatProvider
    calls = []

    def transport(url, headers, body):
        calls.append(json.loads(body.decode("utf-8")))
        return reply
    p = GoChatProvider(name="go-c", model="mimo-v2.6-flash",
                       api_key="k-test", transport=transport)
    return {"go-c": p}, calls


CHAT_SCORE = {"choices": [{"message": {"content":
    '{"score": 2, "confidence": 0.8, "reason": "x"}'}}], "usage": {}}


def test_temp_ab_in_cache_key():
    k_old = m.cache_key_of("i", "a", "b", "c", "c2", 4.0, "off")
    assert "|TAB=" not in k_old  # T3旧格式兼容
    k95 = m.cache_key_of("i", "a", "b", "c", "c2", 4.0, "off", 0.95)
    k70 = m.cache_key_of("i", "a", "b", "c", "c2", 4.0, "off", 0.7)
    assert k95 != k70 and "|TAB=0.95" in k95  # 换温即重打


def test_temp_ab_default_and_c_stays_02():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp-ab", type=float, default=0.95)
    assert ap.parse_args([]).temp_ab == 0.95  # 默认0.95
    provs, calls = _chat_provider(CHAT_SCORE)
    ad = m.Adapter(provs, "go-c", transport=provs["go-c"].transport)
    a = {"value": 1.0, "reason": "差", "confidence": 0.9}
    b = {"value": 8.0, "reason": "好", "confidence": 0.9}
    c, delta = m.arbitrate("sentiment", "t", a, b, ad, "")
    assert delta == 7.0 and c["called"] == "rejudge"
    assert calls[0]["temperature"] == 0.2  # C恒0.2


def test_load_input_ok_and_rejects():
    import tempfile
    rows = [{"id": "x1", "text": "好", "task": "spam", "source": "S",
             "orig_label": "正常", "url": "u", "license": "l"}]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     encoding="utf-8") as f:
        f.write(json.dumps(rows[0], ensure_ascii=False) + "\n")
        p = f.name
    items, rl = m.load_input(p)
    assert items[0]["task"] == "spam" and rl == {}
    bad = [{"id": "x2", "text": "好", "task": "nope"}]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     encoding="utf-8") as f:
        f.write(json.dumps(bad[0], ensure_ascii=False) + "\n")
        p2 = f.name
    try:
        m.load_input(p2)
        raise AssertionError("未知task应抛错")
    except ValueError:
        pass


def test_md5_dedup_and_contam():
    assert fp.md5key("  好用 推荐 ") == fp.md5key("好用 推荐")  # 归一化
    contam = {fp.md5key("污染句")}
    pool = [("污染句", "正常", None), ("干净甲", "正常", None),
            ("干净甲", "正常", None), ("干净乙", "刷评spam", None)]
    picked, dup, near, hit = fp.select_md5(pool, contam, 3, seed=1)
    assert hit == 1 and dup == 1 and len(picked) == 2


def test_pick_balanced_stratified():
    groups = {"正常": [(f"好{i}", "正常", None) for i in range(20)],
              "刷评spam": [(f"夸{i}", "刷评spam", None) for i in range(20)]}
    picked, st = fp.pick_balanced(groups, set(), 5, seed=42)
    from collections import Counter
    assert Counter(lab for _, lab, _ in picked) == {"正常": 5, "刷评spam": 5}


def _dlg(qas, turns):
    return {"DialogueID": 1, "QA": qas,
            "Dialogue": [{"speaker": s, "turn": t, "utterance": u}
                         for s, t, u in turns]}


def test_derive_h1_dup_intent():
    d = _dlg(
        [{"QueSumm": "问A", "AnsSummShort": "答A", "QueSummUttIDs": [2], "intent": "改单"},
         {"QueSumm": "问B", "AnsSummShort": "答B", "QueSummUttIDs": [5], "intent": "改单"}],
        [("Q", 2, "改单怎么改"), ("A", 3, "答A"), ("Q", 5, "还是改单问题"), ("A", 6, "答B")])
    r = dc.derive_one(d)
    assert r["pre"] == "转人工" and r["rule"].startswith("H1")


def test_derive_h2_hollow_then_followup():
    d = _dlg(
        [{"QueSumm": "问地址", "AnsSummShort": "客服说暂时无法修改",
          "QueSummUttIDs": [2], "intent": "改单"},
         {"QueSumm": "问券", "AnsSummShort": "可以退回",
          "QueSummUttIDs": [9], "intent": "退券"}],
        [("Q", 2, "地址错了"), ("A", 3, "暂时无法修改"),
         ("Q", 9, "那券呢"), ("A", 10, "可以退回")])
    r = dc.derive_one(d)
    assert r["pre"] == "转人工" and "H2" in r["rule"] and "那券呢" in r["text"]


def test_derive_no_trigger():
    d = _dlg(
        [{"QueSumm": "问地址", "AnsSummShort": "提供订单后可修改",
          "QueSummUttIDs": [2], "intent": "改单"}],
        [("Q", 2, "地址错了"), ("A", 3, "提供订单后可修改")])
    r = dc.derive_one(d)
    assert r["pre"] == "不转" and r["window_turns"] >= 1


def test_derive_uttid_misaligned_falls_back():
    d = _dlg(
        [{"QueSumm": "问冰箱", "AnsSummShort": "让保持畅通",
          "QueSummUttIDs": [1], "intent": "配送"}],
        [("Q", 0, "冰箱明天能到吗"), ("A", 1, "您好客服"), ("Q", 6, "是的"),
         ("Q", 8, "好")])
    r = dc.derive_one(d)
    assert r["text"].strip() and r["window_turns"] >= 1
    assert "回退" in r["rule"]


def test_gates_out_and_rework(tmp_path):
    scores = []
    for i in range(5):  # FakeReview：3/5需人工=0.6 → OUT
        scores.append({"id": f"fk_{i}", "source": "FakeReview", "task": "spam",
                       "A": {"ok": True}, "B": {"ok": True},
                       "C": {"ok": True, "action": "需人工复核" if i < 3 else "通过",
                             "final": "正常", "confidence": 0.8}})
    for i in range(4):  # CSDS：pre全转人工，C判2对2错 → 一致率0.5 → 回炉
        scores.append({"id": f"csds_{i}", "source": "CSDS", "task": "handoff",
                       "A": {"ok": True}, "B": {"ok": True},
                       "C": {"ok": True, "action": "通过",
                             "final": "转人工" if i < 2 else "不转",
                             "confidence": 0.8}})
    sp = tmp_path / "s.jsonl"
    sp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in scores),
                  encoding="utf-8")
    dp = tmp_path / "d.jsonl"
    dp.write_text("\n".join(
        json.dumps({"id": f"csds_{i}", "orig_label": "转人工"}, ensure_ascii=False)
        for i in range(4)), encoding="utf-8")
    rep = dc.gates(sp, dp, tmp_path / "g.json")
    assert rep["FakeReview"]["verdict"] == "OUT"
    assert rep["CSDS_agree"] == {"n": 4, "agree": 2, "agree_rate": 0.5,
                                 "verdict": "回炉"}
