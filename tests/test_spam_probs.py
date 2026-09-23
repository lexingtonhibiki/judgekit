# -*- coding: utf-8 -*-
"""T18离线测试：sweep边界+推荐确定性+缺分布抛错。全离线，不联网不读key。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from judgekit.engine import Decision  # noqa: E402
from training import eval_spam_probs as m  # noqa: E402
from training import eval_spam120 as t17  # noqa: E402
from training import eval_jev_v4 as jev  # noqa: E402

EXP_SPAM = ("场景：购物评价区。你的身份是潜在买家。垃圾判定：对于你而言，"
            "纯粹情绪称赞（刷好评返现）/谩骂（恶意攻击）等不提供真实消费反馈的均为垃圾；"
            "只看text字段独立判定")


def _dec(probs) -> Decision:
    return Decision("classify", "垃圾", 0.9, "{}", "typesafe", 10, 0.0,
                    probabilities=probs)


def _recs() -> list[dict]:
    # 4行：2垃圾(p高/低)+2正常(p高/低)，τ滑动可分；argmax固定供flips计数。
    return [
        {"id": "a", "gold": "垃圾", "p_spam": 0.9, "pred_argmax": "垃圾"},
        {"id": "b", "gold": "垃圾", "p_spam": 0.4, "pred_argmax": "正常"},
        {"id": "c", "gold": "正常", "p_spam": 0.6, "pred_argmax": "垃圾"},
        {"id": "d", "gold": "正常", "p_spam": 0.1, "pred_argmax": "正常"},
    ]


def test_prompt_verbatim_reuse_t17():
    assert m.SPAM_PROMPT == EXP_SPAM
    assert m.SPAM_PROMPT is t17.SPAM_PROMPT  # import复用，禁复制
    assert m.SPAM_PROMPT is jev.SCNV2_SPAM
    t = m.build_spam_task()
    assert t.primitive == "classify" and t.labels == ["垃圾", "正常"]
    assert t.instruction == EXP_SPAM and t.provider == "typesafe"


def test_extract_p_spam_winsorize():
    assert m.extract_p_spam(_dec({"垃圾": 0.7, "正常": 0.3})) == 0.7
    assert m.extract_p_spam(_dec({"垃圾": 1.5, "正常": -0.5})) == 1.0
    assert m.extract_p_spam(_dec({"垃圾": -2.0})) == 0.0


def test_extract_p_spam_missing_raises():
    for bad in (None, {}, {"正常": 0.5}, {"垃圾": "nan-x"}, "notadict"):
        try:
            m.extract_p_spam(_dec(bad))
        except SystemExit as e:
            assert "NEEDS_CONTEXT" in str(e)
        else:
            raise AssertionError(f"缺分布未停线：{bad!r}")


def test_sweep_boundaries_tau0_all_spam_tau1_all_normal():
    recs = _recs()
    lo = m.sweep_thresholds(recs, [0.0])[0]
    hi = m.sweep_thresholds(recs, [1.0])[0]
    # τ=0：全判垃圾→垃圾召回100%，正常特异度0，flips=2（b翻转？不，verdict全垃圾 vs argmax：b正常→翻1，c垃圾→同，计2）
    assert lo["recall"] == 1.0 and lo["specificity"] == 0.0
    assert lo["flips"] == sum(1 for r in recs if "垃圾" != r["pred_argmax"])
    # τ=1：p_spam≥1才判垃圾（本集最高0.9）→全判正常→召回0，特异度1
    assert hi["recall"] == 0.0 and hi["specificity"] == 1.0
    assert hi["flips"] == sum(1 for r in recs if "正常" != r["pred_argmax"])
    # 阈值单调：τ↑召回非增、特异度非减
    tabs = m.sweep_thresholds(recs, [0.2, 0.5, 0.8])
    assert tabs[0]["recall"] >= tabs[1]["recall"] >= tabs[2]["recall"]
    assert tabs[0]["specificity"] <= tabs[1]["specificity"] <= tabs[2]["specificity"]


def test_recommend_deterministic_youden_f1_small_tau():
    recs = _recs()
    t1 = m.sweep_thresholds(recs)
    t2 = m.sweep_thresholds(list(recs))
    assert m.recommend_threshold(t1) == m.recommend_threshold(t2)  # 确定性
    # 并列取小τ：构造两τ间无p_spam落点→指标全同，推荐取小者
    tie = [{"id": "x", "gold": "垃圾", "p_spam": 0.9, "pred_argmax": "垃圾"},
           {"id": "y", "gold": "正常", "p_spam": 0.1, "pred_argmax": "正常"}]
    tab = m.sweep_thresholds(tie, [0.2, 0.3, 0.8])
    assert tab[0]["youden"] == tab[1]["youden"] == 1.0
    assert tab[0]["f1"] == tab[1]["f1"] == 1.0
    assert m.recommend_threshold(tab)["tau"] == 0.2
