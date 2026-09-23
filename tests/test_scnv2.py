# -*- coding: utf-8 -*-
"""T15用户版提示词verbatim离线测试：两句逐字断言×4路径(Jev×2+ABC×2)，不调网。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import abc_score as abc  # noqa: E402
from training import eval_jev_v4 as jev  # noqa: E402

EXP_SPAM = "场景：购物评价区。你的身份是潜在买家。垃圾判定：对于你而言，纯粹情绪称赞（刷好评返现）/谩骂（恶意攻击）等不提供真实消费反馈的均为垃圾；只看text字段独立判定"
EXP_HANDOFF = "场景：用户对客服说话。你的身份为客服或分诊系统。转人工判定：需要人工介入操作的, 如修改订单,多次发消息催促的(>=2), 非规范化信息回复的；text字段换行表示多段对话.只看text字段独立判定。"


def test_jev_spam_scnv2_verbatim():
    assert jev.SCNV2_SPAM == EXP_SPAM
    assert ">=2" in jev.SCNV2_HANDOFF  # 换行表示多段对话原文保留对照（handoff侧）
    t = jev.build_tasks(scn="scnv2")["spam"]
    assert t.instruction == EXP_SPAM  # 整体替换，非前置叠加
    assert t.label_descriptions["垃圾"] == "无消费反馈的称赞/谩骂/引流其余不判"
    assert t.label_descriptions["正常"] == "其余"
    assert "scnv2" in t.name
    off = jev.build_tasks()["spam"]
    assert off.instruction != EXP_SPAM  # 默认off保旧链


def test_jev_handoff_scnv2_verbatim():
    assert jev.SCNV2_HANDOFF == EXP_HANDOFF
    assert ">=2" in jev.SCNV2_HANDOFF
    assert "换行表示多段对话" in jev.SCNV2_HANDOFF
    t = jev.build_tasks(scn="scnv2")["handoff"]
    assert t.instruction == EXP_HANDOFF  # 整体替换
    assert t.label_descriptions["转人工"] == "需人工介入：改单/连催/非规范回复"
    assert t.label_descriptions["不转"] == "其余"
    assert "scnv2" in t.name
    off = jev.build_tasks()["handoff"]
    assert off.instruction != EXP_HANDOFF


def test_abc_spam_scnv2_both_paths():
    assert abc.SCNV2_SPAM == EXP_SPAM
    p = abc.Adapter._abc_user_prompt("spam", "很好用", [], "", "off", "off", "scnv2")
    assert p.startswith(EXP_SPAM)  # 与旧泛化串并存，位置放最前
    assert "举一反三" in p
    from judgekit.providers.go_openai import GoChatProvider
    pr = GoChatProvider(name="go-c", model="m", api_key="k-test")
    ad = abc.Adapter({"go-c": pr}, "go-c", scn="scnv2")
    t = ad._typesafe_task("spam", [])
    assert t.instruction.startswith(EXP_SPAM)
    assert "举一反三" in t.instruction
    k_off = abc.cache_key_of("i", "a", "b", "c", "c2", 4.0, "off", 0.95)
    k_v2 = abc.cache_key_of("i", "a", "b", "c", "c2", 4.0, "off", 0.95,
                            "off", "off", False, "scnv2")
    assert "SCNv2" in k_v2 and k_v2 != k_off


def test_abc_handoff_scnv2_both_paths():
    assert abc.SCNV2_HANDOFF == EXP_HANDOFF
    assert ">=2" in abc.SCNV2_HANDOFF
    assert "换行表示多段对话" in abc.SCNV2_HANDOFF
    p = abc.Adapter._abc_user_prompt("handoff", "你好", [], "", "off", "off", "scnv2")
    assert p.startswith(EXP_HANDOFF)
    assert "重复催" in p  # 与旧泛化串并存
    from judgekit.providers.go_openai import GoChatProvider
    pr = GoChatProvider(name="go-c", model="m", api_key="k-test")
    ad = abc.Adapter({"go-c": pr}, "go-c", scn="scnv2")
    t = ad._typesafe_task("handoff", [])
    assert t.instruction.startswith(EXP_HANDOFF)
