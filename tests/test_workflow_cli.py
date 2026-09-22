# -*- coding: utf-8 -*-
"""工作流集成面测试：单条 judge / stdin 管道 / fail-under 门禁 / 保留名 rules 免配置。"""
import json
import subprocess
import sys

TRIAGE = "judgekit/examples/triage.yaml"
SCORE = "judgekit/examples/score_comment.yaml"


def _cli(*args, stdin_text=None):
    return subprocess.run(
        [sys.executable, "-m", "judgekit", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        input=stdin_text, timeout=60,
    )


def test_rules_reserved_name_needs_no_providers_yaml():
    """task.provider='rules' 时零配置直跑，不应再报 missing-provider。"""
    from judgekit.engine import Task, run_task
    task = Task.load(TRIAGE)
    dec = run_task(task, {"text": "快递三天了还没到，催单"}, {})
    assert dec.ok and dec.provider == "rules"


def test_judge_single_text_exit0_and_json():
    r = _cli("judge", TRIAGE, "我的订单三天了还没发货，再不处理就投诉了")
    assert r.returncode == 0, r.stderr
    rec = json.loads(r.stdout.strip().splitlines()[-1])
    assert rec["primitive"] == "route" and rec["value"] in ("退款售后", "物流查询", "投诉建议", "技术故障", "咨询其他")


def test_judge_no_hit_exits_1():
    """score 原语规则兜底必失败 → 退出码 1，shell 可作门。"""
    r = _cli("judge", SCORE, "随便一句话")
    assert r.returncode == 1
    rec = json.loads(r.stdout.strip().splitlines()[-1])
    assert rec["ok"] is False


def test_run_stdin_pipe():
    line = json.dumps({"id": 1, "text": "想退货退款"}, ensure_ascii=False)
    r = _cli("run", TRIAGE, "--input", "-", stdin_text=line + "\n")
    assert r.returncode == 0, r.stderr
    rec = json.loads(r.stdout.strip().splitlines()[0])
    assert rec["ok"] is True


def test_run_fail_under_exit_2(tmp_path):
    """score 原语 ok 率 0%，门禁 100% → 退出码 2。"""
    data = tmp_path / "in.jsonl"
    data.write_text(json.dumps({"id": 1, "text": "随便"}, ensure_ascii=False) + "\n", encoding="utf-8")
    r = _cli("run", SCORE, "--input", str(data), "--fail-under", "100")
    assert r.returncode == 2


def test_run_bad_line_does_not_kill_batch(tmp_path):
    """坏 JSON 行记为 error 事件继续跑，批次不炸（借鉴 evals 事件化容错）。"""
    data = tmp_path / "in.jsonl"
    data.write_text('{"id": 1, "text": "想退货退款"}\n这不是JSON\n', encoding="utf-8")
    r = _cli("run", TRIAGE, "--input", str(data))
    assert r.returncode == 0, r.stderr
    recs = [json.loads(l) for l in r.stdout.strip().splitlines() if l.startswith("{")]
    assert len(recs) == 2 and recs[0]["ok"] is True and recs[1]["ok"] is False
    assert recs[1]["error"].startswith("bad-json")


def test_gold_label_not_leaked_into_rules(tmp_path):
    """金标字段不参与规则匹配：text 没命中但 label 字段含关键词 → 仍 no-hit。"""
    data = tmp_path / "in.jsonl"
    data.write_text(json.dumps({"id": 1, "text": "今天天气不错",
                                "label": "退款售后"}, ensure_ascii=False) + "\n", encoding="utf-8")
    r = _cli("run", TRIAGE, "--input", str(data))
    rec = json.loads(r.stdout.strip().splitlines()[0])
    assert rec["ok"] is False and rec["error"].startswith("rules-no-hit")


def test_verify_string_false_is_false():
    """模型把布尔写成字符串 "false" 时必须解析为 False（bool("false") 曾翻转真值）。"""
    from judgekit.engine import Task
    from judgekit.providers.openai_compat import parse_decision
    task = Task.from_dict({"name": "v", "primitive": "verify", "criteria": "成立吗"})
    dec = parse_decision(task, '{"verdict": "false", "confidence": 0.9}', "m", 1, 0.0)
    assert dec.value is False
    for bad in ('{"verdict": "不知道"}', '{"confidence": 0.5}'):
        try:
            parse_decision(task, bad, "m", 1, 0.0)
            assert False, "应抛 ValueError"
        except ValueError:
            pass


def test_label_raw_scan_removed():
    """响应正文顺嘴提候选词不再命中（三级模糊匹配已删）。"""
    from judgekit.engine import Task
    from judgekit.providers.openai_compat import parse_decision
    task = Task.from_dict({"name": "t", "primitive": "route", "labels": ["物流查询", "退款售后"]})
    try:
        parse_decision(task, '{"label": "不知道", "confidence": 0.9}（顺嘴：物流查询挺慢的）', "m", 1, 0.0)
        assert False, "应抛 ValueError"
    except ValueError as e:
        assert "label-not-in-candidates" in str(e)


def test_provider_not_in_registry_stderr_gate_and_exit_contract(tmp_path):
    """provider 不在注册表：stderr 门必须响；有 fallback 时批次经兜底存活（exit 0，防无声假绿）。"""
    prov = tmp_path / "providers.yaml"
    prov.write_text("providers:\n  demo:\n    kind: rules\n", encoding="utf-8")
    task = tmp_path / "ghost.yaml"
    task.write_text(
        "name: ghost\nprimitive: route\nlabels: [退款售后, 物流查询]\n"
        "provider: ghost\nfallback_rules:\n  退款售后: [退款]\n  物流查询: [物流]\n",
        encoding="utf-8")
    data = tmp_path / "in.jsonl"
    data.write_text(json.dumps({"id": 1, "text": "想退货退款"}, ensure_ascii=False) + "\n", encoding="utf-8")
    r = _cli("run", str(task), "--providers", str(prov), "--input", str(data))
    assert r.returncode == 0, r.stderr
    assert "不在注册表" in r.stderr
    rec = json.loads(r.stdout.strip().splitlines()[0])
    assert rec["ok"] is True and rec["provider"] == "rules-after-fail"


def test_judge_missing_provider_no_fallback_exits_1(tmp_path):
    """无 fallback 的 ghost provider：judge 必须 exit 1（shell 门不假绿）。"""
    prov = tmp_path / "providers.yaml"
    prov.write_text("providers:\n  demo:\n    kind: rules\n", encoding="utf-8")
    task = tmp_path / "ghost.yaml"
    task.write_text(
        "name: ghost\nprimitive: score\ncriteria: 拱火度\nprovider: ghost\n",
        encoding="utf-8")
    r = _cli("judge", str(task), "--providers", str(prov), "随便一句话")
    assert r.returncode == 1
    assert "不在注册表" in r.stderr


def test_build_prompt_only_whitelisted_field():
    """prompt builder 只放 input_field 白名单字段，金标/元数据不得进 prompt（防金标泄漏）。"""
    from judgekit.engine import Task
    from judgekit.providers.openai_compat import build_prompt
    t = Task.from_dict({"name": "t", "primitive": "classify", "labels": ["a", "b"]})
    _, user = build_prompt(t, {"text": "正文在此", "label": "泄密标签a", "id": 7})
    assert "正文在此" in user
    assert "泄密标签" not in user


def test_run_empty_input_exits_1(tmp_path):
    """空输入按失败处理：exit 1（防空跑假绿）。"""
    data = tmp_path / "empty.jsonl"
    data.write_text("\n", encoding="utf-8")
    r = _cli("run", TRIAGE, "--input", str(data))
    assert r.returncode == 1
    assert "0 decisions" in r.stderr


def test_from_dict_validation():
    """非映射/缺 labels 显式报错；未知键警告。"""
    import warnings
    from judgekit.engine import Task
    for bad, kw in ((None, "空"), ([], "列表")):
        try:
            Task.from_dict(bad)
            assert False, f"{kw} 应报错"
        except ValueError:
            pass
    try:
        Task.from_dict({"name": "t", "primitive": "classify"})
        assert False, "classify 无 labels 应报错"
    except ValueError:
        pass
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Task.from_dict({"name": "t", "primitive": "route", "labels": ["a"], "lables": ["拼错"]})
        assert any("lables" in str(x.message) for x in w)
