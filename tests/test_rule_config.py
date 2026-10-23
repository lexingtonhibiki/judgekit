"""Offline regressions for rule configuration mistakes in customized tasks."""
import json
import subprocess
import sys

import pytest

from judgekit import Task, run_task


@pytest.mark.parametrize("primitive", ["classify", "route"])
@pytest.mark.parametrize("rules, message", [
    ({"shiping": ["parcel"]}, "shiping.*labels"),
    ({"shipping": "parcel"}, "shipping.*list"),
    ({"shipping": {"parcel": True}}, "shipping.*list"),
    ({"shipping": [42]}, "shipping.*string"),
    ({"shipping": [""]}, "shipping.*non-empty"),
    (["shipping"], "fallback_rules.*mapping"),
])
def test_invalid_rules_rejected_at_load(primitive, rules, message):
    with pytest.raises(ValueError, match=message):
        Task.from_dict({"primitive": primitive, "labels": ["shipping"],
                        "fallback_rules": rules})


@pytest.mark.parametrize("labels", [["shipping", "refund"],
                                   {"shipping": "Parcels", "refund": "Returns"}])
@pytest.mark.parametrize("rules", [None, {}, {"shipping": []}])
def test_optional_and_disabled_rules_remain_valid(labels, rules):
    task = Task.from_dict({"primitive": "route", "labels": labels, "fallback_rules": rules})
    assert not run_task(task, {"text": "parcel"}, {}).ok


def test_valid_rules_keep_case_substring_tie_and_fallback_semantics():
    task = Task.from_dict({"primitive": "route", "labels": ["shipping", "refund"],
                          "provider": "missing", "fallback_rules": {
                              "refund": ["退货", "return"], "shipping": ["parcel"]}})
    assert run_task(task, {"text": "return parcel"}, {}).value == "refund"
    assert run_task(task, {"text": "退货"}, {}).value == "refund"
    assert run_task(task, {"text": "parcels"}, {}).value == "shipping"
    assert not run_task(task, {"text": "PARCEL"}, {}).ok
    assert run_task(task, {"text": "parcel"}, {}).provider == "rules-after-fail"


@pytest.mark.parametrize("command", ["judge", "run"])
@pytest.mark.parametrize("rules, diagnostic", [
    ("  shiping: [parcel]\n", "shiping"),
    ("  shipping: parcel\n", "list"),
])
def test_cli_rejects_invalid_rules_before_output_or_providers(tmp_path, command, rules, diagnostic):
    task = tmp_path / "custom.yaml"
    task.write_text("primitive: route\nlabels: [shipping]\nprovider: rules\nfallback_rules:\n" + rules,
                    encoding="utf-8")
    output = tmp_path / "decisions.jsonl"
    output.write_text("keep previous results\n", encoding="utf-8")
    args = [sys.executable, "-m", "judgekit", command, str(task),
            "--providers", str(tmp_path / "must-not-be-loaded.yaml")]
    if command == "judge":
        args += ["parcel"]
    else:
        args += ["--input", "-", "--out", str(output)]
    result = subprocess.run(args, input=json.dumps({"text": "parcel"}) + "\n",
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 2, result.stderr
    assert result.stdout == ""
    assert "fallback_rules" in result.stderr and diagnostic in result.stderr
    assert repr(str(task)) in result.stderr
    assert "Traceback" not in result.stderr
    assert "must-not-be-loaded" not in result.stderr
    assert output.read_text(encoding="utf-8") == "keep previous results\n"


def test_omitted_rules_remain_optional():
    task = Task.from_dict({"primitive": "route", "labels": ["shipping"]})
    assert task.fallback_rules == {}
    assert not run_task(task, {"text": "parcel"}, {}).ok


@pytest.mark.parametrize("primitive", ["score", "verify"])
@pytest.mark.parametrize("rules", [{"not-a-label": "scalar"}, ["ignored"]])
def test_non_rule_primitives_do_not_gain_rule_validation(primitive, rules):
    task = Task.from_dict({"primitive": primitive, "fallback_rules": rules})
    assert task.fallback_rules == rules
    assert not run_task(task, {"text": "scalar"}, {}).ok


@pytest.mark.parametrize("primitive", ["classify", "route"])
@pytest.mark.parametrize("labels", ["shipping", None, 1, True, 1.5, [True, False],
                                   ["shipping", 7], [""], {False: "No"}])
def test_invalid_label_candidates_rejected_at_load(primitive, labels):
    with pytest.raises(ValueError, match="labels"):
        Task.from_dict({"primitive": primitive, "labels": labels})


@pytest.mark.parametrize("command", ["judge", "run"])
@pytest.mark.parametrize("labels", ["shipping", "null", "7", "[yes, no]", "['']"])
def test_cli_rejects_invalid_labels_before_providers_or_output(tmp_path, command, labels):
    task = tmp_path / "invalid-labels.yaml"
    task.write_text("primitive: route\nlabels: " + labels + "\nprovider: rules\n",
                    encoding="utf-8")
    output = tmp_path / "decisions.jsonl"
    output.write_text("keep previous results\n", encoding="utf-8")
    args = [sys.executable, "-m", "judgekit", command, str(task),
            "--providers", str(tmp_path / "must-not-be-loaded.yaml")]
    if command == "judge":
        args += ["parcel"]
    else:
        args += ["--input", "-", "--out", str(output)]
    result = subprocess.run(args, input='{"text": "parcel"}\n', capture_output=True,
                            text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 2, result.stderr
    assert result.stdout == ""
    assert "labels" in result.stderr
    assert repr(str(task)) in result.stderr
    assert "Traceback" not in result.stderr
    assert "must-not-be-loaded" not in result.stderr
    assert output.read_text(encoding="utf-8") == "keep previous results\n"


@pytest.mark.parametrize("labels", [["yes", "no"], ("yes", "no"),
                                   {"yes": "Accept", "no": "Reject"}])
def test_string_label_candidates_preserve_order_and_rules(labels):
    task = Task.from_dict({"primitive": "route", "labels": labels,
                          "fallback_rules": {"yes": ["parcel"]}})
    assert task.labels == ["yes", "no"]
    assert run_task(task, {"text": "parcel"}, {}).value == "yes"


@pytest.mark.parametrize("primitive", ["score", "verify"])
def test_non_choice_tasks_keep_optional_labels(primitive):
    assert Task.from_dict({"primitive": primitive}).labels == []


@pytest.mark.parametrize("kind", ["generator", "set", "sequence"])
def test_python_iterable_labels_keep_existing_support(kind):
    class Sequence:
        def __getitem__(self, index):
            return ["yes", "no"][index]

    labels = {"generator": (item for item in ["yes", "no"]),
              "set": {"yes", "no"}, "sequence": Sequence()}[kind]
    task = Task.from_dict({"primitive": "classify", "labels": labels})
    assert set(task.labels) == {"yes", "no"}
    if kind != "set":
        assert task.labels == ["yes", "no"]


@pytest.mark.parametrize("stage", ["iter", "next"])
def test_user_label_iterator_errors_are_not_suppressed(stage):
    failure = RuntimeError("user iterator failure")

    class Broken:
        def __iter__(self):
            if stage == "iter":
                raise failure
            return self

        def __next__(self):
            raise failure

    with pytest.raises(RuntimeError) as exc:
        Task.from_dict({"primitive": "route", "labels": Broken()})
    assert exc.value is failure
