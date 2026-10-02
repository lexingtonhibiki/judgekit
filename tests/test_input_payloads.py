"""Offline checks at the HTTP boundary: recipes send content, metadata stays local."""
import importlib
import io
import json
import urllib.request

import pytest

from judgekit.engine import Task, run_task
from judgekit.providers.nanojev import NanoJev
from judgekit.providers.openai_compat import OpenAICompat
from judgekit.providers.typesafe import TypeSafe


def test_learn_next_sends_candidate_and_mastery(monkeypatch):
    prompts = []

    def respond(req, timeout=None):
        prompts.append(json.loads(req.data)["messages"][1]["content"])
        return io.BytesIO(json.dumps({"choices": [{"message": {
            "content": '{"score": 0.8, "confidence": 0.9}'
        }}]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", respond)
    recipe = importlib.import_module("recipes.learn_next.next")
    kg = {"course": "Math-Course", "nodes": [{"id": "a", "name": "Algebra-Node"}]}
    provider = OpenAICompat("offline", "https://offline.invalid", "stub")
    ranked, _ = recipe.judge_rescore(
        kg, {"a": 0.2}, "pass_exam",
        [{"node": "Algebra-Node", "score": 0.4, "why": "needs practice"}],
        {"offline": provider}, "offline",
    )
    assert ranked[0]["ok"]
    assert "Math-Course" in prompts[0] and "Algebra-Node" in prompts[0]
    assert '"Algebra-Node": 0.2' in prompts[0]


def test_resume_lens_sends_resume_and_survey(monkeypatch):
    prompts = []

    def respond(req, timeout=None):
        prompts.append(json.loads(req.data)["messages"][1]["content"])
        answer = '{"score": 0.8, "confidence": 0.9}' if len(prompts) == 1 else \
            '{"verdict": false, "confidence": 0.9}'
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": answer}}]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", respond)
    recipe = importlib.import_module("recipes.resume_lens.match")
    provider = OpenAICompat("offline", "https://offline.invalid", "stub")
    ranked, _ = recipe.judge_rescore(
        {"skills": ["python-sentinel"]},
        [{"company": "Example", "title": "Engineer", "extras": "weekend-sentinel"}],
        {"overtime_tolerance": 1},
        [{"jd": "Example·Engineer", "fit": 0.3, "final": 0.3, "ot_conflict": False}],
        {"offline": provider}, "offline",
    )
    assert ranked[0]["judge_fit"] == 0.8 and ranked[0]["judge_conflict"] is False
    assert "python-sentinel" in prompts[0] and "Engineer" in prompts[0]
    assert '"overtime_tolerance": 1' in prompts[0]
    assert "weekend-sentinel" in prompts[1] and '"问卷加班容忍度": 1' in prompts[1]


@pytest.mark.parametrize("kind", ["typesafe", "nanojev"])
def test_native_runtime_sends_only_declared_field(monkeypatch, kind):
    states = []

    def respond(req, timeout=None):
        body = json.loads(req.data)
        state = body["state"] if kind == "typesafe" else body["states"][0]["state"]
        states.append(json.loads(state))
        answer = {"type": "choice", "choice": "refund", "confidence": 0.9,
                  "probabilities": {"refund": 1.0}}
        response = {"answers": {"d": answer}} if kind == "typesafe" else \
            {"states": [{"answers": {"d": answer}}]}
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(urllib.request, "urlopen", respond)
    provider = TypeSafe("offline") if kind == "typesafe" else NanoJev("offline")
    task = Task(name="routing", primitive="route", labels=["refund"],
                provider="offline", input_field="body")
    row = {"body": "please refund", "text": "unselected", "label": "gold-sentinel"}
    decision = run_task(task, row, {"offline": provider})
    assert decision.ok
    assert states == [{"body": "please refund"}]
    assert row["label"] == "gold-sentinel"
