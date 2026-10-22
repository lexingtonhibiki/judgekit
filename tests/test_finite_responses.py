"""Reject malformed numeric responses without turning them into perfect scores."""
import io
import json
import urllib.request

import pytest

from judgekit.engine import Task, run_task
from judgekit.providers.openai_compat import OpenAICompat, parse_decision


INVALID = [True, False, None, "NaN", "Infinity", "-Infinity", float('nan'), float('inf'), -float('inf'), [], {}]


@pytest.mark.parametrize('value', INVALID)
def test_invalid_score_is_not_a_successful_decision(value):
    with pytest.raises(ValueError, match='score-invalid'):
        parse_decision(Task('t', 'score'), json.dumps({'score': value}), 'offline', 0, 0)


@pytest.mark.parametrize('primitive,payload', [
    ('score', {'score': 0.4}), ('verify', {'verdict': True}),
    ('classify', {'label': 'yes'}), ('route', {'label': 'yes'}),
])
@pytest.mark.parametrize('value', INVALID)
def test_invalid_confidence_is_rejected_for_all_primitives(primitive, payload, value):
    with pytest.raises(ValueError, match='confidence-invalid'):
        parse_decision(Task('t', primitive, labels=['yes']),
                       json.dumps({**payload, 'confidence': value}), 'offline', 0, 0)


@pytest.mark.parametrize('raw', ['{"score": 1e999}', '{"score": 0.5, "confidence": -1e999}'])
def test_overflowing_json_number_is_rejected(raw):
    with pytest.raises(ValueError, match='-invalid'):
        parse_decision(Task('t', 'score'), raw, 'offline', 0, 0)


@pytest.mark.parametrize('value,expected', [(0, 0), (1, 1), (-2.0, 0), (2.0, 1), ('0.375', 0.375)])
def test_finite_number_compatibility(value, expected):
    dec = parse_decision(Task('t', 'score'), json.dumps({'score': value, 'confidence': value}), 'offline', 3, 0.1)
    assert dec.ok and dec.value == expected and dec.confidence == expected
    assert dec.provider == 'offline' and dec.latency_ms == 3 and dec.cost == 0.1


def test_omitted_confidence_preserves_default():
    assert parse_decision(Task('t', 'score'), '{"score": 0.2}', 'offline', 0, 0).confidence == 0.5


@pytest.mark.parametrize('fallback', [True, False])
def test_mock_provider_invalid_confidence_follows_existing_failure_policy(monkeypatch, fallback):
    response = {'choices': [{'message': {'content': '{"label":"yes","confidence":"NaN"}'}}]}
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(json.dumps(response).encode()))
    task = Task('t', 'classify', labels=['yes'], provider='offline', fallback_rules={'yes': ['match']})
    provider = OpenAICompat('offline', 'https://offline.invalid', 'stub')
    dec = run_task(task, {'text': 'match'}, {'offline': provider}, fallback=fallback)
    if fallback:
        assert dec.ok and dec.provider == 'rules-after-fail' and dec.value == 'yes'
        assert 'confidence-invalid' in dec.error and dec.error.endswith('; fallback-ok')
    else:
        assert not dec.ok and dec.value is None and dec.provider == 'offline'
        assert 'confidence-invalid' in dec.error


@pytest.mark.parametrize('kind', ['openai', 'go-chat', 'go-responses'])
@pytest.mark.parametrize('raw', ['{"score":"NaN"}', '{"score":true}'])
def test_provider_score_failure_is_not_reported_as_perfect_score(monkeypatch, kind, raw):
    from judgekit.providers.go_openai import GoChatProvider, GoResponsesProvider

    chat = {'choices': [{'message': {'content': raw}}]}
    if kind == 'openai':
        monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(json.dumps(chat).encode()))
        provider = OpenAICompat('offline', 'https://offline.invalid', 'stub')
    elif kind == 'go-chat':
        provider = GoChatProvider('offline', 'stub', api_key='test-only', session_id='test-only',
                                  transport=lambda *a: chat)
    else:
        response = {'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': raw}]}]}
        provider = GoResponsesProvider('offline', 'stub', api_key='test-only', session_id='test-only',
                                       transport=lambda *a: response)
    dec = run_task(Task('t', 'score', provider='offline'), {'text': 'input'}, {'offline': provider})
    assert not dec.ok and dec.value is None and dec.provider == 'offline'
    assert 'score-invalid' in dec.error
