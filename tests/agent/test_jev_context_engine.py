from __future__ import annotations

import copy
import json
import logging
import urllib.request
from collections.abc import Mapping
from typing import Any

import pytest

from agent.context_compressor import ContextCompressor
from agent.conversation_loop import _apply_context_engine_selection
import plugins.context_engine.jev as jev_module
from plugins.context_engine import discover_context_engines, load_context_engine
from plugins.context_engine.jev import JevClient, JevContextEngine, JevError
from plugins.context_engine.jev.client import _NoRedirectHandler


def _messages() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Investigate current production failure"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_old",
                    "type": "function",
                    "function": {"name": "logs.read", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_old", "content": "old-noise-" * 1_000},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_keep",
                    "type": "function",
                    "function": {"name": "repo.status.read", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_keep", "content": "critical-sha-" * 700},
        {"role": "assistant", "content": "working"},
        {"role": "user", "content": "Only the current failure matters"},
        {"role": "assistant", "content": "continuing"},
        {"role": "tool", "tool_call_id": "recent", "content": "recent-output-" * 500},
    ]


def _valid_response(questions: Mapping[str, Any]) -> dict[str, Any]:
    answers: dict[str, Any] = {}
    for question_id, question in questions.items():
        if question["type"] == "noul":
            noul = 0.05 if question_id.startswith("m3__") else 0.95
            answers[question_id] = {"type": "noul", "noul": noul}
            continue
        answers[question_id] = {
            "type": "choice",
            "choice": "DROP",
            "confidence": 0.99,
            "probabilities": {"KEEP": 0.01, "DROP": 0.99},
        }
    return {
        "model": "jev-test",
        "answers": answers,
        "usage": {"input_tokens": 42, "output_tokens": 6},
    }


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self.body = json.dumps(payload).encode("utf-8")
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]

    def close(self) -> None:
        self.closed = True


def test_shadow_selection_preserves_request_and_transcript(monkeypatch):
    monkeypatch.setattr(jev_module, "_PROTECTED_TAIL_MESSAGES", 4)
    monkeypatch.setattr(jev_module, "_MIN_TOTAL_CANDIDATE_CHARS", 1_000)
    client = JevClient(token="test-token", urlopen=lambda *_args, **_kwargs: None)
    engine = JevContextEngine(client=client)
    original = _messages()
    request = copy.deepcopy(original)
    conversation = copy.deepcopy(original)
    request_before = copy.deepcopy(request)
    conversation_before = copy.deepcopy(conversation)
    monkeypatch.setattr(
        engine, "_post", lambda state, questions: _valid_response(questions)
    )

    class Agent:
        context_compressor = engine
        session_id = "shadow-test"

    selected = _apply_context_engine_selection(
        Agent(),
        request,
        conversation,
        conversation[-1],
        logger=logging.getLogger(__name__),
    )

    assert selected is request
    assert request == request_before
    assert conversation == conversation_before
    assert engine.last_jev_metrics["mode"] == "shadow"
    assert engine.last_jev_metrics["would_drop_count"] == 1
    assert engine.last_jev_metrics["would_reclaim_chars"] == len(original[3]["content"])
    assert engine.last_jev_metrics["usage"] == {"input_tokens": 42, "output_tokens": 6}
    assert len(engine.last_jev_metrics["decisions"]) == 2

    observability = engine.get_observability_status()
    assert list(observability) == [
        "mode",
        "attempted",
        "ok",
        "model",
        "latency_ms",
        "candidates",
        "would_drop_count",
        "would_reclaim_chars",
    ]
    assert observability["mode"] == "shadow"
    assert observability["attempted"] is True
    assert observability["ok"] is True
    assert observability["model"] == "jev-test"
    assert observability["candidates"] == 2
    assert observability["would_drop_count"] == 1
    assert observability["would_reclaim_chars"] == len(original[3]["content"])
    assert "usage" not in observability
    assert "decisions" not in observability
    assert engine.get_status()["jev"] == observability


def test_shadow_engine_keeps_the_normal_compressor():
    engine = JevContextEngine(client=JevClient(token="test-token"))
    assert isinstance(engine, ContextCompressor)
    assert engine.compress.__func__ is ContextCompressor.compress
    engine.update_model(model="test-model", context_length=4_000)
    assert engine.context_length == 4_000
    assert engine.should_compress(1_000_000) is True
    assert engine.get_tool_schemas() == []


def test_below_threshold_and_missing_key_do_not_call_jev(monkeypatch):
    monkeypatch.setattr(jev_module, "_PROTECTED_TAIL_MESSAGES", 4)
    monkeypatch.setattr(jev_module, "_MIN_TOTAL_CANDIDATE_CHARS", 1_000)
    engine = JevContextEngine(client=JevClient(token=""))
    calls = []
    monkeypatch.setattr(engine, "_post", lambda *_args: calls.append(True))

    assert engine.select_context(_messages()[:5]) is None
    assert engine.last_jev_metrics["reason"] == "below_threshold"
    assert engine.select_context(_messages()) is None
    assert engine.last_jev_metrics["reason"] == "not_configured"
    assert calls == []


def test_network_failure_fails_open_and_does_not_mutate_messages(monkeypatch):
    monkeypatch.setattr(jev_module, "_PROTECTED_TAIL_MESSAGES", 4)
    monkeypatch.setattr(jev_module, "_MIN_TOTAL_CANDIDATE_CHARS", 1_000)
    original = _messages()
    before = copy.deepcopy(original)
    client = JevClient(
        token="test-token",
        urlopen=lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError()),
    )
    engine = JevContextEngine(client=client)

    assert engine.select_context(original) is None
    assert original == before
    assert engine.last_jev_metrics["ok"] is False
    assert engine.last_jev_metrics["error"] == "network_error"


def test_client_redacts_and_bounds_all_outbound_text():
    raw_secret = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
    payload_secret = "payload-secret-must-not-leave"
    captured: dict[str, Any] = {}

    def fake_open(request: urllib.request.Request, timeout: float):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = request.data
        decoded = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(_valid_response(decoded["questions"]))

    client = JevClient(token="test-token", urlopen=fake_open)
    result = client.decide(
        {
            "active_user_task": f"Check OPENAI_API_KEY={raw_secret}",
            "credentials": {"api_key": payload_secret},
            f"token={payload_secret}": "field-name-test",
            "candidates": [{"content": f"OPENAI_API_KEY={raw_secret}"}],
        },
        {
            "decision": {
                "type": "choice",
                "instructions": f"Route this request; token={payload_secret}",
                "criteria": {"KEEP": "retain", "DROP": "omit"},
            }
        },
    )

    body = captured["body"].decode("utf-8")
    assert captured["url"] == "https://api.typesafe.ai/v1/systemone"
    assert captured["headers"]["Authorization"] == "Bearer test-token"
    assert raw_secret not in body
    assert payload_secret not in body
    assert "[redacted]" in body or "***" in body
    assert result.payload["answers"]["decision"]["choice"] == "DROP"


def test_oversized_text_is_omitted_before_egress():
    captured: dict[str, bytes] = {}

    def fake_open(request: urllib.request.Request, timeout: float):
        captured["body"] = request.data
        decoded = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(_valid_response(decoded["questions"]))

    client = JevClient(token="test-token", urlopen=fake_open)
    client.decide(
        {"active_user_task": "x" * 9_000},
        {"safe": {"type": "noul", "instructions": "is there a risk?"}},
    )
    body = captured["body"].decode("utf-8")
    assert "x" * 100 not in body
    assert "omitted: oversized text" in body


def test_oversized_serialized_request_is_rejected_before_network():
    calls = []
    client = JevClient(
        token="test-token",
        urlopen=lambda *_args, **_kwargs: calls.append(True),
    )

    with pytest.raises(JevError) as exc_info:
        client.decide(
            {"items": [{"text": "x" * 8_000} for _ in range(8)]},
            {"risk": {"type": "noul", "instructions": "Is this risky?"}},
        )

    assert exc_info.value.code == "request_too_large"
    assert calls == []


@pytest.mark.parametrize(
    "response",
    [
        {"model": "jev", "answers": {"choice": {"type": "choice", "choice": "DROP"}}},
        {
            "model": "jev",
            "answers": {
                "choice": {
                    "type": "choice",
                    "choice": "UNKNOWN",
                    "confidence": 0.8,
                    "probabilities": {"KEEP": 0.1, "DROP": 0.9},
                }
            },
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        {
            "model": "jev",
            "answers": {
                "choice": {
                    "type": "choice",
                    "choice": "DROP",
                    "confidence": 0.9,
                    "probabilities": {"KEEP": 0.1, "DROP": 0.9},
                }
            },
            "usage": {"input_tokens": -1, "output_tokens": 1},
        },
    ],
)
def test_client_rejects_incomplete_or_invalid_typed_responses(response):
    client = JevClient(
        token="test-token",
        urlopen=lambda *_args, **_kwargs: _FakeResponse(response),
    )

    with pytest.raises(JevError) as exc_info:
        client.decide(
            {"task": "diagnose"},
            {
                "choice": {
                    "type": "choice",
                    "criteria": {"KEEP": "keep", "DROP": "drop"},
                }
            },
        )
    assert exc_info.value.code == "invalid_response"


def test_client_fails_closed_without_key_and_rejects_redirects():
    no_key_calls = []
    client = JevClient(
        token="", urlopen=lambda *_args, **_kwargs: no_key_calls.append(True)
    )
    with pytest.raises(JevError) as exc_info:
        client.decide({"task": "diagnose"}, {"risk": {"type": "noul"}})
    assert exc_info.value.code == "missing_api_key"
    assert no_key_calls == []

    assert (
        _NoRedirectHandler().redirect_request(
            None, None, 307, "redirect", {}, "https://evil.invalid"
        )
        is None
    )


def test_discovery_and_load_are_network_free_without_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("network must not be opened")
        ),
    )
    rows = discover_context_engines()
    jev_rows = [row for row in rows if row[0] == "jev"]

    assert len(jev_rows) == 1
    assert jev_rows[0][2] is False
    engine = load_context_engine("jev")
    assert isinstance(engine, JevContextEngine)
    assert engine.is_available() is False
    assert engine.get_tool_schemas() == []


def test_shadow_failure_has_no_raw_error_or_prompt_data_in_status(monkeypatch):
    monkeypatch.setattr(jev_module, "_PROTECTED_TAIL_MESSAGES", 4)
    monkeypatch.setattr(jev_module, "_MIN_TOTAL_CANDIDATE_CHARS", 1_000)
    secret = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
    engine = JevContextEngine(client=JevClient(token="test-token"))
    engine._post = lambda *_args: (_ for _ in ()).throw(
        RuntimeError(f"failed for {secret}")
    )

    engine.select_context(_messages())

    status = json.dumps(engine.get_status())
    assert secret not in status
    assert "failed for" not in status
    assert engine.last_jev_metrics["mode"] == "shadow"
