"""Focused tests for the Jev context-engine plugin."""

from __future__ import annotations

import copy
import json
from typing import Any
from unittest.mock import MagicMock

from agent.conversation_loop import _apply_context_engine_selection
from plugins.context_engine import discover_context_engines, load_context_engine
from plugins.context_engine.jev import JevClient, JevContextEngine


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._body = json.dumps(payload).encode("utf-8")
        self.closed = False

    def read(self) -> bytes:
        return self._body

    def close(self) -> None:
        self.closed = True


def _questions() -> dict[str, dict[str, Any]]:
    return {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Payments and refunds",
                "technical": "Bugs and outages",
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated"],
        },
        "urgent": {
            "type": "noul",
            "instructions": "Does this need urgent handling?",
        },
    }


def _response() -> dict[str, Any]:
    return {
        "model": "jev-latest",
        "answers": {
            "department": {
                "type": "choice",
                "choice": "technical",
                "probabilities": {"billing": 0.2, "technical": 0.8},
                "confidence": 0.8,
            },
            "frustration": {
                "type": "score",
                "score": 0.8,
                "legend": {"0": "Calm", "1": "Frustrated"},
                "probabilities": {"0": 0.2, "1": 0.8},
                "confidence": 0.8,
            },
            "urgent": {"type": "noul", "noul": 0.9},
        },
        "usage": {"input_tokens": 20, "output_tokens": 10},
    }


def test_discovery_and_load_are_network_free_without_api_key(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("JEV_CONTEXT_MAX_MESSAGES", raising=False)
    network_calls = []

    def fail_if_called(*args, **kwargs):
        network_calls.append((args, kwargs))
        raise AssertionError("discovery must not perform network I/O")

    monkeypatch.setattr("urllib.request.urlopen", fail_if_called)

    discovered = discover_context_engines()
    jev_entries = [entry for entry in discovered if entry[0] == "jev"]
    assert len(jev_entries) == 1
    assert "TypeSafe" in jev_entries[0][1]
    assert jev_entries[0][2] is False

    loaded = load_context_engine("jev")
    assert isinstance(loaded, JevContextEngine)
    assert loaded.is_available() is False
    assert network_calls == []


def test_request_only_selection_keeps_system_and_latest_messages(monkeypatch):
    monkeypatch.setenv("JEV_CONTEXT_MAX_MESSAGES", "4")
    engine = JevContextEngine(api_key="test-key")
    request_messages = [
        {"role": "system", "content": [{"type": "text", "text": "system"}]},
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": {"text": "recent user"}},
        {"role": "assistant", "content": [{"text": "recent answer"}]},
    ]
    original = copy.deepcopy(request_messages)

    selected = engine.select_context(
        request_messages,
        conversation_messages=request_messages,
        incoming_message=request_messages[-2],
        budget_tokens=1000,
    )

    assert selected is not None
    assert [message["role"] for message in selected] == [
        "system",
        "assistant",
        "user",
        "assistant",
    ]
    assert selected[0]["content"] == original[0]["content"]
    assert selected[-2]["content"] == original[-2]["content"]
    selected[-1]["content"][0]["text"] = "changed in request copy"
    assert request_messages == original


def test_host_selection_does_not_mutate_persisted_transcript(monkeypatch):
    monkeypatch.setenv("JEV_CONTEXT_MAX_MESSAGES", "3")
    engine = JevContextEngine(api_key="test-key")
    agent = MagicMock()
    agent.session_id = "test-session"
    agent.context_compressor = engine
    history = [
        {"role": "user", "content": {"text": "old"}},
        {"role": "assistant", "content": [{"text": "old answer"}]},
        {"role": "user", "content": {"text": "new"}},
    ]
    request = [
        {"role": "system", "content": [{"text": "system"}]},
        *history,
    ]
    history_snapshot = copy.deepcopy(history)
    request_snapshot = copy.deepcopy(request)

    selected = _apply_context_engine_selection(
        agent,
        request,
        history,
        history[-1],
        logger=MagicMock(),
    )

    assert selected is not request
    assert selected == [
        request[0],
        request[-2],
        request[-1],
    ]
    assert history == history_snapshot
    assert request == request_snapshot


def test_typed_request_and_choice_score_noul_response_use_official_shape():
    received: dict[str, Any] = {}

    def fake_urlopen(request, timeout):
        received["url"] = request.full_url
        received["headers"] = dict(request.header_items())
        received["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(_response())

    client = JevClient(api_key="test-key", urlopen=fake_urlopen)
    result = client.decide(
        state={"message": "The integration is failing"},
        questions=_questions(),
    )

    assert received["url"] == "https://api.typesafe.ai/v1/systemone"
    assert received["headers"]["Authorization"] == "Bearer test-key"
    assert received["body"]["model"] == "jev-latest"
    assert received["body"]["questions"] == _questions()
    assert result["answers"]["department"]["choice"] == "technical"
    assert result["answers"]["frustration"]["score"] == 0.8
    assert result["answers"]["urgent"]["noul"] == 0.9


def test_tool_call_returns_structured_success_and_unknown_tool_error():
    def fake_urlopen(request, timeout):
        return _FakeResponse(_response())

    engine = JevContextEngine(
        client=JevClient(api_key="test-key", urlopen=fake_urlopen),
    )
    success = json.loads(
        engine.handle_tool_call(
            "jev_decide",
            {"state": "hello", "questions": _questions()},
        )
    )
    unknown = json.loads(engine.handle_tool_call("not_jev", {}))

    assert success["ok"] is True
    assert success["result"]["answers"]["urgent"]["noul"] == 0.9
    assert unknown == {
        "ok": False,
        "error": {"code": "unknown_tool", "message": "Unknown Jev tool: not_jev"},
    }


def test_missing_key_and_invalid_input_fail_closed_without_network(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    network_calls = []

    def fail_if_called(*args, **kwargs):
        network_calls.append((args, kwargs))
        raise AssertionError("invalid calls must stop before the network")

    engine = JevContextEngine(
        client=JevClient(api_key="", urlopen=fail_if_called),
    )
    missing_key = json.loads(
        engine.handle_tool_call(
            "jev_decide",
            {"state": "hello", "questions": _questions()},
        )
    )
    invalid_input = json.loads(
        engine.handle_tool_call(
            "jev_decide",
            {"state": "hello", "questions": {}},
        )
    )

    assert missing_key["error"]["code"] == "missing_api_key"
    assert invalid_input["error"]["code"] == "invalid_request"
    assert network_calls == []


def test_invalid_response_fails_closed():
    def fake_urlopen(request, timeout):
        return _FakeResponse(
            {
                "model": "jev-latest",
                "answers": {
                    "urgent": {"type": "noul", "noul": 2.0},
                },
            }
        )

    engine = JevContextEngine(
        client=JevClient(api_key="test-key", urlopen=fake_urlopen),
    )
    result = json.loads(
        engine.handle_tool_call(
            "jev_decide",
            {
                "state": "hello",
                "questions": {
                    "urgent": {
                        "type": "noul",
                        "instructions": "Is this urgent?",
                    }
                },
            },
        )
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_response"
