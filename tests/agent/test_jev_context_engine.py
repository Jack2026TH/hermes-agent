from __future__ import annotations

import json

from plugins.context_engine import discover_context_engines, load_context_engine
from plugins.context_engine.jev import JevContextEngine


def _messages():
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Investigate current production failure"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_old", "type": "function", "function": {"name": "logs.read", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_old", "content": "old-noise-" * 1000},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_keep", "type": "function", "function": {"name": "repo.status.read", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_keep", "content": "critical-sha-" * 700},
        {"role": "assistant", "content": "working"},
        {"role": "user", "content": "Only the current failure matters"},
        {"role": "assistant", "content": "continuing"},
        {"role": "tool", "tool_call_id": "recent", "content": "recent-output-" * 500},
    ]


def test_select_context_is_request_only_and_preserves_critical(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setenv("JEV_CONTEXT_MIN_CHARS", "1000")
    monkeypatch.setenv("JEV_CONTEXT_PROTECT_TAIL_MESSAGES", "4")
    engine = JevContextEngine()
    original = _messages()

    def fake_post(state, questions):
        assert state["active_user_task"] == "Only the current failure matters"
        return {
            "model": "jev-test",
            "answers": {
                "m3__relevance": {"choice": "DROP", "confidence": 0.99},
                "m3__critical": {"noul": 0.05},
                "m5__relevance": {"choice": "DROP", "confidence": 0.99},
                "m5__critical": {"noul": 0.95},
            },
        }

    engine._post = fake_post
    selected = engine.select_context(original)

    assert original[3]["content"].startswith("old-noise-")
    assert selected[3]["content"].startswith("[JEV context selection:")
    assert selected[5]["content"].startswith("critical-sha-")
    assert selected[9]["content"].startswith("recent-output-")
    assert engine.last_jev_metrics["dropped"] == 1
    assert engine.last_jev_metrics["reclaimed_chars"] > 9000


def test_select_context_fails_open_to_original_model_path(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setenv("JEV_CONTEXT_MIN_CHARS", "1000")
    monkeypatch.setenv("JEV_CONTEXT_PROTECT_TAIL_MESSAGES", "4")
    engine = JevContextEngine()

    def broken(*args, **kwargs):
        raise TimeoutError("slow")

    engine._post = broken
    assert engine.select_context(_messages()) is None
    assert engine.last_jev_metrics["ok"] is False


def test_jev_decide_tool(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    engine = JevContextEngine()
    engine._post = lambda state, questions: {
        "model": "jev-test",
        "answers": {"route": {"choice": "logs.read", "confidence": 0.97}},
    }

    raw = engine.handle_tool_call(
        "jev_decide",
        {
            "state": {"task": "diagnose"},
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Choose the next diagnostic capability",
                    "criteria": {"logs.read": "logs", "repo.status.read": "repo"},
                }
            },
        },
    )
    parsed = json.loads(raw)
    assert parsed["ok"] is True
    assert parsed["answers"]["route"]["choice"] == "logs.read"


def test_no_jev_call_below_threshold(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setenv("JEV_CONTEXT_MIN_CHARS", "1000000")
    engine = JevContextEngine()
    assert engine.select_context(_messages()) is None
    assert engine.last_jev_metrics == {"attempted": False, "reason": "below_threshold"}

def test_select_context_redacts_secrets_before_jev(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setenv("JEV_CONTEXT_MIN_CHARS", "100")
    monkeypatch.setenv("JEV_CONTEXT_PROTECT_TAIL_MESSAGES", "2")
    engine = JevContextEngine()
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Check OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz1234567890"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_secret", "type": "function", "function": {"name": "env.read", "arguments": "{}"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_secret",
            "content": ("OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz1234567890\\n" * 20),
        },
        {"role": "user", "content": "Current task only"},
        {"role": "assistant", "content": "working"},
    ]

    seen = {}

    def fake_post(state, questions):
        seen["state"] = state
        return {
            "model": "jev-test",
            "answers": {
                "m3__relevance": {"choice": "KEEP", "confidence": 0.99},
                "m3__critical": {"noul": 0.99},
            },
        }

    engine._post = fake_post
    engine.select_context(messages)

    serialized = json.dumps(seen["state"], ensure_ascii=False)
    assert "sk-abcdefghijklmnopqrstuvwxyz1234567890" not in serialized
    assert "OPENAI_API_KEY=***" in serialized


def test_discovery_and_load_are_network_free_without_token(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_JEV_URL", raising=False)
    rows = discover_context_engines()
    jev_rows = [row for row in rows if row[0] == "jev"]

    assert len(jev_rows) == 1
    assert jev_rows[0][2] is False
    engine = load_context_engine("jev")
    assert isinstance(engine, JevContextEngine)
    assert engine.is_available() is False


def test_jev_decide_fails_closed_without_token(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    engine = JevContextEngine()
    raw = engine.handle_tool_call(
        "jev_decide",
        {
            "state": {"task": "diagnose"},
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Choose the next diagnostic capability",
                    "criteria": {"logs.read": "logs", "repo.status.read": "repo"},
                }
            },
        },
    )
    parsed = json.loads(raw)
    assert parsed["ok"] is False
    assert parsed["error"] == "jev_error:RuntimeError"
