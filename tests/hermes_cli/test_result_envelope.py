import json

from hermes_cli._parser import build_top_level_parser
from hermes_cli import result_envelope as subject


def _result(**overrides):
    value = {
        "session_id": "session-001",
        "model": "deepseek-v4-flash:0731",
        "provider": "ollama-cloud",
        "input_tokens": 1234,
        "output_tokens": 234,
        "cache_read_tokens": 1000,
        "cache_write_tokens": 50,
        "reasoning_tokens": 20,
        "estimated_cost_usd": 0.0123,
        "cost_status": "estimated",
        "api_calls": 3,
        "completed": True,
        "failed": False,
        "partial": False,
        "interrupted": False,
        "turn_exit_reason": "text_response",
        "final_response": "model response must stay outside the envelope",
        "messages": [
            {"role": "user", "content": "test"},
            {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}]},
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": "done"},
        ],
    }
    value.update(overrides)
    return value


def _identity(monkeypatch):
    monkeypatch.setenv("PAPERCLIP_TASK_ID", "issue-001")
    monkeypatch.setenv("PAPERCLIP_RUN_ID", "run-001")
    monkeypatch.setenv("PAPERCLIP_AGENT_ID", "agent-001")
    monkeypatch.setattr(subject, "_active_profile_name", lambda: "tester")


def test_builds_exact_controlled_success_envelope(monkeypatch):
    _identity(monkeypatch)
    envelope = subject.build_result_envelope(
        result=_result(),
        session_id="session-fallback",
        model="model-fallback",
        provider="provider-fallback",
        reasoning_effort="low",
        started_at="2026-08-21T01:00:00Z",
    )

    assert envelope == {
        "schema_version": "1.0",
        "paperclip_issue_id": "issue-001",
        "paperclip_run_id": "run-001",
        "hermes_session_id": "session-001",
        "agent_id": "agent-001",
        "profile": "tester",
        "provider": "ollama-cloud",
        "model": "deepseek-v4-flash:0731",
        "reasoning_effort": "low",
        "input_tokens": 1234,
        "output_tokens": 234,
        "cache_read_tokens": 1000,
        "cache_write_tokens": 50,
        "reasoning_tokens": 20,
        "tool_calls": 2,
        "started_at": "2026-08-21T01:00:00Z",
        "finished_at": envelope["finished_at"],
        "status": "succeeded",
        "termination_reason": "text_response",
        "usage_source": "session_cumulative",
        "cost_currency": "USD",
        "estimated_cost": 0.0123,
        "cost_status": "estimated",
    }
    serialized = json.dumps(envelope)
    assert "model response must stay outside" not in serialized
    assert "messages" not in envelope
    assert "error" not in envelope


def test_missing_usage_and_unknown_cost_remain_null(monkeypatch):
    _identity(monkeypatch)
    envelope = subject.build_result_envelope(
        result=_result(
            input_tokens=None,
            output_tokens=None,
            cache_read_tokens=99,
            cache_write_tokens=10,
            reasoning_tokens=5,
            estimated_cost_usd=0,
            cost_status="unknown",
        ),
        session_id="session-001",
        model="deepseek-v4-flash:0731",
        provider="ollama-cloud",
        reasoning_effort=None,
        started_at="2026-08-21T01:00:00Z",
    )

    assert envelope["usage_source"] == "unknown"
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
    ):
        assert envelope[key] is None
    assert envelope["estimated_cost"] is None
    assert envelope["cost_currency"] is None
    assert envelope["cost_status"] == "unknown"


def test_status_overrides_cover_interrupt_and_startup_failure(monkeypatch):
    _identity(monkeypatch)
    interrupted = subject.build_result_envelope(
        result=None,
        session_id="session-001",
        model="model",
        provider="provider",
        reasoning_effort=None,
        started_at="2026-08-21T01:00:00Z",
        status_override="interrupted",
        termination_reason="keyboard_interrupt",
    )
    assert interrupted["status"] == "interrupted"
    assert interrupted["termination_reason"] == "keyboard_interrupt"
    assert interrupted["usage_source"] == "unknown"

    failed = subject.build_result_envelope(
        result=None,
        session_id=None,
        model="model",
        provider="provider",
        reasoning_effort=None,
        started_at="2026-08-21T01:00:00Z",
        status_override="failed",
        termination_reason="credentials_not_ready",
    )
    assert failed["status"] == "failed"
    assert failed["termination_reason"] == "credentials_not_ready"


def test_emitter_is_at_most_once_and_excludes_secret_and_response(monkeypatch, capsys):
    _identity(monkeypatch)
    synthetic_secret = "-".join(("fixture", "credential", "content"))
    emitter = subject.ResultEnvelopeEmitter(True, started_at="2026-08-21T01:00:00Z")

    first = emitter.emit(
        result=_result(final_response=f"do not emit {synthetic_secret}"),
        session_id="session-001",
        model="model",
        provider="provider",
        reasoning_effort="low",
    )
    second = emitter.emit(
        result=_result(),
        session_id="session-002",
        model="model",
        provider="provider",
        reasoning_effort="low",
    )

    captured = capsys.readouterr()
    lines = [line for line in captured.err.splitlines() if line.startswith(subject.RESULT_ENVELOPE_PREFIX)]
    assert first is not None
    assert second is None
    assert len(lines) == 1
    assert synthetic_secret not in captured.err
    assert "do not emit" not in captured.err


def test_chat_parser_exposes_opt_in_result_envelope_flag():
    parser, _, _ = build_top_level_parser()
    args = parser.parse_args(["chat", "--result-envelope", "-Q", "-q", "synthetic"])
    assert args.result_envelope is True
    assert args.quiet is True
    assert args.query == "synthetic"
