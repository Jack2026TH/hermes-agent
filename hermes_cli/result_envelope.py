"""Controlled machine-readable terminal result for Paperclip runners.

The envelope is opt-in and written to stderr so model-controlled stdout
cannot impersonate runtime telemetry. It deliberately excludes response text,
free-form exceptions, environment values, credentials, and filesystem paths.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
import sys
from typing import Any, Mapping, Optional


RESULT_ENVELOPE_PREFIX = "paperclip_result_envelope:"
RESULT_ENVELOPE_SCHEMA_VERSION = "1.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _nullable_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _counter(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _cost(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _active_profile_name() -> Optional[str]:
    try:
        from hermes_cli.profiles import get_active_profile_name

        return _nullable_text(get_active_profile_name())
    except Exception:
        home = _nullable_text(os.environ.get("HERMES_HOME"))
        if not home:
            return None
        normalized = home.rstrip("/\\")
        parent = os.path.basename(os.path.dirname(normalized))
        return os.path.basename(normalized) if parent == "profiles" else "default"


def _tool_call_count(messages: Any) -> Optional[int]:
    if not isinstance(messages, list):
        return None
    total = 0
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            total += len(tool_calls)
    return total


def _status(result: Mapping[str, Any], override: Optional[str]) -> str:
    if override:
        return override
    if result.get("interrupted"):
        return "interrupted"
    if result.get("failed"):
        return "failed"
    if result.get("partial"):
        return "partial"
    if result.get("completed") or result.get("final_response") is not None:
        return "succeeded"
    return "failed"


def build_result_envelope(
    *,
    result: Optional[Mapping[str, Any]],
    session_id: Any,
    model: Any,
    provider: Any,
    reasoning_effort: Any,
    started_at: Optional[str],
    status_override: Optional[str] = None,
    termination_reason: Optional[str] = None,
) -> dict[str, Any]:
    controlled = result if isinstance(result, Mapping) else {}
    counters = {
        "input_tokens": _counter(controlled.get("input_tokens")),
        "output_tokens": _counter(controlled.get("output_tokens")),
        "cache_read_tokens": _counter(controlled.get("cache_read_tokens")),
        "cache_write_tokens": _counter(controlled.get("cache_write_tokens")),
        "reasoning_tokens": _counter(controlled.get("reasoning_tokens")),
    }
    usage_known = counters["input_tokens"] is not None and counters["output_tokens"] is not None
    if not usage_known:
        counters = {key: None for key in counters}
    estimated_cost = _cost(controlled.get("estimated_cost_usd"))
    raw_cost_status = _nullable_text(controlled.get("cost_status"))
    if not raw_cost_status or raw_cost_status == "unknown":
        estimated_cost = None
        cost_status = "unknown"
    else:
        cost_status = raw_cost_status if estimated_cost is not None else "unknown"

    terminal_reason = termination_reason or _nullable_text(controlled.get("turn_exit_reason"))
    return {
        "schema_version": RESULT_ENVELOPE_SCHEMA_VERSION,
        "paperclip_issue_id": _nullable_text(os.environ.get("PAPERCLIP_TASK_ID")),
        "paperclip_run_id": _nullable_text(os.environ.get("PAPERCLIP_RUN_ID")),
        "hermes_session_id": _nullable_text(controlled.get("session_id")) or _nullable_text(session_id),
        "agent_id": _nullable_text(os.environ.get("PAPERCLIP_AGENT_ID")),
        "profile": _active_profile_name(),
        "provider": _nullable_text(controlled.get("provider")) or _nullable_text(provider),
        "model": _nullable_text(controlled.get("model")) or _nullable_text(model),
        "reasoning_effort": _nullable_text(reasoning_effort),
        **counters,
        "tool_calls": _tool_call_count(controlled.get("messages")),
        "started_at": started_at,
        "finished_at": utc_now(),
        "status": _status(controlled, status_override),
        "termination_reason": terminal_reason,
        "usage_source": "session_cumulative" if usage_known else "unknown",
        "cost_currency": "USD" if estimated_cost is not None else None,
        "estimated_cost": estimated_cost,
        "cost_status": cost_status,
    }


class ResultEnvelopeEmitter:
    """At-most-once stderr emitter used by the single-query finalizer."""

    def __init__(self, enabled: bool, *, started_at: Optional[str] = None) -> None:
        self.enabled = bool(enabled)
        self.started_at = started_at or (utc_now() if self.enabled else None)
        self.emitted = False

    def emit(
        self,
        *,
        result: Optional[Mapping[str, Any]],
        session_id: Any,
        model: Any,
        provider: Any,
        reasoning_effort: Any,
        status_override: Optional[str] = None,
        termination_reason: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        if not self.enabled or self.emitted:
            return None
        envelope = build_result_envelope(
            result=result,
            session_id=session_id,
            model=model,
            provider=provider,
            reasoning_effort=reasoning_effort,
            started_at=self.started_at,
            status_override=status_override,
            termination_reason=termination_reason,
        )
        payload = json.dumps(envelope, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        print(f"{RESULT_ENVELOPE_PREFIX}{payload}", file=sys.stderr, flush=True)
        self.emitted = True
        return envelope
