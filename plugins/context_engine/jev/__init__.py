"""Shadow-only Jev context evaluation for Hermes long-running sessions.

Jev evaluates whether older tool output could be omitted, but this first-stage
integration never changes the messages sent to Hermes' configured provider.
Normal ContextCompressor behavior remains active.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from agent.context_compressor import ContextCompressor
from agent.context_engine import (
    CONTEXT_ENGINE_OBSERVABILITY_FIELDS,
    sanitize_context_engine_observability,
)

from .client import DEFAULT_JEV_MODEL, JevClient, JevError

_MIN_TOTAL_CANDIDATE_CHARS = 8_000
_MIN_CANDIDATE_CHARS = 256
_MAX_CANDIDATES = 8
_MAX_CANDIDATE_CHARS = 1_500
_PROTECTED_TAIL_MESSAGES = 8
_CRITICAL_PROBABILITY_THRESHOLD = 0.5
_ACTIVE_TASK_MAX_CHARS = 4_000


def _tool_name(messages: list[dict[str, Any]], index: int) -> str:
    message = messages[index]
    call_id = str(message.get("tool_call_id") or "")
    if not call_id:
        return "tool"
    for prior in reversed(messages[:index]):
        for call in prior.get("tool_calls") or []:
            if not isinstance(call, Mapping) or str(call.get("id") or "") != call_id:
                continue
            function = call.get("function") or {}
            return str(function.get("name") or "tool")
    return "tool"


class JevContextEngine(ContextCompressor):
    """Keep Hermes compression and collect Jev's counterfactual decisions."""

    def __init__(self, *, client: JevClient | None = None) -> None:
        # The host sets the active model/context length through update_model().
        # Leave context length unresolved here; a fabricated value would alter
        # the existing compressor's threshold and summary budget.
        super().__init__(model="", quiet_mode=True)
        self._client = client or JevClient()
        self.last_jev_metrics: dict[str, Any] = {
            "mode": "shadow",
            "attempted": False,
            "reason": "not_run",
        }

    def configure_host_compression(self, **compressor_kwargs: Any) -> None:
        """Use the same compression policy as the host's built-in compressor."""
        client = self._client
        ContextCompressor.__init__(self, **compressor_kwargs)
        self._client = client
        self.last_jev_metrics = {
            "mode": "shadow",
            "attempted": False,
            "reason": "not_run",
        }

    @property
    def name(self) -> str:
        return "jev"

    def is_available(self) -> bool:
        """Report key availability without performing network I/O."""
        return self._client.is_configured

    def _candidate_indices(self, messages: list[dict[str, Any]]) -> list[int]:
        tail_start = max(0, len(messages) - _PROTECTED_TAIL_MESSAGES)
        candidates: list[int] = []
        total_chars = 0
        for index, message in enumerate(messages[:tail_start]):
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content:
                continue
            total_chars += len(content)
            if len(content) >= _MIN_CANDIDATE_CHARS:
                candidates.append(index)
        if total_chars < _MIN_TOTAL_CANDIDATE_CHARS:
            return []
        return candidates[-_MAX_CANDIDATES:]

    @staticmethod
    def _active_user_text(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user" and isinstance(
                message.get("content"), str
            ):
                return message["content"][-_ACTIVE_TASK_MAX_CHARS:]
        return ""

    def _post(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._client.decide(state, questions).payload

    def select_context(
        self,
        request_messages: list[dict[str, Any]],
        *,
        conversation_messages: list[dict[str, Any]] | None = None,
        incoming_message: dict[str, Any] | None = None,
        budget_tokens: int = 0,
    ) -> list[dict[str, Any]] | None:
        """Evaluate possible pruning and always preserve the original request."""
        candidates = self._candidate_indices(request_messages)
        if not candidates:
            self.last_jev_metrics = {
                "mode": "shadow",
                "attempted": False,
                "reason": "below_threshold",
            }
            return None
        if not self.is_available():
            self.last_jev_metrics = {
                "mode": "shadow",
                "attempted": False,
                "reason": "not_configured",
                "candidates": len(candidates),
            }
            return None

        state_candidates = []
        questions: dict[str, dict[str, Any]] = {}
        for index in candidates:
            content = request_messages[index]["content"]
            state_candidates.append({
                "message_index": index,
                "tool": _tool_name(request_messages, index),
                # The client performs a second, recursive redaction pass and
                # rejects oversized or unsupported payloads before egress.
                "content": content[:_MAX_CANDIDATE_CHARS],
                "original_chars": len(content),
            })
            questions[f"m{index}__relevance"] = {
                "type": "choice",
                "instructions": (
                    "Would this older tool result be needed to correctly handle the active user task? "
                    "Choose KEEP for facts or results still needed; choose DROP only for stale, "
                    "superseded, repetitive, or reproducible output."
                ),
                "criteria": {
                    "KEEP": "Needed for the current task or to avoid repeating or contradicting work",
                    "DROP": "Not needed for the current task; stale, repeated, or reproducible",
                },
            }
            questions[f"m{index}__critical"] = {
                "type": "noul",
                "instructions": (
                    "Would omitting this result risk losing a decision, constraint, identifier, "
                    "result, blocker, rollback fact, or other critical state?"
                ),
                "criteria": {
                    "true": "A critical fact could be lost",
                    "false": "No critical fact is at risk",
                },
            }

        started = time.monotonic()
        try:
            parsed = self._post(
                {
                    "active_user_task": self._active_user_text(request_messages),
                    "candidates": state_candidates,
                },
                questions,
            )
        except JevError as exc:
            self.last_jev_metrics = {
                "mode": "shadow",
                "attempted": True,
                "ok": False,
                "error": exc.code,
                "latency_ms": round((time.monotonic() - started) * 1000, 3),
                "candidates": len(candidates),
            }
            return None
        except Exception as exc:
            # Do not retain/log exception text: HTTP and provider exceptions may
            # contain request details. The host's own hook is also fail-open.
            self.last_jev_metrics = {
                "mode": "shadow",
                "attempted": True,
                "ok": False,
                "error": type(exc).__name__,
                "latency_ms": round((time.monotonic() - started) * 1000, 3),
                "candidates": len(candidates),
            }
            return None

        answers = parsed["answers"]
        decisions = []
        would_drop_count = 0
        would_reclaim_chars = 0
        for index in candidates:
            relevance = answers[f"m{index}__relevance"]
            critical = answers[f"m{index}__critical"]["noul"]
            would_drop = (
                relevance["choice"] == "DROP"
                and critical < _CRITICAL_PROBABILITY_THRESHOLD
            )
            original_chars = len(request_messages[index]["content"])
            decisions.append({
                "message_index": index,
                "relevance": relevance["choice"],
                "critical_probability": critical,
                "would_drop": would_drop,
            })
            if would_drop:
                would_drop_count += 1
                would_reclaim_chars += original_chars

        self.last_jev_metrics = {
            "mode": "shadow",
            "attempted": True,
            "ok": True,
            "model": parsed["model"],
            "latency_ms": round((time.monotonic() - started) * 1000, 3),
            "candidates": len(candidates),
            "would_drop_count": would_drop_count,
            "would_reclaim_chars": would_reclaim_chars,
            "usage": parsed["usage"],
            "decisions": decisions,
        }

        # Critical shadow-mode invariant: this hook only observes. It never
        # replaces or mutates request_messages, conversation_messages, or the
        # persisted transcript. Promotion to active pruning is a separate gate.
        return None

    def get_status(self) -> dict[str, Any]:
        status = super().get_status()
        status["jev"] = self.get_observability_status()
        return status

    def get_observability_status(self) -> dict[str, Any]:
        """Return the content-free Jev metrics safe for gateway status output."""
        defaults: dict[str, Any] = {
            "mode": "shadow",
            "attempted": False,
            "ok": None,
            "model": None,
            "latency_ms": None,
            "candidates": 0,
            "would_drop_count": 0,
            "would_reclaim_chars": 0,
        }
        metrics = {
            field: self.last_jev_metrics.get(field, default)
            for field, default in defaults.items()
        }
        # Keep the field order stable for machine-readable gateway status and
        # guard the plugin boundary even though the gateway repeats the same
        # validation before rendering.
        ordered = {
            field: metrics[field]
            for field in CONTEXT_ENGINE_OBSERVABILITY_FIELDS
        }
        return sanitize_context_engine_observability(ordered)

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self.last_jev_metrics = {
            "mode": "shadow",
            "attempted": False,
            "reason": "not_run",
        }


def register(ctx: Any) -> None:
    """Register Jev through Hermes' context-engine discovery protocol."""
    ctx.register_context_engine(JevContextEngine())


__all__ = ["DEFAULT_JEV_MODEL", "JevClient", "JevContextEngine", "JevError", "register"]
