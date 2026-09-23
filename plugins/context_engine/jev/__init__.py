"""Jev-backed context engine for Hermes.

Uses TypeSafe System One for cheap typed decisions while preserving Hermes'
normal compressor and transcript semantics. The Jev pass is request-only:
persisted conversation history is never rewritten.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
from typing import Any, Dict, List, Mapping

from agent.context_compressor import ContextCompressor
from agent.redact import redact_sensitive_text
from .client import DEFAULT_JEV_MODEL, DEFAULT_JEV_URL, JevClient
_DROP_MARKER = "[JEV context selection: older non-critical tool output omitted for this request]"


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _tool_name(messages: List[Dict[str, Any]], index: int) -> str:
    msg = messages[index]
    call_id = str(msg.get("tool_call_id") or "")
    if not call_id:
        return "tool"
    for prior in reversed(messages[:index]):
        for call in prior.get("tool_calls") or []:
            if not isinstance(call, Mapping) or str(call.get("id") or "") != call_id:
                continue
            fn = call.get("function") or {}
            return str(fn.get("name") or "tool")
    return "tool"


class JevContextEngine(ContextCompressor):
    """Default Hermes compressor plus Jev per-turn System-1 decisions."""

    emit_automatic_compaction_status = False

    def __init__(self) -> None:
        super().__init__(model="", quiet_mode=True, config_context_length=200_000)
        self.jev_url = os.environ.get("TYPESAFE_JEV_URL", DEFAULT_JEV_URL).strip() or DEFAULT_JEV_URL
        self.jev_model = os.environ.get("TYPESAFE_JEV_MODEL", DEFAULT_JEV_MODEL).strip() or DEFAULT_JEV_MODEL
        self.jev_timeout_seconds = _env_float("TYPESAFE_JEV_TIMEOUT_SECONDS", 2.0, 0.1)
        self.jev_context_min_chars = _env_int("JEV_CONTEXT_MIN_CHARS", 8_000, 0)
        self.jev_context_candidate_chars = _env_int("JEV_CONTEXT_CANDIDATE_CHARS", 6_000, 256)
        self.jev_context_max_candidates = _env_int("JEV_CONTEXT_MAX_CANDIDATES", 20, 1)
        self.jev_protect_tail_messages = _env_int("JEV_CONTEXT_PROTECT_TAIL_MESSAGES", 8, 1)
        self.jev_critical_threshold = _env_float("JEV_CRITICAL_THRESHOLD", 0.5, 0.0)
        self.last_jev_metrics: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "jev"

    @property
    def _jev_token(self) -> str:
        return str(os.environ.get("TYPESAFE_API_KEY") or "").strip()

    def is_available(self) -> bool:
        """Report configuration availability without making a network call."""
        return bool(self._jev_token)

    def _post(self, state: Mapping[str, Any], questions: Mapping[str, Any]) -> Dict[str, Any]:
        return JevClient(
            token=self._jev_token,
            url=self.jev_url,
            model=self.jev_model,
            timeout_seconds=self.jev_timeout_seconds,
        ).decide(state, questions).payload

    def _candidate_indices(self, messages: List[Dict[str, Any]]) -> list[int]:
        tail_start = max(0, len(messages) - self.jev_protect_tail_messages)
        candidates: list[int] = []
        total_chars = 0
        for index, msg in enumerate(messages[:tail_start]):
            if msg.get("role") != "tool":
                continue
            content = msg.get("content")
            if not isinstance(content, str) or not content:
                continue
            total_chars += len(content)
            if len(content) >= 256:
                candidates.append(index)
        if total_chars < self.jev_context_min_chars:
            return []
        return candidates[-self.jev_context_max_candidates :]

    @staticmethod
    def _active_user_text(messages: List[Dict[str, Any]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                return msg["content"][-4000:]
        return ""

    def select_context(
        self,
        request_messages: List[Dict[str, Any]],
        *,
        conversation_messages: List[Dict[str, Any]] = None,
        incoming_message: Dict[str, Any] = None,
        budget_tokens: int = 0,
    ) -> List[Dict[str, Any]] | None:
        candidates = self._candidate_indices(request_messages)
        if not candidates:
            self.last_jev_metrics = {"attempted": False, "reason": "below_threshold"}
            return None

        state_candidates = []
        questions: dict[str, dict[str, Any]] = {}
        for index in candidates:
            content = str(request_messages[index].get("content") or "")
            state_candidates.append(
                {
                    "message_index": index,
                    "tool": _tool_name(request_messages, index),
                    "content": redact_sensitive_text(
                        content[: self.jev_context_candidate_chars],
                        force=True,
                        redact_url_credentials=True,
                    ),
                    "original_chars": len(content),
                }
            )
            questions[f"m{index}__relevance"] = {
                "type": "choice",
                "instructions": (
                    f"For candidates message_index={index}, decide whether the old tool result is "
                    "needed to correctly handle the active user task. Prefer DROP for stale, "
                    "superseded, repetitive, or reproducible output; KEEP for facts/results still "
                    "needed by the current task."
                ),
                "criteria": {
                    "KEEP": "Needed for the current task or to avoid repeating/contradicting work",
                    "DROP": "Not needed for the current task; stale, repeated, or reproducible",
                },
            }
            questions[f"m{index}__critical"] = {
                "type": "noul",
                "instructions": (
                    f"For candidates message_index={index}, would omitting this tool result risk "
                    "losing a decision, constraint, identifier, result, blocker, rollback fact, or "
                    "other critical state?"
                ),
                "criteria": {"true": "Critical fact could be lost", "false": "No critical fact"},
            }

        started = time.monotonic()
        try:
            parsed = self._post(
                {
                    "active_user_task": redact_sensitive_text(
                        self._active_user_text(request_messages),
                        force=True,
                        redact_url_credentials=True,
                    ),
                    "candidates": state_candidates,
                },
                questions,
            )
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, TypeError, RuntimeError) as exc:
            self.last_jev_metrics = {
                "attempted": True,
                "ok": False,
                "error": type(exc).__name__,
                "latency_ms": round((time.monotonic() - started) * 1000, 3),
            }
            return None

        answers = parsed["answers"]
        selected = [dict(msg) for msg in request_messages]
        dropped = 0
        reclaimed_chars = 0
        for index in candidates:
            rel = answers.get(f"m{index}__relevance")
            crit = answers.get(f"m{index}__critical")
            choice = str(rel.get("choice") or "KEEP").upper() if isinstance(rel, Mapping) else "KEEP"
            try:
                critical = float(crit.get("noul") or 0.0) >= self.jev_critical_threshold if isinstance(crit, Mapping) else True
            except (TypeError, ValueError):
                critical = True
            if choice != "DROP" or critical:
                continue
            old = str(selected[index].get("content") or "")
            selected[index]["content"] = _DROP_MARKER
            reclaimed_chars += max(0, len(old) - len(_DROP_MARKER))
            dropped += 1

        self.last_jev_metrics = {
            "attempted": True,
            "ok": True,
            "model": str(parsed.get("model") or self.jev_model),
            "latency_ms": round((time.monotonic() - started) * 1000, 3),
            "candidates": len(candidates),
            "dropped": dropped,
            "reclaimed_chars": reclaimed_chars,
        }
        return selected if dropped else None

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "jev_decide",
                "description": (
                    "Use TypeSafe Jev for a small typed System-1 decision. Suitable for Choice, "
                    "Score, or Noul questions; not for prose generation or complex reasoning."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "state": {"type": "object", "description": "Structured state to evaluate."},
                        "questions": {
                            "type": "object",
                            "description": "TypeSafe System One typed questions keyed by question id.",
                        },
                    },
                    "required": ["state", "questions"],
                    "additionalProperties": False,
                },
            }
        ]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        if name != "jev_decide":
            return json.dumps({"ok": False, "error": f"unknown_jev_tool:{name}"})
        state = args.get("state")
        questions = args.get("questions")
        if not isinstance(state, Mapping) or not isinstance(questions, Mapping) or not questions:
            return json.dumps({"ok": False, "error": "invalid_jev_request"})
        started = time.monotonic()
        try:
            parsed = self._post(state, questions)
            return json.dumps(
                {
                    "ok": True,
                    "model": parsed.get("model") or self.jev_model,
                    "answers": parsed["answers"],
                    "latency_ms": round((time.monotonic() - started) * 1000, 3),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        except Exception as exc:
            return json.dumps(
                {
                    "ok": False,
                    "error": f"jev_error:{type(exc).__name__}",
                    "latency_ms": round((time.monotonic() - started) * 1000, 3),
                },
                sort_keys=True,
            )

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status["jev"] = dict(self.last_jev_metrics)
        status["jev_configured"] = bool(self._jev_token)
        return status


def register(ctx: Any) -> None:
    ctx.register_context_engine(JevContextEngine())