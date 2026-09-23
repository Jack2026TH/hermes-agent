"""Jev context-engine plugin for Hermes Agent.

The plugin has two deliberately separate responsibilities:

* select_context is a deterministic, request-only selector. It can limit
  the messages sent on one provider request when explicitly enabled, without
  changing the persisted transcript.
* jev_decide is a typed TypeSafe evaluation tool for Choice, Score, and
  Noul questions. It is lazy and fail-closed when no API key is configured.

Importing or discovering this module never performs network I/O.
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine


DEFAULT_API_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
_API_KEY_ENV_NAMES = ("JEV_API_KEY", "TYPESAFE_API_KEY")
_QUESTION_TYPES = {"choice", "score", "noul"}
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 529}


class JevError(RuntimeError):
    """A safe, structured error that never contains credentials or response bodies."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def _env_value(name: str) -> str:
    return os.environ.get(name, "").strip()


def _configured_api_key() -> str:
    for name in _API_KEY_ENV_NAMES:
        value = _env_value(name)
        if value:
            return value
    return ""


def _bounded_int(value: Any, default: int, *, minimum: int = 0, maximum: int = 100000) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _is_json_value(value: Any, *, allow_none: bool = False) -> bool:
    if value is None:
        return allow_none
    return isinstance(value, (str, dict, list))


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _validate_request(payload: Mapping[str, Any]) -> None:
    state = payload.get("state")
    model = payload.get("model")
    questions = payload.get("questions")

    if not _is_json_value(state):
        raise JevError("invalid_request", "state must be a string, object, or array.")
    if not isinstance(model, str) or not model.strip():
        raise JevError("invalid_request", "model must be a non-empty string.")
    if not isinstance(questions, Mapping) or not questions:
        raise JevError("invalid_request", "questions must be a non-empty object.")

    for question_id, question in questions.items():
        if not isinstance(question_id, str) or not question_id.strip():
            raise JevError("invalid_request", "question ids must be non-empty strings.")
        if not isinstance(question, Mapping):
            raise JevError("invalid_request", f"question '{question_id}' must be an object.")

        question_type = question.get("type")
        instructions = question.get("instructions")
        if question_type not in _QUESTION_TYPES:
            raise JevError(
                "invalid_request",
                f"question '{question_id}' has an unsupported type.",
            )
        if not _is_json_value(instructions):
            raise JevError(
                "invalid_request",
                f"question '{question_id}' instructions must be a string, object, or array.",
            )

        criteria = question.get("criteria")
        if question_type == "choice":
            if not isinstance(criteria, Mapping) or not criteria:
                raise JevError(
                    "invalid_request",
                    f"choice question '{question_id}' needs a non-empty criteria object.",
                )
            if len(criteria) > 255:
                raise JevError(
                    "invalid_request",
                    f"choice question '{question_id}' has more than 255 options.",
                )
            if any(not isinstance(key, str) or not key for key in criteria):
                raise JevError(
                    "invalid_request",
                    f"choice question '{question_id}' has an invalid option id.",
                )
            if any(not _is_json_value(value, allow_none=True) for value in criteria.values()):
                raise JevError(
                    "invalid_request",
                    f"choice question '{question_id}' has invalid option criteria.",
                )
        elif question_type == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise JevError(
                    "invalid_request",
                    f"score question '{question_id}' needs 2 to 10 criteria levels.",
                )
            if any(not _is_json_value(value) for value in criteria):
                raise JevError(
                    "invalid_request",
                    f"score question '{question_id}' has invalid criteria.",
                )
        elif criteria is not None:
            if not isinstance(criteria, Mapping):
                raise JevError(
                    "invalid_request",
                    f"noul question '{question_id}' criteria must be an object.",
                )
            if any(key not in {"true", "false"} for key in criteria):
                raise JevError(
                    "invalid_request",
                    f"noul question '{question_id}' criteria only supports true and false.",
                )
            if any(not _is_json_value(value) for value in criteria.values()):
                raise JevError(
                    "invalid_request",
                    f"noul question '{question_id}' has invalid criteria.",
                )


def _validate_probability_map(value: Any, *, field: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise JevError("invalid_response", f"{field} must be a non-empty object.")
    probabilities = []
    for probability in value.values():
        if not _is_finite_number(probability) or not 0 <= float(probability) <= 1:
            raise JevError("invalid_response", f"{field} contains an invalid probability.")
        probabilities.append(float(probability))
    if abs(sum(probabilities) - 1.0) > 0.02:
        raise JevError("invalid_response", f"{field} probabilities do not sum to one.")


def _validate_response(response: Any, questions: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(response, Mapping):
        raise JevError("invalid_response", "TypeSafe returned a non-object response.")

    answers = response.get("answers")
    if not isinstance(answers, Mapping):
        raise JevError("invalid_response", "TypeSafe response has no answers object.")

    for question_id, question in questions.items():
        answer = answers.get(question_id)
        if not isinstance(answer, Mapping):
            raise JevError(
                "invalid_response",
                f"TypeSafe response is missing answer '{question_id}'.",
            )

        question_type = question["type"]
        if answer.get("type") != question_type:
            raise JevError(
                "invalid_response",
                f"TypeSafe answer '{question_id}' has the wrong type.",
            )

        if question_type == "noul":
            noul = answer.get("noul")
            if not _is_finite_number(noul) or not 0 <= float(noul) <= 1:
                raise JevError(
                    "invalid_response",
                    f"TypeSafe answer '{question_id}' has an invalid noul value.",
                )
            continue

        confidence = answer.get("confidence")
        if not _is_finite_number(confidence) or not 0 <= float(confidence) <= 1:
            raise JevError(
                "invalid_response",
                f"TypeSafe answer '{question_id}' has an invalid confidence.",
            )
        _validate_probability_map(
            answer.get("probabilities"),
            field=f"answer '{question_id}' probabilities",
        )

        if question_type == "choice":
            choice = answer.get("choice")
            criteria = question["criteria"]
            if not isinstance(choice, str) or choice not in criteria:
                raise JevError(
                    "invalid_response",
                    f"TypeSafe answer '{question_id}' chose an unknown option.",
                )
            if set(answer["probabilities"]) != set(criteria):
                raise JevError(
                    "invalid_response",
                    f"TypeSafe answer '{question_id}' probabilities do not match criteria.",
                )
        else:
            score = answer.get("score")
            legend = answer.get("legend")
            if not _is_finite_number(score):
                raise JevError(
                    "invalid_response",
                    f"TypeSafe answer '{question_id}' has an invalid score.",
                )
            if not isinstance(legend, Mapping) or not legend:
                raise JevError(
                    "invalid_response",
                    f"TypeSafe answer '{question_id}' has no legend.",
                )
            if set(answer["probabilities"]) != {str(index) for index in range(len(question["criteria"]))}:
                raise JevError(
                    "invalid_response",
                    f"TypeSafe answer '{question_id}' probabilities have invalid levels.",
                )

    usage = response.get("usage")
    if usage is not None:
        if not isinstance(usage, Mapping):
            raise JevError("invalid_response", "TypeSafe response usage must be an object.")
        for key in ("input_tokens", "output_tokens"):
            if key in usage and (
                not isinstance(usage[key], int) or isinstance(usage[key], bool) or usage[key] < 0
            ):
                raise JevError("invalid_response", f"TypeSafe response usage has invalid {key}.")

    return dict(response)


class JevClient:
    """Small stdlib-only client for the TypeSafe System One endpoint."""

    def __init__(
        self,
        *,
        api_key: str = "",
        api_url: str = "",
        model: str = "",
        timeout: float = 15.0,
        max_retries: int = 0,
        urlopen: Optional[Callable[..., Any]] = None,
        sleep: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.api_key = api_key.strip() or _configured_api_key()
        self.api_url = (api_url.strip() or _env_value("JEV_API_URL") or DEFAULT_API_URL).rstrip("/")
        self.model = model.strip() or _env_value("JEV_MODEL") or DEFAULT_MODEL
        self.timeout = max(0.1, float(timeout))
        self.max_retries = _bounded_int(max_retries, 0, maximum=5)
        self._urlopen = urlopen or urllib.request.urlopen
        self._sleep = sleep or time.sleep

    def decide(
        self,
        *,
        state: Any,
        questions: Mapping[str, Any],
        model: str = "",
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "state": state,
            "model": model.strip() or self.model,
            "questions": questions,
        }
        _validate_request(payload)
        payload["questions"] = {
            str(question_id): dict(question)
            for question_id, question in questions.items()
        }
        return _validate_response(
            self._post(payload),
            payload["questions"],
        )

    def _post(self, payload: Mapping[str, Any]) -> Any:
        if not self.api_key:
            raise JevError(
                "missing_api_key",
                "Jev is not configured: set JEV_API_KEY or TYPESAFE_API_KEY.",
            )

        body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.api_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        for attempt in range(self.max_retries + 1):
            try:
                response = self._urlopen(request, timeout=self.timeout)
                try:
                    raw_body = response.read()
                finally:
                    close = getattr(response, "close", None)
                    if callable(close):
                        close()
                try:
                    decoded = json.loads(raw_body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    raise JevError(
                        "invalid_response",
                        "TypeSafe returned an invalid JSON response.",
                    )
                return decoded
            except urllib.error.HTTPError as exc:
                status = int(getattr(exc, "code", 0) or 0)
                if status in _RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                    self._sleep(min(2.0, 0.25 * (2**attempt)))
                    continue
                raise JevError(
                    "http_error",
                    f"TypeSafe request failed with HTTP {status}.",
                    status=status,
                ) from None
            except urllib.error.URLError:
                if attempt < self.max_retries:
                    self._sleep(min(2.0, 0.25 * (2**attempt)))
                    continue
                raise JevError(
                    "network_error",
                    "TypeSafe request failed before a response.",
                ) from None
            except TimeoutError:
                if attempt < self.max_retries:
                    self._sleep(min(2.0, 0.25 * (2**attempt)))
                    continue
                raise JevError(
                    "timeout",
                    "TypeSafe request timed out.",
                ) from None

        raise JevError("network_error", "TypeSafe request did not complete.")


class JevContextEngine(ContextEngine):
    """Hermes context engine with deterministic request-only selection."""

    def __init__(
        self,
        *,
        client: Optional[JevClient] = None,
        api_key: str = "",
        api_url: str = "",
        model: str = "",
    ) -> None:
        self._client = client
        self._api_key = api_key.strip() or _configured_api_key()
        self._api_url = api_url.strip() or _env_value("JEV_API_URL") or DEFAULT_API_URL
        self._model = model.strip() or _env_value("JEV_MODEL") or DEFAULT_MODEL
        self.context_selection_limit = _bounded_int(
            _env_value("JEV_CONTEXT_MAX_MESSAGES"),
            0,
            maximum=10000,
        )
        self.last_jev_error: Optional[str] = None

    @property
    def name(self) -> str:
        return "jev"

    def is_available(self) -> bool:
        return self._client is not None or bool(self._api_key)

    def _get_client(self) -> JevClient:
        if self._client is None:
            self._client = JevClient(
                api_key=self._api_key,
                api_url=self._api_url,
                model=self._model,
            )
        return self._client

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        self.last_completion_tokens = int(
            usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        )
        self.last_total_tokens = int(
            usage.get("total_tokens", self.last_prompt_tokens + self.last_completion_tokens) or 0
        )

    def should_compress(self, prompt_tokens: int = None) -> bool:
        return False

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
        force: bool = False,
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        return messages

    def select_context(
        self,
        request_messages: List[Dict[str, Any]],
        *,
        conversation_messages: List[Dict[str, Any]] = None,
        incoming_message: Dict[str, Any] = None,
        budget_tokens: int = 0,
    ) -> Optional[List[Dict[str, Any]]]:
        limit = self.context_selection_limit
        if limit <= 0 or len(request_messages) <= limit:
            return None

        system_messages = [
            message for message in request_messages
            if isinstance(message, dict) and message.get("role") == "system"
        ]
        non_system_messages = [
            message for message in request_messages
            if not (isinstance(message, dict) and message.get("role") == "system")
        ]
        non_system_limit = max(0, limit - len(system_messages))
        selected = system_messages + (
            non_system_messages[-non_system_limit:] if non_system_limit else []
        )
        return copy.deepcopy(selected)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "jev_decide",
                    "description": "Evaluate typed Choice, Score, or Noul questions against a state.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "state": {
                                "description": "The text or structured state to evaluate.",
                                "oneOf": [
                                    {"type": "string"},
                                    {"type": "object"},
                                    {"type": "array"},
                                ],
                            },
                            "model": {"type": "string"},
                            "questions": {
                                "type": "object",
                                "description": "Map of typed question ids to question definitions.",
                                "additionalProperties": {"type": "object"},
                            },
                        },
                        "required": ["state", "questions"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        if name != "jev_decide":
            return json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "unknown_tool",
                        "message": f"Unknown Jev tool: {name}",
                    },
                },
                ensure_ascii=False,
            )

        try:
            args = args if isinstance(args, dict) else {}
            result = self._get_client().decide(
                state=args.get("state"),
                questions=args.get("questions"),
                model=args.get("model", ""),
            )
            self.last_jev_error = None
            return json.dumps({"ok": True, "result": result}, ensure_ascii=False)
        except JevError as exc:
            self.last_jev_error = exc.code
            return json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": exc.code,
                        "message": exc.message,
                    },
                },
                ensure_ascii=False,
            )
        except Exception:
            self.last_jev_error = "internal_error"
            return json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "internal_error",
                        "message": "Jev evaluation failed.",
                    },
                },
                ensure_ascii=False,
            )

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status.update(
            {
                "jev_available": self.is_available(),
                "jev_context_selection_limit": self.context_selection_limit,
                "jev_last_error": self.last_jev_error,
            }
        )
        return status


def register(ctx: Any) -> None:
    """Register the engine through Hermes' context-engine discovery protocol."""
    ctx.register_context_engine(JevContextEngine())


__all__ = [
    "DEFAULT_API_URL",
    "DEFAULT_MODEL",
    "JevClient",
    "JevContextEngine",
    "JevError",
    "register",
]
