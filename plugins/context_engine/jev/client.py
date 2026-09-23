"""Bounded, redacting client for TypeSafe System One shadow evaluations."""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from agent.redact import redact_sensitive_text

DEFAULT_JEV_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_JEV_MODEL = "jev-latest"
_REQUEST_TIMEOUT_SECONDS = 2.0
_MAX_REQUEST_BYTES = 32 * 1024
_MAX_RESPONSE_BYTES = 64 * 1024
_MAX_STRING_CHARS = 8_000
_MAX_COLLECTION_ITEMS = 128
_MAX_NESTING_DEPTH = 10
_SECRET_FIELD_PARTS = (
    "apikey",
    "token",
    "secret",
    "password",
    "authorization",
    "credential",
    "privatekey",
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_ -]?key|access[_ -]?token|refresh[_ -]?token|token|secret|password|"
    r"authorization|credential)\b\s*([=:])\s*([^\s,;\"']+)"
)


class JevError(RuntimeError):
    """A safe error code that does not retain request, response, or secret data."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class JevResponse:
    payload: dict[str, Any]
    latency_ms: float


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_secret_field(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return any(part in normalized for part in _SECRET_FIELD_PARTS)


def _redact_text(value: str) -> str:
    try:
        redacted = redact_sensitive_text(
            value,
            force=True,
            redact_url_credentials=True,
        )
    except Exception:
        raise JevError("redaction_failed") from None
    return _SECRET_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[redacted]",
        redacted,
    )


def _sanitize_json_value(value: Any, *, depth: int = 0) -> Any:
    """Redact and bound every value before it crosses the TypeSafe boundary."""
    if depth > _MAX_NESTING_DEPTH:
        raise JevError("request_too_deep")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise JevError("invalid_request")
        return value
    if isinstance(value, str):
        if len(value) > _MAX_STRING_CHARS:
            return "[omitted: oversized text]"
        return _redact_text(value)
    if isinstance(value, Mapping):
        if len(value) > _MAX_COLLECTION_ITEMS:
            raise JevError("request_too_large")
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > 128:
                raise JevError("invalid_request")
            safe_key = _redact_text(key)
            if not safe_key or len(safe_key) > 128 or safe_key in sanitized:
                raise JevError("invalid_request")
            sanitized[safe_key] = (
                "[redacted]"
                if _is_secret_field(key)
                else _sanitize_json_value(item, depth=depth + 1)
            )
        return sanitized
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_COLLECTION_ITEMS:
            raise JevError("request_too_large")
        return [_sanitize_json_value(item, depth=depth + 1) for item in value]
    raise JevError("invalid_request")


def _validate_questions(questions: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(questions, Mapping) or not questions:
        raise JevError("invalid_request")
    if len(questions) > _MAX_COLLECTION_ITEMS:
        raise JevError("request_too_large")

    sanitized = _sanitize_json_value(questions)
    if not isinstance(sanitized, dict):
        raise JevError("invalid_request")

    for question_id, question in sanitized.items():
        if not question_id.strip() or not isinstance(question, dict):
            raise JevError("invalid_request")
        question_type = question.get("type")
        if question_type == "choice":
            criteria = question.get("criteria")
            if not isinstance(criteria, dict) or not criteria:
                raise JevError("invalid_request")
        elif question_type == "noul":
            criteria = question.get("criteria")
            if criteria is not None and not isinstance(criteria, dict):
                raise JevError("invalid_request")
        else:
            raise JevError("invalid_request")
        if (
            "instructions" in question
            and question["instructions"] is not None
            and not isinstance(question["instructions"], (str, dict, list))
        ):
            raise JevError("invalid_request")
    return sanitized


def _validate_probability_map(value: Any, expected_keys: set[str]) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise JevError("invalid_response")
    probabilities: dict[str, float] = {}
    for key, probability in value.items():
        if not _is_finite_number(probability) or not 0 <= float(probability) <= 1:
            raise JevError("invalid_response")
        probabilities[key] = float(probability)
    if abs(sum(probabilities.values()) - 1.0) > 0.02:
        raise JevError("invalid_response")
    return probabilities


def _validate_response(response: Any, questions: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(response, Mapping):
        raise JevError("invalid_response")
    model = response.get("model")
    answers = response.get("answers")
    usage = response.get("usage")
    if not isinstance(model, str) or not model.strip() or len(model) > 128:
        raise JevError("invalid_response")
    if not isinstance(answers, Mapping) or set(answers) != set(questions):
        raise JevError("invalid_response")
    if not isinstance(usage, Mapping):
        raise JevError("invalid_response")
    usage_result: dict[str, int] = {}
    for field in ("input_tokens", "output_tokens"):
        value = usage.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise JevError("invalid_response")
        usage_result[field] = value

    validated_answers: dict[str, dict[str, Any]] = {}
    for question_id, question in questions.items():
        answer = answers.get(question_id)
        if not isinstance(answer, Mapping) or answer.get("type") != question["type"]:
            raise JevError("invalid_response")

        if question["type"] == "noul":
            noul = answer.get("noul")
            if not _is_finite_number(noul) or not 0 <= float(noul) <= 1:
                raise JevError("invalid_response")
            validated_answers[question_id] = {"type": "noul", "noul": float(noul)}
            continue

        criteria = question["criteria"]
        choice = answer.get("choice")
        confidence = answer.get("confidence")
        if not isinstance(choice, str) or choice not in criteria:
            raise JevError("invalid_response")
        if not _is_finite_number(confidence) or not 0 <= float(confidence) <= 1:
            raise JevError("invalid_response")
        probabilities = _validate_probability_map(
            answer.get("probabilities"), set(criteria)
        )
        if probabilities[choice] < max(probabilities.values()):
            raise JevError("invalid_response")
        validated_answers[question_id] = {
            "type": "choice",
            "choice": choice,
            "confidence": float(confidence),
            "probabilities": probabilities,
        }

    return {
        "model": model.strip(),
        "answers": validated_answers,
        "usage": usage_result,
    }


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Do not forward the bearer credential to a redirect target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _open_without_redirects(request: urllib.request.Request, timeout: float):
    opener = urllib.request.build_opener(_NoRedirectHandler())
    return opener.open(request, timeout=timeout)


class JevClient:
    """Small TypeSafe client; only the fixed HTTPS endpoint is permitted."""

    def __init__(
        self,
        *,
        token: str | None = None,
        urlopen: Callable[..., Any] | None = None,
        timeout_seconds: float = _REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self.token = str(
            token if token is not None else os.environ.get("TYPESAFE_API_KEY") or ""
        ).strip()
        self._urlopen = urlopen or _open_without_redirects
        self.timeout_seconds = max(0.1, min(float(timeout_seconds), 5.0))

    @property
    def is_configured(self) -> bool:
        return bool(self.token)

    def decide(self, state: Any, questions: Mapping[str, Any]) -> JevResponse:
        if not self.token:
            raise JevError("missing_api_key")
        if not isinstance(state, (str, Mapping, list, tuple)):
            raise JevError("invalid_request")

        payload = {
            "model": DEFAULT_JEV_MODEL,
            "state": _sanitize_json_value(state),
            "questions": _validate_questions(questions),
        }
        try:
            body = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError):
            raise JevError("invalid_request") from None
        if len(body) > _MAX_REQUEST_BYTES:
            raise JevError("request_too_large")

        request = urllib.request.Request(
            DEFAULT_JEV_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        started = time.monotonic()
        try:
            response = self._urlopen(request, timeout=self.timeout_seconds)
            try:
                raw_body = response.read(_MAX_RESPONSE_BYTES + 1)
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
        except urllib.error.HTTPError as exc:
            raise JevError(f"http_{int(exc.code or 0)}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise JevError("network_error") from None

        if len(raw_body) > _MAX_RESPONSE_BYTES:
            raise JevError("response_too_large")
        try:
            decoded = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise JevError("invalid_response") from None

        validated = _validate_response(decoded, payload["questions"])
        return JevResponse(
            payload=validated,
            latency_ms=round((time.monotonic() - started) * 1000, 3),
        )
