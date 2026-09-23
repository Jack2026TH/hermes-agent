"""Shared TypeSafe Jev System One client used by Hermes and Grok Bot CLI."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

DEFAULT_JEV_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_JEV_MODEL = "jev-latest"

def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default

@dataclass(frozen=True)
class JevResponse:
    payload: dict[str, Any]
    latency_ms: float

class JevClient:
    def __init__(
        self,
        *,
        token: str | None = None,
        url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.token = str(token if token is not None else os.environ.get("TYPESAFE_API_KEY") or "").strip()
        self.url = str(url or os.environ.get("TYPESAFE_JEV_URL") or DEFAULT_JEV_URL).strip() or DEFAULT_JEV_URL
        self.model = str(model or os.environ.get("TYPESAFE_JEV_MODEL") or DEFAULT_JEV_MODEL).strip() or DEFAULT_JEV_MODEL
        self.timeout_seconds = float(timeout_seconds) if timeout_seconds is not None else _env_float("TYPESAFE_JEV_TIMEOUT_SECONDS", 2.0, 0.1)

    def decide(self, state: Mapping[str, Any], questions: Mapping[str, Any]) -> JevResponse:
        if not self.token:
            raise RuntimeError("jev_token_missing")
        payload = json.dumps(
            {"model": self.model, "state": dict(state), "questions": dict(questions)},
            ensure_ascii=False,
        ).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=payload,
            method="POST",
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        started = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
            parsed = json.loads(response.read().decode("utf-8"))
        latency_ms = round((time.monotonic() - started) * 1000, 3)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("answers"), dict):
            raise ValueError("jev_bad_payload")
        return JevResponse(payload=parsed, latency_ms=latency_ms)
