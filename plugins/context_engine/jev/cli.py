"""Stdin/stdout CLI so Grok Bot can call the same Jev client directly."""

from __future__ import annotations

import json
import sys

try:
    from .client import JevClient
except ImportError:
    from client import JevClient

def main() -> int:
    try:
        request = json.load(sys.stdin)
        state = request.get("state")
        questions = request.get("questions")
        if not isinstance(state, dict) or not isinstance(questions, dict) or not questions:
            raise ValueError("invalid_jev_request")
        response = JevClient().decide(state, questions)
        print(json.dumps(
            {
                "ok": True,
                "model": response.payload.get("model"),
                "answers": response.payload["answers"],
                "latency_ms": response.latency_ms,
            },
            ensure_ascii=False,
            sort_keys=True,
        ))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"jev_error:{type(exc).__name__}"}, sort_keys=True))
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
