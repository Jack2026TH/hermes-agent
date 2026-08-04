from pathlib import Path

import pytest

from hermes_cli.session_export_md import (
    append_manifest_entry,
    render_session_markdown,
    safe_session_filename,
    verify_export_file,
    write_session_markdown,
)


def _session(**overrides):
    data = {
        "id": "20260706_123456_abcd1234",
        "title": "Export Test",
        "source": "telegram",
        "model": "gpt-5.5",
        "billing_provider": "openai-codex",
        "cwd": "/tmp/project",
        "started_at": 1783331696.0,
        "last_active": 1783331705.0,
        "ended_at": 1783331710.0,
        "message_count": 3,
        "tool_call_count": 1,
        "archived": 0,
        "messages": [
            {"role": "user", "content": "Hello", "created_at": 1783331697.0},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "terminal", "arguments": "{\"command\": \"pwd\"}"}}
                ],
                "created_at": 1783331698.0,
            },
            {"role": "tool", "name": "terminal", "content": "output", "created_at": 1783331699.0},
        ],
    }
    data.update(overrides)
    return data




def test_safe_session_filename_is_deterministic_and_path_safe():
    filename = safe_session_filename(
        _session(id="20260706_123456_abcd1234", title="Bad / title: * ?"), fmt="qmd"
    )

    assert filename.startswith("20260706_123456_abcd1234-")
    assert filename.endswith(".qmd")
    assert "/" not in filename
    assert ":" not in filename
    assert "*" not in filename
    assert "?" not in filename






def test_verify_export_file_checks_count_and_sha(tmp_path):
    session = _session()
    path = write_session_markdown(session, tmp_path)

    ok, reason = verify_export_file(path, session)
    assert ok is True
    assert reason == "ok"

    path.write_text(path.read_text(encoding="utf-8").replace("Hello", "Tampered"), encoding="utf-8")
    ok, reason = verify_export_file(path, session)
    assert ok is False
    assert "sha256" in reason




# --- Tests for embedded SHA256 lines (cherry-picked fix faf6b739c0) ---

def test_verify_export_with_embedded_sha256_lines(tmp_path):
    """Session content contains SHA256 lines from quoted exports.
    verify_export_file must use the LAST (footer) SHA line, not the first."""
    session = _session()
    session["messages"].append({
        "role": "tool",
        "content": "## Export verification\n- Session id: `20260706_999999_zzzzzzz`\n"
                   "- Exported messages: `42`\n"
                   "- SHA256 of exported body: `deadbeef00000000000000000000000000000000000000000000000000000dead`\n",
        "created_at": 1783331700.0,
    })
    session["message_count"] = 4

    path = write_session_markdown(session, tmp_path)
    ok, reason = verify_export_file(path, session)
    assert ok is True, f"Expected ok but got: {reason}"


def test_verify_export_with_placeholder_in_content(tmp_path):
    """Session content contains the literal __SHA256_PLACEHOLDER__ string.
    render_session_markdown must only replace the footer placeholder, not
    embedded copies."""
    session = _session()
    session["messages"].append({
        "role": "user",
        "content": "The placeholder looks like __SHA256_PLACEHOLDER__",
        "created_at": 1783331701.0,
    })
    session["message_count"] = 4

    path = write_session_markdown(session, tmp_path)
    ok, reason = verify_export_file(path, session)
    assert ok is True, f"Expected ok but got: {reason}"
