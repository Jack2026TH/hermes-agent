"""Test that kanban sessions are sealed by seal_idle_non_webui_sessions.

The archive script's seal_idle_non_webui_sessions() must include "kanban"
in sealable_sources.  Without it, kanban sessions never get ended_at set
and are never archived, accumulating forever in state.db.

This test imports the production archive_sessions.py (which lives outside
the git repo at $HERMES_HOME/scripts/) via importlib.
"""
import importlib.util
import os
import time
from pathlib import Path

import pytest


os.environ.setdefault("HERMES_HOME", "/opt/rent-oleg-runtime/data/hermes_jack_v2")
os.environ.setdefault("HERMES_ARCHIVE_SOURCE", "/opt/rent-oleg-runtime/data/hermes_jack_v2/hermes-agent")
os.environ.setdefault("HERMES_ARCHIVE_SYNC_DEVICE_ID", "test-device")

ARCHIVE_SCRIPT = Path(
    os.environ.get("HERMES_HOME", "/opt/rent-oleg-runtime/data/hermes_jack_v2")
) / "scripts" / "archive_sessions.py"

if not ARCHIVE_SCRIPT.exists():
    pytest.skip(f"archive_sessions.py not found at {ARCHIVE_SCRIPT}", allow_module_level=True)

_spec = importlib.util.spec_from_file_location(
    "hermes_runtime_archive_sessions", ARCHIVE_SCRIPT
)
assert _spec is not None and _spec.loader is not None
archive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(archive)


class _SealableDB:
    """Minimal DB stub for seal_idle_non_webui_sessions."""

    def __init__(self, sessions):
        self.sessions = sessions
        self.sealed = []

    def list_sessions_rich(self, **_kwargs):
        return [
            {
                "id": sid,
                "source": source,
                "ended_at": None,
                "last_active": last_active,
                "started_at": last_active - 100,
            }
            for sid, source, last_active in self.sessions
        ]

    def end_session(self, session_id, reason):
        self.sealed.append((session_id, reason))

    def get_session(self, session_id):
        for sid, _source, last_active in self.sessions:
            if sid == session_id:
                return {"id": sid, "ended_at": last_active + 1}
        return None


def test_kanban_sealed_by_seal_idle_non_webui():
    """kanban sessions must be sealed by seal_idle_non_webui_sessions."""
    old_time = time.time() - 7200  # 2 hours ago
    db = _SealableDB([
        ("kanban-001", "kanban", old_time),
        ("cli-001", "cli", old_time),
        ("telegram-001", "telegram", old_time),
    ])

    sealed = archive.seal_idle_non_webui_sessions(db, min_idle=3600)

    assert "kanban-001" in sealed, f"kanban missing from sealed: {sealed}"
    assert "cli-001" in sealed
    assert "telegram-001" in sealed
    assert len(sealed) == 3