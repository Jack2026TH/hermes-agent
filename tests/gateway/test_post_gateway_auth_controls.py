"""Native control traffic retains the gateway handler after consumer registration."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from gateway.run import GatewayRunner
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from hermes_cli import plugins

@pytest.mark.parametrize("command", ["stop", "approve", "deny"])
@pytest.mark.parametrize("outcome", ["handled", "block", "exception"])
def test_native_controls_reach_actual_message_handler(monkeypatch, command, outcome):
    value = object.__new__(GatewayRunner)
    value.config = SimpleNamespace(multiplex_profiles=False)
    value._scale_to_zero_note_real_inbound = lambda: None
    value._is_user_authorized_for_source = lambda source: source.user_id == "101"
    value._admit_bot_message_for_source = lambda source: True
    value._hm_estop_gate = lambda *args: None
    value._session_key_for_source = lambda source: "telegram:synthetic"
    value._peek_session_state = lambda key: None
    value._is_session_running = lambda key: False
    value._hm_pending_reply_intercepts = AsyncMock(return_value=None)
    value._hm_evict_idle_stale_agent = lambda key: None
    value._hm_dispatch_idle_commands = AsyncMock(return_value=(True, "native control result"))
    called = []
    def consumer(**kwargs):
        called.append(kwargs)
        if outcome == "exception":
            raise TimeoutError("isolated consumer failure")
        return {"action": outcome}
    manager = plugins.PluginManager()
    manager._hooks["post_gateway_auth"] = [consumer]
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
    message = MessageEvent(text="/" + command, message_id="7",
        source=SessionSource(platform=Platform.TELEGRAM, user_id="101", chat_id="101", chat_type="group"))
    assert asyncio.run(value._handle_message(message)) == "native control result"
    assert called == []
    value._hm_dispatch_idle_commands.assert_awaited_once()
    message.source.user_id = "999"
    assert asyncio.run(value._handle_message(message)) is None
    assert value._hm_dispatch_idle_commands.await_count == 1
