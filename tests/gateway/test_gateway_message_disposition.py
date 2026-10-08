"""Gateway terminal disposition closes custody after the actual handler path."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from gateway.run import GatewayRunner
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from hermes_cli import plugins

@pytest.mark.parametrize("kind", ["native", "auth_reject", "bot_reject", "pre_skip", "consumer_failure", "busy_queue", "pending_intercept"])
def test_original_event_gets_terminal_disposition_after_real_message_handler(monkeypatch, kind):
    value = object.__new__(GatewayRunner)
    value.config = SimpleNamespace(multiplex_profiles=False)
    value._scale_to_zero_note_real_inbound = lambda: None
    value._is_user_authorized_for_source = lambda source: kind != "auth_reject"
    value._admit_bot_message_for_source = lambda source: kind != "bot_reject"
    value._hm_estop_gate = lambda *args: None
    value._session_key_for_source = lambda source: "telegram:synthetic"
    value._peek_session_state = lambda key: None
    value._is_session_running = lambda key: kind == "busy_queue"
    value._hm_pending_reply_intercepts = AsyncMock(return_value=None)
    if kind == "pending_intercept":
        del value._hm_pending_reply_intercepts
        value._hm_update_prompt_reply = lambda *args: "resolved inline"
    value._hm_evict_idle_stale_agent = lambda key: None
    value._hm_dispatch_idle_commands = AsyncMock(return_value=(True, "native response"))
    value._hm_handle_running_session_message = AsyncMock(return_value=None)
    value._hm_evict_reaped_agent = lambda key: None
    manager = plugins.PluginManager()
    received = []
    def disposition(event, disposition, **kwargs):
        received.append((event, disposition, value._hm_dispatch_idle_commands.await_count))
    manager._hooks["gateway_message_disposition"] = [disposition]
    if kind == "pre_skip":
        manager._hooks["pre_gateway_dispatch"] = [lambda **kw: {"action": "skip"}]
    if kind == "consumer_failure":
        def fail(**kw):
            raise TimeoutError("synthetic durable consumer failure")
        manager._hooks["post_gateway_auth"] = [fail]
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
    message = MessageEvent(text="/stop" if kind in {"native", "pending_intercept"} else "original",
        raw_message={"original": True}, message_id="7", platform_update_id=8,
        source=SessionSource(platform=Platform.TELEGRAM, user_id="101", chat_id="101", chat_type="group"))
    asyncio.run(value._handle_message(message))
    assert len(received) == 1
    assert received[0][0].raw_message == {"original": True}
    assert received[0][0].platform_update_id == 8
    assert received[0][1] == ("handled" if kind in {"native", "pending_intercept"} else "failed" if kind in {"consumer_failure", "busy_queue"} else "rejected")
    assert received[0][2] == (1 if kind == "native" else 0)

def test_actual_active_adapter_routes_custody_owner_through_wrapped_gateway(monkeypatch):
    from gateway.platforms.base import BasePlatformAdapter
    from hermes_cli import lifecycle
    value = SimpleNamespace(_dispatch_inline_reply=AsyncMock(),
        _busy_session_handler=AsyncMock(return_value=True))
    message = MessageEvent(text="original", platform_update_id=8,
        source=SessionSource(platform=Platform.TELEGRAM, user_id="101", chat_id="101", chat_type="group"))
    monkeypatch.setattr(lifecycle, "has_hook", lambda name: name == "gateway_message_disposition")
    asyncio.run(BasePlatformAdapter._handle_message_while_active(value, message, "synthetic"))
    value._dispatch_inline_reply.assert_awaited_once_with(message)
    value._busy_session_handler.assert_not_awaited()
