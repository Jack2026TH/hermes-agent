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


def _custody_test_runner(monkeypatch, *, outcome=None):
    """Exercise the real terminal handler path without any network or writes."""
    value = object.__new__(GatewayRunner)
    value.config = SimpleNamespace(multiplex_profiles=False)
    value._hm_estop_gate = lambda *args: None
    value._session_key_for_source = lambda source: "telegram:synthetic"
    def admit(event):
        event._gateway_disposition = "failed"
        return (event, event.source, False)
    value._hm_admit_event = AsyncMock(side_effect=admit)
    value._hm_pending_reply_intercepts = AsyncMock(return_value=None)
    value._hm_evict_idle_stale_agent = lambda key: None
    value._is_session_running = lambda key: False
    value._hm_dispatch_idle_commands = AsyncMock(return_value=(False, None))
    value._is_telegram_topic_root_lobby = lambda source: False
    value._external_drain_active = False
    value._claim_active_session_slot = lambda key, source: (None, None)
    value._hm_rescue_orphaned_fifo = lambda event, source, internal, key: (event, source, internal)
    value._session_state = lambda key: SimpleNamespace(turn=SimpleNamespace(agent=None, started_ts=0))
    value._persist_active_agents = lambda: None
    value._begin_session_run_generation = lambda key: 1
    value._handle_message_with_agent = AsyncMock(return_value=outcome)
    value._run_post_turn_hooks = AsyncMock()
    value._restore_pending_one_turn_model_override = lambda *args: None
    value._clear_durable_active_turn = AsyncMock()
    value._release_running_agent_state = lambda *args, **kwargs: None
    value._release_turn_lease = lambda *args, **kwargs: None
    manager = plugins.PluginManager()
    dispositions = []
    manager._hooks["gateway_message_disposition"] = [
        lambda event, disposition, **kwargs: dispositions.append(disposition)
    ]
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
    return value, dispositions


@pytest.mark.parametrize("outcome", [None, "finished answer"])
def test_completed_agent_turn_has_terminal_handled_disposition(monkeypatch, outcome):
    """Even an empty/streamed reply ran tools: replay would run them twice."""
    runner, dispositions = _custody_test_runner(monkeypatch, outcome=outcome)
    event = MessageEvent(text="do work", message_id="7", platform_update_id=8,
        source=SessionSource(platform=Platform.TELEGRAM, user_id="101", chat_id="101", chat_type="group"))
    assert asyncio.run(runner._handle_message(event)) == outcome
    assert runner._handle_message_with_agent.await_count == 1
    assert dispositions == ["handled"]


@pytest.mark.parametrize("text", ["new task", "unknown slash /command"])
def test_busy_adapter_guard_prevents_pre_sentinel_agent_overlap(monkeypatch, text):
    """Adapter's busy guard is authoritative even before runner agent sentinel."""
    runner, dispositions = _custody_test_runner(monkeypatch)
    claims = []
    runner._claim_active_session_slot = lambda *args: claims.append(args) or (None, None)
    event = MessageEvent(text=text, message_id="8", platform_update_id=9,
        source=SessionSource(platform=Platform.TELEGRAM, user_id="101", chat_id="101", chat_type="group"))
    event._gateway_adapter_custody_busy = True
    assert asyncio.run(runner._handle_message(event)) is None
    assert dispositions == ["failed"]  # durable owner retains it for retry
    assert not claims
    runner._handle_message_with_agent.assert_not_awaited()
    runner._hm_dispatch_idle_commands.assert_not_awaited()


def test_busy_adapter_guard_still_allows_native_command(monkeypatch):
    runner, dispositions = _custody_test_runner(monkeypatch)
    runner._hm_dispatch_idle_commands = AsyncMock(return_value=(True, "native response"))
    event = MessageEvent(text="/status", message_id="9", platform_update_id=10,
        source=SessionSource(platform=Platform.TELEGRAM, user_id="101", chat_id="101", chat_type="group"))
    event._gateway_adapter_custody_busy = True
    assert asyncio.run(runner._handle_message(event)) == "native response"
    assert dispositions == ["handled"]
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.parametrize("busy", [False, True])
def test_pre_auth_text_rewrite_preserves_terminal_custody_and_busy_guard(monkeypatch, busy):
    """dataclasses.replace must not strip transport custody metadata or lose ack."""
    runner, dispositions = _custody_test_runner(monkeypatch, outcome="finished")
    runner._hm_admit_event = GatewayRunner._hm_admit_event.__get__(runner)
    runner._hm_estop_turn_allowed = lambda *args: False
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._is_user_authorized_for_source = lambda source: source.user_id == "101"
    runner._admit_bot_message_for_source = lambda source: True
    manager = plugins._delivery_manager()
    manager._hooks["pre_gateway_dispatch"] = [
        lambda **kwargs: {"action": "rewrite", "text": "rewritten"}
    ]
    event = MessageEvent(text="original", message_id="10", platform_update_id=11,
        source=SessionSource(platform=Platform.TELEGRAM, user_id="101", chat_id="101", chat_type="group"))
    if busy:
        event._gateway_adapter_custody_busy = True
    output = asyncio.run(runner._handle_message(event))
    if busy:
        assert output is None
        assert dispositions == ["failed"]
        runner._handle_message_with_agent.assert_not_awaited()
    else:
        assert output == "finished"
        assert dispositions == ["handled"]
        passed_event = runner._handle_message_with_agent.await_args.args[0]
        assert passed_event.text == "rewritten"


def test_completed_turn_not_replayed_on_cleanup_failure(monkeypatch):
    runner, dispositions = _custody_test_runner(monkeypatch, outcome="finished")
    runner._clear_durable_active_turn = AsyncMock(side_effect=RuntimeError("cleanup failed"))
    event = MessageEvent(text="run once", message_id="12", platform_update_id=13,
        source=SessionSource(platform=Platform.TELEGRAM, user_id="101", chat_id="101", chat_type="group"))
    with pytest.raises(RuntimeError, match="cleanup failed"):
        asyncio.run(runner._handle_message(event))
    assert runner._handle_message_with_agent.await_count == 1
    assert dispositions == ["handled"]
