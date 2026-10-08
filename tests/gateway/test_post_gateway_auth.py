"""Post-auth ownership must precede normal gateway dispatch, with failure closed."""
from types import SimpleNamespace
import asyncio

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.run_inbound import GatewayInboundMixin
from hermes_cli import plugins

def runner():
    value = GatewayInboundMixin()
    value.config = SimpleNamespace(multiplex_profiles=False)
    value._scale_to_zero_note_real_inbound = lambda: None
    value._is_user_authorized_for_source = lambda source: source.user_id == "101"
    value._admit_bot_message_for_source = lambda source: True
    return value

def event(actor="101"):
    return MessageEvent(text="original text", message_id="7", platform_update_id=8,
        raw_message={"text": "original text", "from": {"id": int(actor)}},
        source=SessionSource(platform=Platform.TELEGRAM, user_id=actor, chat_id=actor, chat_type="group"))

def manager(tmp_path, monkeypatch):
    home = tmp_path / "home"
    root = home / "plugins" / "postauth"
    root.mkdir(parents=True)
    (root / "plugin.yaml").write_text("name: postauth\nversion: 1.0.0\ndescription: Isolated consumer probe\n")
    (root / "__init__.py").write_text(
        "consumer = None\n"
        "def receive(event, **kwargs):\n"
        "    return consumer(event)\n"
        "def register(ctx):\n"
        "    ctx.register_hook('post_gateway_auth', receive)\n")
    (home / "config.yaml").write_text("plugins:\n  enabled: [postauth]\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    result = plugins.PluginManager()
    result.discover_and_load()
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: result)
    return result

def test_authenticated_consumer_is_discovered_and_exclusive(tmp_path, monkeypatch):
    managed = manager(tmp_path, monkeypatch)
    state = managed._plugins["postauth"]
    assert state.enabled and state.error is None, state.error
    callback = managed._hooks["post_gateway_auth"][0]
    calls = []
    callback.__globals__["consumer"] = lambda event: calls.append(event) or {"action": "handled"}
    admitted = runner()
    assert asyncio.run(admitted._hm_admit_event(event("999"))) is None
    assert calls == []
    internal = event()
    internal.internal = True
    assert asyncio.run(admitted._hm_admit_event(internal))[2] is True
    assert calls == []
    original = event()
    assert asyncio.run(admitted._hm_admit_event(original)) is None
    assert calls[0].raw_message == original.raw_message
    assert calls[0].source.user_id == "101"

def test_consumer_failure_and_mutation_cannot_fall_through(tmp_path, monkeypatch):
    managed = manager(tmp_path, monkeypatch)
    state = managed._plugins["postauth"]
    assert state.enabled and state.error is None, state.error
    callback = managed._hooks["post_gateway_auth"][0]
    original = event()
    def failure(value):
        value.source.user_id = "999"
        value.raw_message["text"] = "forged"
        raise TimeoutError("durable consumer unavailable")
    callback.__globals__["consumer"] = failure
    assert asyncio.run(runner()._hm_admit_event(original)) is None
    assert original.source.user_id == "101"
    assert original.raw_message["text"] == "original text"
    callback.__globals__["consumer"] = lambda value: {"action": "rewrite", "text": "forged"}
    assert asyncio.run(runner()._hm_admit_event(original)) is None
    # A pre-auth plugin cannot launder a different sender into the authorized consumer.
    observed = []
    callback.__globals__["consumer"] = lambda value: observed.append(value) or {"action": "handled"}
    def change_source(event, **kwargs):
        event.source.user_id = "101"
    managed._hooks.setdefault("pre_gateway_dispatch", []).append(change_source)
    assert asyncio.run(runner()._hm_admit_event(event("999"))) is None
    assert observed == []
    managed._hooks["pre_gateway_dispatch"].clear()
    # Reject multiple owners before either gets a persistence opportunity.
    managed._hooks["post_gateway_auth"].append(lambda **kwargs: observed.append(kwargs))
    assert asyncio.run(runner()._hm_admit_event(event())) is None
    assert observed == []
    managed._hooks["post_gateway_auth"].pop()
    callback.__globals__["consumer"] = lambda value: None
    result = asyncio.run(runner()._hm_admit_event(original))
    assert result[0] is original
