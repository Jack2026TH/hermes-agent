"""A configured durable PTB queue must preserve pending provider updates."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as telegram
from plugins.platforms.telegram.adapter import TelegramAdapter

@pytest.mark.asyncio
async def test_durable_queue_preserves_pending_on_every_polling_start():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="synthetic-test-token"))
    queue = asyncio.Queue()
    queue.durable_custody = True
    app = SimpleNamespace(update_queue=queue,
        updater=SimpleNamespace(update_queue=queue, start_polling=AsyncMock()))
    await adapter._start_polling_once(app, drop_pending_updates=True,
        error_callback=None, schedule_verifier=False)
    assert app.updater.start_polling.await_args.kwargs["drop_pending_updates"] is False
    queue.durable_custody = False
    await adapter._start_polling_once(app, drop_pending_updates=True,
        error_callback=None, schedule_verifier=False)
    assert app.updater.start_polling.await_args.kwargs["drop_pending_updates"] is True
    queue.durable_custody = True
    app.updater.update_queue = asyncio.Queue()
    with pytest.raises(RuntimeError):
        await adapter._start_polling_once(app, drop_pending_updates=True,
            error_callback=None, schedule_verifier=False)

@pytest.mark.asyncio
async def test_durable_queue_preserves_pending_webhook_updates(monkeypatch):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="synthetic-test-token"))
    queue = asyncio.Queue()
    queue.durable_custody = True
    app = SimpleNamespace(update_queue=queue,
        updater=SimpleNamespace(update_queue=queue, start_webhook=AsyncMock()))
    adapter._app = app
    monkeypatch.setattr(telegram, "_get_scoped_secret", lambda name: "synthetic-webhook-secret")
    await adapter._start_webhook_mode("https://example.test/telegram", is_reconnect=False)
    assert app.updater.start_webhook.await_args.kwargs["drop_pending_updates"] is False

def test_request_binding_is_strict_and_preserves_existing_request(monkeypatch):
    import importlib
    from hermes_cli import lifecycle
    # Gateway conftest intentionally stubs the SDK; real PTB coverage lives in the isolated eval.
    class Request:
        pass
    monkeypatch.setattr(importlib.import_module("telegram.request"), "BaseRequest", Request)
    monkeypatch.setattr(lifecycle, "has_hook", lambda name: True)
    original = Request()
    wrapped = Request()
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda *args, **kwargs: [{"request": wrapped}])
    assert TelegramAdapter._bind_get_updates_request(original) is wrapped
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda *args, **kwargs: [])
    monkeypatch.setattr(lifecycle, "has_hook", lambda name: False)
    assert TelegramAdapter._bind_get_updates_request(original) is original
    monkeypatch.setattr(lifecycle, "has_hook", lambda name: True)
    with pytest.raises(TypeError):
        TelegramAdapter._bind_get_updates_request(original)
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda *args, **kwargs: [{"request": object()}])
    with pytest.raises(TypeError):
        TelegramAdapter._bind_get_updates_request(original)


def test_queue_validation_accepts_legacy_app_only_without_durable_claim():
    # A legacy test/transport without PTB queue metadata is not a custody owner.
    legacy_app = SimpleNamespace()
    plain_request = SimpleNamespace(durable_custody=False)
    TelegramAdapter._validate_request_custody(legacy_app, plain_request)

    # Declaring durable custody requires the original shared PTB queue.
    with pytest.raises(RuntimeError):
        TelegramAdapter._validate_request_custody(
            legacy_app, SimpleNamespace(durable_custody=True, custody_queue=asyncio.Queue())
        )

    queue = asyncio.Queue()
    queue.durable_custody = True
    app = SimpleNamespace(update_queue=queue, updater=SimpleNamespace(update_queue=queue))
    with pytest.raises(RuntimeError):
        TelegramAdapter._validate_request_custody(app, plain_request)

    owner_request = SimpleNamespace(durable_custody=True, custody_queue=queue)
    TelegramAdapter._validate_request_custody(app, owner_request)

    # Incomplete SDK updater metadata must fail closed instead of silently proceeding.
    incomplete_app = SimpleNamespace(update_queue=queue, updater=SimpleNamespace())
    with pytest.raises(RuntimeError):
        TelegramAdapter._validate_request_custody(incomplete_app, owner_request)
