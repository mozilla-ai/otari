"""Unit tests for the best-effort platform memory client (recall + remember)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from gateway.api.routes import _platform as platform_module
from gateway.api.routes._platform import _recall_platform_memory, _remember_platform_memory


def _config(*, base_url: str | None = "https://platform.local", **platform_extra: Any) -> Any:
    cfg = MagicMock()
    cfg.platform = {"base_url": base_url} if base_url else {}
    cfg.platform.update(platform_extra)
    cfg.platform_token = "gw_test_token"
    return cfg


# ───────────── recall ─────────────


@pytest.mark.asyncio
async def test_recall_returns_facts_and_sends_dual_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_post(
        *, url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        captured.update(url=url, headers=headers, body=body)
        return httpx.Response(200, json={"facts": ["likes metric", "name is Dimitris"]})

    monkeypatch.setattr(platform_module, "_post_platform", fake_post)

    out = await _recall_platform_memory(_config(), "tk_user", "what units?")

    assert out == ["likes metric", "name is Dimitris"]
    assert captured["url"].endswith("/gateway/memory/recall")
    assert captured["headers"]["X-Gateway-Token"] == "gw_test_token"
    assert captured["headers"]["X-User-Token"] == "tk_user"
    assert captured["body"] == {"query": "what units?"}


@pytest.mark.asyncio
async def test_recall_without_base_url_skips_call() -> None:
    assert await _recall_platform_memory(_config(base_url=None), "tk", "q") == []


@pytest.mark.asyncio
async def test_recall_blank_query_skips_call() -> None:
    assert await _recall_platform_memory(_config(), "tk", "   ") == []


@pytest.mark.asyncio
async def test_recall_non_200_yields_no_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(**kwargs: Any) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    monkeypatch.setattr(platform_module, "_post_platform", fake_post)
    assert await _recall_platform_memory(_config(), "tk", "q") == []


@pytest.mark.asyncio
async def test_recall_timeout_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(**kwargs: Any) -> httpx.Response:
        raise httpx.TimeoutException("slow")

    monkeypatch.setattr(platform_module, "_post_platform", fake_post)
    assert await _recall_platform_memory(_config(), "tk", "q") == []


@pytest.mark.asyncio
async def test_recall_malformed_payload_yields_no_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(**kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json={"facts": "not-a-list"})

    monkeypatch.setattr(platform_module, "_post_platform", fake_post)
    assert await _recall_platform_memory(_config(), "tk", "q") == []


@pytest.mark.asyncio
async def test_recall_other_httpx_error_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(**kwargs: Any) -> httpx.Response:
        raise httpx.RemoteProtocolError("server disconnected")

    monkeypatch.setattr(platform_module, "_post_platform", fake_post)
    assert await _recall_platform_memory(_config(), "tk", "q") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_timeout", ["abc", None, "", []])
async def test_recall_malformed_timeout_falls_back(monkeypatch: pytest.MonkeyPatch, bad_timeout: Any) -> None:
    captured: dict[str, Any] = {}

    async def fake_post(*, timeout_seconds: float, **kwargs: Any) -> httpx.Response:
        captured["timeout_seconds"] = timeout_seconds
        return httpx.Response(200, json={"facts": ["ok"]})

    monkeypatch.setattr(platform_module, "_post_platform", fake_post)
    out = await _recall_platform_memory(_config(memory_recall_timeout_ms=bad_timeout), "tk", "q")
    assert out == ["ok"]
    assert captured["timeout_seconds"] == 2.0


# ───────────── remember ─────────────


@pytest.mark.asyncio
async def test_remember_posts_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_post(
        *, url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        captured.update(url=url, headers=headers, body=body)
        return httpx.Response(202)

    monkeypatch.setattr(platform_module, "_post_platform", fake_post)

    messages = [{"role": "user", "content": "My name is Dimitris"}, {"role": "assistant", "content": "Hi"}]
    await _remember_platform_memory(_config(), "tk_user", messages)

    assert captured["url"].endswith("/gateway/memory/remember")
    assert captured["headers"]["X-User-Token"] == "tk_user"
    assert captured["body"] == {"messages": messages}


@pytest.mark.asyncio
async def test_remember_empty_messages_skips_call() -> None:
    # No base_url guard not needed: empty messages short-circuit before any call.
    await _remember_platform_memory(_config(), "tk", [])  # must not raise


@pytest.mark.asyncio
async def test_remember_error_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(**kwargs: Any) -> httpx.Response:
        raise httpx.NetworkError("down")

    monkeypatch.setattr(platform_module, "_post_platform", fake_post)
    # Must not raise.
    await _remember_platform_memory(_config(), "tk", [{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_remember_other_httpx_error_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(**kwargs: Any) -> httpx.Response:
        raise httpx.RemoteProtocolError("server disconnected")

    monkeypatch.setattr(platform_module, "_post_platform", fake_post)
    # Must not raise.
    await _remember_platform_memory(_config(), "tk", [{"role": "user", "content": "x"}])


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_timeout", ["abc", None, "", []])
async def test_remember_malformed_timeout_falls_back(monkeypatch: pytest.MonkeyPatch, bad_timeout: Any) -> None:
    captured: dict[str, Any] = {}

    async def fake_post(*, timeout_seconds: float, **kwargs: Any) -> httpx.Response:
        captured["timeout_seconds"] = timeout_seconds
        return httpx.Response(202)

    monkeypatch.setattr(platform_module, "_post_platform", fake_post)
    await _remember_platform_memory(
        _config(memory_remember_timeout_ms=bad_timeout), "tk", [{"role": "user", "content": "x"}]
    )
    assert captured["timeout_seconds"] == 10.0
