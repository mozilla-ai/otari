"""A provider, and with it its HTTP client, is built once and reused across requests."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from any_llm import AnyLLM

from gateway.services.providers import acompletion, aresponses, provider_for


@pytest.fixture
def created() -> Any:
    """Count the providers built, each a fresh mock."""
    with patch.object(AnyLLM, "create", side_effect=lambda *a, **k: MagicMock()) as create:
        yield create


@pytest.mark.asyncio
async def test_the_same_credentials_reuse_one_provider(created: Any) -> None:
    first = provider_for("openai", "sk-a", "https://api.example/v1", {"timeout": 30})
    second = provider_for("openai", "sk-a", "https://api.example/v1", {"timeout": 30})

    assert first is second
    assert created.call_count == 1


@pytest.mark.asyncio
async def test_a_different_key_base_or_option_builds_its_own(created: Any) -> None:
    base = provider_for("openai", "sk-a", "https://api.example/v1", None)

    assert provider_for("openai", "sk-b", "https://api.example/v1", None) is not base
    assert provider_for("openai", "sk-a", "https://other.example/v1", None) is not base
    assert provider_for("openai", "sk-a", "https://api.example/v1", {"max_retries": 0}) is not base
    assert provider_for("anthropic", "sk-a", "https://api.example/v1", None) is not base
    assert created.call_count == 5


@pytest.mark.asyncio
async def test_nested_plain_options_key_the_cache_by_value(created: Any) -> None:
    first = provider_for("openai", None, None, {"default_headers": {"a": "1"}, "scopes": ["x"]})
    second = provider_for("openai", None, None, {"scopes": ["x"], "default_headers": {"a": "1"}})

    assert first is second
    assert created.call_count == 1


@pytest.mark.asyncio
async def test_an_object_option_builds_a_provider_every_call(created: Any) -> None:
    """An object may be built per request, and keying one by identity could hand its client to another."""
    http_client = object()

    first = provider_for("openai", None, None, {"http_client": http_client})
    second = provider_for("openai", None, None, {"http_client": http_client})

    assert first is not second
    assert created.call_count == 2


@pytest.mark.asyncio
async def test_the_cache_is_bounded(created: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gateway.services.providers._provider_clients._MAX_PROVIDERS", 2)
    oldest = provider_for("openai", "sk-1", None, None)
    provider_for("openai", "sk-2", None, None)
    provider_for("openai", "sk-3", None, None)

    assert provider_for("openai", "sk-1", None, None) is not oldest


@pytest.mark.asyncio
async def test_a_failed_build_is_not_cached() -> None:
    with patch.object(AnyLLM, "create", side_effect=[ValueError("no key"), MagicMock()]):
        with pytest.raises(ValueError, match="no key"):
            provider_for("openai", None, None, None)
        assert provider_for("openai", None, None, None) is not None


@pytest.mark.asyncio
async def test_the_calls_forward_like_any_llms_own(created: Any) -> None:
    done: Any = object()
    resp: Any = object()
    llm = MagicMock(acompletion=AsyncMock(return_value=done), aresponses=AsyncMock(return_value=resp))
    created.side_effect = lambda *a, **k: llm

    assert await acompletion(model="openai:gpt-5", api_key="sk", messages=[], stream=False) is done
    llm.acompletion.assert_awaited_once_with(model="gpt-5", messages=[], stream=False)
    created.assert_called_once_with("openai", api_key="sk", api_base=None)

    assert await aresponses("gpt-5", [], provider="openai", api_key="sk") is resp
    llm.aresponses.assert_awaited_once_with(model="gpt-5", input_data=[])
