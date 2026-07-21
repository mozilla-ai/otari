"""Unit tests for the provider health service.

The service is a thin view over ``discover_provider_models`` (the existing
per-provider test path); these tests pin the mapping from a discovery result to a
``ProviderHealth``, the concurrent fan-out, the forced-refresh cache clear (and
its debounce), and the honest wall-clock ``checked_at`` read from the discovery
cache.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from any_llm.types.model import Model

from gateway.core.config import GatewayConfig
from gateway.services import provider_health_service as phs
from gateway.services.model_discovery_service import ProviderDiscovery, get_model_cache


def _config(providers: dict[str, dict[str, str]] | None = None) -> GatewayConfig:
    return GatewayConfig(providers=providers or {}, model_cache_ttl_seconds=300)


def _model(model_id: str) -> Model:
    return Model(id=model_id, object="model", created=1700000000, owned_by="openai")


@pytest.mark.asyncio
async def test_healthy_provider_reports_model_count_and_checked_at() -> None:
    get_model_cache().clear()
    config = _config({"openai": {"api_key": "x"}})

    async def discover(cfg: GatewayConfig, instance: str) -> ProviderDiscovery:
        get_model_cache().set(instance, [_model("gpt-4o"), _model("gpt-4o-mini")])
        return ProviderDiscovery(provider=instance, models=[_model("gpt-4o"), _model("gpt-4o-mini")])

    with patch.object(phs, "discover_provider_models", side_effect=discover):
        health = await phs.check_provider_health(config, "openai")

    assert health.instance == "openai"
    assert health.ok is True
    assert health.model_count == 2
    assert health.error is None
    assert isinstance(health.checked_at, datetime)


@pytest.mark.asyncio
async def test_unreachable_provider_is_unhealthy_and_keeps_error() -> None:
    get_model_cache().clear()
    config = _config({"anthropic": {"api_key": "x"}})

    async def discover(cfg: GatewayConfig, instance: str) -> ProviderDiscovery:
        return ProviderDiscovery(provider=instance, models=[], error="authentication failed")

    with patch.object(phs, "discover_provider_models", side_effect=discover):
        health = await phs.check_provider_health(config, "anthropic")

    assert health.ok is False
    assert health.model_count == 0
    assert health.error == "authentication failed"


@contextmanager
def _spy_on_clear() -> Iterator[list[str | None]]:
    """Record every clear(instance) call while still clearing for real."""
    cleared: list[str | None] = []
    real_clear = get_model_cache().clear

    def spy_clear(instance: str | None = None) -> None:
        cleared.append(instance)
        real_clear(instance)

    with patch.object(get_model_cache(), "clear", side_effect=spy_clear):
        yield cleared


async def _noop_discover(cfg: GatewayConfig, instance: str) -> ProviderDiscovery:
    return ProviderDiscovery(provider=instance, models=[_model("gpt-4o")])


@pytest.mark.asyncio
async def test_refresh_clears_the_cache_when_the_last_dial_is_stale() -> None:
    get_model_cache().clear()
    config = _config({"openai": {"api_key": "x"}})
    # Prime a cached listing whose dial is older than the debounce window, so a
    # forced refresh must clear it before re-dialing.
    get_model_cache().set("openai", [_model("stale")])
    get_model_cache()._store["openai"].checked_at = datetime.now(UTC) - timedelta(seconds=60)

    with (
        _spy_on_clear() as cleared,
        patch.object(phs, "discover_provider_models", side_effect=_noop_discover),
    ):
        await phs.check_provider_health(config, "openai", refresh=True)

    assert cleared == ["openai"]


@pytest.mark.asyncio
async def test_refresh_is_debounced_within_the_window() -> None:
    """A refresh right after a recent dial is coalesced, not re-cleared/re-dialed."""
    get_model_cache().clear()
    config = _config({"openai": {"api_key": "x"}})
    # A freshly dialed listing (checked_at == now); a refresh within the window
    # must reuse it instead of clearing the cache and detaching single-flight.
    get_model_cache().set("openai", [_model("gpt-4o")])

    with (
        _spy_on_clear() as cleared,
        patch.object(phs, "discover_provider_models", side_effect=_noop_discover),
    ):
        await phs.check_provider_health(config, "openai", refresh=True)

    assert cleared == []  # debounced: the recent dial is reused


@pytest.mark.asyncio
async def test_refresh_redials_a_provider_never_checked() -> None:
    """A refresh for a provider with no cached dial re-dials (no debounce to apply)."""
    get_model_cache().clear()
    config = _config({"openai": {"api_key": "x"}})

    with (
        _spy_on_clear() as cleared,
        patch.object(phs, "discover_provider_models", side_effect=_noop_discover),
    ):
        await phs.check_provider_health(config, "openai", refresh=True)

    assert cleared == ["openai"]


@pytest.mark.asyncio
async def test_check_all_fans_out_and_summarizes() -> None:
    get_model_cache().clear()
    config = _config({"openai": {"api_key": "x"}, "anthropic": {"api_key": "y"}})

    async def discover(cfg: GatewayConfig, instance: str) -> ProviderDiscovery:
        if instance == "anthropic":
            return ProviderDiscovery(provider=instance, models=[], error="boom")
        return ProviderDiscovery(provider=instance, models=[_model("gpt-4o")])

    with patch.object(phs, "discover_provider_models", side_effect=discover):
        results = await phs.check_all_provider_health(config)

    by_instance = {item.instance: item for item in results}
    assert set(by_instance) == {"openai", "anthropic"}
    assert by_instance["openai"].ok is True
    assert by_instance["anthropic"].ok is False
    assert by_instance["anthropic"].error == "boom"


@pytest.mark.asyncio
async def test_check_all_surfaces_a_stray_exception_without_sinking_others() -> None:
    """A provider that somehow raises is reported as an error, not propagated."""
    get_model_cache().clear()
    config = _config({"good": {"api_key": "x"}, "bad": {"api_key": "y"}})

    async def discover(cfg: GatewayConfig, instance: str) -> ProviderDiscovery:
        if instance == "bad":
            raise RuntimeError("unexpected")
        return ProviderDiscovery(provider=instance, models=[_model("gpt-4o")])

    with patch.object(phs, "discover_provider_models", side_effect=discover):
        results = await phs.check_all_provider_health(config)

    by_instance = {item.instance: item for item in results}
    assert by_instance["good"].ok is True
    assert by_instance["bad"].ok is False
    assert by_instance["bad"].error is not None
