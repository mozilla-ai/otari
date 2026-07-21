"""Unit tests for the provider health service.

The service is a thin view over ``discover_provider_models`` (the existing
per-provider test path); these tests pin the mapping from a discovery result to a
``ProviderHealth``, the concurrent fan-out, the forced-refresh cache clear, and
the honest wall-clock ``checked_at`` read from the discovery cache.
"""

from datetime import datetime
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


@pytest.mark.asyncio
async def test_refresh_clears_the_cache_before_dialing() -> None:
    get_model_cache().clear()
    config = _config({"openai": {"api_key": "x"}})
    # Prime a stale cached listing; refresh must clear it before the new dial.
    get_model_cache().set("openai", [_model("stale")])

    cleared: list[str | None] = []
    real_clear = get_model_cache().clear

    def spy_clear(instance: str | None = None) -> None:
        cleared.append(instance)
        real_clear(instance)

    async def discover(cfg: GatewayConfig, instance: str) -> ProviderDiscovery:
        return ProviderDiscovery(provider=instance, models=[_model("gpt-4o")])

    with (
        patch.object(get_model_cache(), "clear", side_effect=spy_clear),
        patch.object(phs, "discover_provider_models", side_effect=discover),
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
