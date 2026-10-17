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
from gateway.services.model_discovery_service import DiscoveryDial, ProviderDiscovery, get_model_cache


def _config(providers: dict[str, dict[str, str]] | None = None) -> GatewayConfig:
    return GatewayConfig(providers=providers or {}, model_cache_ttl_seconds=300)


def _model(model_id: str) -> Model:
    return Model(id=model_id, object="model", created=1700000000, owned_by="openai")


@pytest.mark.asyncio
async def test_healthy_provider_reports_model_count_and_checked_at() -> None:
    get_model_cache().clear()
    config = _config({"openai": {"api_key": "x"}})

    async def discover(
        cfg: GatewayConfig, instance: str, *, serve_stale: bool = False, dial: object = None
    ) -> ProviderDiscovery:
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

    async def discover(
        cfg: GatewayConfig, instance: str, *, serve_stale: bool = False, dial: object = None
    ) -> ProviderDiscovery:
        return ProviderDiscovery(provider=instance, models=[], error="authentication failed")

    with patch.object(phs, "discover_provider_models", side_effect=discover):
        health = await phs.check_provider_health(config, "anthropic")

    assert health.ok is False
    assert health.model_count == 0
    assert health.error == "authentication failed"
    assert health.discovery_unsupported is False


@pytest.mark.asyncio
async def test_provider_without_a_models_endpoint_is_flagged_not_just_unhealthy() -> None:
    """A missing /v1/models is carried through so the dashboard can warn (otari#447)."""
    get_model_cache().clear()
    config = _config({"otari": {"api_key": "x"}})

    async def discover(
        cfg: GatewayConfig, instance: str, *, serve_stale: bool = False, dial: object = None
    ) -> ProviderDiscovery:
        return ProviderDiscovery(
            provider=instance,
            models=[],
            error="Error code: 404",
            discovery_unsupported=True,
        )

    with patch.object(phs, "discover_provider_models", side_effect=discover):
        health = await phs.check_provider_health(config, "otari")

    assert health.ok is False
    assert health.discovery_unsupported is True


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


async def _noop_discover(
    cfg: GatewayConfig, instance: str, *, serve_stale: bool = False, dial: object = None
) -> ProviderDiscovery:
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

    async def discover(
        cfg: GatewayConfig, instance: str, *, serve_stale: bool = False, dial: object = None
    ) -> ProviderDiscovery:
        if instance == "anthropic":
            return ProviderDiscovery(provider=instance, models=[], error="boom")
        return ProviderDiscovery(provider=instance, models=[_model("gpt-4o")])

    with patch.object(phs, "discover_provider_models", side_effect=discover):
        results = await phs.check_all_provider_health(config, hosted=None)

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

    async def discover(
        cfg: GatewayConfig, instance: str, *, serve_stale: bool = False, dial: object = None
    ) -> ProviderDiscovery:
        if instance == "bad":
            raise RuntimeError("unexpected")
        return ProviderDiscovery(provider=instance, models=[_model("gpt-4o")])

    with patch.object(phs, "discover_provider_models", side_effect=discover):
        results = await phs.check_all_provider_health(config, hosted=None)

    by_instance = {item.instance: item for item in results}
    assert by_instance["good"].ok is True
    assert by_instance["bad"].ok is False
    assert by_instance["bad"].error is not None


@pytest.mark.asyncio
async def test_polled_health_serves_the_cache_instead_of_dialing() -> None:
    """The hourly poll must not be what pays an unreachable provider's timeout.

    Health fans out over every configured instance, so a dial here costs
    ``model_discovery_timeout_seconds`` per unreachable provider and holds the
    request open for it. The background discovery refresher owns the dialing.
    """
    get_model_cache().clear()
    config = _config({"openai": {"api_key": "x"}})
    seen: list[bool] = []

    async def discover(
        cfg: GatewayConfig, instance: str, *, serve_stale: bool = False, dial: object = None
    ) -> ProviderDiscovery:
        seen.append(serve_stale)
        return ProviderDiscovery(provider=instance, models=[_model("gpt-4o")])

    with patch.object(phs, "discover_provider_models", side_effect=discover):
        await phs.check_all_provider_health(config, hosted=None, serve_stale=True)

    assert seen == [True]


@pytest.mark.asyncio
async def test_explicit_recheck_still_dials() -> None:
    """An explicit "re-check now" is an operator asking for a fresh probe; honor it."""
    get_model_cache().clear()
    config = _config({"openai": {"api_key": "x"}})
    seen: list[bool] = []

    async def discover(
        cfg: GatewayConfig, instance: str, *, serve_stale: bool = False, dial: object = None
    ) -> ProviderDiscovery:
        seen.append(serve_stale)
        return ProviderDiscovery(provider=instance, models=[_model("gpt-4o")])

    with patch.object(phs, "discover_provider_models", side_effect=discover):
        await phs.check_all_provider_health(config, hosted=None, refresh=True, serve_stale=True)

    # refresh wins over serve_stale, or the button would return the cached verdict.
    assert seen == [False]


@pytest.mark.asyncio
async def test_check_all_dials_a_hosted_provider_on_its_own_dial() -> None:
    """A hosted provider is checked beside the configured instances, on the dial the providers domain hands over."""
    get_model_cache().clear()
    config = _config({"openai": {"api_key": "x"}})
    dialed: list[str] = []

    async def mistral() -> ProviderDiscovery:
        dialed.append("mistral")
        return ProviderDiscovery(provider="mistral", models=[_model("mistral-large"), _model("mistral-small")])

    async def hosted() -> dict[str, DiscoveryDial]:
        return {"mistral": mistral}

    async def configured(cfg: GatewayConfig, instance: str) -> ProviderDiscovery:
        return ProviderDiscovery(provider=instance, models=[_model("gpt-4o")])

    with patch("gateway.services.model_discovery_service._discover_uncached", side_effect=configured):
        results = await phs.check_all_provider_health(config, hosted=hosted)

    by_instance = {item.instance: item for item in results}
    assert set(by_instance) == {"openai", "mistral"}
    assert by_instance["mistral"].ok is True
    assert by_instance["mistral"].model_count == 2
    assert by_instance["mistral"].checked_at is not None
    assert dialed == ["mistral"]


@pytest.mark.asyncio
async def test_a_configured_instance_shadows_the_hosted_provider_of_its_name() -> None:
    """The instance serves every request naming it, so its health is the one to report."""
    get_model_cache().clear()
    config = _config({"openai": {"api_key": "x"}})

    async def hosted_openai() -> ProviderDiscovery:
        raise AssertionError("a shadowed hosted provider must not be dialed")

    async def hosted() -> dict[str, DiscoveryDial]:
        return {"openai": hosted_openai}

    async def configured(cfg: GatewayConfig, instance: str) -> ProviderDiscovery:
        return ProviderDiscovery(provider=instance, models=[_model("gpt-4o")])

    with patch("gateway.services.model_discovery_service._discover_uncached", side_effect=configured):
        results = await phs.check_all_provider_health(config, hosted=hosted)

    assert [(item.instance, item.ok) for item in results] == [("openai", True)]
