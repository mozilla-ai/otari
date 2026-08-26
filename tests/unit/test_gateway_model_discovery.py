"""Unit tests for the model discovery cache and service."""

import asyncio
import time
from collections.abc import Callable
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from any_llm.exceptions import ModelNotFoundError
from any_llm.types.model import Model

from gateway.core.config import GatewayConfig
from gateway.services.model_discovery_service import (
    _ERROR_MAX_CHARS,
    _MIN_REFRESH_INTERVAL_SECONDS,
    ModelCache,
    ProviderDiscovery,
    _CacheEntry,
    _is_missing_models_endpoint,
    _refresh_interval,
    _short_error,
    _supports_list_models,
    background_discovery_enabled,
    discover_all_models,
    discover_models_with_status,
    discover_provider_models,
    refresh_discovery_cache,
    run_discovery_refresher,
)
from gateway.services.model_discovery_service import (
    test_provider_credentials as run_credentials_test,  # aliased so pytest does not collect it
)
from gateway.services.provider_kwargs import _KEYLESS_PLACEHOLDER_API_KEY


def _make_model(model_id: str, owned_by: str = "openai", created: int = 1700000000) -> Model:
    """Create a minimal Model instance for testing."""
    return Model(id=model_id, object="model", created=created, owned_by=owned_by)


# ---------------------------------------------------------------------------
# ModelCache tests
# ---------------------------------------------------------------------------


class TestShortError:
    def test_strips_matching_provider_tag(self) -> None:
        # any-llm prefixes provider errors with "[provider] "; it is redundant
        # in the dashboard's provider-specific surfaces, so it is dropped.
        exc = ValueError("[anthropic] No anthropic API key provided.")
        assert _short_error(exc, provider="anthropic") == "No anthropic API key provided."

    def test_keeps_tag_without_provider(self) -> None:
        exc = ValueError("[anthropic] No anthropic API key provided.")
        assert _short_error(exc) == "[anthropic] No anthropic API key provided."

    def test_keeps_tag_for_a_different_provider(self) -> None:
        # Only the provider being tested is stripped; any other bracketed text stays.
        exc = ValueError("[anthropic] boom")
        assert _short_error(exc, provider="openai") == "[anthropic] boom"

    def test_falls_back_to_class_name_for_empty_message(self) -> None:
        assert _short_error(ValueError()) == "ValueError"

    def test_falls_back_to_class_name_when_message_is_only_the_tag(self) -> None:
        # Stripping the tag off a bare "[anthropic]" (or "[anthropic]   ") must not
        # leave an empty string, which would render as a blank error.
        assert _short_error(ValueError("[anthropic]"), provider="anthropic") == "ValueError"
        assert _short_error(ValueError("[anthropic]   "), provider="anthropic") == "ValueError"

    def test_caps_long_message_after_stripping_tag(self) -> None:
        # Words rather than one long run of characters: redaction masks an
        # unbroken alphanumeric run as key material, which would shorten the
        # message before the cap this asserts could apply.
        body = " ".join(["boom"] * 80)
        result = _short_error(ValueError(f"[anthropic] {body}"), provider="anthropic")
        assert len(result) == _ERROR_MAX_CHARS
        assert result.endswith("\u2026")
        # The tag was removed before the cap, so none of it survives.
        assert not result.startswith("[anthropic]")

    def test_masks_a_credential_in_the_upstream_message(self) -> None:
        # otari-ai#1880: this text comes from a call made with the deployment's
        # own credential and lands in a provider health or test response, so it
        # goes through the same redaction a client-facing provider error does.
        exc = ValueError("[openai] Incorrect API key provided: sk-proj-abcd1234efgh5678ijkl")
        result = _short_error(exc, provider="openai")
        assert "sk-proj-abcd1234efgh5678ijkl" not in result
        assert "Incorrect API key provided" in result

    def test_masks_a_self_hosted_api_base(self) -> None:
        exc = ValueError("Connection refused to https://llm.internal.example.com/v1/models")
        assert "llm.internal.example.com" not in _short_error(exc)

    def test_falls_back_to_class_name_when_redaction_rejects_the_message(self) -> None:
        # A message that echoes the request back is rejected whole rather than
        # masked, which would otherwise render as a blank error in the dashboard.
        exc = ValueError('messages: [{"role": "user", "content": "my private prompt"}]')
        assert _short_error(exc) == "ValueError"


class TestModelCache:
    def test_get_returns_none_on_empty_cache(self) -> None:
        cache = ModelCache()
        assert cache.get("openai", ttl=300) is None

    def test_set_and_get_returns_cached_models(self) -> None:
        cache = ModelCache()
        models = [_make_model("gpt-4o"), _make_model("gpt-4o-mini")]
        cache.set("openai", models)

        result = cache.get("openai", ttl=300)
        assert result is not None
        assert len(result) == 2
        assert result[0].id == "gpt-4o"

    def test_get_returns_none_when_ttl_zero(self) -> None:
        cache = ModelCache()
        cache.set("openai", [_make_model("gpt-4o")])
        assert cache.get("openai", ttl=0) is None

    def test_cache_expires_after_ttl(self) -> None:
        cache = ModelCache()
        cache.set("openai", [_make_model("gpt-4o")])

        # Manually backdate the cached_at timestamp.
        entry = cache._store["openai"]
        entry.cached_at = time.monotonic() - 400

        assert cache.get("openai", ttl=300) is None

    def test_cache_still_valid_before_ttl(self) -> None:
        cache = ModelCache()
        cache.set("openai", [_make_model("gpt-4o")])

        result = cache.get("openai", ttl=300)
        assert result is not None
        assert len(result) == 1

    def test_clear_specific_provider(self) -> None:
        cache = ModelCache()
        cache.set("openai", [_make_model("gpt-4o")])
        cache.set("anthropic", [_make_model("claude-3-opus")])

        cache.clear("openai")
        assert cache.get("openai", ttl=300) is None
        assert cache.get("anthropic", ttl=300) is not None

    def test_clear_all_providers(self) -> None:
        cache = ModelCache()
        cache.set("openai", [_make_model("gpt-4o")])
        cache.set("anthropic", [_make_model("claude-3-opus")])

        cache.clear()
        assert cache.get("openai", ttl=300) is None
        assert cache.get("anthropic", ttl=300) is None

    def test_get_returns_shallow_copy(self) -> None:
        """Mutating the returned list should not affect the cache."""
        cache = ModelCache()
        cache.set("openai", [_make_model("gpt-4o")])

        returned = cache.get("openai", ttl=300)
        assert returned is not None
        returned.append(_make_model("gpt-3.5"))

        # Internal cache should still have only 1 model.
        assert len(cache._store["openai"].result.models) == 1

    def test_set_copies_list(self) -> None:
        """Mutating the original list should not affect the cached copy."""
        cache = ModelCache()
        models = [_make_model("gpt-4o")]
        cache.set("openai", models)
        models.append(_make_model("gpt-3.5"))

        cached = cache.get("openai", ttl=300)
        assert cached is not None
        assert len(cached) == 1


# ---------------------------------------------------------------------------
# _supports_list_models tests
# ---------------------------------------------------------------------------


class TestSupportsListModels:
    def test_returns_true_for_supported_provider(self) -> None:
        metadata = MagicMock()
        metadata.list_models = True
        provider_class = MagicMock()
        provider_class.get_provider_metadata.return_value = metadata

        with patch("gateway.services.model_discovery_service.AnyLLM") as mock_any:
            mock_any.get_provider_class.return_value = provider_class
            assert _supports_list_models("openai") is True

    def test_returns_false_for_unsupported_provider(self) -> None:
        metadata = MagicMock()
        metadata.list_models = False
        provider_class = MagicMock()
        provider_class.get_provider_metadata.return_value = metadata

        with patch("gateway.services.model_discovery_service.AnyLLM") as mock_any:
            mock_any.get_provider_class.return_value = provider_class
            assert _supports_list_models("sagemaker") is False

    def test_returns_false_on_import_error(self) -> None:
        with patch("gateway.services.model_discovery_service.AnyLLM") as mock_any:
            mock_any.get_provider_class.side_effect = ImportError("no such provider")
            assert _supports_list_models("nonexistent") is False


# ---------------------------------------------------------------------------
# discover_all_models tests
# ---------------------------------------------------------------------------


class TestDiscoverAllModels:
    def _make_config(
        self, providers: dict[str, Any] | None = None, discovery: bool = True, ttl: int = 300
    ) -> GatewayConfig:
        return GatewayConfig(
            providers=providers or {},
            model_discovery=discovery,
            model_cache_ttl_seconds=ttl,
        )

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_providers(self) -> None:
        config = self._make_config()
        with patch("gateway.services.model_discovery_service.get_model_cache") as mock_cache:
            mock_cache.return_value = ModelCache()
            result = await discover_all_models(config)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_provider_qualified_tuples(self) -> None:
        """discover_all_models returns (provider_name, Model) tuples."""
        config = self._make_config(providers={"openai": {"api_key": "sk-test"}})
        expected_models = [_make_model("gpt-4o")]

        with (
            patch("gateway.services.model_discovery_service.get_model_cache") as mock_cache_fn,
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch(
                "gateway.services.model_discovery_service.alist_models",
                new_callable=AsyncMock,
                return_value=expected_models,
            ),
            patch("gateway.services.model_discovery_service.get_provider_kwargs", return_value={"api_key": "sk-test"}),
        ):
            cache = ModelCache()
            mock_cache_fn.return_value = cache
            result = await discover_all_models(config)

        assert len(result) == 1
        provider_name, model = result[0]
        assert provider_name == "openai"
        assert model.id == "gpt-4o"

    @pytest.mark.asyncio
    async def test_discovers_models_from_provider(self) -> None:
        config = self._make_config(providers={"openai": {"api_key": "sk-test"}})
        expected_models = [_make_model("gpt-4o"), _make_model("gpt-4o-mini")]

        with (
            patch("gateway.services.model_discovery_service.get_model_cache") as mock_cache_fn,
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch(
                "gateway.services.model_discovery_service.alist_models",
                new_callable=AsyncMock,
                return_value=expected_models,
            ),
            patch("gateway.services.model_discovery_service.get_provider_kwargs", return_value={"api_key": "sk-test"}),
        ):
            cache = ModelCache()
            mock_cache_fn.return_value = cache
            result = await discover_all_models(config)

        assert len(result) == 2
        assert result[0][0] == "openai"
        assert result[0][1].id == "gpt-4o"

    @pytest.mark.asyncio
    async def test_uses_cache_on_hit(self) -> None:
        config = self._make_config(providers={"openai": {"api_key": "sk-test"}})
        cached_models = [_make_model("gpt-4o")]

        cache = ModelCache()
        cache.set("openai", cached_models)

        with (
            patch("gateway.services.model_discovery_service.get_model_cache", return_value=cache),
            patch("gateway.services.model_discovery_service._supports_list_models") as mock_supports,
            patch("gateway.services.model_discovery_service.alist_models") as mock_alist,
        ):
            result = await discover_all_models(config)

        assert len(result) == 1
        assert result[0][0] == "openai"
        assert result[0][1].id == "gpt-4o"
        mock_supports.assert_not_called()
        mock_alist.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_unsupported_providers(self) -> None:
        config = self._make_config(providers={"sagemaker": {"region": "us-east-1"}})

        with (
            patch("gateway.services.model_discovery_service.get_model_cache") as mock_cache_fn,
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=False),
            patch("gateway.services.model_discovery_service.alist_models") as mock_alist,
        ):
            mock_cache_fn.return_value = ModelCache()
            result = await discover_all_models(config)

        assert result == []
        mock_alist.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_provider_failure_gracefully(self) -> None:
        config = self._make_config(
            providers={
                "openai": {"api_key": "sk-test"},
                "mistral": {"api_key": "mk-test"},
            }
        )

        openai_models = [_make_model("gpt-4o")]

        async def mock_alist(provider: Any, **kwargs: Any) -> list[Model]:
            if provider.value == "openai":
                return openai_models
            raise ConnectionError("upstream unreachable")

        with (
            patch("gateway.services.model_discovery_service.get_model_cache") as mock_cache_fn,
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch("gateway.services.model_discovery_service.alist_models", side_effect=mock_alist),
            patch(
                "gateway.services.model_discovery_service.get_provider_kwargs",
                return_value={"api_key": "test"},
            ),
        ):
            mock_cache_fn.return_value = ModelCache()
            result = await discover_all_models(config)

        # Only openai models should be returned; mistral failure is swallowed.
        assert len(result) == 1
        assert result[0][0] == "openai"
        assert result[0][1].id == "gpt-4o"

    @pytest.mark.asyncio
    async def test_provider_filter_limits_query(self) -> None:
        config = self._make_config(
            providers={
                "openai": {"api_key": "sk-test"},
                "anthropic": {"api_key": "ak-test"},
            }
        )
        openai_models = [_make_model("gpt-4o")]

        with (
            patch("gateway.services.model_discovery_service.get_model_cache") as mock_cache_fn,
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch(
                "gateway.services.model_discovery_service.alist_models",
                new_callable=AsyncMock,
                return_value=openai_models,
            ) as mock_alist,
            patch("gateway.services.model_discovery_service.get_provider_kwargs", return_value={"api_key": "test"}),
        ):
            mock_cache_fn.return_value = ModelCache()
            result = await discover_all_models(config, provider_filter="openai")

        assert len(result) == 1
        # alist_models should only be called once (for openai).
        assert mock_alist.await_count == 1

    @pytest.mark.asyncio
    async def test_provider_filter_nonexistent_returns_empty(self) -> None:
        config = self._make_config(providers={"openai": {"api_key": "sk-test"}})

        with patch("gateway.services.model_discovery_service.get_model_cache") as mock_cache_fn:
            mock_cache_fn.return_value = ModelCache()
            result = await discover_all_models(config, provider_filter="nonexistent")

        assert result == []

    @pytest.mark.asyncio
    async def test_named_instance_lists_under_instance_name(self) -> None:
        """A custom instance lists its models keyed on the instance, not the impl."""
        config = self._make_config(
            providers={"home_lab": {"provider_type": "openai", "api_base": "http://box/v1", "api_key": "ht"}}
        )
        captured: dict[str, Any] = {}

        async def mock_alist(provider: Any, **kwargs: Any) -> list[Model]:
            captured["provider"] = provider
            captured["api_base"] = kwargs.get("api_base")
            return [_make_model("deepseek-v4-flash", owned_by="openai")]

        with (
            patch("gateway.services.model_discovery_service.get_model_cache") as mock_cache_fn,
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch("gateway.services.model_discovery_service.alist_models", side_effect=mock_alist),
        ):
            mock_cache_fn.return_value = ModelCache()
            result = await discover_all_models(config)

        # any-llm is queried as the implementation, against the instance's api_base.
        assert captured["provider"].value == "openai"
        assert captured["api_base"] == "http://box/v1"
        # ...but the result is keyed on the instance name (so model_key is home_lab:...).
        assert result == [("home_lab", result[0][1])]
        assert result[0][1].id == "deepseek-v4-flash"

    @pytest.mark.asyncio
    async def test_declared_models_used_when_listing_unsupported(self) -> None:
        """An instance whose backend has no /v1/models serves its declared models: list."""
        config = self._make_config(
            providers={"edge": {"provider_type": "openai", "api_base": "http://edge/v1", "models": ["m1", "m2"]}}
        )

        with (
            patch("gateway.services.model_discovery_service.get_model_cache") as mock_cache_fn,
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=False),
            patch("gateway.services.model_discovery_service.alist_models") as mock_alist,
        ):
            mock_cache_fn.return_value = ModelCache()
            result = await discover_all_models(config)

        mock_alist.assert_not_called()
        assert sorted(model.id for _, model in result) == ["m1", "m2"]
        assert all(name == "edge" for name, _ in result)

    @pytest.mark.asyncio
    async def test_declared_models_fallback_on_list_failure(self) -> None:
        """A list_models failure falls back to the declared models: list when present."""
        config = self._make_config(
            providers={"edge": {"provider_type": "openai", "api_base": "http://edge/v1", "models": ["m1"]}}
        )

        async def mock_alist(provider: Any, **kwargs: Any) -> list[Model]:
            raise ConnectionError("no /v1/models on this backend")

        with (
            patch("gateway.services.model_discovery_service.get_model_cache") as mock_cache_fn,
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch("gateway.services.model_discovery_service.alist_models", side_effect=mock_alist),
        ):
            mock_cache_fn.return_value = ModelCache()
            result = await discover_all_models(config)

        assert result == [("edge", result[0][1])]
        assert result[0][1].id == "m1"


class TestDiscoveryStallGuards:
    """Regressions for the discovery stall / over-calling that hung a DELETE.

    Several defects compounded: a live ``list_models`` call had no timeout, so an
    unreachable provider blocked for the client's default (~60s); the cache lock
    was held across the whole fanout, serializing consumers; failures were never
    cached, so a broken provider was re-dialed on every request; and concurrent
    consumers each fired their own call. The fix bounds the call, caches failures
    (negative TTL), and single-flights concurrent discoveries of one provider.
    """

    def _config(
        self,
        providers: dict[str, Any],
        timeout: float = 10.0,
        negative_ttl: float = 30.0,
    ) -> GatewayConfig:
        return GatewayConfig(
            providers=providers,
            model_discovery=True,
            model_cache_ttl_seconds=300,
            model_discovery_timeout_seconds=timeout,
            model_discovery_negative_ttl_seconds=negative_ttl,
        )

    @pytest.mark.asyncio
    async def test_unreachable_provider_bounded_by_timeout(self) -> None:
        """A provider whose list_models never returns is dropped after the timeout.

        The whole call is wrapped in a generous 2s guard: if the per-provider
        timeout regressed, the inner 5s sleep would trip that guard and fail the
        test instead of hanging the suite.
        """
        config = self._config({"openai": {"api_key": "sk-test"}}, timeout=0.05)

        async def never_returns(provider: Any, **kwargs: Any) -> list[Model]:
            await asyncio.sleep(5)
            return [_make_model("unreachable")]

        with (
            patch("gateway.services.model_discovery_service.get_model_cache", return_value=ModelCache()),
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch("gateway.services.model_discovery_service.alist_models", side_effect=never_returns),
            patch("gateway.services.model_discovery_service.get_provider_kwargs", return_value={"api_key": "sk-test"}),
        ):
            result = await asyncio.wait_for(discover_all_models(config), timeout=2)

        # No declared models: fallback, so the timed-out provider is dropped.
        assert result == []

    @pytest.mark.asyncio
    async def test_slow_provider_does_not_block_concurrent_discovery(self) -> None:
        """The cache lock is not held across the network fanout.

        Two concurrent ``discover_all_models`` calls share one cache (hence one
        lock). The slow call is started first, so under the old code (lock held
        across the fetch) it would acquire the lock and pin the fast call behind
        it, completing first. With the fix the fast call finishes first.
        """
        cache = ModelCache()
        order: list[str] = []

        async def fake_discover(name: str, cfg: GatewayConfig) -> tuple[str, list[Model]]:
            if name == "slow":
                await asyncio.sleep(0.3)
            order.append(name)
            return name, [_make_model(f"{name}-m", owned_by=name)]

        slow_cfg = self._config({"slow": {"api_key": "x"}})
        fast_cfg = self._config({"fast": {"api_key": "y"}})

        with (
            patch("gateway.services.model_discovery_service.get_model_cache", return_value=cache),
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch("gateway.services.model_discovery_service._discover_for_provider", side_effect=fake_discover),
        ):
            slow_task = asyncio.create_task(discover_all_models(slow_cfg))
            fast_task = asyncio.create_task(discover_all_models(fast_cfg))
            await asyncio.gather(slow_task, fast_task)

        assert order[0] == "fast"

    @pytest.mark.asyncio
    async def test_failure_is_negatively_cached(self) -> None:
        """A failed provider is remembered, not re-dialed on every call."""
        config = self._config({"openai": {"api_key": "sk-test"}}, negative_ttl=30.0)
        cache = ModelCache()
        calls = 0

        async def failing(provider: Any, **kwargs: Any) -> list[Model]:
            nonlocal calls
            calls += 1
            raise ConnectionError("unreachable")

        with (
            patch("gateway.services.model_discovery_service.get_model_cache", return_value=cache),
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch("gateway.services.model_discovery_service.alist_models", side_effect=failing),
            patch("gateway.services.model_discovery_service.get_provider_kwargs", return_value={"api_key": "sk-test"}),
        ):
            first = await discover_provider_models(config, "openai")
            second = await discover_provider_models(config, "openai")
            assert calls == 1  # the negative cache prevented a second dial
            assert first.error is not None
            assert second.error is not None

            # Expire the negative entry; the next call re-dials.
            cache._store["openai"].cached_at -= 31
            await discover_provider_models(config, "openai")
            assert calls == 2

    @pytest.mark.asyncio
    async def test_negative_ttl_zero_disables_caching(self) -> None:
        """Setting the negative TTL to 0 restores retry-every-time."""
        config = self._config({"openai": {"api_key": "sk-test"}}, negative_ttl=0.0)
        cache = ModelCache()
        calls = 0

        async def failing(provider: Any, **kwargs: Any) -> list[Model]:
            nonlocal calls
            calls += 1
            raise ConnectionError("unreachable")

        with (
            patch("gateway.services.model_discovery_service.get_model_cache", return_value=cache),
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch("gateway.services.model_discovery_service.alist_models", side_effect=failing),
            patch("gateway.services.model_discovery_service.get_provider_kwargs", return_value={"api_key": "sk-test"}),
        ):
            await discover_provider_models(config, "openai")
            await discover_provider_models(config, "openai")

        assert calls == 2

    @pytest.mark.asyncio
    async def test_concurrent_callers_share_one_upstream_call(self) -> None:
        """Concurrent discoveries of the same provider dial it once (single-flight).

        This is what stops /v1/models and /v1/models/discoverable from each firing
        a full fanout when the Models page mounts both at once.
        """
        config = self._config({"openai": {"api_key": "sk-test"}})
        cache = ModelCache()
        calls = 0

        async def slow_list(provider: Any, **kwargs: Any) -> list[Model]:
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.2)
            return [_make_model("gpt-4o")]

        with (
            patch("gateway.services.model_discovery_service.get_model_cache", return_value=cache),
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch("gateway.services.model_discovery_service.alist_models", side_effect=slow_list),
            patch("gateway.services.model_discovery_service.get_provider_kwargs", return_value={"api_key": "sk-test"}),
        ):
            a, b = await asyncio.gather(
                discover_provider_models(config, "openai"),
                discover_provider_models(config, "openai"),
            )

        assert calls == 1
        assert [m.id for m in a.models] == ["gpt-4o"]
        assert [m.id for m in b.models] == ["gpt-4o"]

    @pytest.mark.asyncio
    async def test_clear_during_inflight_forces_fresh_discovery(self) -> None:
        """clear() detaches an in-flight discovery so a later caller re-fetches.

        Models a credential change (or a Test) landing while a discovery is in
        flight: the stale in-flight result must neither be served to nor cached
        for callers that arrive after the clear, so the post-write "test
        connection" stays a live check.
        """
        cache = ModelCache()
        calls = 0

        async def stale() -> ProviderDiscovery:
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.2)
            return ProviderDiscovery(provider="p", models=[], error="OLD")

        async def fresh() -> ProviderDiscovery:
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.02)
            return ProviderDiscovery(provider="p", models=[], error="NEW")

        inflight = asyncio.ensure_future(
            cache.get_or_discover("p", positive_ttl=300, negative_ttl=30, discover=stale)
        )
        await asyncio.sleep(0.02)  # let the stale discovery register as in-flight
        cache.clear("p")  # a credential change invalidates it
        late = await cache.get_or_discover("p", positive_ttl=300, negative_ttl=30, discover=fresh)
        await inflight

        assert late.error == "NEW"  # did not ride the stale in-flight
        assert cache._store["p"].result.error == "NEW"  # stale result did not repopulate
        assert calls == 2

    @pytest.mark.asyncio
    async def test_returned_discovery_is_isolated_from_cache(self) -> None:
        """Mutating a returned discovery must not corrupt the cached entry."""
        cache = ModelCache()

        async def disc() -> ProviderDiscovery:
            return ProviderDiscovery(provider="p", models=[_make_model("a")], error=None)

        first = await cache.get_or_discover("p", positive_ttl=300, negative_ttl=30, discover=disc)
        first.models.append(_make_model("b"))
        first.error = "tampered"

        second = await cache.get_or_discover("p", positive_ttl=300, negative_ttl=30, discover=disc)
        assert [m.id for m in second.models] == ["a"]
        assert second.error is None

    @pytest.mark.asyncio
    async def test_one_provider_raising_does_not_sink_the_listing(self) -> None:
        """A provider whose discovery raises is dropped/surfaced, never propagated.

        discover_models_with_status feeds the operator's /v1/models/discoverable,
        which awaits it with no guard, so an escaped exception must not 500 it.
        """
        config = self._config({"good": {"api_key": "x"}, "bad": {"api_key": "y"}})

        async def flaky(cfg: GatewayConfig, instance: str) -> ProviderDiscovery:
            if instance == "bad":
                raise RuntimeError("boom")
            return ProviderDiscovery(provider=instance, models=[_make_model("m", owned_by=instance)])

        from gateway.services import model_discovery_service as mds

        with (
            patch.object(mds, "get_model_cache", return_value=ModelCache()),
            patch.object(mds, "_discover_uncached", side_effect=flaky),
        ):
            all_models = await discover_all_models(config)
            statuses = await discover_models_with_status(config)

        assert [name for name, _ in all_models] == ["good"]  # bad dropped from catalog
        by_provider = {d.provider: d for d in statuses}
        assert by_provider["good"].error is None
        assert by_provider["bad"].error is not None  # surfaced as an error, not raised


class TestKeylessProviderConnection:
    """test_provider_credentials must honor the optional key for custom endpoints (otari#421)."""

    @pytest.mark.asyncio
    async def test_keyless_custom_endpoint_supplies_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A keyless "Test connection" for a custom endpoint would otherwise be
        # rejected by any-llm with MissingApiKeyError; the ad-hoc test path injects
        # the same placeholder the saved path uses so the endpoint is dialed.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        mock_alist = AsyncMock(return_value=[_make_model("local-model")])
        with (
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch("gateway.services.model_discovery_service.alist_models", mock_alist),
        ):
            result = await run_credentials_test("openai", api_key=None, api_base="http://localhost:8000/v1")

        assert result.error is None
        assert [m.id for m in result.models] == ["local-model"]
        assert mock_alist.await_args is not None
        assert mock_alist.await_args.kwargs["api_key"] == _KEYLESS_PLACEHOLDER_API_KEY

    @pytest.mark.asyncio
    async def test_explicit_key_is_used_verbatim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        mock_alist = AsyncMock(return_value=[])
        with (
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch("gateway.services.model_discovery_service.alist_models", mock_alist),
        ):
            await run_credentials_test("openai", api_key="sk-real", api_base="http://localhost:8000/v1")

        assert mock_alist.await_args is not None
        assert mock_alist.await_args.kwargs["api_key"] == "sk-real"


class TestProviderApiBaseSsrfGate:
    """Opt-in SSRF gate for the operator-supplied provider api_base (otari#316).

    Default is allow-all (home-lab keeps working); when an operator sets
    OTARI_PROVIDER_ALLOW_PRIVATE_HOSTS=false the internal api_base is refused
    before the endpoint is dialed, on both the ad-hoc test path and runtime
    discovery.
    """

    def _make_config(self, providers: dict[str, Any]) -> GatewayConfig:
        return GatewayConfig(providers=providers, model_discovery=True, model_cache_ttl_seconds=300)

    @pytest.mark.asyncio
    async def test_credentials_test_allows_internal_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTARI_PROVIDER_ALLOW_PRIVATE_HOSTS", raising=False)
        mock_alist = AsyncMock(return_value=[_make_model("local-model")])
        with (
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch("gateway.services.model_discovery_service.alist_models", mock_alist),
        ):
            result = await run_credentials_test("openai", api_key="sk", api_base="http://127.0.0.1:8000/v1")

        assert result.error is None
        assert mock_alist.await_count == 1

    @pytest.mark.asyncio
    async def test_credentials_test_blocks_internal_when_gated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OTARI_PROVIDER_ALLOW_PRIVATE_HOSTS", "false")
        mock_alist = AsyncMock(return_value=[_make_model("local-model")])
        with (
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch("gateway.services.model_discovery_service.alist_models", mock_alist),
        ):
            result = await run_credentials_test("openai", api_key="sk", api_base="http://127.0.0.1:8000/v1")

        # Blocked before the endpoint is dialed, surfaced as a test failure.
        assert result.error is not None
        assert "loopback" in result.error
        mock_alist.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_discovery_allows_internal_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTARI_PROVIDER_ALLOW_PRIVATE_HOSTS", raising=False)
        config = self._make_config(
            {"home_lab": {"provider_type": "openai", "api_base": "http://10.0.0.5/v1", "api_key": "ht"}}
        )
        with (
            patch("gateway.services.model_discovery_service.get_model_cache") as mock_cache_fn,
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch(
                "gateway.services.model_discovery_service.alist_models",
                new_callable=AsyncMock,
                return_value=[_make_model("m1")],
            ),
        ):
            mock_cache_fn.return_value = ModelCache()
            result = await discover_provider_models(config, "home_lab")

        assert result.error is None
        assert [m.id for m in result.models] == ["m1"]

    @pytest.mark.asyncio
    async def test_discovery_blocks_internal_when_gated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OTARI_PROVIDER_ALLOW_PRIVATE_HOSTS", "false")
        config = self._make_config(
            {"home_lab": {"provider_type": "openai", "api_base": "http://10.0.0.5/v1", "api_key": "ht"}}
        )
        mock_alist = AsyncMock(return_value=[_make_model("m1")])
        with (
            patch("gateway.services.model_discovery_service.get_model_cache") as mock_cache_fn,
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch("gateway.services.model_discovery_service.alist_models", mock_alist),
        ):
            mock_cache_fn.return_value = ModelCache()
            result = await discover_provider_models(config, "home_lab")

        assert result.models == []
        assert result.error is not None
        assert "private" in result.error
        mock_alist.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_gate_on_does_not_affect_provider_without_api_base(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A standard provider (no api_base, SDK default endpoint) is dialed even when gated."""
        monkeypatch.setenv("OTARI_PROVIDER_ALLOW_PRIVATE_HOSTS", "false")
        config = self._make_config({"openai": {"api_key": "sk-test"}})
        with (
            patch("gateway.services.model_discovery_service.get_model_cache") as mock_cache_fn,
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch(
                "gateway.services.model_discovery_service.alist_models",
                new_callable=AsyncMock,
                return_value=[_make_model("gpt-4o")],
            ) as mock_alist,
        ):
            mock_cache_fn.return_value = ModelCache()
            result = await discover_provider_models(config, "openai")

        assert result.error is None
        assert [m.id for m in result.models] == ["gpt-4o"]
        mock_alist.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_discovery_block_skips_declared_fallback_when_gated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A blocked api_base fails outright; it must not fall back to declared models."""
        monkeypatch.setenv("OTARI_PROVIDER_ALLOW_PRIVATE_HOSTS", "false")
        config = self._make_config(
            {"edge": {"provider_type": "openai", "api_base": "http://10.0.0.5/v1", "models": ["declared-1"]}}
        )
        with (
            patch("gateway.services.model_discovery_service.get_model_cache") as mock_cache_fn,
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch("gateway.services.model_discovery_service.alist_models", new_callable=AsyncMock) as mock_alist,
        ):
            mock_cache_fn.return_value = ModelCache()
            result = await discover_provider_models(config, "edge")

        assert result.models == []
        assert result.error is not None
        mock_alist.assert_not_awaited()


class TestMissingModelsEndpoint:
    """A backend with no /v1/models is a discovery gap, not an unusable provider.

    otari.ai shipped without a model-listing endpoint, which made the dashboard
    report a perfectly good key as unreachable (otari#447). Discovery now
    distinguishes "this deployment serves no listing" from "these credentials do
    not work", so the dashboard can warn instead of condemning the provider.
    """

    def _make_config(self, providers: dict[str, Any]) -> GatewayConfig:
        return GatewayConfig(providers=providers, model_discovery=True, model_cache_ttl_seconds=300)

    @staticmethod
    def _status_error(status: int, message: str = "Not Found") -> Exception:
        """An OpenAI-SDK-shaped error: the status lives on the exception."""
        exc = RuntimeError(message)
        exc.status_code = status  # type: ignore[attr-defined]
        return exc

    async def _discover(self, exc: Exception) -> ProviderDiscovery:
        config = self._make_config({"otari": {"provider_type": "openai", "api_key": "sk-test"}})
        with (
            patch("gateway.services.model_discovery_service.get_model_cache") as mock_cache_fn,
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch("gateway.services.model_discovery_service.alist_models", side_effect=exc),
            patch("gateway.services.model_discovery_service.get_provider_kwargs", return_value={"api_key": "sk-test"}),
        ):
            mock_cache_fn.return_value = ModelCache()
            return await discover_provider_models(config, "otari")

    @pytest.mark.asyncio
    async def test_404_marks_discovery_unsupported_not_unreachable(self) -> None:
        result = await self._discover(self._status_error(404))

        assert result.models == []
        assert result.error is not None
        assert result.discovery_unsupported is True

    @pytest.mark.asyncio
    async def test_httpx_shaped_404_is_recognized(self) -> None:
        # httpx-style clients carry the status on the response, not the exception.
        exc = RuntimeError("Not Found")
        exc.response = SimpleNamespace(status_code=404)  # type: ignore[attr-defined]

        assert (await self._discover(exc)).discovery_unsupported is True

    @pytest.mark.asyncio
    async def test_any_llm_wrapped_404_is_recognized(self) -> None:
        # any-llm keeps the SDK exception on original_exception.
        wrapper = RuntimeError("[openai] Not Found")
        wrapper.original_exception = self._status_error(404)  # type: ignore[attr-defined]

        assert (await self._discover(wrapper)).discovery_unsupported is True

    def test_unified_any_llm_exception_shape_is_recognized(self) -> None:
        """Pin the shape any-llm produces under ANY_LLM_UNIFIED_EXCEPTIONS.

        That flag replaces the SDK error with ``ModelNotFoundError``, which carries
        no ``status_code`` of its own and keeps the original on
        ``original_exception`` (and as ``__cause__``). It is slated to become the
        default, so losing this hop would silently turn every missing-listing
        provider back into "unreachable".
        """
        original = self._status_error(404)
        unified = ModelNotFoundError("Model not found", original_exception=original, provider_name="openai")

        assert getattr(unified, "status_code", None) is None
        assert _is_missing_models_endpoint(unified) is True

    def test_doubly_wrapped_status_is_still_found(self) -> None:
        # A wrapper around a wrapper, and a chain built with `raise ... from`,
        # both still resolve rather than falling through to "unreachable".
        inner = self._status_error(404)
        middle = RuntimeError("wrapped once")
        middle.original_exception = inner  # type: ignore[attr-defined]
        outer = RuntimeError("wrapped twice")
        outer.original_exception = middle  # type: ignore[attr-defined]
        assert _is_missing_models_endpoint(outer) is True

        chained = RuntimeError("raised from")
        chained.__cause__ = self._status_error(501)
        assert _is_missing_models_endpoint(chained) is True

    def test_status_lookup_terminates_on_a_cyclic_chain(self) -> None:
        # A self-referential chain must not hang the discovery path.
        a = RuntimeError("a")
        b = RuntimeError("b")
        a.__cause__ = b
        b.__cause__ = a
        assert _is_missing_models_endpoint(a) is False

    @pytest.mark.asyncio
    async def test_bad_credentials_stay_unreachable(self) -> None:
        result = await self._discover(self._status_error(401, "invalid api key"))

        assert result.error is not None
        assert result.discovery_unsupported is False

    @pytest.mark.asyncio
    async def test_connection_failure_stays_unreachable(self) -> None:
        # No status at all: nothing answered, so the provider really is unreachable.
        result = await self._discover(ConnectionError("upstream unreachable"))

        assert result.error is not None
        assert result.discovery_unsupported is False

    @pytest.mark.asyncio
    async def test_provider_that_cannot_list_models_is_discovery_unsupported(self) -> None:
        config = self._make_config({"sagemaker": {"region": "us-east-1"}})
        with (
            patch("gateway.services.model_discovery_service.get_model_cache") as mock_cache_fn,
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=False),
        ):
            mock_cache_fn.return_value = ModelCache()
            result = await discover_provider_models(config, "sagemaker")

        assert result.error is not None
        assert result.discovery_unsupported is True

    @pytest.mark.asyncio
    async def test_credentials_test_reports_missing_endpoint(self) -> None:
        with (
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch("gateway.services.model_discovery_service.alist_models", side_effect=self._status_error(404)),
        ):
            result = await run_credentials_test("openai", api_key="sk", api_base="https://api.otari.ai/v1")

        assert result.error is not None
        assert result.discovery_unsupported is True

    @pytest.mark.asyncio
    async def test_credentials_test_keeps_auth_failure_unreachable(self) -> None:
        with (
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch(
                "gateway.services.model_discovery_service.alist_models",
                side_effect=self._status_error(401, "invalid api key"),
            ),
        ):
            result = await run_credentials_test("openai", api_key="sk")

        assert result.error is not None
        assert result.discovery_unsupported is False

    def test_only_missing_endpoint_statuses_classify(self) -> None:
        # A server error is the provider failing, not the endpoint being absent.
        assert _is_missing_models_endpoint(self._status_error(404)) is True
        assert _is_missing_models_endpoint(self._status_error(405)) is True
        assert _is_missing_models_endpoint(self._status_error(501)) is True
        assert _is_missing_models_endpoint(self._status_error(500)) is False
        assert _is_missing_models_endpoint(ValueError("no status here")) is False


# ---------------------------------------------------------------------------
# Background refresh: reads serve the cache, the refresher owns the dialing
# ---------------------------------------------------------------------------


class TestBackgroundDiscovery:
    """Discovery must not be dialed on the request path.

    ``model_discovery_timeout_seconds`` (10s per unreachable provider) used to be
    paid by whoever's read happened to arrive after the TTL lapsed, which put it
    on a dashboard page load. These pin the read/refresh split that removed that.
    """

    @staticmethod
    def _config(ttl: int = 300) -> GatewayConfig:
        return GatewayConfig(providers={"openai": {"api_key": "sk-test"}}, model_cache_ttl_seconds=ttl)

    @pytest.mark.asyncio
    async def test_stale_entry_is_served_without_dialing(self) -> None:
        """The regression: an expired entry is served, not re-dialed, on a read."""
        config = self._config()
        cache = ModelCache()
        cache.set("openai", [_make_model("gpt-4o")])
        # Age the entry well past the 300s TTL, which is what used to force the
        # next reader to pay the provider timeout.
        cache._store["openai"].cached_at = time.monotonic() - 10_000

        with (
            patch("gateway.services.model_discovery_service.get_model_cache", return_value=cache),
            patch("gateway.services.model_discovery_service.alist_models") as mock_alist,
        ):
            result = await discover_all_models(config, serve_stale=True)

        assert [model.id for _, model in result] == ["gpt-4o"]
        mock_alist.assert_not_called()

    @pytest.mark.asyncio
    async def test_serve_stale_still_dials_a_provider_never_checked(self) -> None:
        """A cold worker must dial rather than claim the provider has no models."""
        config = self._config()
        cache = ModelCache()

        with (
            patch("gateway.services.model_discovery_service.get_model_cache", return_value=cache),
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch(
                "gateway.services.model_discovery_service.alist_models",
                new=AsyncMock(return_value=[_make_model("gpt-4o")]),
            ) as mock_alist,
        ):
            result = await discover_all_models(config, serve_stale=True)

        assert [model.id for _, model in result] == ["gpt-4o"]
        assert mock_alist.await_count == 1

    @pytest.mark.asyncio
    async def test_stale_read_does_not_hide_a_cached_failure(self) -> None:
        """A negatively cached provider stays failed; it must not look healthy."""
        config = self._config()
        cache = ModelCache()
        cache._store["openai"] = _CacheEntry(
            result=ProviderDiscovery(provider="openai", models=[], error="bad key"),
            cached_at=time.monotonic() - 10_000,
            checked_at=datetime.now(UTC),
        )

        with (
            patch("gateway.services.model_discovery_service.get_model_cache", return_value=cache),
            patch("gateway.services.model_discovery_service.alist_models") as mock_alist,
        ):
            discovery = await discover_provider_models(config, "openai", serve_stale=True)

        assert discovery.error == "bad key"
        mock_alist.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_redials_a_fresh_entry(self) -> None:
        """The refresher's own read ignores freshness, or the cache never moves."""
        config = self._config()
        cache = ModelCache()
        cache.set("openai", [_make_model("stale-model")])

        with (
            patch("gateway.services.model_discovery_service.get_model_cache", return_value=cache),
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch(
                "gateway.services.model_discovery_service.alist_models",
                new=AsyncMock(return_value=[_make_model("fresh-model")]),
            ) as mock_alist,
        ):
            discoveries = await discover_models_with_status(config, force=True)

        assert mock_alist.await_count == 1
        assert [model.id for model in discoveries[0].models] == ["fresh-model"]
        # And the refreshed result is what a later stale read serves.
        assert [model.id for model in (cache.stale("openai") or ProviderDiscovery("openai", [])).models] == [
            "fresh-model"
        ]

    @staticmethod
    async def _run_refresher_until(
        config: GatewayConfig,
        on_call: "Callable[[int], None]",
        rounds: int,
    ) -> int:
        """Drive the refresher until ``rounds`` calls land, then cancel it.

        Waits on an event rather than a wall-clock sleep: the failure path logs a
        full traceback, and rendering that can outlast any fixed window, which
        would make a timing-based assertion flaky under load.
        """
        seen = 0
        reached = asyncio.Event()

        async def refresh(cfg: GatewayConfig) -> None:
            nonlocal seen
            seen += 1
            on_call(seen)
            if seen >= rounds:
                reached.set()

        with patch("gateway.services.model_discovery_service.refresh_discovery_cache", side_effect=refresh):
            task = asyncio.create_task(run_discovery_refresher(config, interval=0.001))
            try:
                await asyncio.wait_for(reached.wait(), timeout=5)
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        return seen

    @pytest.mark.asyncio
    async def test_refresher_primes_immediately_then_ticks(self) -> None:
        """Priming is the first thing the refresher does, before any sleep."""
        first_call_delay: list[float] = []
        started = time.monotonic()

        def record(_: int) -> None:
            first_call_delay.append(time.monotonic() - started)

        seen = await self._run_refresher_until(self._config(), record, rounds=2)

        assert seen == 2
        # The prime lands before the first sleep, so the catalog is warm without
        # waiting out an interval after boot.
        assert first_call_delay[0] < 1.0

    @pytest.mark.asyncio
    async def test_refresher_survives_a_failing_round(self) -> None:
        """One bad round must not kill the refresher and freeze the catalog."""

        def blow_up_once(call: int) -> None:
            if call == 1:
                raise RuntimeError("provider fanout blew up")

        seen = await self._run_refresher_until(self._config(), blow_up_once, rounds=2)

        assert seen == 2

    def test_zero_ttl_keeps_dialing_on_every_read(self) -> None:
        """`model_cache_ttl_seconds = 0` documents "no caching"; honor it."""
        assert background_discovery_enabled(self._config(ttl=0)) is False
        assert background_discovery_enabled(self._config(ttl=300)) is True

    def test_refresh_interval_has_a_floor(self) -> None:
        """A tiny TTL must not become a re-dial storm against every provider."""
        assert _refresh_interval(self._config(ttl=1)) == _MIN_REFRESH_INTERVAL_SECONDS
        assert _refresh_interval(self._config(ttl=600)) == 600.0

    @pytest.mark.asyncio
    async def test_refresher_skips_dialing_while_caching_is_off_but_keeps_ticking(self) -> None:
        """The setting is read per tick, not once at startup.

        ``model_cache_ttl_seconds`` is runtime-settable from the Settings page, and
        raising it from 0 flips every read onto the serve-from-cache path at once.
        A refresher that had been skipped at startup would leave that cache filled
        once and never refreshed for the life of the worker. So the loop always
        runs and decides per tick: no dial while caching is off (the reads dial for
        themselves then, and a second dialer would be pure duplication), and it
        picks straight back up when the setting changes.
        """
        config = self._config(ttl=0)
        rounds = 0
        reached = asyncio.Event()

        async def refresh(_cfg: GatewayConfig) -> None:
            nonlocal rounds
            rounds += 1
            reached.set()

        with patch("gateway.services.model_discovery_service.refresh_discovery_cache", side_effect=refresh):
            task = asyncio.create_task(run_discovery_refresher(config, interval=0.001))
            try:
                # Several ticks' worth of time, with caching off the whole way.
                await asyncio.sleep(0.05)
                assert rounds == 0, "refresher dialed while model_cache_ttl_seconds was 0"

                # An operator turns caching on; the running loop must notice.
                config.model_cache_ttl_seconds = 300
                await asyncio.wait_for(reached.wait(), timeout=5)
                assert rounds >= 1
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task


class TestRefresherCadence:
    """A failed provider must not stay failed for a whole success interval.

    Reads serve a cached failure at any age, so the refresh interval is what
    decides how quickly a recovered provider reappears. Before the refresher
    existed, ``model_discovery_negative_ttl_seconds`` bounded that at 30s via the
    next read's re-dial; the refresher has to keep that promise.
    """

    @staticmethod
    def _config(ttl: int = 300, negative_ttl: float = 30.0, discovery: bool = True) -> GatewayConfig:
        return GatewayConfig(
            providers={"openai": {"api_key": "sk-test"}},
            model_cache_ttl_seconds=ttl,
            model_discovery_negative_ttl_seconds=negative_ttl,
            model_discovery=discovery,
        )

    def test_a_failed_round_comes_back_on_the_negative_ttl(self) -> None:
        config = self._config(ttl=300, negative_ttl=30.0)
        assert _refresh_interval(config, had_failure=False) == 300.0
        assert _refresh_interval(config, had_failure=True) == 30.0

    def test_the_floor_still_applies_to_a_failed_round(self) -> None:
        """A tiny negative TTL must not become a re-dial storm."""
        config = self._config(ttl=300, negative_ttl=1.0)
        assert _refresh_interval(config, had_failure=True) == _MIN_REFRESH_INTERVAL_SECONDS

    def test_a_failure_never_lengthens_the_interval(self) -> None:
        """A negative TTL above the success TTL must not slow recovery down."""
        config = self._config(ttl=600, negative_ttl=3600.0)
        assert _refresh_interval(config, had_failure=True) == 600.0

    @pytest.mark.asyncio
    async def test_refresh_reports_whether_a_provider_failed(self) -> None:
        config = self._config()
        cache = ModelCache()

        with (
            patch("gateway.services.model_discovery_service.get_model_cache", return_value=cache),
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch(
                "gateway.services.model_discovery_service.alist_models",
                new=AsyncMock(side_effect=RuntimeError("provider down")),
            ),
        ):
            assert await refresh_discovery_cache(config) is True

        cache.clear()
        with (
            patch("gateway.services.model_discovery_service.get_model_cache", return_value=cache),
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch(
                "gateway.services.model_discovery_service.alist_models",
                new=AsyncMock(return_value=[_make_model("gpt-4o")]),
            ),
        ):
            assert await refresh_discovery_cache(config) is False

    def test_discovery_disabled_stops_the_background_dialing_too(self) -> None:
        """`model_discovery: false` means "do not dial providers", unattended included.

        A refresher fanning out every interval for the life of the process is new
        unrequested traffic against a provider that may meter list_models, on a
        deployment that explicitly opted out.
        """
        assert background_discovery_enabled(self._config(discovery=False)) is False
        assert background_discovery_enabled(self._config(discovery=True)) is True
