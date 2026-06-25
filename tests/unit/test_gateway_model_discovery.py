"""Unit tests for the model discovery cache and service."""

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from any_llm.types.model import Model

from gateway.core.config import GatewayConfig
from gateway.services.model_discovery_service import (
    ModelCache,
    _supports_list_models,
    discover_all_models,
)


def _make_model(model_id: str, owned_by: str = "openai", created: int = 1700000000) -> Model:
    """Create a minimal Model instance for testing."""
    return Model(id=model_id, object="model", created=created, owned_by=owned_by)


# ---------------------------------------------------------------------------
# ModelCache tests
# ---------------------------------------------------------------------------


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

    def test_get_all_cached(self) -> None:
        cache = ModelCache()
        cache.set("openai", [_make_model("gpt-4o")])
        cache.set("anthropic", [_make_model("claude-3-opus")])

        all_cached = cache.get_all_cached()
        assert "openai" in all_cached
        assert "anthropic" in all_cached
        assert len(all_cached["openai"]) == 1
        assert len(all_cached["anthropic"]) == 1

    def test_get_all_cached_respects_ttl(self) -> None:
        """get_all_cached with ttl filters out expired entries."""
        cache = ModelCache()
        cache.set("openai", [_make_model("gpt-4o")])
        cache.set("anthropic", [_make_model("claude-3-opus")])

        # Backdate the openai entry so it appears expired.
        cache._store["openai"].cached_at = time.monotonic() - 400

        all_cached = cache.get_all_cached(ttl=300)
        assert "openai" not in all_cached
        assert "anthropic" in all_cached

    def test_get_all_cached_returns_shallow_copy(self) -> None:
        """Mutating the returned list should not affect the cache."""
        cache = ModelCache()
        cache.set("openai", [_make_model("gpt-4o")])

        returned = cache.get_all_cached()
        returned["openai"].append(_make_model("gpt-3.5"))

        # Internal cache should still have only 1 model.
        assert len(cache._store["openai"].models) == 1

    def test_get_returns_shallow_copy(self) -> None:
        """Mutating the returned list should not affect the cache."""
        cache = ModelCache()
        cache.set("openai", [_make_model("gpt-4o")])

        returned = cache.get("openai", ttl=300)
        assert returned is not None
        returned.append(_make_model("gpt-3.5"))

        # Internal cache should still have only 1 model.
        assert len(cache._store["openai"].models) == 1

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
