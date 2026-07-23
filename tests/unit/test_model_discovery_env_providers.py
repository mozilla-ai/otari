"""Unit tests for env-var-based provider discovery.

Regression coverage for the discovery vs. completion-routing disagreement
(issue #221): a provider is callable for completions the moment its native
credential env var is set (``get_provider_kwargs`` returns ``{}`` and any-llm
reads the env var), but GET /v1/models used to only see ``config.providers``.
Discovery now also picks up providers made callable by an env var alone.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.model import Model

from gateway.core.config import GatewayConfig
from gateway.services.model_discovery_service import (
    ModelCache,
    _discoverable_instances,
    _env_provider_instances,
    discover_all_models,
    discover_models_with_status,
)


def _make_model(model_id: str, owned_by: str, created: int = 1700000000) -> Model:
    return Model(id=model_id, object="model", created=created, owned_by=owned_by)


class TestEnvProviderInstances:
    def test_detects_provider_with_credential_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        assert "anthropic" in _env_provider_instances(configured_impls=set())

    def test_absent_env_var_is_not_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert "anthropic" not in _env_provider_instances(configured_impls=set())

    def test_blank_env_var_is_treated_as_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A blank credential (common from container templating) would only fail to
        # authenticate, so it must not add the provider.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        assert "anthropic" not in _env_provider_instances(configured_impls=set())

    def test_whitespace_only_env_var_is_treated_as_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A whitespace-only value is as unusable as a blank one and must not add
        # the provider.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
        assert "anthropic" not in _env_provider_instances(configured_impls=set())

    def test_already_configured_implementation_is_not_duplicated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        assert "anthropic" not in _env_provider_instances(configured_impls={"anthropic"})

    def test_accepts_any_of_slash_separated_alternatives(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # gemini's ENV_API_KEY_NAME is "GEMINI_API_KEY/GOOGLE_API_KEY"; either works.
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "goog-test")
        assert "gemini" in _env_provider_instances(configured_impls=set())

    def test_registry_failure_falls_back_to_no_env_providers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A metadata-registry hiccup must not raise (it would 500 GET /v1/models);
        # env detection degrades to configured-only discovery.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        with patch(
            "gateway.services.model_discovery_service.AnyLLM.get_all_provider_metadata",
            side_effect=RuntimeError("registry down"),
        ):
            assert _env_provider_instances(configured_impls=set()) == []


class TestDiscoverableInstances:
    def test_configured_come_first_then_env_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        config = GatewayConfig(providers={"home_lab": {"provider_type": "openai", "api_key": "x"}})

        instances = _discoverable_instances(config)

        assert instances[0] == "home_lab"
        assert "anthropic" in instances

    def test_custom_instance_of_same_impl_suppresses_env_duplicate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A custom-named anthropic instance plus ANTHROPIC_API_KEY must not also
        # yield a bare "anthropic": that would dial the same credential twice and
        # list the same models under two keys.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        config = GatewayConfig(providers={"my-anthropic": {"provider_type": "anthropic", "api_key": "x"}})

        instances = _discoverable_instances(config)

        assert instances == ["my-anthropic"]
        assert "anthropic" not in instances

    def test_env_detection_can_be_stubbed_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        config = GatewayConfig(providers={"openai": {"api_key": "x"}})
        with patch(
            "gateway.services.model_discovery_service._env_provider_instances",
            return_value=[],
        ):
            assert _discoverable_instances(config) == ["openai"]


class TestEnvProviderDiscovery:
    """The bug: an env-only provider is callable but missing from the catalog."""

    def _config(self, providers: dict[str, Any] | None = None) -> GatewayConfig:
        return GatewayConfig(
            providers=providers or {},
            model_discovery=True,
            model_cache_ttl_seconds=300,
        )

    @pytest.mark.asyncio
    async def test_env_only_provider_appears_in_discovery(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Only ANTHROPIC_API_KEY is set; anthropic is NOT in config.providers.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        config = self._config(providers={})

        async def fake_alist(provider: Any, **kwargs: Any) -> list[Model]:
            if provider.value == "anthropic":
                return [_make_model("claude-3-5-sonnet", owned_by="anthropic")]
            return []

        with (
            patch("gateway.services.model_discovery_service.get_model_cache", return_value=ModelCache()),
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch("gateway.services.model_discovery_service.alist_models", side_effect=fake_alist),
            patch("gateway.services.model_discovery_service.get_provider_kwargs", return_value={}),
        ):
            result = await discover_all_models(config)

        assert ("anthropic", result[0][1]) == result[0]
        assert result[0][1].id == "claude-3-5-sonnet"

    @pytest.mark.asyncio
    async def test_env_only_provider_honored_by_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        config = self._config(providers={})

        with (
            patch("gateway.services.model_discovery_service.get_model_cache", return_value=ModelCache()),
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch(
                "gateway.services.model_discovery_service.alist_models",
                new_callable=AsyncMock,
                return_value=[_make_model("claude-3-5-sonnet", owned_by="anthropic")],
            ),
            patch("gateway.services.model_discovery_service.get_provider_kwargs", return_value={}),
        ):
            result = await discover_all_models(config, provider_filter="anthropic")

        assert [name for name, _ in result] == ["anthropic"]

    @pytest.mark.asyncio
    async def test_env_only_provider_appears_in_operator_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        config = self._config(providers={})

        with (
            patch("gateway.services.model_discovery_service.get_model_cache", return_value=ModelCache()),
            patch("gateway.services.model_discovery_service._supports_list_models", return_value=True),
            patch(
                "gateway.services.model_discovery_service.alist_models",
                new_callable=AsyncMock,
                return_value=[_make_model("claude-3-5-sonnet", owned_by="anthropic")],
            ),
            patch("gateway.services.model_discovery_service.get_provider_kwargs", return_value={}),
        ):
            statuses = await discover_models_with_status(config)

        by_provider = {d.provider: d for d in statuses}
        assert "anthropic" in by_provider
        assert by_provider["anthropic"].error is None
        assert [m.id for m in by_provider["anthropic"].models] == ["claude-3-5-sonnet"]
