"""Provider client retries: the deployment and instance settings, and what each SDK is handed."""

from __future__ import annotations

from typing import Any

import pytest
from any_llm import AnyLLM, LLMProvider

from gateway.core.config import GatewayConfig
from gateway.services.provider_kwargs import (
    _GOOGLE_RETRY_PROVIDERS,
    _MAX_RETRIES_CLIENT_PROVIDERS,
    get_provider_kwargs,
    with_provider_retries,
)


def _client_args(config: GatewayConfig, provider: LLMProvider, instance: str | None = None) -> dict[str, Any]:
    client_args: dict[str, Any] = get_provider_kwargs(config, provider, instance=instance).get("client_args") or {}
    return client_args


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_unset_leaves_the_sdk_default() -> None:
    config = GatewayConfig(providers={"openai": {"api_key": "sk"}})

    assert "max_retries" not in _client_args(config, LLMProvider.OPENAI)


def test_the_deployment_setting_reaches_every_instance() -> None:
    config = GatewayConfig(
        provider_max_retries=0,
        providers={"openai": {"api_key": "sk"}, "anthropic": {"api_key": "sk-ant"}},
    )

    assert _client_args(config, LLMProvider.OPENAI)["max_retries"] == 0
    assert _client_args(config, LLMProvider.ANTHROPIC)["max_retries"] == 0


def test_an_instance_setting_overrides_the_deployment() -> None:
    config = GatewayConfig(
        provider_max_retries=0,
        providers={"home_lab": {"provider_type": "openai", "api_key": "sk", "max_retries": 4}},
    )

    kwargs = get_provider_kwargs(config, LLMProvider.OPENAI, instance="home_lab")

    assert kwargs["client_args"]["max_retries"] == 4
    # Otari's own key, so it is not forwarded to the completion call.
    assert "max_retries" not in kwargs


def test_client_args_win_and_the_config_is_not_mutated() -> None:
    client_args = {"max_retries": 5, "timeout": 30}
    config = GatewayConfig(provider_max_retries=0, providers={"openai": {"api_key": "sk", "client_args": client_args}})

    assert _client_args(config, LLMProvider.OPENAI) == {"max_retries": 5, "timeout": 30}

    other = GatewayConfig(
        provider_max_retries=2, providers={"openai": {"api_key": "sk", "client_args": {"timeout": 30}}}
    )
    original = other.providers["openai"]["client_args"]
    assert _client_args(other, LLMProvider.OPENAI) == {"timeout": 30, "max_retries": 2}
    assert original == {"timeout": 30}


def test_a_bare_provider_with_no_config_entry_gets_the_deployment_setting() -> None:
    config = GatewayConfig(provider_max_retries=1)

    assert _client_args(config, LLMProvider.OPENAI)["max_retries"] == 1


def test_google_counts_attempts_through_http_options() -> None:
    config = GatewayConfig(
        provider_max_retries=2,
        providers={"gemini": {"api_key": "k", "client_args": {"http_options": {"api_version": "v1"}}}},
    )

    assert _client_args(config, LLMProvider.GEMINI)["http_options"] == {
        "api_version": "v1",
        "retry_options": {"attempts": 3},
    }


def test_google_retry_options_already_set_win() -> None:
    kwargs = {"client_args": {"http_options": {"retry_options": {"attempts": 5}}}}

    assert with_provider_retries(LLMProvider.GEMINI, kwargs, 0) is kwargs


@pytest.mark.parametrize("provider", [LLMProvider.MISTRAL, LLMProvider.BEDROCK, LLMProvider.OLLAMA, LLMProvider.OTARI])
def test_a_provider_with_no_known_retry_setting_is_left_alone(provider: LLMProvider) -> None:
    kwargs = {"api_key": "sk", "client_args": {"timeout": 30}}

    assert with_provider_retries(provider, kwargs, 0) is kwargs


@pytest.mark.parametrize("value", [-1, True, "2", 1.5])
def test_an_invalid_instance_setting_fails_at_load(value: object) -> None:
    config = GatewayConfig(providers={"openai": {"api_key": "sk", "max_retries": value}})

    with pytest.raises(ValueError, match=r"providers\.openai\.max_retries must be a non-negative integer"):
        config.validate_provider_instances()


def test_a_negative_deployment_setting_fails_at_load() -> None:
    with pytest.raises(ValueError, match="provider_max_retries"):
        GatewayConfig(provider_max_retries=-1)


# ---------------------------------------------------------------------------
# What the installed any-llm builds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", sorted(_MAX_RETRIES_CLIENT_PROVIDERS, key=str))
def test_each_listed_sdk_takes_max_retries(provider: LLMProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    """The list is only true of the installed any-llm, so a provider that stops taking the keyword fails here."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "project")
    kwargs = with_provider_retries(provider, {"api_key": "sk", "api_base": "http://localhost:1/v1"}, 0)

    llm = AnyLLM.create(provider, api_key=kwargs["api_key"], api_base=kwargs["api_base"], **kwargs["client_args"])

    assert llm.client.max_retries == 0  # type: ignore[attr-defined]


@pytest.mark.parametrize("retries", [0, 2])
def test_google_sdk_receives_the_attempt_count(retries: int) -> None:
    (provider,) = _GOOGLE_RETRY_PROVIDERS - {LLMProvider.VERTEXAI}
    kwargs = with_provider_retries(provider, {"api_key": "k"}, retries)

    llm = AnyLLM.create(provider, api_key=kwargs["api_key"], **kwargs["client_args"])

    retry_options = llm.client._api_client._http_options.retry_options  # type: ignore[attr-defined]
    assert retry_options is not None
    assert retry_options.attempts == retries + 1
