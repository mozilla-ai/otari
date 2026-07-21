"""Unit tests for provider metadata and model context-window lookups.

These read only the bundled any-llm and genai-prices datasets, so they need no
database or network.
"""

import pytest

from gateway.core.config import GatewayConfig
from gateway.services.pricing_service import model_context_window
from gateway.services.provider_metadata_service import (
    KnownProvider,
    list_known_providers,
    list_provider_info,
    provider_info,
)


def _config(providers: dict[str, dict[str, object]]) -> GatewayConfig:
    return GatewayConfig(master_key="test", providers=providers)


def test_context_window_known_model() -> None:
    """A model genai-prices knows reports its context window."""
    assert model_context_window("openai", "gpt-4o") == 128000


def test_context_window_bare_name_resolves() -> None:
    """A provider-agnostic name still resolves when it is unambiguous."""
    assert model_context_window(None, "gpt-4o") == 128000


def test_context_window_unknown_model_is_none() -> None:
    """An unknown model yields None rather than raising."""
    assert model_context_window("openai", "totally-made-up-model-xyz") is None


def test_provider_info_openai_metadata() -> None:
    """OpenAI resolves a display name, doc URL, pricing links, and capabilities."""
    info = provider_info(_config({"openai": {"api_key": "sk-x"}}), "openai")

    assert info.instance == "openai"
    assert info.provider_type == "openai"
    assert info.name == "OpenAI"
    assert info.doc_url is not None and info.doc_url.startswith("http")
    assert info.pricing_urls  # genai-prices lists OpenAI pricing pages
    assert info.env_key == "OPENAI_API_KEY"
    # any-llm reports OpenAI as vision-capable and able to list models.
    assert info.capabilities.vision is True
    assert info.capabilities.list_models is True


def test_provider_info_uses_instance_type_for_named_instance() -> None:
    """A named instance backed by openai gets openai's metadata, keeps its name."""
    config = _config(
        {"my-openai": {"provider_type": "openai", "api_key": "sk-x", "api_base": "http://x/v1"}}
    )
    info = provider_info(config, "my-openai")

    assert info.instance == "my-openai"
    assert info.provider_type == "openai"
    assert info.name == "OpenAI"
    assert info.capabilities.list_models is True


def test_provider_info_unknown_type_is_graceful() -> None:
    """An unknown provider type falls back to the instance name and empty caps."""
    info = provider_info(_config({"mystery": {"api_key": "x"}}), "mystery")

    assert info.instance == "mystery"
    assert info.name == "mystery"
    assert info.capabilities.vision is False
    assert info.capabilities.list_models is False


def test_list_provider_info_sorted_by_instance() -> None:
    """Providers come back sorted by their configured instance name."""
    config = _config(
        {
            "openai": {"api_key": "sk-x"},
            "anthropic": {"api_key": "sk-y"},
        }
    )
    names = [info.instance for info in list_provider_info(config)]
    assert names == ["anthropic", "openai"]


def _known_by_id(providers: list[KnownProvider]) -> dict[str, KnownProvider]:
    return {p.id: p for p in providers}


def test_known_provider_env_key_present_false_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key-based provider whose env var is unset reports env_key_present False."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    openai = _known_by_id(list_known_providers())["openai"]

    assert openai.env_key == "OPENAI_API_KEY"
    assert openai.requires_api_key is True
    assert openai.env_key_present is False


def test_known_provider_env_key_present_true_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key-based provider whose env var is set reports env_key_present True."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    openai = _known_by_id(list_known_providers())["openai"]

    assert openai.requires_api_key is True
    assert openai.env_key_present is True


def test_known_provider_env_key_present_false_for_blank_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """A whitespace-only env var counts as absent, not a usable key."""
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    openai = _known_by_id(list_known_providers())["openai"]

    assert openai.env_key_present is False


def test_known_provider_keyless_backend_never_present() -> None:
    """A keyless backend (ollama) has no env var, so env_key_present stays False."""
    ollama = _known_by_id(list_known_providers())["ollama"]

    assert ollama.env_key is None
    assert ollama.requires_api_key is False
    assert ollama.env_key_present is False
