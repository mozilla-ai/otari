"""Unit tests for provider metadata and model context-window lookups.

These read only the bundled any-llm and genai-prices datasets, so they need no
database or network.
"""

import sys

import pytest
from any_llm import AnyLLM

from gateway.core.config import GatewayConfig
from gateway.services.pricing_service import model_context_window
from gateway.services.provider_metadata_service import (
    known_provider_detail,
    list_known_provider_summaries,
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


def test_provider_summaries_import_no_provider_sdks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Listing the picker's providers must not import any provider SDK module.

    This is the whole point of the lazy split (issue #365): the picker list is
    built from the any-llm registry plus the bundled genai-prices names, so it
    stays cheap. get_provider_class is what pulls in a provider SDK, so it must
    never be called while building the summary list.
    """

    def fail_get_provider_class(provider_type: str) -> object:
        raise AssertionError(f"summary listing imported provider SDK for {provider_type!r}")

    monkeypatch.setattr(AnyLLM, "get_provider_class", staticmethod(fail_get_provider_class))

    summaries = list_known_provider_summaries()

    ids = {s.id for s in summaries}
    assert "openai" in ids and "anthropic" in ids
    # Sorted by display name, case-insensitively.
    names = [s.name for s in summaries]
    assert names == sorted(names, key=str.lower)


def test_provider_summaries_do_not_import_provider_submodules() -> None:
    """The summary path leaves any-llm's per-provider submodules unimported."""
    before = {m for m in sys.modules if m.startswith("any_llm.providers.") and m.count(".") == 2}
    list_known_provider_summaries()
    after = {m for m in sys.modules if m.startswith("any_llm.providers.") and m.count(".") == 2}
    assert after == before


def test_provider_detail_openai() -> None:
    """Detail for a single provider carries its autofill hints."""
    openai = known_provider_detail("openai")

    assert openai is not None
    assert openai.id == "openai"
    assert openai.env_key == "OPENAI_API_KEY"
    assert openai.requires_api_key is True


def test_provider_detail_env_key_present_false_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key-based provider whose env var is unset reports env_key_present False."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    openai = known_provider_detail("openai")

    assert openai is not None
    assert openai.env_key_present is False


def test_provider_detail_env_key_present_true_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key-based provider whose env var is set reports env_key_present True."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    openai = known_provider_detail("openai")

    assert openai is not None
    assert openai.env_key_present is True


def test_provider_detail_env_key_present_false_for_blank_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """A whitespace-only env var counts as absent, not a usable key."""
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    openai = known_provider_detail("openai")

    assert openai is not None
    assert openai.env_key_present is False


def test_provider_detail_keyless_backend_never_present() -> None:
    """A keyless backend (ollama) has no env var, so env_key_present stays False."""
    ollama = known_provider_detail("ollama")

    assert ollama is not None
    assert ollama.env_key is None
    assert ollama.requires_api_key is False
    assert ollama.env_key_present is False


def test_provider_detail_unknown_id_is_none() -> None:
    """An unknown provider id yields None (the route maps this to a 404)."""
    assert known_provider_detail("definitely-not-a-provider") is None
