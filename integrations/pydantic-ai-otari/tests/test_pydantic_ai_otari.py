from __future__ import annotations

import pytest
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai_otari import DEFAULT_BASE_URL, OtariModel, OtariProvider


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with no Otari or OpenAI variables set."""
    for name in ("OTARI_BASE_URL", "OTARI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_reads_the_gateway_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """OTARI_BASE_URL and OTARI_API_KEY configure the provider."""
    monkeypatch.setenv("OTARI_BASE_URL", "https://otari.example.com/api/v1")
    monkeypatch.setenv("OTARI_API_KEY", "otari-key")
    provider = OtariProvider()
    assert provider.base_url == "https://otari.example.com/api/v1/"
    assert provider.client.api_key == "otari-key"
    assert provider.name == "otari"


def test_arguments_win_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit arguments take precedence over the environment."""
    monkeypatch.setenv("OTARI_BASE_URL", "https://otari.example.com/api/v1")
    monkeypatch.setenv("OTARI_API_KEY", "otari-key")
    provider = OtariProvider(base_url="http://gateway:8000/api/v1", api_key="explicit-key")
    assert provider.base_url == "http://gateway:8000/api/v1/"
    assert provider.client.api_key == "explicit-key"


def test_defaults_to_otari_ai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without OTARI_BASE_URL, requests go to otari.ai's gateway."""
    monkeypatch.setenv("OTARI_API_KEY", "otari-key")
    assert OtariProvider().base_url == DEFAULT_BASE_URL + "/"


def test_never_falls_back_to_the_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """An OpenAI key in the environment is never sent to the gateway."""
    monkeypatch.setenv("OPENAI_API_KEY", "an-openai-key-that-must-not-reach-the-gateway")
    with pytest.raises(UserError, match="OTARI_API_KEY"):
        OtariProvider()


def test_model_uses_an_otari_provider_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """OtariModel builds its own provider from the environment."""
    monkeypatch.setenv("OTARI_API_KEY", "otari-key")
    model = OtariModel("openai:gpt-4o")
    assert model.system == "otari"
    assert model.model_name == "openai:gpt-4o"


@pytest.mark.parametrize("model_name", ["anthropic:claude-sonnet-4-6", "anthropic/claude-sonnet-4-6"])
def test_profile_follows_the_model_family(model_name: str) -> None:
    """A Claude model gets Claude's capabilities, with Chat Completions schema rules."""
    profile = OtariProvider.model_profile(model_name)
    assert profile is not None
    for field, value in anthropic_model_profile("claude-sonnet-4-6").items():
        assert profile[field] == value, field
    assert profile["json_schema_transformer"] is OpenAIJsonSchemaTransformer


@pytest.mark.parametrize("model_name", ["support-agent", "unknown-vendor:model"])
def test_profile_of_a_policy_or_unknown_family_is_openai_compatible(model_name: str) -> None:
    """A policy name or an unknown family gets an OpenAI-compatible profile."""
    profile = OtariProvider.model_profile(model_name)
    assert profile is not None
    assert profile["json_schema_transformer"] is OpenAIJsonSchemaTransformer
