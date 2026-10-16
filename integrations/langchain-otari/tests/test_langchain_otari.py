from __future__ import annotations

import pytest
from langchain_otari import DEFAULT_BASE_URL, ChatOtari


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with no Otari or OpenAI variables set."""
    for name in ("OTARI_BASE_URL", "OTARI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_reads_the_gateway_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """OTARI_BASE_URL and OTARI_API_KEY configure the model, and streams report usage."""
    monkeypatch.setenv("OTARI_BASE_URL", "https://otari.example.com/api/v1")
    monkeypatch.setenv("OTARI_API_KEY", "otari-key")
    llm = ChatOtari(model="openai:gpt-4o")
    assert llm.openai_api_base == "https://otari.example.com/api/v1"
    assert llm.root_client.api_key == "otari-key"
    assert llm.stream_usage is True
    assert llm._llm_type == "otari-chat"


def test_arguments_win_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit arguments take precedence over the environment."""
    monkeypatch.setenv("OTARI_BASE_URL", "https://otari.example.com/api/v1")
    monkeypatch.setenv("OTARI_API_KEY", "otari-key")
    llm = ChatOtari(model="openai:gpt-4o", base_url="http://gateway:8000/api/v1", api_key="explicit-key")
    assert llm.openai_api_base == "http://gateway:8000/api/v1"
    assert llm.root_client.api_key == "explicit-key"


def test_defaults_to_otari_ai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without OTARI_BASE_URL, requests go to otari.ai's gateway."""
    monkeypatch.setenv("OTARI_API_KEY", "otari-key")
    assert ChatOtari(model="openai:gpt-4o").openai_api_base == DEFAULT_BASE_URL


def test_never_falls_back_to_the_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """An OpenAI key in the environment is never sent to the gateway."""
    monkeypatch.setenv("OPENAI_API_KEY", "an-openai-key-that-must-not-reach-the-gateway")
    with pytest.raises(ValueError, match="OTARI_API_KEY"):
        ChatOtari(model="openai:gpt-4o")


def test_serializes_its_secret_under_the_otari_name() -> None:
    """A serialized model points at OTARI_API_KEY, not OPENAI_API_KEY."""
    assert ChatOtari(model="openai:gpt-4o", api_key="otari-key").lc_secrets == {"openai_api_key": "OTARI_API_KEY"}
