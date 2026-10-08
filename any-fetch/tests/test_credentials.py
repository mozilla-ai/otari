"""Where a provider's key and base URL come from."""

import pytest

from any_fetch import MissingCredentialError, ProviderMetadata
from any_fetch._credentials import resolve_api_base, resolve_api_key

KEYED = ProviderMetadata(
    name="keyed",
    doc_url="https://example.com/docs",
    env_key="KEYED_API_KEY",
    env_api_base="KEYED_API_BASE",
    requires_api_key=True,
    requires_api_base=False,
    default_api_base="https://api.keyed.example",
    tier="production",
    max_urls_per_call=1,
    renders_javascript=False,
    formats=["text"],
    options=[],
)


def test_an_explicit_key_wins_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEYED_API_KEY", "from-env")
    assert resolve_api_key(KEYED, "explicit") == "explicit"


def test_the_environment_key_is_read_when_none_is_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEYED_API_KEY", "from-env")
    assert resolve_api_key(KEYED, None) == "from-env"


def test_a_missing_key_names_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KEYED_API_KEY", raising=False)
    with pytest.raises(MissingCredentialError, match="KEYED_API_KEY"):
        resolve_api_key(KEYED, None)


def test_an_explicit_empty_key_never_falls_back_to_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEYED_API_KEY", "from-env")
    with pytest.raises(MissingCredentialError):
        resolve_api_key(KEYED, "")


def test_a_key_is_optional_where_the_provider_needs_none() -> None:
    assert resolve_api_key(KEYED.model_copy(update={"requires_api_key": False, "env_key": None}), None) is None


def test_the_base_url_is_explicit_then_environment_then_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KEYED_API_BASE", raising=False)
    assert resolve_api_base(KEYED, None) == "https://api.keyed.example"
    monkeypatch.setenv("KEYED_API_BASE", "https://env.example")
    assert resolve_api_base(KEYED, None) == "https://env.example"
    assert resolve_api_base(KEYED, "https://explicit.example") == "https://explicit.example"
    assert resolve_api_base(KEYED, "") == "https://api.keyed.example"


def test_a_required_base_url_that_is_missing_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KEYED_API_BASE", raising=False)
    metadata = KEYED.model_copy(update={"requires_api_base": True, "default_api_base": None})
    with pytest.raises(MissingCredentialError, match="api_base"):
        resolve_api_base(metadata, None)
