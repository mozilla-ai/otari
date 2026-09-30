"""The provider account a dispatch reaches, as the providers domain names it."""

import uuid

import pytest
from any_llm import LLMProvider

from gateway.services.provider_kwargs import effective_credential, provider_account
from gateway.types.provider_account import ResolvedCredential

_WORKSPACE = uuid.uuid4()


def _account(credential: ResolvedCredential, *, instance: str = "anthropic") -> str:
    return provider_account(LLMProvider.ANTHROPIC, instance, _WORKSPACE, credential).identity


def test_the_credential_is_the_one_the_dispatch_carries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")

    credential = effective_credential(LLMProvider.ANTHROPIC, {"api_key": "sk-configured", "api_base": "https://a"})

    assert credential.api_key == "sk-configured"
    assert credential.api_base == "https://a"


def test_a_dispatch_carrying_no_key_uses_the_sdk_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")

    assert effective_credential(LLMProvider.ANTHROPIC, {}).api_key == "sk-from-env"


def test_no_key_anywhere_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(LookupError):
        effective_credential(LLMProvider.ANTHROPIC, {"api_key": None})


def test_one_credential_names_one_account() -> None:
    credential = ResolvedCredential(api_key="sk-one")

    assert _account(credential) == _account(ResolvedCredential(api_key="sk-one"))


def test_a_changed_key_names_another_account() -> None:
    assert _account(ResolvedCredential(api_key="sk-one")) != _account(ResolvedCredential(api_key="sk-two"))


def test_a_changed_base_names_another_account() -> None:
    before = _account(ResolvedCredential(api_key="sk-one", api_base="https://a"))

    assert before != _account(ResolvedCredential(api_key="sk-one", api_base="https://b"))


def test_renaming_the_instance_keeps_the_account() -> None:
    credential = ResolvedCredential(api_key="sk-one")

    assert _account(credential, instance="anthropic") == _account(credential, instance="claude")


def test_neither_the_account_nor_the_credential_shows_the_key() -> None:
    credential = ResolvedCredential(api_key="sk-secret-value")
    account = provider_account(LLMProvider.ANTHROPIC, "anthropic", _WORKSPACE, credential)

    assert "sk-secret-value" not in account.identity
    assert "sk-secret-value" not in repr(account)
    assert "sk-secret-value" not in repr(credential)
