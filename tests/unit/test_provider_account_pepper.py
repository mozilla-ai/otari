"""The pepper that names provider accounts, which a deployment making provider copies must set."""

import uuid
from pathlib import Path

import pytest
from any_llm import LLMProvider
from fastapi.testclient import TestClient
from pydantic import ValidationError

from gateway.core.config import GatewayConfig
from gateway.main import _validate_provider_account_pepper, create_app
from gateway.services.provider_kwargs import ProviderAccounts
from gateway.types.provider_account import ResolvedCredential

_PEPPER = "p" * 32
_CREDENTIAL = ResolvedCredential(api_key="sk-one")


def _identity(pepper: str) -> str:
    accounts = ProviderAccounts(pepper=pepper, workspace_id=uuid.uuid4())
    return accounts.name(LLMProvider.ANTHROPIC, "anthropic", _CREDENTIAL).identity


def test_the_account_name_depends_on_the_pepper() -> None:
    assert _identity(_PEPPER) == _identity(_PEPPER)
    assert _identity(_PEPPER) != _identity("q" * 32)


def test_a_deployment_making_copies_needs_a_pepper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTARI_PROVIDER_ACCOUNT_PEPPER", raising=False)

    with pytest.raises(ValueError, match="OTARI_PROVIDER_ACCOUNT_PEPPER"):
        _validate_provider_account_pepper(GatewayConfig())


@pytest.mark.parametrize(("files_enabled", "copies_enabled"), [(True, False), (False, True)])
def test_a_deployment_making_no_copies_needs_none(
    monkeypatch: pytest.MonkeyPatch, files_enabled: bool, copies_enabled: bool
) -> None:
    monkeypatch.delenv("OTARI_PROVIDER_ACCOUNT_PEPPER", raising=False)

    _validate_provider_account_pepper(
        GatewayConfig(files_enabled=files_enabled, files_provider_upload_enabled=copies_enabled)
    )


def test_a_short_pepper_is_refused() -> None:
    with pytest.raises(ValidationError):
        GatewayConfig(provider_account_pepper="short")


def test_a_pepper_shared_with_the_master_key_is_refused() -> None:
    with pytest.raises(ValueError, match="master key") as exc_info:
        _validate_provider_account_pepper(GatewayConfig(provider_account_pepper=_PEPPER, master_key=_PEPPER))

    assert _PEPPER not in str(exc_info.value)


def test_a_pepper_shared_with_the_secret_key_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", f"other-key,{_PEPPER}")

    with pytest.raises(ValueError, match="OTARI_SECRET_KEY") as exc_info:
        _validate_provider_account_pepper(GatewayConfig(provider_account_pepper=_PEPPER))

    assert _PEPPER not in str(exc_info.value)


def test_a_distinct_pepper_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTARI_SECRET_KEY", raising=False)

    _validate_provider_account_pepper(GatewayConfig(provider_account_pepper=_PEPPER, master_key="m" * 32))


def test_an_app_builds_without_a_pepper_and_refuses_to_serve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Building the app is not serving it: a tool that only reads the schema needs no pepper."""
    monkeypatch.delenv("OTARI_PROVIDER_ACCOUNT_PEPPER", raising=False)

    app = create_app(GatewayConfig(database_url=f"sqlite:///{tmp_path / 'no-pepper.db'}"))
    assert app.openapi()["info"]["title"] == "otari"

    with pytest.raises(ValueError, match="OTARI_PROVIDER_ACCOUNT_PEPPER"), TestClient(app):
        pass
