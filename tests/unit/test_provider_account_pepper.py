"""The pepper that names provider accounts, without which a deployment makes no provider copies."""

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


def test_a_deployment_without_a_pepper_turns_copies_off(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("OTARI_PROVIDER_ACCOUNT_PEPPER", raising=False)
    config = GatewayConfig()

    _validate_provider_account_pepper(config)

    assert config.files_provider_upload_enabled is False
    assert "OTARI_PROVIDER_ACCOUNT_PEPPER" in caplog.text


def test_a_deployment_with_a_pepper_keeps_copies_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTARI_SECRET_KEY", raising=False)
    config = GatewayConfig(provider_account_pepper=_PEPPER, master_key="m" * 32)

    _validate_provider_account_pepper(config)

    assert config.files_provider_upload_enabled is True


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


def test_an_app_serves_without_a_pepper_with_copies_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The first run needs no pepper: the app serves, and provider copies stay off."""
    monkeypatch.delenv("OTARI_PROVIDER_ACCOUNT_PEPPER", raising=False)
    config = GatewayConfig(database_url=f"sqlite:///{tmp_path / 'no-pepper.db'}")

    with TestClient(create_app(config)) as client:
        assert client.get("/api/v1/health").status_code == 200

    assert config.files_provider_upload_enabled is False
