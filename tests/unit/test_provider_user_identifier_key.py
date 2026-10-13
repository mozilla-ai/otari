"""The key that makes the end-user identifier sent to providers unlinkable across deployments."""

import logging

import pytest
from pydantic import ValidationError

from gateway.core.config import GatewayConfig
from gateway.main import _validate_provider_user_identifier_key

_KEY = "k" * 32


def test_the_identifier_is_on_by_default_and_warns_without_a_key(caplog: pytest.LogCaptureFixture) -> None:
    config = GatewayConfig()

    assert config.provider_user_identifier_enabled is True
    assert config.provider_user_identifier_key is None
    with caplog.at_level(logging.WARNING, logger="gateway"):
        _validate_provider_user_identifier_key(config)

    assert "OTARI_PROVIDER_USER_IDENTIFIER_KEY" in caplog.text


def test_no_warning_while_the_identifier_is_off(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="gateway"):
        _validate_provider_user_identifier_key(GatewayConfig(provider_user_identifier_enabled=False))

    assert "OTARI_PROVIDER_USER_IDENTIFIER_KEY" not in caplog.text


def test_a_short_key_is_refused() -> None:
    with pytest.raises(ValidationError):
        GatewayConfig(provider_user_identifier_key="short")


def test_a_key_shared_with_the_master_key_is_refused() -> None:
    with pytest.raises(ValueError, match="master key") as exc_info:
        _validate_provider_user_identifier_key(GatewayConfig(provider_user_identifier_key=_KEY, master_key=_KEY))

    assert _KEY not in str(exc_info.value)


def test_a_key_shared_with_the_secret_key_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", f"other-key,{_KEY}")

    with pytest.raises(ValueError, match="OTARI_SECRET_KEY") as exc_info:
        _validate_provider_user_identifier_key(GatewayConfig(provider_user_identifier_key=_KEY))

    assert _KEY not in str(exc_info.value)


def test_a_distinct_key_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTARI_SECRET_KEY", raising=False)

    _validate_provider_user_identifier_key(GatewayConfig(provider_user_identifier_key=_KEY, master_key="m" * 32))
