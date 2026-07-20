"""Unit tests for the stored-provider overlay merged over config providers."""

import time
from collections.abc import Iterator
from typing import Any

import pytest

from gateway.core.config import GatewayConfig
from gateway.models.entities import ProviderCredential
from gateway.services import provider_store_service as store
from gateway.services.provider_store_service import apply_to_config, reset_provider_cache
from gateway.services.secret_box import (
    SecretDecryptionError,
    encrypt_secret,
    generate_secret_key,
)


@pytest.fixture(autouse=True)
def _clean_cache() -> Iterator[None]:
    reset_provider_cache()
    yield
    reset_provider_cache()


def _prime(overlay: dict[str, dict[str, Any]]) -> None:
    """Stand in for a database load without needing a session."""
    store._cache.clear()
    store._cache.update(overlay)
    store._cached_at = time.monotonic()


def test_config_providers_untouched_when_no_stored() -> None:
    config = GatewayConfig(providers={"openai": {"api_key": "sk-config"}})
    assert apply_to_config(config) == set()
    assert config.providers == {"openai": {"api_key": "sk-config"}}


def test_stored_provider_is_added_alongside_config() -> None:
    config = GatewayConfig(providers={"openai": {"api_key": "sk-config"}})
    _prime({"anthropic": {"api_key": "sk-stored"}})
    assert apply_to_config(config) == set()
    assert config.providers["openai"] == {"api_key": "sk-config"}
    assert config.providers["anthropic"] == {"api_key": "sk-stored"}


def test_stored_provider_shadows_config_of_same_name() -> None:
    config = GatewayConfig(providers={"openai": {"api_key": "sk-config"}})
    _prime({"openai": {"api_key": "sk-stored"}})
    assert apply_to_config(config) == {"openai"}
    assert config.providers["openai"] == {"api_key": "sk-stored"}


def test_removed_stored_row_restores_config_on_reapply() -> None:
    config = GatewayConfig(providers={"openai": {"api_key": "sk-config"}})
    _prime({"openai": {"api_key": "sk-stored"}})
    apply_to_config(config)
    # Simulate the row being deleted: cache empties, overlay re-applied.
    store._cache.clear()
    apply_to_config(config)
    assert config.providers["openai"] == {"api_key": "sk-config"}


def test_cache_reset_does_not_bake_overlay_into_baseline() -> None:
    # Regression: reset must not cause the next apply to capture the baseline from
    # an already-merged providers map, which would make a stored row permanent.
    config = GatewayConfig(providers={"openai": {"api_key": "sk-config"}})
    _prime({"openai": {"api_key": "sk-stored"}})
    apply_to_config(config)
    assert config.providers["openai"] == {"api_key": "sk-stored"}
    reset_provider_cache()
    apply_to_config(config)
    assert config.providers["openai"] == {"api_key": "sk-config"}


def test_row_to_entry_decrypts_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    row = ProviderCredential(
        instance="openai",
        encrypted_api_key=encrypt_secret("sk-live"),
        last4="live",
        client_args={},
    )
    assert store._row_to_entry(row) == {"api_key": "sk-live"}


def test_row_to_entry_carries_type_base_and_client_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    row = ProviderCredential(
        instance="home_lab",
        provider_type="openai",
        api_base="http://x/v1",
        encrypted_api_key=encrypt_secret("tok"),
        client_args={"timeout": 30},
    )
    assert store._row_to_entry(row) == {
        "provider_type": "openai",
        "api_base": "http://x/v1",
        "client_args": {"timeout": 30},
        "api_key": "tok",
    }


def test_row_to_entry_raises_when_key_cannot_be_decrypted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    row = ProviderCredential(instance="openai", encrypted_api_key=encrypt_secret("sk"), client_args={})
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    with pytest.raises(SecretDecryptionError):
        store._row_to_entry(row)
