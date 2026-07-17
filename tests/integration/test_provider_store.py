"""Integration tests for the stored-provider CRUD + overlay against a real DB.

Exercises the ``provider_credentials`` migration, encryption at rest, the merge
onto ``config.providers``, and graceful skipping of rows that can no longer be
decrypted.
"""

from collections.abc import Iterator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.services.provider_store_service import (
    delete_credential,
    get_credential,
    list_credentials,
    refresh_provider_cache,
    reset_provider_cache,
    save_credential,
)
from gateway.services.secret_box import generate_secret_key


@pytest.fixture(autouse=True)
def _secret_key_and_clean_cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    reset_provider_cache()
    yield
    reset_provider_cache()


@pytest.mark.asyncio
async def test_crud_round_trip(async_db: AsyncSession) -> None:
    await save_credential(
        async_db, instance="openai", api_key="sk-live-1234", api_base="https://api.openai.com/v1"
    )
    await async_db.commit()

    row = await get_credential(async_db, "openai")
    assert row is not None
    assert row.last4 == "1234"
    assert row.encrypted_api_key and "sk-live-1234" not in row.encrypted_api_key
    assert [r.instance for r in await list_credentials(async_db)] == ["openai"]

    # PATCH the api_base only; the stored key is left in place.
    await save_credential(async_db, instance="openai", api_base="https://proxy/v1")
    await async_db.commit()
    row = await get_credential(async_db, "openai")
    assert row is not None
    assert row.api_base == "https://proxy/v1"
    assert row.last4 == "1234"

    assert await delete_credential(async_db, "openai") is True
    await async_db.commit()
    assert await get_credential(async_db, "openai") is None
    assert await delete_credential(async_db, "openai") is False


@pytest.mark.asyncio
async def test_refresh_overlays_and_shadows_config(async_db: AsyncSession) -> None:
    await save_credential(async_db, instance="openai", api_key="sk-stored-9999")
    await async_db.commit()

    config = GatewayConfig(providers={"openai": {"api_key": "sk-config"}, "mistral": {"api_key": "sk-m"}})
    shadowed = await refresh_provider_cache(async_db, config)

    assert shadowed == {"openai"}
    assert config.providers["openai"]["api_key"] == "sk-stored-9999"
    assert config.providers["mistral"] == {"api_key": "sk-m"}


@pytest.mark.asyncio
async def test_refresh_skips_undecryptable_and_restores_config(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    await save_credential(async_db, instance="home_lab", provider_type="openai", api_key="tok")
    await async_db.commit()

    config = GatewayConfig(providers={"openai": {"api_key": "sk-config"}})
    await refresh_provider_cache(async_db, config)
    assert "home_lab" in config.providers

    # Rotate the encryption key so the stored row can no longer be decrypted.
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    reset_provider_cache()
    await refresh_provider_cache(async_db, config)

    assert "home_lab" not in config.providers
    assert config.providers["openai"] == {"api_key": "sk-config"}
