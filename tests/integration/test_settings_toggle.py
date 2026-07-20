"""Tests for the writable runtime settings endpoint (PATCH /v1/settings)."""

import asyncio
from collections.abc import AsyncGenerator, Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from gateway.core.config import GatewayConfig
from gateway.core.database import get_db
from gateway.main import create_app
from gateway.services.pricing_service import configure_default_pricing, default_pricing_enabled

from .conftest import _run_alembic_migrations, _to_async_url


@pytest.fixture(autouse=True)
def _isolate_runtime_settings(postgres_url: str) -> Generator[None]:
    """Clear persisted overrides and the process pricing flag around each test.

    These custom-built apps do not drop tables between tests, and an override is
    applied on startup, so a leftover row would leak into the next test here or
    in another file. Reset before and after to keep tests independent.
    """
    _run_alembic_migrations(postgres_url)

    def _reset() -> None:
        engine = create_engine(postgres_url, pool_pre_ping=True)
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM runtime_settings"))
        engine.dispose()
        configure_default_pricing(False)

    _reset()
    yield
    _reset()


def _make_client(config: GatewayConfig) -> Generator[TestClient]:
    _run_alembic_migrations(config.database_url)
    async_engine = create_async_engine(_to_async_url(config.database_url), pool_pre_ping=True)
    async_session_factory = async_sessionmaker(async_engine, expire_on_commit=False)
    app = create_app(config)

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with async_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            yield client
    finally:
        try:
            asyncio.run(async_engine.dispose())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(async_engine.dispose())
            loop.close()


def _config(postgres_url: str) -> GatewayConfig:
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        model_discovery=True,
        default_pricing=False,
    )


@pytest.fixture
def master_header() -> dict[str, str]:
    return {"Authorization": "Bearer test-master-key"}


def test_get_settings_reports_toggle_flags(postgres_url: str, master_header: dict[str, str]) -> None:
    client_gen = _make_client(_config(postgres_url))
    client = next(client_gen)
    try:
        body = client.get("/v1/settings", headers=master_header).json()
        assert body["model_discovery"] is True
        assert body["default_pricing"] is False
    finally:
        client_gen.close()


def test_patch_toggles_and_applies_settings(postgres_url: str, master_header: dict[str, str]) -> None:
    client_gen = _make_client(_config(postgres_url))
    client = next(client_gen)
    try:
        resp = client.patch(
            "/v1/settings",
            json={"model_discovery": False, "default_pricing": True},
            headers=master_header,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_discovery"] is False
        assert body["default_pricing"] is True
        # default_pricing applies to the process-wide pricing flag immediately.
        assert default_pricing_enabled() is True
        # And the change is visible on a fresh GET.
        assert client.get("/v1/settings", headers=master_header).json()["model_discovery"] is False
    finally:
        client_gen.close()


def test_patch_leaves_omitted_fields_unchanged(postgres_url: str, master_header: dict[str, str]) -> None:
    client_gen = _make_client(_config(postgres_url))
    client = next(client_gen)
    try:
        body = client.patch("/v1/settings", json={"default_pricing": True}, headers=master_header).json()
        assert body["default_pricing"] is True
        # model_discovery was not in the request, so it keeps its configured value.
        assert body["model_discovery"] is True
    finally:
        client_gen.close()


def test_override_survives_a_restart(postgres_url: str, master_header: dict[str, str]) -> None:
    # Toggle discovery off, then rebuild the app against the same database: the
    # persisted override is loaded and applied on startup.
    first_gen = _make_client(_config(postgres_url))
    first = next(first_gen)
    try:
        first.patch("/v1/settings", json={"model_discovery": False}, headers=master_header)
    finally:
        first_gen.close()

    second_gen = _make_client(_config(postgres_url))
    second = next(second_gen)
    try:
        assert second.get("/v1/settings", headers=master_header).json()["model_discovery"] is False
    finally:
        second_gen.close()


def test_patch_requires_master_key(postgres_url: str) -> None:
    client_gen = _make_client(_config(postgres_url))
    client = next(client_gen)
    try:
        resp = client.patch("/v1/settings", json={"model_discovery": False})
        assert resp.status_code in (401, 403)
    finally:
        client_gen.close()
