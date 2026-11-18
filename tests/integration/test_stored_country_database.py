"""The stored country file as each replica reads it: loaded at startup, swapped when it changes, and outranked."""

import hashlib
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient

from _mmdb_support import write_country_database
from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.core.unit_of_work import create_unit_of_work
from gateway.log_config import logger
from gateway.repositories.inference import CountryDatabaseRepository
from gateway.repositories.providers import HostedProviderRepository
from gateway.services.inference import (
    CountryDatabaseError,
    _client_country,
    _country_database,
    client_country,
    request_country_gate,
)

from .conftest import build_test_client

_MASTER = {API_KEY_HEADER: "Bearer test-master-key"}
_URL = f"{API_ROOT}/settings/country-database"
_ADDRESS = "192.0.2.7"


def _config(postgres_url: str, **overrides: Any) -> GatewayConfig:
    settings: dict[str, Any] = {
        "database_url": postgres_url,
        "master_key": "test-master-key",
        "auto_migrate": False,
        "require_pricing": False,
        "model_discovery": False,
    }
    return GatewayConfig(**{**settings, **overrides})


@pytest.fixture
def client(postgres_url: str) -> Generator[TestClient]:
    yield from build_test_client(_config(postgres_url))


def _file(tmp_path: Path, country: str) -> bytes:
    return write_country_database(tmp_path / f"{country}.mmdb", {"192.0.2.0/24": country}).read_bytes()


def _upload(client: TestClient, content: bytes) -> httpx2.Response:
    return client.put(_URL, files={"file": ("countries.mmdb", content)}, headers=_MASTER)


def _stored_country() -> str | None:
    return client_country({}, _ADDRESS, header=None, database=None)


def _store_from_another_replica(client: TestClient, content: bytes) -> None:
    """Write the row as another replica's upload would, which this process has not applied."""

    async def write() -> None:
        async with create_unit_of_work() as uow, uow:
            await CountryDatabaseRepository(uow).replace(
                content=content,
                database_type="Test-Country",
                built_at=datetime.fromtimestamp(1, UTC),
                sha256=hashlib.sha256(content).hexdigest(),
                uploaded_at=datetime.now(UTC),
            )

    assert client.portal is not None
    client.portal.call(write)


def _refresh(client: TestClient) -> None:
    async def tick() -> None:
        async with create_unit_of_work() as uow:
            await _country_database.refresh_stored_country_database(uow)

    assert client.portal is not None
    client.portal.call(tick)


def test_an_upload_is_looked_up_at_once(client: TestClient, tmp_path: Path) -> None:
    assert _stored_country() is None

    assert _upload(client, _file(tmp_path, "DE")).status_code == 200

    assert _stored_country() == "DE"


def test_a_removed_file_is_no_longer_looked_up(client: TestClient, tmp_path: Path) -> None:
    _upload(client, _file(tmp_path, "DE"))

    assert client.delete(_URL, headers=_MASTER).status_code == 204

    assert _stored_country() is None


def test_another_replicas_upload_arrives_on_the_next_refresh(client: TestClient, tmp_path: Path) -> None:
    _upload(client, _file(tmp_path, "DE"))
    _store_from_another_replica(client, _file(tmp_path, "NL"))
    assert _stored_country() == "DE"

    _refresh(client)

    assert _stored_country() == "NL"


def test_an_unchanged_file_is_not_opened_again(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _upload(client, _file(tmp_path, "DE"))
    opened: list[str] = []
    monkeypatch.setattr(_country_database, "open_stored_country_database", lambda sha256, _: opened.append(sha256))

    _refresh(client)

    assert opened == []


def test_a_stored_file_that_fails_to_open_keeps_the_file_open_before(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _upload(client, _file(tmp_path, "DE"))
    _store_from_another_replica(client, b"not a database")
    warnings: list[object] = []
    monkeypatch.setattr(logger, "warning", lambda *args: warnings.append(args))

    _refresh(client)
    _refresh(client)

    assert _stored_country() == "DE"
    assert len(warnings) == 1


def test_a_temporary_file_that_cannot_be_written_is_tried_again(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _upload(client, _file(tmp_path, "DE"))
    _store_from_another_replica(client, _file(tmp_path, "NL"))
    real_open = _client_country.open_stored_country_database

    def disk_full(sha256: str, content: bytes) -> object:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(_country_database, "open_stored_country_database", disk_full)
    _refresh(client)
    assert _stored_country() == "DE"

    monkeypatch.setattr(_country_database, "open_stored_country_database", real_open)
    _refresh(client)

    assert _stored_country() == "NL"


def test_a_replica_that_starts_with_a_stored_file_loads_it(postgres_url: str, tmp_path: Path) -> None:
    for client in build_test_client(_config(postgres_url)):
        _upload(client, _file(tmp_path, "DE"))
    _client_country.reset_stored_country_database()

    for _client in build_test_client(_config(postgres_url)):
        assert _stored_country() == "DE"


def test_a_configured_file_wins_over_the_stored_one(postgres_url: str, tmp_path: Path) -> None:
    for client in build_test_client(_config(postgres_url)):
        _upload(client, _file(tmp_path, "DE"))
    mounted = write_country_database(tmp_path / "mounted.mmdb", {"192.0.2.0/24": "NL"})
    config = _config(postgres_url, client_country_database=str(mounted))

    for client in build_test_client(config):
        assert request_country_gate(config, {}, _ADDRESS).country == "NL"
        assert _stored_country() is None
        _upload(client, _file(tmp_path, "FR"))
        assert _stored_country() is None


def test_startup_refuses_a_country_list_with_no_source(postgres_url: str) -> None:
    config = _config(postgres_url, provider_supported_countries={"openai": ["DE"]})

    with pytest.raises(CountryDatabaseError, match="provider_supported_countries"):
        for _client in build_test_client(config):
            pass


def test_a_stored_file_is_a_source_for_the_country_list(postgres_url: str, tmp_path: Path) -> None:
    for client in build_test_client(_config(postgres_url)):
        _upload(client, _file(tmp_path, "DE"))
    config = _config(postgres_url, provider_supported_countries={"openai": ["DE"]})

    for client in build_test_client(config):
        assert request_country_gate(config, {}, _ADDRESS).country == "DE"
        response = client.delete(_URL, headers=_MASTER)
        assert response.status_code == 409
        assert "provider_supported_countries" in response.json()["detail"]
        assert client.get(_URL, headers=_MASTER).json()["stored"] is not None


def _store_hosted_provider(client: TestClient, *, enabled: bool = True) -> None:
    async def write() -> None:
        async with create_unit_of_work() as uow, uow:
            await HostedProviderRepository(uow).insert(
                name="eu-openai", provider="openai", enabled=enabled, supported_countries=["DE"]
            )

    assert client.portal is not None
    client.portal.call(write)


def test_a_stored_file_a_stored_provider_needs_is_kept(client: TestClient, tmp_path: Path) -> None:
    _upload(client, _file(tmp_path, "DE"))
    _store_hosted_provider(client)

    response = client.delete(_URL, headers=_MASTER)

    assert response.status_code == 409
    assert "eu-openai" in response.json()["detail"]
    assert client.get(_URL, headers=_MASTER).json()["stored"] is not None


def test_a_stored_file_a_configured_provider_needs_is_kept(postgres_url: str, tmp_path: Path) -> None:
    config = _config(
        postgres_url,
        hosted_providers={
            "eu-anthropic": {"provider": "anthropic", "api_key": "sk-test", "supported_countries": ["DE"]}
        },
    )

    for client in build_test_client(config):
        _upload(client, _file(tmp_path, "DE"))
        response = client.delete(_URL, headers=_MASTER)
        assert response.status_code == 409
        assert "eu-anthropic" in response.json()["detail"]


def test_a_disabled_provider_does_not_keep_the_stored_file(client: TestClient, tmp_path: Path) -> None:
    _upload(client, _file(tmp_path, "DE"))
    _store_hosted_provider(client, enabled=False)

    assert client.delete(_URL, headers=_MASTER).status_code == 204


def test_a_country_header_lets_the_stored_file_go(postgres_url: str, tmp_path: Path) -> None:
    config = _config(postgres_url, client_country_header="CF-IPCountry")

    for client in build_test_client(config):
        _upload(client, _file(tmp_path, "DE"))
        _store_hosted_provider(client)
        assert client.delete(_URL, headers=_MASTER).status_code == 204
