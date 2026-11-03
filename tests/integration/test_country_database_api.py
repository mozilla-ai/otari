"""``/api/v1/settings/country-database``: the country file an operator stores in the database."""

import hashlib
from collections.abc import Generator
from pathlib import Path

import httpx2
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from _mmdb_support import write_country_database
from gateway.api.routes import country_database as country_database_route
from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.services.inference import _country_database

from .conftest import build_test_client

_MASTER = {API_KEY_HEADER: "Bearer test-master-key"}
_URL = f"{API_ROOT}/settings/country-database"


def _config(postgres_url: str, *, client_country_database: str | None = None) -> GatewayConfig:
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        client_country_database=client_country_database,
    )


@pytest.fixture
def client(postgres_url: str) -> Generator[TestClient]:
    yield from build_test_client(_config(postgres_url))


def _file(tmp_path: Path, name: str = "countries.mmdb", **kwargs: str) -> bytes:
    return write_country_database(tmp_path / name, {"192.0.2.0/24": "DE"}, **kwargs).read_bytes()


def _upload(client: TestClient, content: bytes) -> httpx2.Response:
    return client.put(_URL, files={"file": ("countries.mmdb", content)}, headers=_MASTER)


def test_no_file_is_stored_at_first(client: TestClient) -> None:
    response = client.get(_URL, headers=_MASTER)

    assert response.status_code == 200
    assert response.json() == {"stored": None, "client_country_database_set": False}


def test_an_uploaded_country_file_is_stored(client: TestClient, tmp_path: Path) -> None:
    content = _file(tmp_path)

    response = _upload(client, content)

    assert response.status_code == 200, response.text
    stored = response.json()["stored"]
    assert stored["database_type"] == "Test-Country"
    assert stored["built_at"].startswith("1970-01-01T00:00:01")
    assert stored["size_bytes"] == len(content)
    assert stored["sha256"] == hashlib.sha256(content).hexdigest()
    assert stored["uploaded_at"] is not None
    assert client.get(_URL, headers=_MASTER).json()["stored"] == stored


def test_a_second_upload_replaces_the_first(client: TestClient, tmp_path: Path) -> None:
    _upload(client, _file(tmp_path, "first.mmdb"))
    second = _file(tmp_path, "second.mmdb", database_type="DBIP-Country-Lite")

    response = _upload(client, second)

    assert response.status_code == 200
    assert response.json()["stored"]["database_type"] == "DBIP-Country-Lite"
    assert response.json()["stored"]["sha256"] == hashlib.sha256(second).hexdigest()


def test_a_file_that_is_no_database_is_refused_and_the_stored_one_stays(client: TestClient, tmp_path: Path) -> None:
    _upload(client, _file(tmp_path))
    before = client.get(_URL, headers=_MASTER).json()

    response = _upload(client, b"not a database")

    assert response.status_code == 400
    assert "MaxMind DB" in response.json()["detail"]
    assert client.get(_URL, headers=_MASTER).json() == before


def test_a_database_that_holds_no_countries_is_refused(client: TestClient, tmp_path: Path) -> None:
    response = _upload(client, _file(tmp_path, database_type="GeoLite2-City"))

    assert response.status_code == 400
    assert "not a country database" in response.json()["detail"]


def test_a_file_over_the_limit_is_refused(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    content = _file(tmp_path)
    monkeypatch.setattr(_country_database, "COUNTRY_DATABASE_MAX_BYTES", len(content) - 1)

    response = _upload(client, content)

    assert response.status_code == 413
    assert client.get(_URL, headers=_MASTER).json()["stored"] is None


def test_a_body_over_the_limit_is_refused_as_it_arrives(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(country_database_route, "_MAX_BODY_BYTES", 1024)

    response = _upload(client, _file(tmp_path) + b"\0" * 2048)

    assert response.status_code == 413
    assert client.get(_URL, headers=_MASTER).json()["stored"] is None


def test_an_upload_is_not_read_before_the_operator_gate(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A File parameter would be parsed, and spooled to disk, before the gate refused the caller."""

    def unread(self: Request, **_: object) -> object:
        raise AssertionError("the body was read before the caller was authenticated")

    monkeypatch.setattr(Request, "form", unread)

    response = client.put(_URL, files={"file": ("countries.mmdb", b"\0" * 1024)})

    assert response.status_code == 401


def test_an_upload_with_no_file_is_refused(client: TestClient) -> None:
    response = client.put(_URL, files={"other": ("countries.mmdb", b"content")}, headers=_MASTER)

    assert response.status_code == 422


def test_the_stored_file_is_removed(client: TestClient, tmp_path: Path) -> None:
    _upload(client, _file(tmp_path))

    assert client.delete(_URL, headers=_MASTER).status_code == 204
    assert client.get(_URL, headers=_MASTER).json()["stored"] is None


def test_removing_with_no_file_stored_is_not_found(client: TestClient) -> None:
    assert client.delete(_URL, headers=_MASTER).status_code == 404


def test_the_status_says_when_a_configured_file_is_read_instead(postgres_url: str, tmp_path: Path) -> None:
    path = write_country_database(tmp_path / "mounted.mmdb", {"192.0.2.0/24": "DE"})
    for client in build_test_client(_config(postgres_url, client_country_database=str(path))):
        assert client.get(_URL, headers=_MASTER).json()["client_country_database_set"] is True
