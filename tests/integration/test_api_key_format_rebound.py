"""A rebound ``ApiKeyFormatPort`` changes what every mint site produces and how keys route.

The overlay hook is the whole point of the port: a hosted build binds a format
of its own and Otari's key routes, verify path and first-run key all follow it
without an Otari file changing. The probe bound here mints ``probe-`` keys,
routes ``elsewhere-`` keys to another host and ``broken-`` keys to a 401, and
checks everything else locally, which is what keeps a legacy key working.
"""

import hashlib
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import update
from sqlalchemy.orm import Session

from gateway.api.deps import misdirected_key_detail
from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.models.api_keys import APIKey

from .conftest import build_test_client

EU_HOST = "api.eu.otari.example"
MODULE = "probe_rebound_key_format"
PROBE_BOOTSTRAP = f'''
from gateway.container import Container
from gateway.ports.api_key_format_port import ApiKeyFormatPort, Local, Malformed, Misdirected
import secrets


class ProbeKeyFormat:
    def __init__(self, session):
        self.session = session

    def mint(self):
        return "probe-" + secrets.token_urlsafe(48)

    def fingerprint(self, api_key):
        return api_key[:13]

    def route(self, presented):
        if presented.startswith("elsewhere-"):
            return Misdirected(host="{EU_HOST}")
        if presented.startswith("broken-"):
            return Malformed()
        return Local()


def register(container: Container) -> None:
    container.bind(ApiKeyFormatPort, ProbeKeyFormat)
'''
MASTER = {API_KEY_HEADER: "Bearer test-master-key"}
LEGACY_KEY = "gw-" + "legacy0" * 9


@pytest.fixture
def rebound_client(
    postgres_url: str, clean_database: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient]:
    (tmp_path / f"{MODULE}.py").write_text(PROBE_BOOTSTRAP)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(MODULE, None)
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        bootstrap=f"{MODULE}:register",
    )
    try:
        yield from build_test_client(config)
    finally:
        sys.modules.pop(MODULE, None)


def _create(client: TestClient) -> dict[str, Any]:
    response = client.post(f"{API_ROOT}/keys", json={"key_name": "probe"}, headers=MASTER)
    assert response.status_code == status.HTTP_200_OK, response.text
    created: dict[str, Any] = response.json()
    return created


def test_a_created_key_carries_the_bound_format(rebound_client: TestClient) -> None:
    created = _create(rebound_client)

    assert created["key"].startswith("probe-")
    assert created["key_prefix"] == created["key"][:13]
    assert created["key_suffix"] == created["key"][-4:]


def test_a_rotated_key_carries_the_bound_format(rebound_client: TestClient) -> None:
    created = _create(rebound_client)

    response = rebound_client.post(f"{API_ROOT}/keys/{created['id']}/rotate", headers=MASTER)

    assert response.status_code == status.HTTP_200_OK, response.text
    rotated = response.json()
    assert rotated["key"] != created["key"]
    assert rotated["key"].startswith("probe-")
    assert rotated["key_prefix"] == rotated["key"][:13]


def test_a_minted_key_authenticates(rebound_client: TestClient) -> None:
    created = _create(rebound_client)

    response = rebound_client.get(f"{API_ROOT}/models", headers={API_KEY_HEADER: created["key"]})

    assert response.status_code == status.HTTP_200_OK


def test_a_key_for_another_host_is_421_naming_it(rebound_client: TestClient) -> None:
    response = rebound_client.get(f"{API_ROOT}/models", headers={API_KEY_HEADER: "elsewhere-" + "x" * 60})

    assert response.status_code == status.HTTP_421_MISDIRECTED_REQUEST
    assert response.json()["detail"] == misdirected_key_detail(EU_HOST)


def test_a_malformed_key_is_the_ordinary_401(rebound_client: TestClient) -> None:
    response = rebound_client.get(f"{API_ROOT}/models", headers={API_KEY_HEADER: "broken-" + "x" * 60})

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"] == "Invalid API key"


def test_a_legacy_key_still_authenticates_by_hash(rebound_client: TestClient, db_session: Session) -> None:
    """A key minted before the format existed is not the format's to refuse."""
    created = _create(rebound_client)
    db_session.execute(
        update(APIKey)
        .where(APIKey.id == created["id"])
        .values(key_hash=hashlib.sha256(LEGACY_KEY.encode()).hexdigest()),
    )
    db_session.commit()

    response = rebound_client.get(f"{API_ROOT}/models", headers={API_KEY_HEADER: LEGACY_KEY})

    assert response.status_code == status.HTTP_200_OK


def test_the_mcp_contract_answers_a_misdirected_key_with_its_own_code(rebound_client: TestClient) -> None:
    """MCP forwards no detail, so the 421 keeps its status and drops the host."""
    response = rebound_client.post(
        f"{API_ROOT}/mcp/execute",
        headers={API_KEY_HEADER: "elsewhere-" + "x" * 60},
        json={
            "mcp_server_id": "2c948a61-dc96-4cd8-96bb-8e1434bf424e",
            "tool_name": "create_issue",
            "arguments": {},
            "server_revision": "rev-1",
            "client_execution_id": "11111111-1111-1111-1111-111111111111",
        },
    )

    assert response.status_code == status.HTTP_421_MISDIRECTED_REQUEST, response.text
    body = response.json()
    assert body["code"] == "misdirected_request"
    assert body["execution_state"] == "not_started"
    assert EU_HOST not in response.text
