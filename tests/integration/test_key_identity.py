"""``GET /api/v1/key-identity`` validates a forwarded workspace API key and says whose it is.

The keys in states no route produces on demand (expired, ownerless, owned by a
soft-deleted user whose keys are still active) are written straight into the
tables. A key another deployment minted needs a key format that says so, which
the open-source format never does, so that case boots on a rebound
``ApiKeyFormatPort`` the way ``test_api_key_format_rebound`` does.
"""

import sys
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.api.deps import reset_config
from gateway.core.config import API_KEY_HEADER, API_ROOT, X_API_KEY_HEADER, GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app
from gateway.models.api_keys import APIKey
from gateway.models.budgets import BudgetReservation
from gateway.models.tenancy import Workspace
from gateway.models.usage import UsageLog
from gateway.models.users import User

from .conftest import build_test_client

PATH = f"{API_ROOT}/key-identity"
REFUSED = {"detail": "Could not validate credentials."}
MODULE = "probe_key_identity_key_format"
PROBE_BOOTSTRAP = """
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
            return Misdirected(host="api.eu.otari.example")
        if presented.startswith("broken-"):
            return Malformed()
        return Local()


def register(container: Container) -> None:
    container.bind(ApiKeyFormatPort, ProbeKeyFormat)
"""


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _refuses_caching(response: Any) -> bool:
    """``private, no-store`` at least; the security middleware also adds ``no-cache``."""
    directives = {part.strip() for part in response.headers.get("Cache-Control", "").split(",")}
    return {"private", "no-store"} <= directives


def _mint(client: TestClient, master_key_header: dict[str, str], user_id: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"key_name": "forwarded"}
    if user_id is not None:
        body["user_id"] = user_id
    response = client.post(f"{API_ROOT}/keys", json=body, headers=master_key_header)
    assert response.status_code == status.HTTP_200_OK, response.text
    created: dict[str, Any] = response.json()
    return created


def _expected(db_session: Session, key_id: str) -> dict[str, Any]:
    """What the lookup should answer for ``key_id``, read from the tables directly."""
    user_id, workspace_id, organization_id = db_session.execute(
        select(APIKey.user_id, APIKey.workspace_id, col(Workspace.organization_id))
        .join(Workspace, col(Workspace.id) == APIKey.workspace_id)
        .where(APIKey.id == key_id)
    ).one()
    return {
        "api_key_id": key_id,
        "user_id": user_id,
        "workspace_id": str(workspace_id),
        "organization_id": str(organization_id),
    }


def _set_key(db_session: Session, key_id: str, **values: Any) -> None:
    db_session.execute(update(APIKey).where(APIKey.id == key_id).values(**values))
    db_session.commit()


def _records(db_session: Session) -> tuple[int, int]:
    usage = db_session.execute(select(func.count()).select_from(UsageLog)).scalar_one()
    reservations = db_session.execute(select(func.count()).select_from(BudgetReservation)).scalar_one()
    return usage, reservations


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param(lambda key: {"Authorization": f"Bearer {key}"}, id="authorization-bearer"),
        pytest.param(lambda key: {API_KEY_HEADER: key}, id="otari-key"),
        pytest.param(lambda key: {API_KEY_HEADER: f"Bearer {key}"}, id="otari-key-bearer"),
        pytest.param(lambda key: {X_API_KEY_HEADER: key}, id="x-api-key"),
    ],
)
def test_a_live_key_answers_with_its_owner_and_tenancy(
    client: TestClient,
    db_session: Session,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
    headers: Any,
) -> None:
    created = _mint(client, master_key_header, user_id=test_user["user_id"])

    response = client.get(PATH, headers=headers(created["key"]))

    assert response.status_code == status.HTTP_200_OK, response.text
    assert _refuses_caching(response)
    body = response.json()
    assert body == _expected(db_session, created["id"])
    assert body["user_id"] == test_user["user_id"]
    assert all(isinstance(value, str) for value in body.values())


def test_a_key_minted_without_a_user_answers_with_the_default_owner(
    client: TestClient, db_session: Session, api_key_obj: dict[str, Any]
) -> None:
    response = client.get(PATH, headers=_bearer(api_key_obj["key"]))

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == _expected(db_session, api_key_obj["id"])


def test_a_key_with_no_owner_reports_a_null_user(
    client: TestClient, db_session: Session, api_key_obj: dict[str, Any]
) -> None:
    _set_key(db_session, api_key_obj["id"], user_id=None)

    response = client.get(PATH, headers=_bearer(api_key_obj["key"]))

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["user_id"] is None
    assert response.json() == _expected(db_session, api_key_obj["id"])


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({}, id="missing"),
        pytest.param({"Authorization": "Basic dXNlcjpwYXNz"}, id="not-bearer"),
        pytest.param(_bearer("gw-not-a-real-key"), id="unknown"),
        pytest.param(_bearer("test-master-key"), id="master-key"),
        pytest.param({API_KEY_HEADER: "   "}, id="blank"),
    ],
)
def test_a_credential_that_names_no_key_is_one_401(client: TestClient, headers: dict[str, str]) -> None:
    response = client.get(PATH, headers=headers)

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json() == REFUSED
    assert _refuses_caching(response)


def test_a_damaged_key_is_one_401(client: TestClient, api_key_obj: dict[str, Any]) -> None:
    key = api_key_obj["key"]
    damaged = key[:-1] + ("A" if key[-1] != "A" else "B")

    response = client.get(PATH, headers=_bearer(damaged))

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json() == REFUSED


def test_a_dashboard_session_is_not_a_credential_here(client: TestClient) -> None:
    """Signed in with the master key, the browser holds a session cookie and no key."""
    signed_in = client.post(f"{API_ROOT}/auth/session", json={"master_key": "test-master-key"})
    assert signed_in.status_code < 300, signed_in.text
    assert client.cookies

    response = client.get(PATH)

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json() == REFUSED


@pytest.mark.parametrize(
    "values",
    [
        pytest.param({"is_active": False}, id="inactive"),
        pytest.param({"expires_at": datetime.now(UTC) - timedelta(days=1)}, id="expired"),
    ],
)
def test_a_key_that_is_no_longer_live_is_one_401(
    client: TestClient, db_session: Session, api_key_obj: dict[str, Any], values: dict[str, Any]
) -> None:
    _set_key(db_session, api_key_obj["id"], **values)

    response = client.get(PATH, headers=_bearer(api_key_obj["key"]))

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json() == REFUSED


def test_a_blocked_owner_is_one_401_and_unblocking_restores_the_key(
    client: TestClient, master_key_header: dict[str, str], test_user: dict[str, Any]
) -> None:
    created = _mint(client, master_key_header, user_id=test_user["user_id"])
    user_path = f"{API_ROOT}/users/{test_user['user_id']}"

    assert client.patch(user_path, json={"blocked": True}, headers=master_key_header).status_code == 200
    blocked = client.get(PATH, headers=_bearer(created["key"]))
    assert client.patch(user_path, json={"blocked": False}, headers=master_key_header).status_code == 200
    unblocked = client.get(PATH, headers=_bearer(created["key"]))

    assert blocked.status_code == status.HTTP_401_UNAUTHORIZED
    assert blocked.json() == REFUSED
    assert unblocked.status_code == status.HTTP_200_OK


def test_a_deleted_owner_is_one_401_even_while_the_key_is_active(
    client: TestClient, db_session: Session, master_key_header: dict[str, str], test_user: dict[str, Any]
) -> None:
    """Deleting a user through the API also deactivates its keys; this is the owner check on its own."""
    created = _mint(client, master_key_header, user_id=test_user["user_id"])
    db_session.execute(update(User).where(User.user_id == test_user["user_id"]).values(deleted_at=datetime.now(UTC)))
    db_session.commit()

    response = client.get(PATH, headers=_bearer(created["key"]))

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json() == REFUSED


def test_identifying_a_key_records_no_usage_and_reserves_nothing(
    client: TestClient, db_session: Session, api_key_obj: dict[str, Any]
) -> None:
    before = _records(db_session)

    assert client.get(PATH, headers=_bearer(api_key_obj["key"])).status_code == status.HTTP_200_OK
    assert client.get(PATH, headers=_bearer("gw-not-a-real-key")).status_code == status.HTTP_401_UNAUTHORIZED

    assert _records(db_session) == before


def test_the_route_is_published_with_its_refusal(client: TestClient) -> None:
    operation = client.get(f"{API_ROOT}/openapi.json").json()["paths"][PATH]["get"]

    assert operation["operationId"] == "key-identity-read_key_identity"
    assert {"200", "401", "503"} <= set(operation["responses"])


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
        auto_migrate=False,
        require_pricing=False,
        bootstrap=f"{MODULE}:register",
    )
    try:
        yield from build_test_client(config)
    finally:
        sys.modules.pop(MODULE, None)


@pytest.mark.parametrize(
    "prefix", [pytest.param("elsewhere-", id="misdirected"), pytest.param("broken-", id="malformed")]
)
def test_a_key_the_format_routes_away_is_one_401(rebound_client: TestClient, prefix: str) -> None:
    """Another deployment's key is a 421 elsewhere, but the forwarding service has nowhere to redirect to."""
    response = rebound_client.get(PATH, headers=_bearer(prefix + "x" * 60))

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json() == REFUSED


@pytest.fixture
def hosted_client(postgres_url: str, clean_database: None) -> Generator[TestClient]:
    yield from build_test_client(
        GatewayConfig(
            mode="hosted",
            database_url=postgres_url,
            master_key="test-master-key",
            auto_migrate=False,
            require_pricing=False,
            model_discovery=False,
        )
    )


def test_a_hosted_control_plane_serves_the_lookup(
    hosted_client: TestClient, db_session: Session, master_key_header: dict[str, str]
) -> None:
    created = _mint(hosted_client, master_key_header)

    response = hosted_client.get(PATH, headers=_bearer(created["key"]))

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.json() == _expected(db_session, created["id"])


def test_a_hybrid_gateway_does_not_serve_the_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hybrid gateway holds no key table; the control plane it reports to answers for its keys."""
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
    app = create_app(GatewayConfig(mode="hybrid", platform={"base_url": "http://localhost:8100/api/v1"}))

    try:
        with TestClient(app) as hybrid_client:
            response = hybrid_client.get(PATH, headers=_bearer("gw-any-key"))
    finally:
        reset_config()
        reset_db()

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert PATH not in app.openapi()["paths"]
