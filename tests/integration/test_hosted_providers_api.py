"""The hosted-providers operator API over HTTP.

The service's rules are pinned in `test_hosted_providers_service.py`; this file
pins the surface: who reaches it and how it refuses, that no response body ever
carries key material, and that each route reaches its use case. Model discovery
is stubbed, so no case dials a provider.
"""

import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from any_llm.types.model import Model
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.core.config import API_ROOT
from gateway.models.tenancy import DashboardSession, Organization, OrganizationMember, User
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.secret_box import generate_secret_key

PATH = f"{API_ROOT}/hosted-providers"
KEY = "sk-live-hosted-1234"


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


@pytest.fixture(autouse=True)
def _discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stub(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(
            provider=impl_name, models=[Model(id="gpt-4o", object="model", created=0, owned_by=impl_name)]
        )

    monkeypatch.setattr("gateway.services.providers._hosted_provider_service.test_provider_credentials", _stub)


def _session_for(
    session_factory: Callable[[], Session], *, email: str, role: str = "member", is_superuser: bool = False
) -> str:
    """An identity in the default organization holding a live dashboard session, and its cookie."""
    session = session_factory()
    try:
        organization_id = session.query(Organization).filter(col(Organization.slug) == "default").one().id
        user = User(
            email=email,
            full_name=email.split("@")[0].title(),
            active_organization_id=organization_id,
            is_superuser=is_superuser,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        session.add(OrganizationMember(organization_id=organization_id, user_id=user.id, role=role, status="active"))
        token = f"otari-sess-{email}"
        session.add(
            DashboardSession(
                token_hash=hash_session_token(token),
                user_id=user.id,
                created_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(hours=12),
            )
        )
        session.commit()
        return token
    finally:
        session.close()


def _create(client: TestClient, headers: dict[str, str], provider: str = "openai", **extra: Any) -> dict[str, Any]:
    response = client.post(PATH, json={"provider": provider, "api_key": KEY, **extra}, headers=headers)
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def _walk(client: TestClient, headers: dict[str, str]) -> list[tuple[str, int, str]]:
    """Every route once, in the order the page uses them, returning each response body."""
    seen: list[tuple[str, int, str]] = []

    def record(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = client.request(method, path, headers=headers, **kwargs)
        seen.append((f"{method} {path}", response.status_code, response.text))
        return response.json() if response.content else {}

    record("POST", PATH, json={"provider": "openai", "api_key": KEY, "client_args": {"aws_secret_access_key": "shh"}})
    record("GET", PATH)
    record("GET", f"{PATH}/openai/models")
    record("GET", f"{PATH}/openai/available-models")
    model = record(
        "POST",
        f"{PATH}/openai/models",
        json={"model": "o3", "input_price_per_million": 1, "output_price_per_million": 2},
    )
    record("PATCH", f"{PATH}/openai/models/{model['id']}", json={"enabled": False})
    record("POST", f"{PATH}/openai/models/refresh")
    record("POST", f"{PATH}/catalog-update", json={})
    record("PATCH", f"{PATH}/openai", json={"api_key": "sk-live-rotated-9876", "enabled": False})
    record("DELETE", f"{PATH}/openai/models/{model['id']}")
    record("DELETE", f"{PATH}/openai")
    return seen


def test_no_response_body_anywhere_in_the_walk_carries_key_material(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    for call, status_code, body in _walk(client, master_key_header):
        assert status_code < 300, (call, body)
        assert "sk-live" not in body, call
        assert "shh" not in body, call
        assert "encrypted_api_key" not in body, call


def test_no_error_body_carries_the_key_material_the_request_sent(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """A refusal is typed, so it names what was wrong and never echoes what was sent."""
    _create(client, master_key_header, api_base="https://proxy.example.com/v1")
    sent = "sk-live-refused-5678"
    secret = "shh-refused"
    refusals = [
        # Already configured: 409.
        ("POST", PATH, {"provider": "openai", "api_key": sent, "client_args": {"aws_secret_access_key": secret}}),
        # No such implementation: 400.
        ("POST", PATH, {"provider": "nope", "api_key": sent, "client_args": {"aws_secret_access_key": secret}}),
        # Too short a body to be a provider: 422, whose detail echoes input by default.
        ("POST", PATH, {"api_key": sent, "client_args": {"aws_secret_access_key": secret}}),
        # No such provider: 404.
        ("PATCH", f"{PATH}/anthropic", {"api_key": sent}),
        # Moved to another origin without the key: 400.
        ("PATCH", f"{PATH}/openai", {"api_base": f"https://user:{secret}@elsewhere.example.com/v1"}),
        # The mask, edited: 400.
        ("PATCH", f"{PATH}/openai", {"api_base": "https://user:***@proxy.example.com/v2", "api_key": sent}),
        # Half a price: 422.
        ("POST", f"{PATH}/openai/models", {"model": sent, "input_price_per_million": 1}),
    ]

    for method, path, body in refusals:
        response = client.request(method, path, json=body, headers=master_key_header)

        assert 400 <= response.status_code < 500, (method, path, response.text)
        assert sent not in response.text, (method, path, response.text)
        assert secret not in response.text, (method, path, response.text)
        assert KEY not in response.text, (method, path, response.text)


def test_a_credential_in_the_base_url_is_masked_and_an_echo_keeps_it(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    """The edit form sends back the base it was shown, so the mask has to read as "keep what is stored"."""
    stored = "https://proxy-user:proxy-pass@proxy.example.com/v1?token=t0ken"
    masked = "https://proxy-user:***@proxy.example.com/v1?token=***"

    created = _create(client, master_key_header, api_base=stored)
    listed = client.get(PATH, headers=master_key_header).json()["data"][0]
    echoed = client.patch(f"{PATH}/openai", json={"api_base": masked, "enabled": False}, headers=master_key_header)

    assert created["api_base"] == masked
    assert listed["api_base"] == masked
    assert echoed.status_code == 200, echoed.text
    assert echoed.json()["api_base"] == masked
    assert _stored_api_base(db_session_factory) == stored

    edited = client.patch(
        f"{PATH}/openai", json={"api_base": "https://proxy-user:***@proxy.example.com/v2"}, headers=master_key_header
    )
    assert edited.status_code == 400
    assert _stored_api_base(db_session_factory) == stored

    replaced = client.patch(
        f"{PATH}/openai", json={"api_base": "https://proxy.example.com/v2"}, headers=master_key_header
    )
    assert replaced.json()["api_base"] == "https://proxy.example.com/v2"


def test_a_base_url_that_will_not_parse_is_shown_as_none(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    """Nothing can find the credential in an unparseable base to mask it, so none of it is shown."""
    stored = "https://proxy-user:proxy-pass@[proxy.example.com/v1?token=t0ken"

    created = _create(client, master_key_header, api_base=stored)
    listed = client.get(PATH, headers=master_key_header).json()["data"][0]

    assert (created["api_base"], listed["api_base"]) == (None, None)
    assert _stored_api_base(db_session_factory) == stored


def _stored_api_base(session_factory: Callable[[], Session]) -> str | None:
    session = session_factory()
    try:
        value: str | None = session.execute(text("SELECT api_base FROM hosted_providers")).scalar_one()
        return value
    finally:
        session.close()


def test_a_configured_provider_comes_back_by_its_tail_and_offers_what_it_listed(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    created = _create(client, master_key_header, api_base="https://api.openai.com/v1")

    assert created["provider"] == "openai"
    assert created["api_key_last4"] == "1234"
    assert "api_key" not in created
    listed = client.get(PATH, headers=master_key_header).json()
    assert listed["count"] == 1
    models = client.get(f"{PATH}/openai/models", headers=master_key_header).json()
    assert [row["model"] for row in models["data"]] == ["gpt-4o"]


def test_a_second_provider_for_the_same_implementation_is_a_409(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _create(client, master_key_header)

    response = client.post(PATH, json={"provider": "openai-compatible", "api_key": KEY}, headers=master_key_header)

    assert response.status_code == 409


def test_an_unknown_provider_is_a_400_and_an_unknown_row_a_404(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    refused = client.post(PATH, json={"provider": "nope", "api_key": KEY}, headers=master_key_header)
    assert refused.status_code == 400

    missing = client.patch(f"{PATH}/anthropic", json={"enabled": False}, headers=master_key_header)
    assert missing.status_code == 404
    assert client.get(f"{PATH}/anthropic/models", headers=master_key_header).status_code == 404
    assert client.delete(f"{PATH}/openai/models/{uuid.uuid4()}", headers=master_key_header).status_code == 404


def test_half_a_price_is_a_422(client: TestClient, master_key_header: dict[str, str]) -> None:
    _create(client, master_key_header)

    response = client.post(
        f"{PATH}/openai/models", json={"model": "o3", "input_price_per_million": 1}, headers=master_key_header
    )

    assert response.status_code == 422


def test_without_a_secret_key_a_create_is_a_500_that_names_no_library(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing secret key is a deployment gap the caller cannot fix, so it is not blamed on the request."""
    monkeypatch.delenv("OTARI_SECRET_KEY")

    response = client.post(PATH, json={"provider": "openai", "api_key": KEY}, headers=master_key_header)

    assert response.status_code == 500
    assert "Fernet" not in response.text
    assert client.get(PATH, headers=master_key_header).json()["count"] == 0


def test_the_surface_refuses_a_request_with_no_credential(client: TestClient) -> None:
    assert client.get(PATH).status_code == 401


def test_a_signed_in_member_is_told_the_surface_is_not_there(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    """404 rather than 403, like /api/v1/admin: the deployment's own credentials are
    not a surface to confirm to a member, on a read or on a write."""
    _create(client, master_key_header)
    token = _session_for(db_session_factory, email="ada@example.com")

    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        read = client.get(PATH)
        write = client.post(PATH, json={"provider": "anthropic", "api_key": KEY})
    finally:
        client.cookies.clear()

    assert read.status_code == 404
    assert write.status_code == 404
    assert client.get(PATH, headers=master_key_header).json()["count"] == 1


def test_an_organization_owner_is_still_not_the_operator(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    _create(client, master_key_header)
    token = _session_for(db_session_factory, email="owner@example.com", role="owner")

    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        assert client.get(PATH).status_code == 404
    finally:
        client.cookies.clear()


def test_a_deployment_superuser_reaches_the_surface_with_a_session(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    _create(client, master_key_header)
    token = _session_for(db_session_factory, email="root@example.com", is_superuser=True)

    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        response = client.get(PATH)
    finally:
        client.cookies.clear()

    assert response.status_code == 200
    assert response.json()["count"] == 1
