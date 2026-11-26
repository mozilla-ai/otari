"""Rotating ``OTARI_SECRET_KEY`` re-seals every organization provider credential.

Through the route, because the claim is about the whole path: a key stores
three encrypted values, the operator re-seals them, and the retired secret is
then removed. A rotation that covered two of the three would pass every check
until that last step, which is the one nobody undoes.
"""

from collections.abc import Generator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.models.provider_keys import OrgProviderKey
from gateway.services.secret_box import decrypt_secret, generate_secret_key
from gateway.services.tenancy.org_provider_key_service import reset_org_provider_cache

from .conftest import shared_test_config

pytestmark = pytest.mark.usefixtures("public_dns")

_MASTER_HEADER = {API_KEY_HEADER: "Bearer test-master-key"}
_KEYS = f"{API_ROOT}/organizations/me/provider-keys"
_REENCRYPT = f"{API_ROOT}/organization-provider-keys/reencrypt"

_API_KEY = "sk-live-abcd"
_ENDPOINT_SECRET = "edge-s3cret"
_CLIENT_ARGS = {"timeout": 30}


@pytest.fixture(scope="module")
def test_config(postgres_url: str) -> GatewayConfig:
    """The shared configuration declaring no provider, so the organization's key stands alone."""
    return shared_test_config(postgres_url, {})


@pytest.fixture(autouse=True)
def _clean_overlay() -> Generator[None]:
    reset_org_provider_cache()
    yield
    reset_org_provider_cache()


def _create_key(client: TestClient) -> None:
    """One key holding all three of the encrypted values a key can hold."""
    response = client.post(
        _KEYS,
        headers=_MASTER_HEADER,
        json={
            "provider": "openai",
            "name": "Production",
            "api_key": _API_KEY,
            "api_base": "https://example.com/v1",
            "client_args": _CLIENT_ARGS,
            "endpoint_auth": {
                "method": "static_header",
                "header_name": "Fastly-Key",
                "secret": _ENDPOINT_SECRET,
                "expires_at": (datetime.now(UTC) + timedelta(days=90)).isoformat(),
            },
        },
    )
    assert response.status_code == status.HTTP_201_CREATED, response.text


def _stored(db_session: Session) -> OrgProviderKey:
    db_session.expire_all()
    return db_session.execute(select(OrgProviderKey)).scalars().one()


def test_every_encrypted_value_survives_removing_the_retired_secret(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    retired, incoming = generate_secret_key(), generate_secret_key()
    monkeypatch.setenv("OTARI_SECRET_KEY", retired)
    _create_key(client)

    monkeypatch.setenv("OTARI_SECRET_KEY", f"{incoming},{retired}")
    response = client.post(_REENCRYPT, headers=_MASTER_HEADER)

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.json() == {"reencrypted": 1, "unreadable": 0, "skipped": 0}
    monkeypatch.setenv("OTARI_SECRET_KEY", incoming)
    row = _stored(db_session)
    assert decrypt_secret(row.encrypted_api_key or "") == _API_KEY
    assert decrypt_secret(row.encrypted_client_args or "") == '{"timeout": 30}'
    assert decrypt_secret(row.encrypted_endpoint_auth_secret or "") == _ENDPOINT_SECRET


def test_a_key_no_configured_secret_reads_is_counted_and_left_alone(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _create_key(client)
    before = _stored(db_session).encrypted_api_key

    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    response = client.post(_REENCRYPT, headers=_MASTER_HEADER)

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.json() == {"reencrypted": 0, "unreadable": 1, "skipped": 0}
    assert _stored(db_session).encrypted_api_key == before
