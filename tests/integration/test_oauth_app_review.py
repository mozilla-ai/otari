"""Consent origins, token revocation, and approval transaction boundaries."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlsplit

import pytest
from apron_auth.pkce import generate_code_challenge
from fastapi.testclient import TestClient
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from gateway.auth.models import hash_key
from gateway.core.config import API_KEY_HEADER, API_ROOT, X_API_KEY_HEADER, GatewayConfig
from gateway.models.api_keys import APIKey, DisabledReason
from gateway.models.identity import OAuthAppGrant
from gateway.repositories.api_keys.api_key_repository import ApiKeyRepository
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME

from .test_oauth_apps import REDIRECT, VERIFIER, approve, code_from, redeem
from .test_oauth_apps import test_config as test_config
from .test_organization_member_keys import _World
from .test_organization_member_keys import world as world


def _consent(world: _World) -> dict[str, str]:
    return dict(
        client_id="thunderbolt",
        redirect_uri=REDIRECT,
        state="s" * 43,
        code_challenge=generate_code_challenge(VERIFIER),
        code_challenge_method="S256",
        workspace_id=str(world.workspaces["alpha_one"]),
    )


@pytest.mark.parametrize("site", ["same-site", "cross-site", None])
def test_allowed_cors_client_cannot_approve_using_dashboard_cookie(
    client: TestClient,
    world: _World,
    test_config: GatewayConfig,
    site: str | None,
    db_session_factory: Callable[[], Session],
) -> None:
    test_config.cors_allow_origins = ["https://evil.example", "https://dashboard.example"]
    test_config.ui_base_url = "https://dashboard.example"
    headers = {"Origin": "https://evil.example"}
    if site is not None:
        headers["Sec-Fetch-Site"] = site
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions["alpha_member"])
    response = client.post(f"{API_ROOT}/oauth/apps/authorize", json=_consent(world), headers=headers)
    assert response.status_code == (401 if site == "cross-site" else 403)
    with db_session_factory() as db:
        assert db.scalars(select(OAuthAppGrant)).first() is None


def test_configured_dashboard_can_approve_across_same_site_origins(
    client: TestClient,
    world: _World,
    test_config: GatewayConfig,
) -> None:
    test_config.ui_base_url = "https://dashboard.example/ui"
    test_config.cors_allow_origins = ["https://dashboard.example"]
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions["alpha_member"])
    response = client.post(
        f"{API_ROOT}/oauth/apps/authorize",
        json=_consent(world),
        headers={"Origin": "https://dashboard.example", "Sec-Fetch-Site": "same-site"},
    )
    assert response.status_code == 200, response.text
    callback = parse_qs(urlsplit(response.json()["redirect_uri"]).query)
    assert callback["iss"] == ["http://testserver"]
    details = client.get(
        f"{API_ROOT}/oauth/apps/clients", params={"client_id": "thunderbolt", "redirect_uri": REDIRECT}
    )
    assert details.json()["issuer"] == "http://testserver"


def test_cookie_consent_without_origin_or_fetch_metadata_is_refused(client: TestClient, world: _World) -> None:
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions["alpha_member"])
    response = client.post(f"{API_ROOT}/oauth/apps/authorize", json=_consent(world))
    assert response.status_code == 403


@pytest.mark.parametrize(
    ("header", "prefix"),
    [(API_KEY_HEADER, ""), (API_KEY_HEADER, "Bearer "), ("Authorization", "Bearer "), (X_API_KEY_HEADER, "")],
)
def test_header_authenticated_scripts_can_approve_consent(
    client: TestClient, test_config: GatewayConfig, header: str, prefix: str
) -> None:
    """Every supported credential header works without browser fetch metadata."""
    headers = {header: f"{prefix}{test_config.master_key}"}
    workspaces = client.get(f"{API_ROOT}/oauth/apps/workspaces", headers=headers)
    assert workspaces.status_code == 200, workspaces.text
    response = client.post(
        f"{API_ROOT}/oauth/apps/authorize",
        headers=headers,
        json={
            "client_id": "thunderbolt",
            "redirect_uri": REDIRECT,
            "state": "s" * 43,
            "code_challenge": generate_code_challenge(VERIFIER),
            "code_challenge_method": "S256",
            "workspace_id": workspaces.json()["data"][0]["id"],
        },
    )
    assert response.status_code == 200, response.text
    assert redeem(client, code_from(response)).status_code == 200


@pytest.mark.parametrize("metadata", [{"Origin": "https://evil.example"}, {"Sec-Fetch-Site": "same-site"}])
def test_credential_headers_do_not_exempt_browser_requests_from_consent_origin_checks(
    client: TestClient,
    world: _World,
    test_config: GatewayConfig,
    metadata: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """A header-authenticated browser still needs the dashboard's consent origin."""
    assert test_config.master_key is not None
    response = client.post(
        f"{API_ROOT}/oauth/apps/authorize",
        json=_consent(world),
        headers={API_KEY_HEADER: test_config.master_key, **metadata},
    )
    assert response.status_code == 403
    with db_session_factory() as db:
        assert db.scalars(select(OAuthAppGrant)).first() is None


@pytest.mark.parametrize("clear_metadata", [False, True])
def test_token_revocation_is_self_scoped_idempotent_and_does_not_require_a_session(
    client: TestClient,
    world: _World,
    db_session_factory: Callable[[], Session],
    clear_metadata: bool,
) -> None:
    first = redeem(client, code_from(approve(client, world))).json()["access_token"]
    second = redeem(client, code_from(approve(client, world))).json()["access_token"]
    if clear_metadata:
        with db_session_factory() as db:
            db.execute(update(APIKey).values(metadata_={}))
            db.commit()
    for hint in ({"token_type_hint": "access_token"}, {"token_type_hint": "unknown"}, {}):
        response = client.post(f"{API_ROOT}/oauth/apps/revoke", data={"token": first, **hint})
        assert response.status_code == 200
        assert response.content == b""
        assert "no-store" in response.headers["cache-control"]
    assert client.get(f"{API_ROOT}/models", headers={"Authorization": f"Bearer {first}"}).status_code == 401
    assert client.get(f"{API_ROOT}/models", headers={"Authorization": f"Bearer {second}"}).status_code == 200
    assert client.post(f"{API_ROOT}/oauth/apps/revoke", data={"token": "unknown"}).status_code == 200


def test_revocation_never_disables_the_master_key(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    token = next(iter(master_key_header.values())).removeprefix("Bearer ")
    assert client.post(f"{API_ROOT}/oauth/apps/revoke", data={"token": token}).status_code == 200
    assert client.get(f"{API_ROOT}/models", headers=master_key_header).status_code == 200


def test_failed_approval_rolls_back_expired_grant_pruning(
    client: TestClient,
    world: _World,
    db_session_factory: Callable[[], Session],
) -> None:
    approve(client, world)
    with db_session_factory() as db:
        db.execute(update(OAuthAppGrant).values(expires_at=datetime.now(UTC) - timedelta(minutes=1)))
        db.commit()
    refused = approve(client, world, who="alpha_viewer")
    assert refused.status_code == 403
    with db_session_factory() as db:
        assert len(db.scalars(select(OAuthAppGrant)).all()) == 1


def test_token_refusals_log_a_reason_without_credentials(
    client: TestClient,
    world: _World,
    caplog: pytest.LogCaptureFixture,
) -> None:
    code = code_from(approve(client, world))
    assert redeem(client, code, code_verifier="w" * 43).status_code == 400
    assert "OAuth app exchange refused" in caplog.text
    assert "client_id=thunderbolt" in caplog.text
    assert "Invalid or expired authorization grant" in caplog.text
    assert code not in caplog.text
    assert "w" * 43 not in caplog.text


def test_unexpected_token_failure_is_not_disguised_as_an_invalid_grant(
    client: TestClient,
    world: _World,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway.exceptions import TenancyError

    code = code_from(approve(client, world))
    with monkeypatch.context() as patch:
        patch.setattr(ApiKeyRepository, "issue_app_key", AsyncMock(side_effect=TenancyError("Internal failure")))
        response = redeem(client, code)
    assert response.status_code == 500
    assert "Internal failure" not in response.text
    assert redeem(client, code).status_code == 200


@pytest.mark.parametrize("active", [False, True])
def test_revocation_permanently_disables_a_presented_manual_key_only(
    client: TestClient,
    world: _World,
    db_session_factory: Callable[[], Session],
    active: bool,
) -> None:
    other = redeem(client, code_from(approve(client, world))).json()["access_token"]
    token = "tk-" + "m" * 48
    with db_session_factory() as db:
        db.add(
            APIKey(
                id="manual",
                key_hash=hash_key(token),
                key_prefix="tk-mmm",
                user_id=str(world.users["alpha_member"]),
                workspace_id=world.workspaces["alpha_one"],
                is_active=active,
                self_service=True,
            )
        )
        db.commit()
    result = client.post(f"{API_ROOT}/oauth/apps/revoke", data={"token": token})
    assert result.status_code == 200
    with db_session_factory() as db:
        key = db.get(APIKey, "manual")
        assert key is not None and not key.is_active and key.disabled_reason is DisabledReason.REVOKED
    assert client.get(f"{API_ROOT}/models", headers={"Authorization": f"Bearer {other}"}).status_code == 200


@pytest.mark.parametrize(
    "content,content_type",
    [
        ("token=one&token=two", "application/x-www-form-urlencoded"),
        ("token=", "application/x-www-form-urlencoded"),
        ('{"token":"value"}', "application/json"),
    ],
)
def test_invalid_revocation_uses_oauth_error_contract(client: TestClient, content: str, content_type: str) -> None:
    result = client.post(f"{API_ROOT}/oauth/apps/revoke", content=content, headers={"Content-Type": content_type})
    assert result.status_code == 400
    assert result.json()["error"] == "invalid_request"
    assert "no-store" in result.headers["cache-control"]
