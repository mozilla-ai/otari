"""App grants keep member authority and require the original S256 proof."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Event
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest
from apron_auth.pkce import generate_code_challenge as pkce_challenge
from fastapi.testclient import TestClient
from httpx2 import Response
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.settings.identity import OAuthAppConfig
from gateway.models.api_keys import APIKey
from gateway.models.identity import OAuthAppGrant
from gateway.models.tenancy import OrganizationMember, User, Workspace, WorkspaceMember
from gateway.models.users import User as BillingUser
from gateway.rate_limit import RateLimiter
from gateway.repositories.api_keys.api_key_repository import ApiKeyRepository
from gateway.repositories.identity.oauth_app_grant_repository import OAuthAppGrantRepository
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME

from .conftest import shared_test_config
from .test_organization_member_keys import _World
from .test_organization_member_keys import world as world

REDIRECT = "http://localhost:5199/oauth/otari"
VERIFIER = "v" * 43


@pytest.fixture
def test_config(postgres_url: str) -> GatewayConfig:
    config = shared_test_config(postgres_url, {})
    config.oauth_apps = {"thunderbolt": OAuthAppConfig(name="Thunderbolt", redirect_uris=[REDIRECT])}
    return config


def approve(
    client: TestClient, world: _World, who: str = "alpha_member", workspace: str = "alpha_one", **changes: str
) -> Response:
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions[who])
    body = dict(
        client_id="thunderbolt",
        redirect_uri=REDIRECT,
        state="s" * 43,
        code_challenge=pkce_challenge(VERIFIER),
        code_challenge_method="S256",
        workspace_id=str(world.workspaces[workspace]),
    )
    body.update(changes)
    return client.post(f"{API_ROOT}/oauth/apps/authorize", json=body, headers={"Sec-Fetch-Site": "same-origin"})


def redeem(client: TestClient, code: str, **changes: str) -> Response:
    body = dict(
        grant_type="authorization_code",
        client_id="thunderbolt",
        redirect_uri=REDIRECT,
        code=code,
        code_verifier=VERIFIER,
    )
    body.update(changes)
    client.cookies.clear()
    return client.post(f"{API_ROOT}/oauth/apps/token", data=body)


def code_from(response: Response) -> str:
    assert response.status_code == 200, response.text
    return parse_qs(urlsplit(str(response.json()["redirect_uri"])).query)["code"][0]


def test_approved_key_is_owned_budget_enforced_and_single_use(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    code = code_from(approve(client, world))
    with db_session_factory() as db:
        assert db.scalars(select(APIKey).where(APIKey.user_id == str(world.users["alpha_member"]))).first() is None
        grant = db.scalars(select(OAuthAppGrant)).one()
        assert code not in grant.code_hash
        assert grant.user_id == world.users["alpha_member"]
        assert grant.workspace_id == world.workspaces["alpha_one"]
    wrong = redeem(client, code, code_verifier="x" * 43)
    assert wrong.status_code == 400
    result = redeem(client, code)
    assert result.status_code == 200, result.text
    assert "no-store" in result.headers["cache-control"]
    assert result.json()["token_type"] == "Bearer"
    with db_session_factory() as db:
        key = db.scalars(select(APIKey).where(APIKey.user_id == str(world.users["alpha_member"]))).one()
        assert key.workspace_id == world.workspaces["alpha_one"]
        assert key.self_service and not key.exclude_from_budget and not key.is_service_key
        assert key.metadata_["oauth_client_id"] == "thunderbolt"
        # The registration names no scopes, so the key calls models and nothing else.
        assert key.scopes == ["inference"]
        assert db.scalars(select(OAuthAppGrant)).first() is None
    assert redeem(client, code).status_code == 400
    headers = {"Authorization": f"Bearer {result.json()['access_token']}"}
    assert client.get(f"{API_ROOT}/models", headers=headers).status_code == 200
    with db_session_factory() as db:
        db.execute(update(APIKey).where(APIKey.id == key.id).values(is_active=False))
        db.commit()
    assert client.get(f"{API_ROOT}/models", headers=headers).status_code in (401, 403)


@pytest.mark.parametrize(
    "who,workspace",
    [
        ("alpha_viewer", "alpha_one"),
        ("alpha_workspace_viewer", "alpha_one"),
        ("alpha_viewer_promoted", "alpha_one"),
        ("alpha_newcomer", "alpha_one"),
        ("alpha_member", "alpha_two"),
        ("alpha_member", "beta_one"),
        ("impostor", "alpha_one"),
    ],
)
def test_cannot_expand_workspace_authority(client: TestClient, world: _World, who: str, workspace: str) -> None:
    assert approve(client, world, who, workspace).status_code in (403, 404)


@pytest.mark.parametrize(
    "changes",
    [
        {"redirect_uri": "https://attacker.invalid/callback"},
        {"client_id": "unknown"},
        {"code_challenge_method": "plain"},
    ],
)
def test_refuses_unregistered_apps_redirects_and_plain_pkce(
    client: TestClient, world: _World, changes: dict[str, str]
) -> None:
    assert approve(client, world, **changes).status_code in (400, 422)


def test_expired_grant_and_wrong_redirect(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    code = code_from(approve(client, world))
    assert redeem(client, code, redirect_uri="http://localhost:5199/other").status_code == 400
    with db_session_factory() as db:
        db.execute(update(OAuthAppGrant).values(expires_at=datetime.now(UTC) - timedelta(seconds=1)))
        db.commit()
    assert redeem(client, code).status_code == 400


def test_consent_requires_authenticated_same_site_session(client: TestClient, world: _World) -> None:
    client.cookies.clear()
    assert client.post(f"{API_ROOT}/oauth/apps/authorize", json={}).status_code in (401, 403)
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions["alpha_member"])
    assert client.post(
        f"{API_ROOT}/oauth/apps/authorize", json={}, headers={"Sec-Fetch-Site": "cross-site"}
    ).status_code in (401, 403)


def test_concurrent_redemption_has_one_winner(client: TestClient, world: _World) -> None:
    code = code_from(approve(client, world))
    client.cookies.clear()
    body = dict(
        grant_type="authorization_code",
        client_id="thunderbolt",
        redirect_uri=REDIRECT,
        code=code,
        code_verifier=VERIFIER,
    )

    def exchange() -> int:
        return client.post(f"{API_ROOT}/oauth/apps/token", data=body).status_code

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(exchange)
        second = executor.submit(exchange)
        assert sorted([first.result(), second.result()]) == [200, 400]


@pytest.mark.parametrize("change", ["identity", "organization", "workspace", "blocked", "deleted"])
def test_authority_is_rechecked_at_exchange(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session], change: str
) -> None:
    code = code_from(approve(client, world))
    user_id = world.users["alpha_member"]
    with db_session_factory() as db:
        if change == "identity":
            db.execute(update(User).where(col(User.id) == user_id).values(is_active=False))
        elif change == "organization":
            db.execute(
                update(OrganizationMember).where(col(OrganizationMember.user_id) == user_id).values(status="suspended")
            )
        elif change == "workspace":
            db.execute(update(WorkspaceMember).where(col(WorkspaceMember.user_id) == user_id).values(role="viewer"))
        else:
            db.add(
                BillingUser(
                    user_id=str(user_id),
                    blocked=change == "blocked",
                    deleted_at=datetime.now(UTC) if change == "deleted" else None,
                )
            )
        db.commit()
    refused = redeem(client, code)
    assert refused.status_code == 400
    assert refused.json()["error"] == "invalid_grant"
    with db_session_factory() as db:
        assert db.scalars(select(APIKey).where(APIKey.user_id == str(user_id))).first() is None


def test_app_key_inherits_current_and_future_owner_restrictions(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    code = code_from(approve(client, world))
    user_id = str(world.users["alpha_member"])
    with db_session_factory() as db:
        db.add(BillingUser(user_id=user_id, allowed_models=["openai:gpt-4o"], allowed_tools=[]))
        db.commit()
    result = redeem(client, code)
    assert result.status_code == 200
    headers = {"Authorization": f"Bearer {result.json()['access_token']}"}
    with db_session_factory() as db:
        key = db.scalars(select(APIKey).where(APIKey.user_id == user_id)).one()
        assert key.allowed_models is None and key.allowed_tools is None
        db.execute(update(BillingUser).where(BillingUser.user_id == user_id).values(allowed_models=[]))
        db.commit()
    response = client.get(f"{API_ROOT}/models", headers=headers)
    assert response.status_code == 200
    assert response.json()["data"] == []
    refused = client.post(
        f"{API_ROOT}/chat/completions",
        headers=headers,
        json={"model": "openai:gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert refused.status_code == 403
    assert "not permitted" in str(refused.json())


def test_exchange_is_bound_to_client_and_exact_callback(
    client: TestClient, world: _World, test_config: GatewayConfig
) -> None:
    test_config.oauth_apps["other"] = OAuthAppConfig(name="Other", redirect_uris=[REDIRECT])
    code = code_from(approve(client, world))
    assert redeem(client, code, client_id="other").status_code == 400
    # Both ports are registered through the loopback exception, but one code binds exactly one.
    assert redeem(client, code, redirect_uri="http://localhost:5200/oauth/otari").status_code == 400
    assert redeem(client, code).status_code == 200


def test_removed_app_cannot_exchange_a_pending_approval(
    client: TestClient, world: _World, test_config: GatewayConfig
) -> None:
    code = code_from(approve(client, world))
    test_config.oauth_apps.clear()
    assert redeem(client, code).status_code == 400


def test_expired_approvals_are_pruned_without_creating_keys(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    old = code_from(approve(client, world))
    with db_session_factory() as db:
        db.execute(update(OAuthAppGrant).values(expires_at=datetime.now(UTC) - timedelta(seconds=1)))
        db.commit()
    code_from(approve(client, world))
    assert redeem(client, old).status_code == 400
    with db_session_factory() as db:
        assert len(db.scalars(select(OAuthAppGrant)).all()) == 1
        assert db.scalars(select(APIKey).where(APIKey.user_id == str(world.users["alpha_member"]))).first() is None


def test_key_issuance_failure_rolls_back_grant_consumption(
    client: TestClient, world: _World, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = code_from(approve(client, world))
    with monkeypatch.context() as context:
        context.setattr(ApiKeyRepository, "issue_app_key", AsyncMock(side_effect=RuntimeError("test issuance failure")))
        with pytest.raises(RuntimeError, match="test issuance failure"):
            redeem(client, code)
    assert redeem(client, code).status_code == 200


@pytest.mark.parametrize(
    "redirect", ["http://[bad/callback", "http://localhost:bad/oauth/otari", "http://localhost:999999/oauth/otari"]
)
def test_malformed_callback_is_a_client_error(client: TestClient, world: _World, redirect: str) -> None:
    assert (
        client.get(
            f"{API_ROOT}/oauth/apps/clients", params={"client_id": "thunderbolt", "redirect_uri": redirect}
        ).status_code
        == 400
    )
    assert approve(client, world, redirect_uri=redirect).status_code == 400


def test_public_token_exchange_is_rate_limited(
    client: TestClient, world: _World, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = code_from(approve(client, world))
    monkeypatch.setattr(client.app.state, "public_auth_rate_limiter", RateLimiter(1))  # type: ignore[attr-defined]
    assert redeem(client, code, code_verifier="x" * 43).status_code == 400
    assert redeem(client, code).status_code == 429


def test_authorization_returns_original_state_without_a_credential(client: TestClient, world: _World) -> None:
    response = approve(client, world, state="state.with-allowed_characters~123")
    assert response.status_code == 200
    assert "no-store" in response.headers["cache-control"]
    callback = urlsplit(response.json()["redirect_uri"])
    values = parse_qs(callback.query)
    assert values["state"] == ["state.with-allowed_characters~123"]
    assert set(values) == {"code", "state", "iss"}
    assert values["iss"] == ["http://testserver"]
    assert len(values["code"][0]) == 43


@pytest.mark.parametrize("change", ["delete", "expire", "demote"])
def test_exchange_does_not_deadlock_or_use_stale_authority_while_waiting_for_workspace(
    client: TestClient,
    world: _World,
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    code = code_from(approve(client, world))
    observed = Event()
    original = OAuthAppGrantRepository.find_bound

    async def found(
        self: OAuthAppGrantRepository,
        *,
        code_hash: str,
        client_id: str,
        redirect_uri: str,
        challenge: str,
        now: datetime,
    ) -> OAuthAppGrant | None:
        grant = await original(
            self, code_hash=code_hash, client_id=client_id, redirect_uri=redirect_uri, challenge=challenge, now=now
        )
        observed.set()
        return grant

    monkeypatch.setattr(OAuthAppGrantRepository, "find_bound", found)
    with db_session_factory() as db, ThreadPoolExecutor(max_workers=1) as executor:
        db.execute(select(Workspace).where(col(Workspace.id) == world.workspaces["alpha_one"]).with_for_update())
        pending = executor.submit(redeem, client, code)
        try:
            assert observed.wait(timeout=5), "exchange did not read the pending grant"
            if change == "delete":
                db.execute(delete(Workspace).where(col(Workspace.id) == world.workspaces["alpha_one"]))
            elif change == "expire":
                db.execute(update(OAuthAppGrant).values(expires_at=datetime.now(UTC) - timedelta(seconds=1)))
            else:
                db.execute(
                    update(WorkspaceMember)
                    .where(col(WorkspaceMember.user_id) == world.users["alpha_member"])
                    .values(role="viewer")
                )
            db.commit()
        finally:
            db.rollback()  # Release the lock even when an assertion fails, before joining the worker.
        response = pending.result(timeout=10)
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_grant"


def test_standard_authorization_url_preserves_request_for_consent(client: TestClient, world: _World) -> None:
    params = dict(
        client_id="thunderbolt",
        redirect_uri=REDIRECT,
        state="s" * 43,
        response_type="code",
        code_challenge=pkce_challenge(VERIFIER),
        code_challenge_method="S256",
    )
    response = client.get(f"{API_ROOT}/oauth/apps/authorize", params=params, follow_redirects=False)
    assert response.status_code == 302
    assert "no-store" in response.headers["cache-control"]
    assert response.headers["location"].startswith("/#/oauth/authorize?")
    assert (
        dict((key, values[0]) for key, values in parse_qs(response.headers["location"].split("?", 1)[1]).items())
        == params
    )
    params["redirect_uri"] = "https://attacker.invalid/callback"
    refused = client.get(f"{API_ROOT}/oauth/apps/authorize", params=params, follow_redirects=False)
    assert refused.status_code == 400
    assert "location" not in refused.headers


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"grant_type": "password"}, "unsupported_grant_type"),
        ({"code_verifier": "short"}, "invalid_request"),
        ({"client_id": "unknown"}, "invalid_client"),
        ({"code": "c" * 43}, "invalid_grant"),
    ],
)
def test_token_errors_follow_oauth_and_are_not_cached(
    client: TestClient, world: _World, changes: dict[str, str], error: str
) -> None:
    code = code_from(approve(client, world))
    response = redeem(
        client, changes.get("code", code), **{key: value for key, value in changes.items() if key != "code"}
    )
    assert response.status_code == 400
    assert response.json()["error"] == error
    assert set(response.json()) == {"error", "error_description"}
    assert "no-store" in response.headers["cache-control"]
    assert response.headers["pragma"] == "no-cache"


@pytest.mark.parametrize("change", ["workspace_removed", "workspace_viewer", "organization_viewer", "identity_revoked"])
def test_issued_app_keys_follow_existing_membership_and_identity_revocation(
    client: TestClient,
    world: _World,
    db_session_factory: Callable[[], Session],
    change: str,
) -> None:
    result = redeem(client, code_from(approve(client, world)))
    assert result.status_code == 200
    headers = {"Authorization": f"Bearer {result.json()['access_token']}"}
    assert client.get(f"{API_ROOT}/models", headers=headers).status_code == 200
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions["alpha_operator"])
    member = f"{API_ROOT}/workspaces/{world.workspaces['alpha_one']}/members/{world.users['alpha_member']}"
    if change == "workspace_removed":
        changed = client.delete(member)
    elif change == "workspace_viewer":
        changed = client.patch(member, params={"role": "viewer"})
    elif change == "organization_viewer":
        with db_session_factory() as db:
            membership = db.scalars(
                select(OrganizationMember).where(col(OrganizationMember.user_id) == world.users["alpha_member"])
            ).one()
        changed = client.patch(f"{API_ROOT}/organizations/me/members/{membership.id}", json={"role": "viewer"})
    else:
        changed = client.delete(f"{API_ROOT}/users/{world.users['alpha_member']}")
    assert changed.status_code in (200, 204), changed.text
    client.cookies.clear()
    assert client.get(f"{API_ROOT}/models", headers=headers).status_code in (401, 403)
    refused = client.post(
        f"{API_ROOT}/chat/completions",
        headers=headers,
        json={"model": "openai:gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert refused.status_code in (401, 403)


def test_duplicate_authorization_parameters_are_refused_without_redirect(client: TestClient) -> None:
    params = urlencode([("client_id", "thunderbolt"), ("client_id", "other")])
    response = client.get(f"{API_ROOT}/oauth/apps/authorize", params=params, follow_redirects=False)
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"
    assert "location" not in response.headers


def test_duplicate_token_parameters_do_not_consume_the_grant(client: TestClient, world: _World) -> None:
    code = code_from(approve(client, world))
    fields = [
        ("grant_type", "authorization_code"),
        ("client_id", "thunderbolt"),
        ("redirect_uri", REDIRECT),
        ("code", code),
        ("code_verifier", VERIFIER),
        ("code_verifier", "x" * 43),
    ]
    response = client.post(
        f"{API_ROOT}/oauth/apps/token",
        content=urlencode(fields),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"
    assert redeem(client, code).status_code == 200


def test_client_metadata_names_do_not_collide_with_protocol_endpoints(
    client: TestClient, test_config: GatewayConfig
) -> None:
    test_config.oauth_apps["authorize"] = OAuthAppConfig(name="Authorize", redirect_uris=[REDIRECT])
    response = client.get(f"{API_ROOT}/oauth/apps/clients", params={"client_id": "authorize", "redirect_uri": REDIRECT})
    assert response.status_code == 200
    assert response.json()["name"] == "Authorize"


def test_authorization_uses_the_deployments_configured_dashboard(
    client: TestClient, test_config: GatewayConfig
) -> None:
    test_config.ui_base_url = "https://example.com/dashboard?edge=one"
    params = dict(
        client_id="thunderbolt",
        redirect_uri=REDIRECT,
        state="s" * 43,
        response_type="code",
        code_challenge=pkce_challenge(VERIFIER),
        code_challenge_method="S256",
    )
    response = client.get(f"{API_ROOT}/oauth/apps/authorize", params=params, follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"].startswith("https://example.com/dashboard/?edge=one#/oauth/authorize?")
