"""An unregistered site connects by its origin when the operator opens app authorization."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from urllib.parse import parse_qs, urlsplit

import pytest
from apron_auth.pkce import generate_code_challenge as pkce_challenge
from fastapi.testclient import TestClient
from httpx2 import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.api_keys import APIKey
from gateway.models.budgets import SCOPE_API_TOKEN, Budget, ScopedBudget
from gateway.models.tenancy import Workspace
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME
from gateway.types.key_scope import DEFAULT_KEY_SCOPES

from .conftest import shared_test_config
from .test_organization_member_keys import _World
from .test_organization_member_keys import world as world

SITE = "https://someone.github.io"
REDIRECT = f"{SITE}/pizza/"
VERIFIER = "v" * 43
STATE = "s" * 43


@pytest.fixture
def test_config(postgres_url: str) -> GatewayConfig:
    config = shared_test_config(postgres_url, {})
    config.oauth_open_apps = True
    return config


def authorization_query(**changes: str) -> dict[str, str]:
    query = dict(
        client_id=SITE,
        redirect_uri=REDIRECT,
        response_type="code",
        state=STATE,
        code_challenge=pkce_challenge(VERIFIER),
        code_challenge_method="S256",
    )
    query.update(changes)
    return query


def connect(client: TestClient, world: _World, **consent: object) -> Response:
    """Approve the site for alpha_member's first workspace and redeem the code, as the site would."""
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions["alpha_member"])
    approved = client.post(
        f"{API_ROOT}/oauth/apps/authorize",
        json={
            **authorization_query(),
            "workspace_id": str(world.workspaces["alpha_one"]),
            "scopes": [scope.value for scope in DEFAULT_KEY_SCOPES],
            **consent,
        },
        headers={"Sec-Fetch-Site": "same-origin"},
    )
    assert approved.status_code == 200, approved.text
    back = urlsplit(approved.json()["redirect_uri"])
    assert f"{back.scheme}://{back.netloc}{back.path}" == REDIRECT
    client.cookies.clear()
    return client.post(
        f"{API_ROOT}/oauth/apps/token",
        data=dict(
            grant_type="authorization_code",
            client_id=SITE,
            redirect_uri=REDIRECT,
            code=parse_qs(back.query)["code"][0],
            code_verifier=VERIFIER,
        ),
    )


def the_key(world: _World, db: Session) -> APIKey:
    return db.scalars(select(APIKey).where(APIKey.user_id == str(world.users["alpha_member"]))).one()


def test_a_site_connects_by_its_origin_and_its_key_is_named_for_it(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    start = client.get(f"{API_ROOT}/oauth/apps/authorize", params=authorization_query(), follow_redirects=False)
    assert start.status_code == 302, start.text
    details = client.get(f"{API_ROOT}/oauth/apps/clients", params={"client_id": SITE, "redirect_uri": REDIRECT})
    assert details.status_code == 200, details.text
    assert details.json()["name"] == "someone.github.io"
    assert details.json()["verified"] is False

    token = connect(client, world)
    assert token.status_code == 200, token.text
    assert token.json()["manage_url"].endswith("/keys")
    with db_session_factory() as db:
        key = the_key(world, db)
        assert key.key_name == "someone.github.io (OAuth)"
        assert key.metadata_["oauth_client_id"] == SITE
        assert key.scopes.stored == [scope.value for scope in DEFAULT_KEY_SCOPES]
        # Nothing was asked for, so nothing caps or ends the key but its holder's own limits.
        assert key.expires_at is None
        assert db.scalars(select(ScopedBudget).where(ScopedBudget.scope_id == key.id)).first() is None
    headers = {"Authorization": f"Bearer {token.json()['access_token']}"}
    assert client.get(f"{API_ROOT}/models", headers=headers).status_code == 200


def test_the_suggested_limit_and_expiry_reach_the_consent_screen(client: TestClient) -> None:
    start = client.get(
        f"{API_ROOT}/oauth/apps/authorize",
        params=authorization_query(limit="12.50", expiration="7-days"),
        follow_redirects=False,
    )
    assert start.status_code == 302, start.text
    forwarded = parse_qs(urlsplit(start.headers["location"]).query)
    assert forwarded["limit"] == ["12.50"]
    assert forwarded["expiration"] == ["7-days"]


def test_an_approved_limit_and_expiry_cap_and_end_the_key(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    before = datetime.now(UTC)
    token = connect(client, world, limit="12.50", expiration="1-day")
    assert token.status_code == 200, token.text
    with db_session_factory() as db:
        key = the_key(world, db)
        assert key.expires_at is not None
        expires_at = key.expires_at.replace(tzinfo=key.expires_at.tzinfo or UTC)
        assert before + timedelta(days=1) <= expires_at <= datetime.now(UTC) + timedelta(days=1)
        ceiling = db.scalars(
            select(ScopedBudget).where(ScopedBudget.scope_type == SCOPE_API_TOKEN, ScopedBudget.scope_id == key.id)
        ).one()
        assert ceiling.provider_key_id is None
        assert ceiling.period_end is None
        budget = db.get(Budget, ceiling.budget_id)
        assert budget is not None
        assert budget.max_budget == Decimal("12.50")
        assert budget.budget_duration_sec is None and budget.reset_alignment is None
        assert budget.name == "someone.github.io (OAuth)"
        workspace = db.get(Workspace, world.workspaces["alpha_one"])
        assert workspace is not None
        assert budget.organization_id == workspace.organization_id


@pytest.mark.parametrize(
    "changes",
    [
        {"redirect_uri": "https://other.github.io/pizza/"},
        {"client_id": "someone"},
        {"client_id": SITE, "redirect_uri": "http://someone.github.io/pizza/"},
    ],
)
def test_a_callback_off_the_sites_origin_is_never_followed(client: TestClient, changes: dict[str, str]) -> None:
    start = client.get(
        f"{API_ROOT}/oauth/apps/authorize", params=authorization_query(**changes), follow_redirects=False
    )
    assert start.status_code == 400
    assert "location" not in start.headers
