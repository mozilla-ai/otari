"""An unregistered site connects by its origin when the operator opens app authorization."""

from collections.abc import Callable
from urllib.parse import parse_qs, urlsplit

import pytest
from apron_auth.pkce import generate_code_challenge as pkce_challenge
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.api_keys import APIKey
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


def test_a_site_connects_by_its_origin_and_its_key_is_named_for_it(
    client: TestClient, world: _World, db_session_factory: Callable[[], Session]
) -> None:
    start = client.get(f"{API_ROOT}/oauth/apps/authorize", params=authorization_query(), follow_redirects=False)
    assert start.status_code == 302, start.text
    details = client.get(f"{API_ROOT}/oauth/apps/clients", params={"client_id": SITE, "redirect_uri": REDIRECT})
    assert details.status_code == 200, details.text
    assert details.json()["name"] == "someone.github.io"
    assert details.json()["verified"] is False

    client.cookies.set(SESSION_COOKIE_NAME, world.sessions["alpha_member"])
    approved = client.post(
        f"{API_ROOT}/oauth/apps/authorize",
        json={
            **authorization_query(),
            "workspace_id": str(world.workspaces["alpha_one"]),
            "scopes": details.json()["scopes"],
        },
        headers={"Sec-Fetch-Site": "same-origin"},
    )
    assert approved.status_code == 200, approved.text
    back = urlsplit(approved.json()["redirect_uri"])
    assert f"{back.scheme}://{back.netloc}{back.path}" == REDIRECT
    code = parse_qs(back.query)["code"][0]

    client.cookies.clear()
    token = client.post(
        f"{API_ROOT}/oauth/apps/token",
        data=dict(
            grant_type="authorization_code",
            client_id=SITE,
            redirect_uri=REDIRECT,
            code=code,
            code_verifier=VERIFIER,
        ),
    )
    assert token.status_code == 200, token.text
    with db_session_factory() as db:
        key = db.scalars(select(APIKey).where(APIKey.user_id == str(world.users["alpha_member"]))).one()
        assert key.key_name == "someone.github.io (OAuth)"
        assert key.metadata_["oauth_client_id"] == SITE
        assert key.scopes.stored == [scope.value for scope in DEFAULT_KEY_SCOPES]
    headers = {"Authorization": f"Bearer {token.json()['access_token']}"}
    assert client.get(f"{API_ROOT}/models", headers=headers).status_code == 200


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
