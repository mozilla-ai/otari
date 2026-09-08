"""Endpoint tests for /v1/connections and the hosted OAuth callback."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from apron_auth.models import IdentityProfile, OAuthPendingState, TokenSet
from fastapi.testclient import TestClient

from gateway.api.deps import reset_config
from gateway.core.config import GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app
from gateway.services.secret_box import generate_secret_key
from gateway.services.tenancy import connected_account_service as svc

AUTH = {"Authorization": "Bearer sk-test-master"}
USER = "alice@acme.test"


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield
    reset_config()
    reset_db()


class _FakeClient:
    def __init__(self, config: Any, state_store: Any = None, **_: Any) -> None:
        self.store = state_store

    async def get_authorization_url(
        self, redirect_uri: str | None = None, metadata: dict[str, Any] | None = None
    ) -> tuple[str, OAuthPendingState]:
        pending = OAuthPendingState(
            state=uuid.uuid4().hex,
            redirect_uri="https://otari.example.com/cb",
            code_verifier="v",
            created_at=time.time(),
            metadata=metadata or {},
        )
        await self.store.save(pending)
        return f"https://github.com/login/oauth/authorize?state={pending.state}", pending

    async def exchange_code(self, code: str, state: str | None = None, **_: Any) -> TokenSet:
        assert state and await self.store.consume(state) is not None
        return TokenSet(access_token="gho_secret", scope="repo read:user")

    async def fetch_identity(self, tokens: TokenSet) -> IdentityProfile:
        return IdentityProfile(provider="github", username="octocat", name="Octo Cat")

    async def revoke_token(self, token: str) -> bool:
        return True


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> TestClient:
    monkeypatch.setattr(svc, "OAuthClient", _FakeClient)
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'connections.db'}",
        master_key="sk-test-master",
        public_base_url="https://otari.example.com",
        connected_apps={"github": {"client_id": "id", "client_secret": "secret", "scopes": ["repo"]}},
        **overrides,
    )
    return TestClient(create_app(config))


def test_requires_auth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(tmp_path, monkeypatch) as client:
        assert client.get("/v1/connections", params={"user": USER}).status_code == 401
        assert client.get("/v1/connections/apps").status_code == 401


def test_full_flow_over_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(tmp_path, monkeypatch) as client:
        apps = client.get("/v1/connections/apps", headers=AUTH, params={"user": USER}).json()
        assert [app["provider"] for app in apps] == ["github"]
        assert apps[0]["connected_accounts"] == 0 and "repo" in apps[0]["scopes"]

        started = client.post(
            "/v1/connections/github/authorize",
            headers=AUTH,
            json={"user": USER, "return_url": "https://myapp.test/settings?tab=apps"},
        )
        assert started.status_code == 200, started.text
        url = started.json()["authorization_url"]
        assert url.startswith("https://github.com/login/oauth/authorize")
        state = parse_qs(urlsplit(url).query)["state"][0]

        # The provider sends the user's browser back; no credential on that request.
        landed = client.get(f"/connected-accounts/github/callback?code=abc&state={state}", follow_redirects=False)
        assert landed.status_code == 302
        location = urlsplit(landed.headers["location"])
        query = parse_qs(location.query)
        assert (location.scheme, location.netloc, location.path) == ("https", "myapp.test", "/settings")
        assert query["tab"] == ["apps"] and query["connection"] == ["ok"] and query["provider"] == ["github"]
        connection_id = query["connection_id"][0]

        listed = client.get("/v1/connections", headers=AUTH, params={"user": USER}).json()
        assert listed["count"] == 1 and listed["data"][0]["account_identifier"] == "octocat"
        assert "gho_secret" not in client.get("/v1/connections", headers=AUTH, params={"user": USER}).text

        token = client.get("/v1/connections/github/token", headers=AUTH, params={"user": USER}).json()
        assert token["token"] == "gho_secret" and token["provider"] == "github"
        assert client.get("/v1/connections/github/token", headers=AUTH, params={"user": "nobody"}).status_code == 404

        patched = client.patch(
            f"/v1/connections/{connection_id}", headers=AUTH, params={"user": USER}, json={"label": "Work"}
        )
        assert patched.json()["label"] == "Work"
        assert client.get(f"/v1/connections/{connection_id}", headers=AUTH, params={"user": "bob"}).status_code == 404
        assert client.delete(f"/v1/connections/{connection_id}", headers=AUTH, params={"user": USER}).status_code == 204
        assert client.get(f"/v1/connections/{connection_id}", headers=AUTH, params={"user": USER}).status_code == 404

        # A reused state lands on the return page with an error rather than a raw 400.
        reused = client.get(f"/connected-accounts/github/callback?code=abc&state={state}", follow_redirects=False)
        assert parse_qs(urlsplit(reused.headers["location"]).query)["connection"] == ["error"]
        assert parse_qs(urlsplit(reused.headers["location"]).query)["reason"] == ["stateinvalid"]


def test_provider_error_and_bad_requests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(tmp_path, monkeypatch) as client:
        started = client.post("/v1/connections/github/authorize", headers=AUTH, json={"user": USER}).json()
        state = parse_qs(urlsplit(started["authorization_url"]).query)["state"][0]
        denied = client.get(
            f"/connected-accounts/github/callback?error=access_denied&state={state}", follow_redirects=False
        )
        assert denied.status_code == 302
        location = urlsplit(denied.headers["location"])
        # No return_url was given, so the browser lands on this deployment's own hash-routed page,
        # with the outcome inside the fragment where a hash router reads it.
        assert (location.netloc, location.path) == ("otari.example.com", "/")
        route, _, fragment_query = location.fragment.partition("?")
        assert route == "/connections"
        query = parse_qs(fragment_query)
        assert query["connection"] == ["error"] and query["reason"] == ["access_denied"]

        assert client.post("/v1/connections/slack/authorize", headers=AUTH, json={"user": USER}).status_code == 400
        assert client.post("/v1/connections/myspace/authorize", headers=AUTH, json={"user": USER}).status_code == 422
        bad_return = client.post(
            "/v1/connections/github/authorize", headers=AUTH, json={"user": USER, "return_url": "http://evil.test/"}
        )
        assert bad_return.status_code == 400
        assert client.post("/v1/connections/github/authorize", headers=AUTH, json={}).status_code == 422
