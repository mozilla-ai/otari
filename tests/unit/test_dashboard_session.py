"""Dashboard sign-in sessions: mint, cookie auth, revocation, and rotation.

Covers the fix for the dashboard losing its sign-in on every tab close or
browser restart (issue #338): the master key is exchanged once for an HttpOnly
session cookie that the master-key auth dependencies accept when a request
carries no header credentials.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from gateway.api.routes import auth_session as auth_session_route
from gateway.core.config import GatewayConfig
from gateway.main import create_app
from gateway.models.entities import DashboardSession
from gateway.services import master_key_service
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME

MASTER_KEY = "sk-test-master"


def _config(tmp_path: Path) -> GatewayConfig:
    return GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'session-test.db'}",
        master_key=MASTER_KEY,
        require_pricing=False,
    )


def _sign_in(client: TestClient, key: str = MASTER_KEY) -> None:
    response = client.post("/v1/auth/session", json={"master_key": key})
    assert response.status_code == 200, response.text


def test_sign_in_sets_cookie_and_cookie_authenticates_management_reads(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with TestClient(create_app(config)) as client:
        response = client.post("/v1/auth/session", json={"master_key": MASTER_KEY})
        assert response.status_code == 200, response.text
        assert SESSION_COOKIE_NAME in response.cookies
        # The opaque token never contains the master key.
        assert MASTER_KEY not in response.cookies[SESSION_COOKIE_NAME]
        assert "expires_at" in response.json()

        set_cookie = response.headers["set-cookie"]
        assert "HttpOnly" in set_cookie
        assert "SameSite=strict" in set_cookie.lower() or "samesite=strict" in set_cookie.lower()
        # Plain-HTTP deployment (TestClient) must still receive the cookie back.
        assert "secure" not in set_cookie.lower()

        # The cookie alone (no Authorization header) now opens the management API.
        settings = client.get("/v1/settings")
        assert settings.status_code == 200, settings.text


def test_sign_in_rejects_a_wrong_key_without_setting_a_cookie(tmp_path: Path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        response = client.post("/v1/auth/session", json={"master_key": "not-the-master-key"})
        assert response.status_code == 401
        assert SESSION_COOKIE_NAME not in response.cookies

        assert client.get("/v1/settings").status_code == 401


def test_header_credentials_win_over_the_cookie(tmp_path: Path) -> None:
    # An explicit-but-wrong header must fail even when a valid cookie rides along:
    # API clients keep exactly the pre-cookie behavior.
    with TestClient(create_app(_config(tmp_path))) as client:
        _sign_in(client)
        response = client.get("/v1/settings", headers={"Authorization": "Bearer wrong-key"})
        assert response.status_code == 401


def test_cross_site_requests_cannot_ride_the_cookie(tmp_path: Path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        _sign_in(client)
        response = client.get("/v1/settings", headers={"Sec-Fetch-Site": "cross-site"})
        assert response.status_code == 401
        # Same-origin fetches (the dashboard itself) stay accepted.
        assert client.get("/v1/settings", headers={"Sec-Fetch-Site": "same-origin"}).status_code == 200


def test_sign_out_revokes_the_session_server_side(tmp_path: Path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        _sign_in(client)
        stolen_cookie = client.cookies[SESSION_COOKIE_NAME]

        response = client.delete("/v1/auth/session")
        assert response.status_code == 204

        # Even a kept copy of the cookie is dead after sign-out (server-side
        # revocation, not just cookie deletion in the browser).
        client.cookies.set(SESSION_COOKIE_NAME, stolen_cookie)
        assert client.get("/v1/settings").status_code == 401


def test_sign_out_without_a_session_is_a_no_op(tmp_path: Path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        assert client.delete("/v1/auth/session").status_code == 204


def test_sign_out_clears_the_cookie_even_when_revocation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A DB failure during revocation must not leave the browser holding a live
    # cookie: sign-out stays best-effort (204 + cookie cleared) and the
    # unrevoked session dies on its TTL.
    async def _boom(db: object, token: str) -> None:
        raise SQLAlchemyError("db down")

    monkeypatch.setattr(auth_session_route, "revoke_dashboard_session", _boom)
    with TestClient(create_app(_config(tmp_path))) as client:
        _sign_in(client)
        response = client.delete("/v1/auth/session")
        assert response.status_code == 204
        set_cookie = response.headers.get("set-cookie", "")
        assert SESSION_COOKIE_NAME in set_cookie
        assert 'expires=' in set_cookie.lower() or "max-age=0" in set_cookie.lower()


def test_expired_sessions_stop_authenticating(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with TestClient(create_app(config)) as client:
        _sign_in(client)
        assert client.get("/v1/settings").status_code == 200

        # Age the stored session past its TTL directly in the database.
        engine = create_engine(config.database_url)
        with sessionmaker(bind=engine)() as db:
            db.execute(update(DashboardSession).values(expires_at=datetime.now(UTC) - timedelta(hours=1)))
            db.commit()
        engine.dispose()

        assert client.get("/v1/settings").status_code == 401


def test_sessions_survive_a_restart_but_not_a_configured_key_change(tmp_path: Path) -> None:
    db_url = f"sqlite:///{tmp_path / 'restart.db'}"

    def config_with(key: str) -> GatewayConfig:
        return GatewayConfig(database_url=db_url, master_key=key, require_pricing=False)

    with TestClient(create_app(config_with(MASTER_KEY))) as client:
        _sign_in(client)
        cookie = client.cookies[SESSION_COOKIE_NAME]

    # Same key across a restart: the session (the whole point of #338) survives.
    with TestClient(create_app(config_with(MASTER_KEY))) as client:
        client.cookies.set(SESSION_COOKIE_NAME, cookie)
        assert client.get("/v1/settings").status_code == 200

    # Rotating OTARI_MASTER_KEY across a restart revokes every session: a
    # session only proves possession of the old key and must die with it.
    with TestClient(create_app(config_with("sk-rotated-master"))) as client:
        client.cookies.set(SESSION_COOKIE_NAME, cookie)
        assert client.get("/v1/settings").status_code == 401


def test_rotation_then_restart_keeps_the_reminted_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_url = f"sqlite:///{tmp_path / 'rotate-restart.db'}"
    monkeypatch.setattr(master_key_service, "generate_master_key", lambda: "otari-mk-first")

    with TestClient(create_app(GatewayConfig(database_url=db_url, require_pricing=False))) as client:
        _sign_in(client, "otari-mk-first")
        monkeypatch.setattr(master_key_service, "generate_master_key", lambda: "otari-mk-second")
        assert client.post("/v1/settings/master-key/rotate").status_code == 200
        reminted = client.cookies[SESSION_COOKIE_NAME]

    # The startup key-change check must recognize the rotated key as current
    # and keep the session the rotation re-minted.
    with TestClient(create_app(GatewayConfig(database_url=db_url, require_pricing=False))) as client:
        client.cookies.set(SESSION_COOKIE_NAME, reminted)
        assert client.get("/v1/settings").status_code == 200


def test_rotation_revokes_other_sessions_and_reminting_keeps_the_caller_signed_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(master_key_service, "generate_master_key", lambda: "otari-mk-first")
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'rotation-session.db'}",
        require_pricing=False,
    )
    with TestClient(create_app(config)) as client:
        # Two sign-ins model two dashboard tabs; the cookie jar holds one at a
        # time, so keep the first session's token aside.
        _sign_in(client, "otari-mk-first")
        other_tab_session = client.cookies[SESSION_COOKIE_NAME]
        _sign_in(client, "otari-mk-first")
        rotating_session = client.cookies[SESSION_COOKIE_NAME]
        assert rotating_session != other_tab_session

        monkeypatch.setattr(master_key_service, "generate_master_key", lambda: "otari-mk-second")
        # Rotate via the session cookie alone: the dashboard tab that signed in
        # with a cookie has no raw key to send.
        rotated = client.post("/v1/settings/master-key/rotate")
        assert rotated.status_code == 200, rotated.text
        assert rotated.json() == {"master_key": "otari-mk-second"}

        # The rotating tab got a fresh session on the response and stays signed in.
        assert client.cookies[SESSION_COOKIE_NAME] != rotating_session
        assert client.get("/v1/settings").status_code == 200

        # Every session minted before the rotation died with the old key.
        for stale in (other_tab_session, rotating_session):
            client.cookies.set(SESSION_COOKIE_NAME, stale)
            assert client.get("/v1/settings").status_code == 401
