"""Dashboard sign-in sessions: mint, identity, cookie auth, revocation, rotation.

Covers the fix for the dashboard losing its sign-in on every tab close or
browser restart (issue #338): the master key is exchanged once for an HttpOnly
session cookie that the master-key auth dependencies accept when a request
carries no header credentials.

It also covers what issue #647 added on top: a session names the identity it was
minted for, so a cookie-authenticated request resolves a user and that user's
active organization rather than only proving the master key was presented once.
The revocation paths are re-asserted here rather than trusted, since binding an
identity to a session is the change most likely to break them silently.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from gateway.api import deps
from gateway.api.routes import auth_session as auth_session_route
from gateway.core.config import GatewayConfig
from gateway.main import create_app
from gateway.models.entities import DashboardSession
from gateway.services import dashboard_session_service, master_key_service
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME
from gateway.services.tenancy.provisioning_service import BOOTSTRAP_IDENTITY_KEY

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


def test_https_requests_get_a_secure_cookie(tmp_path: Path) -> None:
    with TestClient(create_app(_config(tmp_path)), base_url="https://testserver") as client:
        response = client.post("/v1/auth/session", json={"master_key": MASTER_KEY})
        assert response.status_code == 200, response.text
        assert "secure" in response.headers["set-cookie"].lower()


def test_forwarded_https_gets_a_secure_cookie(tmp_path: Path) -> None:
    # Behind a TLS-terminating proxy the ASGI scheme often reads "http"
    # (uvicorn only trusts X-Forwarded-Proto from loopback); the Secure
    # decision must honor the forwarded proto so the common PaaS deployment
    # is not silently downgraded.
    with TestClient(create_app(_config(tmp_path))) as client:
        response = client.post(
            "/v1/auth/session",
            json={"master_key": MASTER_KEY},
            headers={"X-Forwarded-Proto": "https"},
        )
        assert response.status_code == 200, response.text
        assert "secure" in response.headers["set-cookie"].lower()


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


def _sessions(config: GatewayConfig) -> list[tuple[str, str]]:
    """Every stored session as ``(token_hash, user_id)``, read outside the app.

    A sync engine of its own, like the expiry test above: what is under test is
    the row the request handler committed, not the ORM state of a live session.
    """
    engine = create_engine(config.database_url)
    try:
        with engine.connect() as connection:
            return [
                (str(row[0]), str(row[1]))
                for row in connection.execute(text("SELECT token_hash, user_id FROM dashboard_sessions"))
            ]
    finally:
        engine.dispose()


def test_a_session_names_the_operator_identity(tmp_path: Path) -> None:
    """The point of #647: the minted row resolves a user, and the response says who.

    The identity is the deployment's bootstrap operator, the same one a header
    master key resolves to, so both credentials answer for the same organization.
    """
    config = _config(tmp_path)
    with TestClient(create_app(config)) as client:
        signed_in = client.post("/v1/auth/session", json={"master_key": MASTER_KEY})
        assert signed_in.status_code == 200, signed_in.text
        body = signed_in.json()

        operator = client.get("/v1/organizations/me", headers={"Otari-Key": MASTER_KEY})
        assert operator.status_code == 200, operator.text
        assert body["active_organization_id"] == operator.json()["organization"]["id"]

        stored = _sessions(config)
        assert [user_id for _, user_id in stored] == [body["user_id"].replace("-", "")]


def test_the_cookie_resolves_that_identity_on_every_later_request(tmp_path: Path) -> None:
    """A tenancy surface reads its scope off the session, with no key in sight."""
    with TestClient(create_app(_config(tmp_path))) as client:
        _sign_in(client)

        by_cookie = client.get("/v1/organizations/me")
        by_key = client.get("/v1/organizations/me", headers={"Otari-Key": MASTER_KEY})

        assert by_cookie.status_code == 200, by_cookie.text
        assert by_cookie.json()["organization"]["id"] == by_key.json()["organization"]["id"]
        assert by_cookie.json()["role"] == "owner"


def test_deactivating_the_identity_ends_its_sessions(tmp_path: Path) -> None:
    """Deactivation has to end dashboard access now, not when the TTL runs out.

    Which is why the identity is loaded on every cookie-authenticated request
    rather than trusted from the session row.
    """
    config = _config(tmp_path)
    with TestClient(create_app(config)) as client:
        _sign_in(client)
        assert client.get("/v1/settings").status_code == 200

        engine = create_engine(config.database_url)
        with engine.begin() as connection:
            connection.execute(text('UPDATE "user" SET is_active = 0'))
        engine.dispose()

        assert client.get("/v1/settings").status_code == 401


def test_deleting_the_identity_revokes_its_sessions(tmp_path: Path) -> None:
    """The foreign key is CASCADE, so no cookie outlives the identity it names.

    ``PRAGMA foreign_keys`` is enabled on this connection by hand: the gateway's
    own engine sets it (``core.database``), and a plain ``create_engine`` here
    would otherwise leave the session row behind with SQLite's default off.
    """
    config = _config(tmp_path)
    with TestClient(create_app(config)) as client:
        _sign_in(client)

        engine = create_engine(config.database_url)
        with engine.begin() as connection:
            connection.execute(text("PRAGMA foreign_keys=ON"))
            connection.execute(text('DELETE FROM "user"'))
        engine.dispose()

        assert _sessions(config) == []
        assert client.get("/v1/settings").status_code == 401


def test_a_first_sign_in_provisions_the_identity_it_binds_to(tmp_path: Path) -> None:
    """Sign-in is the first tenancy-touching request on a fresh deployment.

    Provisioning is lazy, so the session cannot be bound to an identity that
    exists yet; it has to run before the row is staged. Without that the sign-in
    would fail its foreign key on every fresh install.
    """
    config = _config(tmp_path)
    with TestClient(create_app(config)) as client:
        engine = create_engine(config.database_url)
        with engine.connect() as connection:
            assert connection.execute(text('SELECT COUNT(*) FROM "user"')).scalar_one() == 0

        signed_in = client.post("/v1/auth/session", json={"master_key": MASTER_KEY})
        assert signed_in.status_code == 200, signed_in.text

        with engine.connect() as connection:
            marker = connection.execute(
                text("SELECT value FROM runtime_settings WHERE key = :key"),
                {"key": BOOTSTRAP_IDENTITY_KEY},
            ).scalar_one()
        engine.dispose()

        assert marker.replace("-", "") == signed_in.json()["user_id"].replace("-", "")


def test_rotation_re_mints_the_session_for_the_same_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Revoke-then-re-mint must not lose track of who the tab was signed in as."""
    monkeypatch.setattr(master_key_service, "generate_master_key", lambda: "otari-mk-first")
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'rotation-identity.db'}",
        require_pricing=False,
    )
    with TestClient(create_app(config)) as client:
        signed_in = client.post("/v1/auth/session", json={"master_key": "otari-mk-first"})
        assert signed_in.status_code == 200, signed_in.text
        identity = signed_in.json()["user_id"].replace("-", "")

        monkeypatch.setattr(master_key_service, "generate_master_key", lambda: "otari-mk-second")
        assert client.post("/v1/settings/master-key/rotate").status_code == 200

        assert [user_id for _, user_id in _sessions(config)] == [identity]
        assert client.get("/v1/organizations/me").status_code == 200


_COMPLETION_BODY = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}


def test_the_cookie_authenticates_the_request_plane_too(tmp_path: Path) -> None:
    """The two auth sites that resolve the cookie by hand rather than by injection.

    ``resolve_request_context`` and ``/v1/messages/count_tokens`` call the auth
    dependency directly, so FastAPI cannot inject the session identity and each
    has to resolve it itself. The cookie check used to sit inside
    ``verify_api_key_or_master_key``, which meant every caller got it for free; a
    site left un-updated when it moved out would stop accepting the cookie and
    nothing else in this file would notice.

    Chat stops at the master-key user gate rather than at 401: a cookie carries
    master-key authority, and a master key has to name the user it spends for.
    """
    with TestClient(create_app(_config(tmp_path))) as client:
        _sign_in(client)

        assert client.post("/v1/messages/count_tokens", json=_COMPLETION_BODY).status_code == 200

        chat = client.post("/v1/chat/completions", json=_COMPLETION_BODY)
        assert chat.status_code == 400, chat.text
        assert "'user' field is required" in chat.json()["detail"]


def test_the_request_plane_still_refuses_anonymous_and_cross_site_callers(tmp_path: Path) -> None:
    """Hoisting the cookie check out of the dependency must not have loosened it."""
    with TestClient(create_app(_config(tmp_path))) as client:
        assert client.post("/v1/chat/completions", json=_COMPLETION_BODY).status_code == 401
        assert client.post("/v1/messages/count_tokens", json=_COMPLETION_BODY).status_code == 401

        _sign_in(client)
        cross_site = {"Sec-Fetch-Site": "cross-site"}
        assert client.post("/v1/chat/completions", json=_COMPLETION_BODY, headers=cross_site).status_code == 401
        counted = client.post("/v1/messages/count_tokens", json=_COMPLETION_BODY, headers=cross_site)
        assert counted.status_code == 401


def _rate_limited_config(tmp_path: Path, *, limit: int | None) -> GatewayConfig:
    return GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'session-rate-limit-test.db'}",
        master_key=MASTER_KEY,
        require_pricing=False,
        dashboard_login_rate_limit_per_minute=limit,
    )


def test_repeated_failed_sign_ins_get_throttled(tmp_path: Path) -> None:
    with TestClient(create_app(_rate_limited_config(tmp_path, limit=2))) as client:
        # First two failures are real 401s (attempt count is still under the cap).
        for _ in range(2):
            response = client.post("/v1/auth/session", json={"master_key": "wrong"})
            assert response.status_code == 401

        # The third failed attempt within the window is throttled, not a 401.
        throttled = client.post("/v1/auth/session", json={"master_key": "wrong"})
        assert throttled.status_code == 429
        assert "Retry-After" in throttled.headers


def test_successful_sign_in_is_never_throttled(tmp_path: Path) -> None:
    with TestClient(create_app(_rate_limited_config(tmp_path, limit=2))) as client:
        # Exhaust the limit with failures...
        for _ in range(2):
            assert client.post("/v1/auth/session", json={"master_key": "wrong"}).status_code == 401
        # ...then the real key still gets in: only failures count against the cap.
        assert client.post("/v1/auth/session", json={"master_key": MASTER_KEY}).status_code == 200


def test_rate_limit_buckets_are_isolated_per_client_ip(tmp_path: Path) -> None:
    """A regression to a single global bucket (instead of per-IP) would let one
    noisy/attacked IP lock out every other client; prove two IPs get separate
    quotas, not just that throttling happens at all.
    """
    app = create_app(_rate_limited_config(tmp_path, limit=2))
    with TestClient(app, client=("10.0.0.1", 12345)) as client_a:
        # A second TestClient on the same app reuses its already-started
        # lifespan (DB, rate limiter, etc.); only the wrapper differs, with
        # its own fixed client IP for this test.
        client_b = TestClient(app, client=("10.0.0.2", 12345))

        for _ in range(2):
            assert client_a.post("/v1/auth/session", json={"master_key": "wrong"}).status_code == 401
        assert client_a.post("/v1/auth/session", json={"master_key": "wrong"}).status_code == 429

        # A different IP is unaffected by A's usage: still real 401s, not
        # inheriting A's throttled state from a shared/global bucket.
        for _ in range(2):
            assert client_b.post("/v1/auth/session", json={"master_key": "wrong"}).status_code == 401
        assert client_b.post("/v1/auth/session", json={"master_key": "wrong"}).status_code == 429


def test_login_rate_limit_disabled_by_config(tmp_path: Path) -> None:
    with TestClient(create_app(_rate_limited_config(tmp_path, limit=None))) as client:
        # Many failures in a row, none throttled: the limiter is off entirely.
        for _ in range(5):
            assert client.post("/v1/auth/session", json={"master_key": "wrong"}).status_code == 401


def test_reactivating_the_identity_does_not_restore_its_old_sessions(tmp_path: Path) -> None:
    """Deactivation deletes the sessions rather than only refusing them.

    Refusing alone leaves the rows alive, so flipping ``is_active`` back would
    hand every cookie the identity held before its access again, which is the
    opposite of what deactivating it for a lost laptop was for.
    """
    config = _config(tmp_path)
    with TestClient(create_app(config)) as client:
        signed_in = client.post("/v1/auth/session", json={"master_key": MASTER_KEY})
        assert signed_in.status_code == 200, signed_in.text
        assert client.get("/v1/settings").status_code == 200
        # Addressed by id rather than with a bare ``UPDATE "user"``: a fixture
        # that grows a second identity would otherwise deactivate it too, and
        # the assertion below would pass on sessions this test never minted.
        identity = signed_in.json()["user_id"].replace("-", "")

        engine = create_engine(config.database_url)
        with engine.begin() as connection:
            connection.execute(text('UPDATE "user" SET is_active = 0 WHERE id = :id'), {"id": identity})

        # The refusal is what triggers the revocation, so present the cookie once.
        assert client.get("/v1/settings").status_code == 401
        assert [user_id for _, user_id in _sessions(config) if user_id == identity] == []

        with engine.begin() as connection:
            connection.execute(text('UPDATE "user" SET is_active = 1 WHERE id = :id'), {"id": identity})
        engine.dispose()

        assert client.get("/v1/settings").status_code == 401


def test_a_failed_revocation_still_answers_401_rather_than_503(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cleanup is best-effort, and a failed one must not change the answer.

    A deactivated identity's sessions are deleted on the read that refuses
    them, and that write can fail. The caller's answer is "not authenticated"
    either way, so the failure must not escape into ``get_session_identity``,
    which maps a ``SQLAlchemyError`` to a 503.

    The revocation runs on its own session, so the request's transaction is not
    what is being protected here; what is, is that the dependency keeps
    answering 401 when the cleanup cannot be written at all.
    """

    async def _boom(db: object, user_id: object, **kwargs: object) -> None:
        raise SQLAlchemyError("db down")

    monkeypatch.setattr(dashboard_session_service, "revoke_user_dashboard_sessions", _boom)

    config = _config(tmp_path)
    with TestClient(create_app(config)) as client:
        signed_in = client.post("/v1/auth/session", json={"master_key": MASTER_KEY})
        assert signed_in.status_code == 200, signed_in.text
        identity = signed_in.json()["user_id"].replace("-", "")

        engine = create_engine(config.database_url)
        with engine.begin() as connection:
            connection.execute(text('UPDATE "user" SET is_active = 0 WHERE id = :id'), {"id": identity})
        engine.dispose()

        assert client.get("/v1/settings").status_code == 401


def test_a_failed_revocation_leaves_the_request_session_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The revocation writes on its own session, not the caller's.

    ``_bump_last_used_at`` is the precedent: a best-effort write on the auth
    path takes a short-lived session so it can never commit, or dirty, the one
    the request is using. Asserting the structure rather than trusting the
    comment, since the failure it prevents would be a stray commit of whatever
    a future call site had staged, which no status code would reveal.
    """
    used: list[str] = []
    real_revoke = dashboard_session_service.revoke_user_dashboard_sessions

    async def _record(db: object, user_id: object, **kwargs: object) -> None:
        used.append(f"{id(db):x}")
        await real_revoke(db, user_id, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(dashboard_session_service, "revoke_user_dashboard_sessions", _record)

    config = _config(tmp_path)
    with TestClient(create_app(config)) as client:
        signed_in = client.post("/v1/auth/session", json={"master_key": MASTER_KEY})
        assert signed_in.status_code == 200, signed_in.text
        identity = signed_in.json()["user_id"].replace("-", "")

        engine = create_engine(config.database_url)
        with engine.begin() as connection:
            connection.execute(text('UPDATE "user" SET is_active = 0 WHERE id = :id'), {"id": identity})
        engine.dispose()

        seen: list[str] = []
        real_resolve = dashboard_session_service.resolve_dashboard_session

        async def _capture(db: object, token: str) -> object:
            seen.append(f"{id(db):x}")
            return await real_resolve(db, token)  # type: ignore[arg-type]

        monkeypatch.setattr(deps, "resolve_dashboard_session", _capture)

        assert client.get("/v1/settings").status_code == 401
        assert used and seen, (used, seen)
        assert used[-1] != seen[-1], "the revocation reused the request's session"
        assert _sessions(config) == []
