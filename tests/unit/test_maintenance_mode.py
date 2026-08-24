"""The login freeze: maintenance mode refusing new dashboard sign-ins.

Issue #717 asks for a switch a platform admin flips so a deployment can be
redeployed without anyone signing in mid-migration. otari.ai's flag of the same
name does not do this (it swaps a view in the browser *after* login and its
`login.tsx` checks nothing), so what is asserted here is the behavior the issue
describes rather than a ported one:

* while it is on, ``POST /v1/auth/session`` refuses both credentials with 503;
* a session already minted keeps working, so the operator who flipped the switch
  is not locked out of the dashboard by it;
* the master key through the header is never frozen, which is what guarantees a
  way back out even from a fresh browser;
* the data plane and the rest of the management API are untouched;
* ``GET /v1/bootstrap`` publishes it, so the sign-in screen can say what is
  happening rather than presenting a form that can only be refused.

Unit rather than integration: all of it is route and service behavior that runs
unchanged on the SQLite file each test stands up.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from gateway.core.config import GatewayConfig
from gateway.main import create_app
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME

MASTER_KEY = "sk-test-master"
EMAIL = "operator@example.com"
PASSWORD = "a-real-password"  # pragma: allowlist secret
HEADER = {"Otari-Key": MASTER_KEY}


def _client(tmp_path: Path) -> TestClient:
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'maintenance-test.db'}",
        master_key=MASTER_KEY,
        require_pricing=False,
    )
    return TestClient(create_app(config))


def _freeze(client: TestClient, *, enabled: bool = True) -> None:
    response = client.patch(
        "/v1/settings/maintenance-mode",
        json={"enabled": enabled},
        headers=HEADER,
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"enabled": enabled}


def _claim(client: TestClient) -> None:
    """Claim the deployment, which retires the master key as the *sign-in*."""
    response = client.put(
        "/v1/auth/password",
        json={"email": EMAIL, "new_password": PASSWORD},
        headers=HEADER,
    )
    assert response.status_code == 200, response.text


# =============================================================================
# The freeze itself
# =============================================================================


def test_off_by_default_so_a_fresh_deployment_signs_in(tmp_path: Path) -> None:
    """Absent row means not frozen: nothing has to be turned off first."""
    with _client(tmp_path) as client:
        assert client.get("/v1/settings/maintenance-mode", headers=HEADER).json() == {"enabled": False}
        assert client.post("/v1/auth/session", json={"master_key": MASTER_KEY}).status_code == 200


def test_the_freeze_refuses_master_key_sign_in(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        _freeze(client)

        response = client.post("/v1/auth/session", json={"master_key": MASTER_KEY})

        assert response.status_code == 503
        assert "maintenance mode" in response.json()["detail"]
        assert SESSION_COOKIE_NAME not in client.cookies


def test_the_freeze_refuses_password_sign_in(tmp_path: Path) -> None:
    """The other credential, refused the same way. Neither is an exemption."""
    with _client(tmp_path) as client:
        _claim(client)
        _freeze(client)

        response = client.post("/v1/auth/session", json={"email": EMAIL, "password": PASSWORD})

        assert response.status_code == 503
        assert SESSION_COOKIE_NAME not in client.cookies


def test_a_wrong_credential_is_refused_as_maintenance_not_as_wrong(tmp_path: Path) -> None:
    """The check runs before verification, so a frozen gateway spends no bcrypt.

    It also means a frozen deployment answers a bad credential and a good one
    identically, which is the honest answer: neither one can sign in.
    """
    with _client(tmp_path) as client:
        _claim(client)
        _freeze(client)

        wrong = client.post("/v1/auth/session", json={"email": EMAIL, "password": "not-the-password"})

        assert wrong.status_code == 503


def test_lifting_the_freeze_restores_sign_in(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        _freeze(client)
        assert client.post("/v1/auth/session", json={"master_key": MASTER_KEY}).status_code == 503

        _freeze(client, enabled=False)

        assert client.post("/v1/auth/session", json={"master_key": MASTER_KEY}).status_code == 200


# =============================================================================
# What the freeze deliberately does not touch
# =============================================================================


def test_a_session_minted_before_the_freeze_keeps_working(tmp_path: Path) -> None:
    """New logins only. Booting the operator mid-redeploy is the failure mode."""
    with _client(tmp_path) as client:
        assert client.post("/v1/auth/session", json={"master_key": MASTER_KEY}).status_code == 200
        _freeze(client)

        # No header: this is the cookie the sign-in above set, on its own.
        assert client.get("/v1/settings").status_code == 200


def test_the_master_key_in_the_header_is_never_frozen(tmp_path: Path) -> None:
    """The way back out, and why no identity needs an exemption on sign-in.

    A fresh client holds no cookie, so this is the state an operator is in after
    closing the browser: the switch is still reachable, which is what keeps it
    from being able to lock its own operator out.
    """
    with _client(tmp_path) as client:
        _freeze(client)
        client.cookies.clear()

        assert client.get("/v1/settings", headers=HEADER).status_code == 200
        assert client.get("/v1/keys", headers=HEADER).status_code == 200

        _freeze(client, enabled=False)
        assert client.post("/v1/auth/session", json={"master_key": MASTER_KEY}).status_code == 200


def test_the_freeze_survives_a_restart(tmp_path: Path) -> None:
    """It is a stored row, not process state, which is what makes it hold.

    A second app over the same database stands in for the second replica: it
    never saw the PATCH, and refuses anyway. That is the whole reason this is
    read from the row instead of from the in-memory config the way every
    ``runtime_settings_service`` key is.
    """
    with _client(tmp_path) as client:
        _freeze(client)

    with _client(tmp_path) as restarted:
        assert restarted.post("/v1/auth/session", json={"master_key": MASTER_KEY}).status_code == 503
        assert restarted.get("/v1/settings/maintenance-mode", headers=HEADER).json() == {"enabled": True}


# =============================================================================
# What the browser is told
# =============================================================================


def test_the_bootstrap_publishes_the_freeze_unauthenticated(tmp_path: Path) -> None:
    """So the sign-in screen renders a notice instead of a doomed form."""
    with _client(tmp_path) as client:
        assert client.get("/v1/bootstrap").json()["maintenance_mode"] is False

        _freeze(client)
        client.cookies.clear()

        assert client.get("/v1/bootstrap").json()["maintenance_mode"] is True


# =============================================================================
# The switch itself
# =============================================================================


def test_the_switch_is_master_key_gated(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        assert client.get("/v1/settings/maintenance-mode").status_code == 401
        assert client.patch("/v1/settings/maintenance-mode", json={"enabled": True}).status_code == 401
        # And the refusal changed nothing.
        assert client.get("/v1/settings/maintenance-mode", headers=HEADER).json() == {"enabled": False}


def test_the_switch_refuses_a_body_that_names_no_state(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        assert client.patch("/v1/settings/maintenance-mode", json={}, headers=HEADER).status_code == 422
