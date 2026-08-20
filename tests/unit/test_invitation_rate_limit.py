"""The public invitation routes reuse the sign-in rate limiter.

`_throttle` (`api/routes/invitations.py`) has no test of its own elsewhere:
`test_invitations_api.py` runs against the shared Postgres-backed `client`
fixture, whose config leaves `dashboard_login_rate_limit_per_minute` at its
default, too high to exercise in a handful of requests. Without a test
pinned to a low limit, a refactor that dropped the `_throttle` call from
either route would stay green everywhere else.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from gateway.core.config import GatewayConfig
from gateway.main import create_app

MASTER_KEY = "sk-test-master"


def _rate_limited_config(tmp_path: Path, *, limit: int | None) -> GatewayConfig:
    return GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'invitation-rate-limit-test.db'}",
        master_key=MASTER_KEY,
        require_pricing=False,
        dashboard_login_rate_limit_per_minute=limit,
    )


def test_repeated_invitation_validate_calls_get_throttled(tmp_path: Path) -> None:
    with TestClient(create_app(_rate_limited_config(tmp_path, limit=2))) as client:
        for _ in range(2):
            response = client.post("/v1/invitations/validate", json={"token": "not-a-real-token"})
            assert response.status_code == 404

        throttled = client.post("/v1/invitations/validate", json={"token": "not-a-real-token"})
        assert throttled.status_code == 429
        assert "Retry-After" in throttled.headers


def test_repeated_invitation_accept_calls_get_throttled(tmp_path: Path) -> None:
    with TestClient(create_app(_rate_limited_config(tmp_path, limit=2))) as client:
        for _ in range(2):
            response = client.post("/v1/invitations/accept", json={"token": "not-a-real-token"})
            assert response.status_code == 404

        throttled = client.post("/v1/invitations/accept", json={"token": "not-a-real-token"})
        assert throttled.status_code == 429


def test_invitation_validate_and_accept_share_one_budget(tmp_path: Path) -> None:
    """Both routes draw on the one limiter, not one budget each."""
    with TestClient(create_app(_rate_limited_config(tmp_path, limit=2))) as client:
        assert client.post("/v1/invitations/validate", json={"token": "x"}).status_code == 404
        assert client.post("/v1/invitations/accept", json={"token": "x"}).status_code == 404

        # The budget is already spent between the two routes; a third call to
        # either one is throttled.
        assert client.post("/v1/invitations/validate", json={"token": "x"}).status_code == 429


def test_invitation_rate_limit_disabled_by_config(tmp_path: Path) -> None:
    with TestClient(create_app(_rate_limited_config(tmp_path, limit=None))) as client:
        for _ in range(5):
            response = client.post("/v1/invitations/validate", json={"token": "not-a-real-token"})
            assert response.status_code == 404
