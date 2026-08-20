"""The operator's mail settings surface and its honest refusal.

Unit rather than integration: the route reads configuration and sends through a
transport, with no database behind it beyond the one the app needs to boot.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from gateway.core.config import GatewayConfig
from gateway.main import create_app

AUTH = {"Authorization": "Bearer sk-test-master"}


def _client(tmp_path: Path, **mail: object) -> TestClient:
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'mail-test.db'}",
        master_key="sk-test-master",
        require_pricing=False,
        **mail,  # type: ignore[arg-type]
    )
    return TestClient(create_app(config))


def test_mail_settings_require_the_master_key(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        assert client.get("/v1/settings/mail").status_code == 401
        assert client.post("/v1/settings/mail/test", json={"to": "ada@example.com"}).status_code == 401


def test_an_unconfigured_deployment_reports_itself_unavailable_and_says_what_is_missing(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        body = client.get("/v1/settings/mail", headers=AUTH).json()

    assert body["transport"] == "none"
    assert body["enabled"] is False
    assert body["ready"] is False
    assert body["missing"] == ["smtp_host", "mail_from_email", "public_base_url"]


def test_a_test_send_is_refused_up_front_rather_than_failing_at_send_time(tmp_path: Path) -> None:
    """503 naming the missing settings, not a 200 for a message nobody will receive."""
    with _client(tmp_path) as client:
        response = client.post("/v1/settings/mail/test", json={"to": "ada@example.com"}, headers=AUTH)

    assert response.status_code == 503
    assert "smtp_host" in response.json()["detail"]


def test_a_transport_without_a_public_url_is_still_not_ready(tmp_path: Path) -> None:
    """A link in an inbox has to be absolute, so the address of this deployment is part of readiness."""
    with _client(tmp_path, smtp_host="smtp.example.com", mail_from_email="otari@example.com") as client:
        body = client.get("/v1/settings/mail", headers=AUTH).json()
        refused = client.post("/v1/settings/mail/test", json={"to": "ada@example.com"}, headers=AUTH)

    assert body["transport"] == "smtp"
    assert body["enabled"] is True
    assert body["ready"] is False
    assert body["missing"] == ["public_base_url"]
    assert refused.status_code == 503


def test_a_configured_deployment_sends_a_templated_test_message(tmp_path: Path) -> None:
    with _client(tmp_path, mail_transport="console", public_base_url="https://otari.example.com") as client:
        body = client.get("/v1/settings/mail", headers=AUTH).json()
        sent = client.post("/v1/settings/mail/test", json={"to": "ada@example.com"}, headers=AUTH)

    assert body["ready"] is True
    assert body["missing"] == []
    assert sent.status_code == 200
    assert sent.json() == {"ok": True, "transport": "console", "reason": None}


def test_a_send_that_fails_reports_why_instead_of_erroring(tmp_path: Path) -> None:
    """The operator needs the transport's own reason; this endpoint is master-key gated."""
    with _client(
        tmp_path,
        smtp_host="127.0.0.1",
        smtp_port=1,  # nothing listens here
        mail_from_email="otari@example.com",
        public_base_url="https://otari.example.com",
    ) as client:
        response = client.post("/v1/settings/mail/test", json={"to": "ada@example.com"}, headers=AUTH)

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["transport"] == "smtp"
    assert body["reason"]


def test_a_malformed_recipient_is_refused_before_any_transport_is_touched(tmp_path: Path) -> None:
    with _client(tmp_path, mail_transport="console", public_base_url="https://otari.example.com") as client:
        response = client.post("/v1/settings/mail/test", json={"to": "not-an-address"}, headers=AUTH)

    assert response.status_code == 422


def test_the_mail_surface_never_echoes_the_smtp_password(tmp_path: Path) -> None:
    with _client(
        tmp_path,
        smtp_host="smtp.example.com",
        smtp_user="otari",
        smtp_password="hunter2",
        mail_from_email="otari@example.com",
        public_base_url="https://otari.example.com",
    ) as client:
        mail = client.get("/v1/settings/mail", headers=AUTH).text
        settings = client.get("/v1/settings", headers=AUTH).text

    assert "hunter2" not in mail
    assert "hunter2" not in settings
    # The config viewer shows the non-secret half so an operator can see what is set.
    assert "smtp_host" in settings
