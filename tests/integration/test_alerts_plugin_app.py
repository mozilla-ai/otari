"""The alerts plugin on a booted app, reached the way a deployment reaches it.

``OTARI_BOOTSTRAP=otari_alerts:register`` is the whole of the wiring, so what
this file asks is whether that one setting is enough: does the contributed
router answer over HTTP, does it refuse an unauthenticated caller, and does a
plain build with the selector unset serve nothing at all.

The plugin's tables come from its contributed Alembic chain, which the
integration conftest runs beside Otari's own; see ``_contributed_chains``
there.
"""

from collections.abc import Generator, Iterator

import pytest
from fastapi.testclient import TestClient

from gateway.container import Container
from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.services.secret_box import generate_secret_key

from .conftest import build_test_client

RULES_PATH = f"{API_ROOT}/organizations/me/alert-rules"
SLACK_DESTINATION = "slack://xoxb-AAA/xoxb-BBB/xoxb-CCC/#alerts"


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The destination column is encrypted, so a write needs a key to encrypt with."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


@pytest.fixture
def alerts_client(postgres_url: str) -> Generator[TestClient]:
    """An app booted with the plugin's bootstrap selector set, and nothing else."""
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        bootstrap="otari_alerts:register",
    )
    yield from build_test_client(config)


def test_the_router_is_mounted_and_answers_the_operator(alerts_client: TestClient) -> None:
    response = alerts_client.get(RULES_PATH, headers={API_KEY_HEADER: "test-master-key"})

    assert response.status_code == 200, response.text
    assert response.json() == {"data": [], "count": 0}


def test_the_contributed_router_refuses_an_unauthenticated_caller(alerts_client: TestClient) -> None:
    """The mount adds no credential, so the router's own ``verify_master_key`` is the gate."""
    assert alerts_client.get(RULES_PATH).status_code == 401


def test_an_unsupported_destination_is_a_400_with_no_handler_registered(alerts_client: TestClient) -> None:
    """The whole of the plugin's error registration, end to end.

    Its error classes subclass Otari's ``TenancyError`` family, and each
    carries its own ``status_code``, so the one handler ``gateway.main``
    registers for the base renders them. The plugin registers nothing: a 400
    rather than a 500 here is the proof that works from out of tree.
    """
    response = alerts_client.post(
        RULES_PATH,
        headers={API_KEY_HEADER: "test-master-key"},
        json={"name": "Bad", "destination": "definitelynotascheme://host"},
    )

    assert response.status_code == 400, response.text
    assert "destination" in response.json()["detail"].lower() or "apprise" in response.json()["detail"].lower()


def test_a_rule_round_trips_over_http_with_its_destination_redacted(alerts_client: TestClient) -> None:
    headers = {API_KEY_HEADER: "test-master-key"}

    created = alerts_client.post(
        RULES_PATH,
        headers=headers,
        json={"name": "Platform Slack", "destination": SLACK_DESTINATION},
    )
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["warn_at_percent"] == 80
    # The single assertion worth having on this path: an Apprise URL is a live
    # bot token and the API must never echo one back.
    assert "xoxb-AAA" not in body["destination"]
    assert body["destination"].startswith("slack://")

    listed = alerts_client.get(RULES_PATH, headers=headers)
    assert listed.status_code == 200
    assert listed.json()["count"] == 1

    deleted = alerts_client.delete(f"{RULES_PATH}/{body['id']}", headers=headers)
    assert deleted.status_code == 200
    assert alerts_client.get(RULES_PATH, headers=headers).json()["count"] == 0


def test_the_container_records_all_three_contributions(alerts_client: TestClient) -> None:
    container: Container = alerts_client.app.state.container  # type: ignore[attr-defined]

    assert container.summary == (
        "otari_alerts:register rebound no ports, contributed routers for ungated, "
        "contributed background tasks budget-alerts, contributed migration chains alerts"
    )


def test_a_build_without_the_selector_serves_no_alert_surface(client: TestClient) -> None:
    """The acceptance case for every deployment that has not installed the plugin."""
    assert client.get(RULES_PATH, headers={API_KEY_HEADER: "test-master-key"}).status_code == 404
    container: Container = client.app.state.container  # type: ignore[attr-defined]
    assert container.router_contributions() == ()
    assert container.migration_contributions() == ()
