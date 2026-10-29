"""``POST /api/v1/routing/recommend`` answered by a stored subagent router.

The upstream decision call is stubbed at ``request_decision`` where the core
adapter calls it. These pin which router answers and what the caller is
charged: the row switched on at its rates, the hold when the upstream reports
no tokens, the configured model with no row stored, and nothing with every
row off. Nothing the operator stored, and nothing the caller sent, reaches a
log or a usage row.
"""

import logging
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from gateway.adapters.agent_model_recommender_adapter import NOT_CONFIGURED
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.log_config import logger as gateway_logger
from gateway.services.inference import DecisionProviderError
from gateway.services.routing.recommend import RECOMMENDATION_QUESTION
from gateway.services.secret_box import generate_secret_key

RECOMMEND = f"{API_ROOT}/routing/recommend"
ROUTERS = f"{API_ROOT}/routing/recommenders"
STORED_KEY = "sk-or-stored-router-1234"
TASK = "List the files under src/ and report what each module does."
ANSWER: dict[str, Any] = {
    "model": "decider-1.0",
    "answers": {
        RECOMMENDATION_QUESTION: {
            "type": "choice",
            "choice": "sonnet",
            "probabilities": {"haiku": 0.2, "sonnet": 0.7, "opus": 0.1},
        }
    },
    "usage": {"input_tokens": 800, "output_tokens": 0},
}


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


@pytest.fixture
def test_config(postgres_url: str) -> GatewayConfig:
    """A TypeSafe decisions provider, which answers only while no router is stored."""
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        decision_providers={"typesafe": {"api_key": "ts-secret"}},
        agent_recommender_model="typesafe:jev-latest",
    )


@pytest.fixture
def gateway_logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG, logger="gateway")
    try:
        yield caplog
    finally:
        gateway_logger.removeHandler(caplog.handler)


def _spawn() -> dict[str, Any]:
    return {
        "harness": "claude-code",
        "session_id": "bf05abe2-5ff2-4eb0-8459-ab578d5c9468",
        "tool_use_id": "toolu_01",
        "agent_type": "Explore",
        "description": "List files",
        "prompt": TASK,
        "parent_model": "claude-opus-5",
        "requested_model": None,
    }


def _mock_decision(answer: dict[str, Any] | None = None, *, side_effect: Exception | None = None) -> Any:
    mock = AsyncMock(return_value=answer if answer is not None else ANSWER, side_effect=side_effect)
    return patch("gateway.adapters.agent_model_recommender_adapter.request_decision", mock)


def _store(client: TestClient, master_key_header: dict[str, str], **fields: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "provider": "openrouter",
        "model": "stored-decider",
        "api_key": STORED_KEY,
        "input_price_per_million": 20,
    }
    body.update(fields)
    response = client.post(ROUTERS, json=body, headers=master_key_header)
    assert response.status_code == 201, response.text
    stored: dict[str, Any] = response.json()
    return stored


def _usage_rows(client: TestClient, master_key_header: dict[str, str], user_id: str) -> list[dict[str, Any]]:
    rows = client.get(
        f"{API_ROOT}/usage", params={"user_id": user_id, "endpoint": "/v1/decisions"}, headers=master_key_header
    )
    assert rows.status_code == 200, rows.text
    return [dict(row) for row in rows.json()]


def test_the_stored_router_answers_at_its_rates(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    _store(client, master_key_header)

    with _mock_decision() as mock:
        response = client.post(RECOMMEND, json=_spawn(), headers=api_key_header)

    assert response.status_code == 200, response.text
    assert response.json()["model"] == "sonnet"
    provider, sent = mock.await_args.args
    assert (provider.provider, provider.api_key) == ("openrouter", STORED_KEY)
    assert sent["model"] == "stored-decider"

    (row,) = _usage_rows(client, master_key_header, api_key_obj["user_id"])
    assert (row["status"], row["provider"], row["model"]) == ("success", "otari", "subagent-router")
    # 800 input tokens at $20 per million.
    assert float(row["cost"]) == pytest.approx(0.016)
    user = client.get(f"{API_ROOT}/users/{api_key_obj['user_id']}", headers=master_key_header).json()
    assert user["spend"] == pytest.approx(0.016)
    assert user["reserved"] == pytest.approx(0.0)


def test_an_answer_with_no_token_counts_is_charged_the_hold(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    _store(client, master_key_header)
    uncounted = {key: value for key, value in ANSWER.items() if key != "usage"}

    with _mock_decision(uncounted):
        response = client.post(RECOMMEND, json=_spawn(), headers=api_key_header)

    assert response.status_code == 200, response.text
    (row,) = _usage_rows(client, master_key_header, api_key_obj["user_id"])
    # 65,536 input tokens at $20 per million.
    assert float(row["cost"]) == pytest.approx(1.31072)


@pytest.mark.parametrize(
    ("upstream", "answered"),
    [(500, 502), (503, 502), (429, 429), (None, 502)],
)
def test_an_upstream_failure_is_not_charged_and_names_no_vendor(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    upstream: int | None,
    answered: int,
) -> None:
    _store(client, master_key_header)
    failure = DecisionProviderError("openrouter decisions returned HTTP 500 with a long body", status_code=upstream)

    with _mock_decision(side_effect=failure):
        response = client.post(RECOMMEND, json=_spawn(), headers=api_key_header)

    assert response.status_code == answered, response.text
    assert "openrouter" not in response.text
    (row,) = _usage_rows(client, master_key_header, api_key_obj["user_id"])
    assert (row["status"], row["cost"]) == ("error", None)
    user = client.get(f"{API_ROOT}/users/{api_key_obj['user_id']}", headers=master_key_header).json()
    assert user["spend"] == pytest.approx(0.0)
    assert user["reserved"] == pytest.approx(0.0)


def test_an_unreadable_answer_is_a_bad_gateway(
    client: TestClient, api_key_header: dict[str, str], master_key_header: dict[str, str]
) -> None:
    _store(client, master_key_header)

    with _mock_decision({"model": "decider", "answers": {}}):
        response = client.post(RECOMMEND, json=_spawn(), headers=api_key_header)

    assert response.status_code == 502, response.text


def test_with_every_router_off_nothing_answers(
    client: TestClient, api_key_header: dict[str, str], master_key_header: dict[str, str]
) -> None:
    stored = _store(client, master_key_header)
    off = client.patch(f"{ROUTERS}/{stored['id']}", json={"enabled": False}, headers=master_key_header)
    assert off.status_code == 200, off.text

    with _mock_decision() as mock:
        response = client.post(RECOMMEND, json=_spawn(), headers=api_key_header)

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == NOT_CONFIGURED
    mock.assert_not_awaited()


def test_with_no_router_stored_the_configured_model_answers_unpriced(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    with _mock_decision() as mock:
        response = client.post(RECOMMEND, json=_spawn(), headers=api_key_header)

    assert response.status_code == 200, response.text
    provider, sent = mock.await_args.args
    assert (provider.name, provider.api_key, sent["model"]) == ("typesafe", "ts-secret", "jev-latest")
    (row,) = _usage_rows(client, master_key_header, api_key_obj["user_id"])
    assert (row["provider"], row["model"], row["cost"]) == ("typesafe", "jev-latest", None)


def test_a_router_removed_hands_back_to_the_configured_model(
    client: TestClient, api_key_header: dict[str, str], master_key_header: dict[str, str]
) -> None:
    stored = _store(client, master_key_header)
    assert client.delete(f"{ROUTERS}/{stored['id']}", headers=master_key_header).status_code == 204

    with _mock_decision() as mock:
        response = client.post(RECOMMEND, json=_spawn(), headers=api_key_header)

    assert response.status_code == 200, response.text
    assert mock.await_args.args[1]["model"] == "jev-latest"


def test_neither_the_key_nor_the_task_reaches_a_log_or_a_usage_row(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    gateway_logs: pytest.LogCaptureFixture,
) -> None:
    _store(client, master_key_header)

    with _mock_decision():
        assert client.post(RECOMMEND, json=_spawn(), headers=api_key_header).status_code == 200
    with _mock_decision(side_effect=DecisionProviderError(f"refused {STORED_KEY} for {TASK}", status_code=500)):
        assert client.post(RECOMMEND, json=_spawn(), headers=api_key_header).status_code == 502

    logged = "\n".join(record.getMessage() for record in gateway_logs.records)
    rows = str(_usage_rows(client, master_key_header, api_key_obj["user_id"]))
    for secret in (STORED_KEY, TASK):
        assert secret not in logged
        assert secret not in rows
