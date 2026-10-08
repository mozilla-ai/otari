"""Integration tests for POST /api/v1/routing/recommend.

The upstream decision call is stubbed at ``request_decision`` where the core
recommender adapter calls it, as the decisions tests do, so these cover the
route's own job: who may ask, the question the core puts to the decision model,
how the answer comes back, the request's strictness, and that the decision is
billed to the caller like any other decision. A recommender an overlay binds in
its place is covered by ``test_agent_model_recommender_rebound.py``.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, API_ROOT, DEFAULT_AGENT_MODEL_CANDIDATES, GatewayConfig
from gateway.services.inference import DecisionProviderError
from gateway.services.routing.recommend import RECOMMENDATION_QUESTION

PATH = f"{API_ROOT}/routing/recommend"

ANSWER: dict[str, Any] = {
    "model": "jev-1.13.0",
    "answers": {
        RECOMMENDATION_QUESTION: {
            "type": "choice",
            "choice": "sonnet",
            "probabilities": {"haiku": 0.2, "sonnet": 0.7, "opus": 0.1},
            "confidence": 0.7,
        }
    },
    "usage": {"input_tokens": 800, "output_tokens": 0},
}


@pytest.fixture
def test_config(postgres_url: str) -> GatewayConfig:
    """Override the shared config with a TypeSafe decisions provider for the recommender to ask."""
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        decision_providers={"typesafe": {"api_key": "ts-secret"}},
    )


def _spawn(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "harness": "claude-code",
        "session_id": "bf05abe2-5ff2-4eb0-8459-ab578d5c9468",
        "tool_use_id": "toolu_01",
        "agent_type": "Explore",
        "description": "List files",
        "prompt": "List the files under src/ and report what each module does.",
        "parent_model": "claude-opus-5",
        "requested_model": None,
    }
    body.update(overrides)
    return body


def _mock_decision(answer: dict[str, Any] | None = None, *, side_effect: Exception | None = None) -> Any:
    mock = AsyncMock(return_value=answer if answer is not None else ANSWER, side_effect=side_effect)
    return patch("gateway.adapters.agent_model_recommender_adapter.request_decision", mock)


def test_api_key_gets_the_decision_models_pick(client: TestClient, api_key_header: dict[str, str]) -> None:
    with _mock_decision() as mock:
        response = client.post(PATH, json=_spawn(), headers=api_key_header)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "sonnet"
    assert body["probabilities"] == {"haiku": 0.2, "sonnet": 0.7, "opus": 0.1}
    assert body["reason"] == "jev-1.13.0 chose sonnet with 70%"

    provider, sent = mock.await_args.args
    assert provider.name == "typesafe"
    assert sent["model"] == "jev-latest"
    assert list(sent["questions"]) == [RECOMMENDATION_QUESTION]
    question = sent["questions"][RECOMMENDATION_QUESTION]
    assert question["type"] == "choice"
    assert question["criteria"] == DEFAULT_AGENT_MODEL_CANDIDATES
    assert "Subagent type: Explore" in sent["state"]
    assert "List the files under src/" in sent["state"]
    assert "user" not in sent


def test_caller_request_does_not_reach_the_decision_model(client: TestClient, api_key_header: dict[str, str]) -> None:
    with _mock_decision() as mock:
        response = client.post(PATH, json=_spawn(requested_model="requested-xyz"), headers=api_key_header)

    assert response.status_code == 200, response.text
    assert response.json()["model"] == "sonnet"
    _, sent = mock.await_args.args
    assert "requested-xyz" not in sent["state"]


def test_master_key_bills_the_named_user(
    client: TestClient, master_key_header: dict[str, str], test_user: dict[str, Any]
) -> None:
    with _mock_decision():
        response = client.post(PATH, json=_spawn(user=test_user["user_id"]), headers=master_key_header)

    assert response.status_code == 200, response.text
    assert response.json()["model"] == "sonnet"


def test_master_key_without_a_user_is_a_bad_request(client: TestClient, master_key_header: dict[str, str]) -> None:
    with _mock_decision():
        response = client.post(PATH, json=_spawn(), headers=master_key_header)

    assert response.status_code == 400


def test_no_credential_is_unauthorized(client: TestClient) -> None:
    response = client.post(PATH, json=_spawn())

    assert response.status_code == 401


def test_missing_prompt_is_unprocessable(client: TestClient, api_key_header: dict[str, str]) -> None:
    body = _spawn()
    del body["prompt"]

    response = client.post(PATH, json=body, headers=api_key_header)

    assert response.status_code == 422


def test_unknown_field_is_unprocessable(client: TestClient, api_key_header: dict[str, str]) -> None:
    response = client.post(PATH, json=_spawn(budget_usd=5), headers=api_key_header)

    assert response.status_code == 422


def test_decision_provider_failure_is_a_bad_gateway(client: TestClient, api_key_header: dict[str, str]) -> None:
    with _mock_decision(side_effect=DecisionProviderError("upstream exploded", status_code=500)):
        response = client.post(PATH, json=_spawn(), headers=api_key_header)

    assert response.status_code == 502
    assert "exploded" not in response.text


def test_a_pick_outside_the_candidates_is_a_bad_gateway(client: TestClient, api_key_header: dict[str, str]) -> None:
    answer = {"model": "jev-1.13.0", "answers": {RECOMMENDATION_QUESTION: {"type": "choice", "choice": "gpt-5"}}}
    with _mock_decision(answer):
        response = client.post(PATH, json=_spawn(), headers=api_key_header)

    assert response.status_code == 502


def test_the_decision_is_recorded_as_the_callers_usage(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    with _mock_decision():
        assert client.post(PATH, json=_spawn(), headers=api_key_header).status_code == 200

    rows = client.get(
        f"{API_ROOT}/usage",
        params={"user_id": api_key_obj["user_id"], "endpoint": "/v1/decisions"},
        headers=master_key_header,
    )
    assert rows.status_code == 200, rows.text
    models = {row["model"] for row in rows.json()}
    assert "jev-latest" in models


def test_the_hold_is_sized_from_the_whole_question(client: TestClient, master_key_header: dict[str, str]) -> None:
    """A short prompt beside a long description reserves for the question as sent, not for the prompt alone."""
    budget = client.post(f"{API_ROOT}/budgets", json={"token_limit": 100}, headers=master_key_header).json()
    client.post(
        f"{API_ROOT}/users",
        json={"user_id": "token-capped", "budget_id": budget["budget_id"]},
        headers=master_key_header,
    )
    key = client.post(
        f"{API_ROOT}/keys", json={"key_name": "token-key", "user_id": "token-capped"}, headers=master_key_header
    ).json()
    spawn = _spawn(prompt="List files.", description="y" * 1000)

    with _mock_decision() as mock:
        response = client.post(PATH, json=spawn, headers={API_KEY_HEADER: f"Bearer {key['key']}"})

    assert response.status_code == 403, response.text
    assert "token" in response.json()["detail"]
    mock.assert_not_awaited()
