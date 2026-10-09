"""Integration tests for POST /api/v1/decisions and its /api/v1/systemone alias.

The upstream call is stubbed at ``request_decision`` so these exercise the route's
own job: auth, provider selection, the key allow-list, budget reservation and
settlement, the usage row, and how an upstream failure reaches the caller.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.services.inference import DecisionProviderError

DECISIONS_USAGE_LABEL = "/v1/decisions"

PAYLOAD: dict[str, Any] = {
    "model": "typesafe:jev-latest",
    "state": "I've been trying to connect Stripe for 3 days and I'm losing sales.",
    "questions": {
        "urgency": {"type": "noul", "instructions": "Does this message express urgency?"},
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": None, "technical": "Integrations and outages"},
        },
    },
}

ANSWER: dict[str, Any] = {
    "model": "jev-1.13.0",
    "answers": {
        "urgency": {"type": "noul", "noul": 0.97},
        "team": {
            "type": "choice",
            "choice": "technical",
            "probabilities": {"billing": 0.1, "technical": 0.9},
            "confidence": 0.8,
        },
    },
    "usage": {"input_tokens": 1000, "output_tokens": 10},
}


@pytest.fixture
def test_config(postgres_url: str) -> GatewayConfig:
    """Override the shared config with a TypeSafe and an OpenRouter decisions provider."""
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        decision_providers={
            "typesafe": {"api_key": "ts-secret"},
            "openrouter": {"api_key": "or-secret"},
        },
    )


def _mock_decision(answer: dict[str, Any] | None = None, *, side_effect: Exception | None = None) -> Any:
    mock = AsyncMock(return_value=answer if answer is not None else ANSWER, side_effect=side_effect)
    return patch("gateway.api.routes._passthrough.request_decision", mock)


def _decision_rows(client: TestClient, headers: dict[str, str], user_id: str) -> list[dict[str, Any]]:
    resp = client.get(
        f"{API_ROOT}/usage", params={"user_id": user_id, "endpoint": DECISIONS_USAGE_LABEL}, headers=headers
    )
    assert resp.status_code == 200, resp.text
    return [dict(row) for row in resp.json()]


def test_decisions_requires_auth(client: TestClient) -> None:
    assert client.post(f"{API_ROOT}/decisions", json=PAYLOAD).status_code == 401


def test_decisions_returns_the_provider_answer(client: TestClient, api_key_header: dict[str, str]) -> None:
    with _mock_decision() as mock:
        resp = client.post(f"{API_ROOT}/decisions", json=PAYLOAD, headers=api_key_header)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["answers"]["team"]["choice"] == "technical"
    assert body["answers"]["urgency"]["noul"] == pytest.approx(0.97)
    provider, sent = mock.await_args.args
    assert provider.name == "typesafe"
    # The bare model reaches the provider, and the caller's attribution field does not.
    assert sent == {"model": "jev-latest", "state": PAYLOAD["state"], "questions": PAYLOAD["questions"]}


def test_systemone_alias_serves_the_same_request(client: TestClient, api_key_header: dict[str, str]) -> None:
    with _mock_decision():
        resp = client.post(f"{API_ROOT}/systemone", json=PAYLOAD, headers=api_key_header)
    assert resp.status_code == 200, resp.text
    assert resp.json()["model"] == "jev-1.13.0"


def test_decisions_forwards_images(client: TestClient, api_key_header: dict[str, str]) -> None:
    images = ["data:image/png;base64,iVBORw0KGgo="]
    with _mock_decision() as mock:
        resp = client.post(f"{API_ROOT}/decisions", json={**PAYLOAD, "images": images}, headers=api_key_header)
    assert resp.status_code == 200, resp.text
    assert mock.await_args.args[1]["images"] == images


def test_decisions_refuses_a_remote_image_url(client: TestClient, api_key_header: dict[str, str]) -> None:
    with _mock_decision() as mock:
        resp = client.post(
            f"{API_ROOT}/decisions",
            json={**PAYLOAD, "images": ["https://example.com/a.png"]},
            headers=api_key_header,
        )
    assert resp.status_code == 422
    mock.assert_not_awaited()


def test_decisions_unknown_provider_is_400_and_logged(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    with _mock_decision() as mock:
        resp = client.post(f"{API_ROOT}/decisions", json={**PAYLOAD, "model": "nope:jev"}, headers=api_key_header)
    assert resp.status_code == 400
    assert "decision provider" in resp.json()["detail"]
    mock.assert_not_awaited()

    rows = _decision_rows(client, master_key_header, api_key_obj["user_id"])
    assert [(row["status"], row["status_code"]) for row in rows] == [("error", 400)]


def test_decisions_master_key_requires_user(client: TestClient, master_key_header: dict[str, str]) -> None:
    with _mock_decision():
        resp = client.post(f"{API_ROOT}/decisions", json=PAYLOAD, headers=master_key_header)
    assert resp.status_code == 400


def test_decisions_honors_the_keys_model_allowlist(client: TestClient, master_key_header: dict[str, str]) -> None:
    client.post(f"{API_ROOT}/users", json={"user_id": "narrow-user"}, headers=master_key_header)
    key = client.post(
        f"{API_ROOT}/keys",
        json={"key_name": "narrow-key", "user_id": "narrow-user", "allowed_models": ["openai:gpt-4o"]},
        headers=master_key_header,
    ).json()
    headers = {API_KEY_HEADER: f"Bearer {key['key']}"}

    with _mock_decision():
        assert client.post(f"{API_ROOT}/decisions", json=PAYLOAD, headers=headers).status_code == 403

    granted = client.patch(
        f"{API_ROOT}/keys/{key['id']}",
        json={"allowed_models": ["openai:gpt-4o", "typesafe:jev-latest"]},
        headers=master_key_header,
    )
    assert granted.status_code == 200, granted.text
    with _mock_decision():
        assert client.post(f"{API_ROOT}/decisions", json=PAYLOAD, headers=headers).status_code == 200


def test_decisions_bills_the_configured_token_rate(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": "typesafe:jev-latest", "input_price_per_million": 2.0, "output_price_per_million": 10.0},
        headers=master_key_header,
    )
    user_id = api_key_obj["user_id"]

    with _mock_decision():
        assert client.post(f"{API_ROOT}/decisions", json=PAYLOAD, headers=api_key_header).status_code == 200

    (row,) = _decision_rows(client, master_key_header, user_id)
    assert row["status"] == "success"
    assert (row["model"], row["provider"]) == ("jev-latest", "typesafe")
    assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (1000, 10, 1010)
    # 1000 input tokens at $2/M plus 10 output tokens at $10/M.
    assert float(row["cost"]) == pytest.approx(0.0021)
    user = client.get(f"{API_ROOT}/users/{user_id}", headers=master_key_header).json()
    assert user["reserved"] == pytest.approx(0.0)


def test_decisions_falls_back_to_the_provider_reported_cost(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    answer = {**ANSWER, "usage": {**ANSWER["usage"], "cost": 0.0042}}
    with _mock_decision(answer):
        resp = client.post(
            f"{API_ROOT}/decisions",
            json={**PAYLOAD, "model": "openrouter:typesafe/jev-1.13"},
            headers=api_key_header,
        )
    assert resp.status_code == 200, resp.text

    (row,) = _decision_rows(client, master_key_header, api_key_obj["user_id"])
    assert (row["model"], row["provider"]) == ("typesafe/jev-1.13", "openrouter")
    assert float(row["cost"]) == pytest.approx(0.0042)


@pytest.mark.parametrize(
    ("upstream_status", "caller_status"),
    [(422, 400), (400, 400), (501, 400), (429, 429), (401, 502), (529, 502), (None, 502)],
)
def test_decisions_maps_provider_failures_and_refunds(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    upstream_status: int | None,
    caller_status: int,
) -> None:
    failure = DecisionProviderError("typesafe decisions failed", status_code=upstream_status)
    with _mock_decision(side_effect=failure):
        resp = client.post(f"{API_ROOT}/decisions", json=PAYLOAD, headers=api_key_header)
    assert resp.status_code == caller_status
    assert "typesafe" not in resp.json()["detail"], "the upstream text stays out of the response"

    user_id = api_key_obj["user_id"]
    (row,) = _decision_rows(client, master_key_header, user_id)
    assert row["status"] == "error"
    user = client.get(f"{API_ROOT}/users/{user_id}", headers=master_key_header).json()
    assert user["reserved"] == pytest.approx(0.0)


def test_decisions_unreadable_answer_is_502(client: TestClient, api_key_header: dict[str, str]) -> None:
    with _mock_decision({"answers": "not a map"}):
        resp = client.post(f"{API_ROOT}/decisions", json=PAYLOAD, headers=api_key_header)
    assert resp.status_code == 502


def test_decisions_negative_provider_cost_is_502(client: TestClient, api_key_header: dict[str, str]) -> None:
    answer = {"model": "jev", "answers": {}, "usage": {"input_tokens": 1, "output_tokens": 1, "cost": -1.0}}
    with _mock_decision(answer):
        resp = client.post(f"{API_ROOT}/decisions", json=PAYLOAD, headers=api_key_header)
    assert resp.status_code == 502


def test_decisions_is_budget_enforced(client: TestClient, master_key_header: dict[str, str]) -> None:
    client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": "typesafe:jev-latest", "input_price_per_million": 2.0, "output_price_per_million": 10.0},
        headers=master_key_header,
    )
    budget = client.post(f"{API_ROOT}/budgets", json={"max_budget": 0.0}, headers=master_key_header).json()
    client.post(
        f"{API_ROOT}/users", json={"user_id": "capped", "budget_id": budget["budget_id"]}, headers=master_key_header
    )
    key = client.post(
        f"{API_ROOT}/keys", json={"key_name": "capped-key", "user_id": "capped"}, headers=master_key_header
    ).json()

    with _mock_decision() as mock:
        resp = client.post(f"{API_ROOT}/decisions", json=PAYLOAD, headers={API_KEY_HEADER: f"Bearer {key['key']}"})
    assert resp.status_code == 403
    mock.assert_not_awaited()

    rows = _decision_rows(client, master_key_header, "capped")
    assert [(row["status"], row["status_code"]) for row in rows] == [("error", 403)]


@pytest.mark.parametrize(("model", "refused"), [("jev-latest", True), ("another-model", False)])
def test_decisions_is_held_to_a_model_ceiling(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    model: str,
    refused: bool,
) -> None:
    """A ceiling narrowed to the decision's model binds it, and one on another model does not."""
    budget = client.post(f"{API_ROOT}/budgets", json={"max_budget": 0.0}, headers=master_key_header).json()
    created = client.post(
        f"{API_ROOT}/scoped-budgets",
        json={
            "scope_type": "api_token",
            "scope_id": api_key_obj["id"],
            "provider_key_id": "typesafe",
            "model": model,
            "budget_id": budget["budget_id"],
        },
        headers=master_key_header,
    )
    assert created.status_code == 200, created.text

    with _mock_decision():
        resp = client.post(f"{API_ROOT}/decisions", json=PAYLOAD, headers=api_key_header)
    assert (resp.status_code == 403) is refused, resp.text


def test_decisions_reserves_against_the_token_limit(client: TestClient, master_key_header: dict[str, str]) -> None:
    budget = client.post(f"{API_ROOT}/budgets", json={"token_limit": 10}, headers=master_key_header).json()
    client.post(
        f"{API_ROOT}/users",
        json={"user_id": "token-capped", "budget_id": budget["budget_id"]},
        headers=master_key_header,
    )
    key = client.post(
        f"{API_ROOT}/keys", json={"key_name": "token-key", "user_id": "token-capped"}, headers=master_key_header
    ).json()

    with _mock_decision() as mock:
        resp = client.post(f"{API_ROOT}/decisions", json=PAYLOAD, headers={API_KEY_HEADER: f"Bearer {key['key']}"})
    assert resp.status_code == 403
    assert "token" in resp.json()["detail"]
    mock.assert_not_awaited()


def test_decisions_require_pricing_refuses_an_unpriced_model(
    client: TestClient, test_config: GatewayConfig, api_key_header: dict[str, str]
) -> None:
    test_config.require_pricing = True
    try:
        with _mock_decision() as mock:
            resp = client.post(f"{API_ROOT}/decisions", json=PAYLOAD, headers=api_key_header)
    finally:
        test_config.require_pricing = False
    assert resp.status_code == 402
    mock.assert_not_awaited()
