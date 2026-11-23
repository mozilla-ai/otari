"""Integration tests for POST /api/v1/decisions and its /api/v1/systemone alias.

The upstream call is stubbed at ``request_decision`` so these exercise the route's
own job: auth, provider selection, the key allow-list, budget reservation and
settlement, the usage row, and how an upstream failure reaches the caller.
"""

import json
import time
from collections.abc import Sequence
from typing import Any
from unittest.mock import AsyncMock, patch

import anthropic
import httpx2
import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from fastapi.testclient import TestClient

from gateway.core.config import (
    API_KEY_HEADER,
    API_ROOT,
    ATTEMPT_COUNT_HEADER,
    FALLBACK_HEADER,
    MODEL_HEADER,
    PROVIDER_HEADER,
    REQUEST_ID_HEADER,
    GatewayConfig,
)
from gateway.models.routing import RoutingConfig
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
        model_discovery=False,
        providers={"claude-api": {"provider_type": "anthropic", "api_key": "sk-ant"}},
        decision_providers={
            "typesafe": {"api_key": "ts-secret"},
            "openrouter": {"api_key": "or-secret"},
        },
        routing=RoutingConfig.model_validate(
            {
                "policies": {
                    "tagger": {
                        "select": [{"default": "typesafe:jev-latest"}],
                        "on_failure": ["openrouter:typesafe/jev-1.13"],
                    },
                    "chat-first": {
                        "select": [{"default": "claude-api:claude-haiku-5-5"}],
                        "on_failure": ["typesafe:jev-latest"],
                    },
                }
            }
        ),
    )


def _mock_decision(
    answer: dict[str, Any] | None = None, *, side_effect: Exception | Sequence[object] | None = None
) -> Any:
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


def test_a_decision_line_names_the_rules_that_changed_its_price(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    long = {"context": {"gt": 500}}
    created = client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": "typesafe:jev-latest",
            "rates": {
                "text": {"input": {"price": 2, "overrides": [{"when": long, "price": 4}]}, "output": {"price": 10}}
            },
        },
        headers=master_key_header,
    )
    assert created.status_code == 200, created.text

    with _mock_decision():
        assert client.post(f"{API_ROOT}/decisions", json=PAYLOAD, headers=api_key_header).status_code == 200

    (row,) = _decision_rows(client, master_key_header, api_key_obj["user_id"])
    # 1000 input tokens at the long-context $4/M plus 10 output tokens at $10/M.
    assert float(row["cost"]) == pytest.approx(0.0041)
    assert {line["meter"]: line.get("rules") for line in row["pricing_breakdown"]} == {"input": [long], "output": None}


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
    [(422, 400), (400, 400), (501, 400), (429, 429), (401, 424), (403, 424), (529, 502), (None, 502)],
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
    assert resp.json()["code"] == "pricing_required"
    mock.assert_not_awaited()


def test_a_decision_is_answered_and_recorded_under_its_request_id(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    with _mock_decision():
        resp = client.post(f"{API_ROOT}/decisions", json=PAYLOAD, headers=api_key_header)
    assert resp.status_code == 200, resp.text
    request_id = resp.headers[REQUEST_ID_HEADER]

    deadline = time.monotonic() + 3.0
    while (
        settled := client.get(f"{API_ROOT}/usage/requests/{request_id}", headers=api_key_header)
    ).status_code == 404 and time.monotonic() < deadline:
        time.sleep(0.1)
    assert settled.status_code == 200, settled.text
    assert settled.json()["status"] == "success"
    assert settled.json()["total_tokens"] == 1010


def test_a_refused_decision_carries_its_request_id(client: TestClient, api_key_header: dict[str, str]) -> None:
    with _mock_decision():
        resp = client.post(f"{API_ROOT}/decisions", json={**PAYLOAD, "model": "nope:jev"}, headers=api_key_header)
    assert resp.status_code == 400
    assert resp.headers[REQUEST_ID_HEADER]


POLICY_PAYLOAD: dict[str, Any] = {**PAYLOAD, "model": "tagger"}


def _key_for(client: TestClient, master_key_header: dict[str, str], user_id: str, allowed: list[str]) -> dict[str, str]:
    client.post(f"{API_ROOT}/users", json={"user_id": user_id}, headers=master_key_header)
    key = client.post(
        f"{API_ROOT}/keys",
        json={"key_name": f"{user_id}-key", "user_id": user_id, "allowed_models": allowed},
        headers=master_key_header,
    ).json()
    return {API_KEY_HEADER: f"Bearer {key['key']}"}


def test_a_policy_fails_over_to_its_next_target_and_records_the_one_that_answered(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": "openrouter:typesafe/jev-1.13",
            "input_price_per_million": 3.0,
            "output_price_per_million": 10.0,
        },
        headers=master_key_header,
    )
    rate_limited = DecisionProviderError("typesafe decisions returned HTTP 429", status_code=429)
    with _mock_decision(side_effect=[rate_limited, ANSWER]) as mock:
        resp = client.post(f"{API_ROOT}/decisions", json=POLICY_PAYLOAD, headers=api_key_header)

    assert resp.status_code == 200, resp.text
    assert [call.args[0].name for call in mock.await_args_list] == ["typesafe", "openrouter"]
    assert mock.await_args_list[1].args[1]["model"] == "typesafe/jev-1.13"
    # The answer names the policy, and the headers name the target that gave it.
    assert resp.json()["model"] == "tagger"
    assert resp.json()["answers"]["team"]["choice"] == "technical"
    served_by = (resp.headers[PROVIDER_HEADER], resp.headers[MODEL_HEADER])
    assert served_by == ("openrouter", "typesafe/jev-1.13")
    assert (resp.headers[ATTEMPT_COUNT_HEADER], resp.headers[FALLBACK_HEADER]) == ("2", "true")

    user_id = api_key_obj["user_id"]
    rows = _decision_rows(client, master_key_header, user_id)
    (served,) = [row for row in rows if row["status"] == "success"]
    (absorbed,) = [row for row in rows if row["status"] == "absorbed"]
    assert (served["provider"], served["model"]) == ("openrouter", "typesafe/jev-1.13")
    assert (served["policy_name"], served["selection_reason"]) == ("tagger", "on_failure")
    assert (served["attempt_position"], served["attempt_count"]) == (2, 2)
    # Priced at the rate of the target that answered: 1000 input at $3/M plus 10 output at $10/M.
    assert float(served["cost"]) == pytest.approx(0.0031)
    assert (absorbed["provider"], absorbed["status_code"]) == ("typesafe", 429)
    assert absorbed["counts_toward_budget"] is False
    assert absorbed["request_group_id"] == served["request_group_id"] == resp.headers[REQUEST_ID_HEADER]
    user = client.get(f"{API_ROOT}/users/{user_id}", headers=master_key_header).json()
    assert user["reserved"] == pytest.approx(0.0)


def test_a_policy_whose_every_target_fails_is_a_502_and_refunds(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    outage = [DecisionProviderError("down", status_code=503), DecisionProviderError("down", status_code=500)]
    with _mock_decision(side_effect=outage) as mock:
        resp = client.post(f"{API_ROOT}/decisions", json=POLICY_PAYLOAD, headers=api_key_header)

    assert resp.status_code == 502, resp.text
    assert mock.await_count == 2
    user_id = api_key_obj["user_id"]
    rows = _decision_rows(client, master_key_header, user_id)
    assert sorted((row["status"], row["provider"]) for row in rows) == [
        ("absorbed", "typesafe"),
        ("error", "openrouter"),
    ]
    user = client.get(f"{API_ROOT}/users/{user_id}", headers=master_key_header).json()
    assert user["reserved"] == pytest.approx(0.0)


def test_a_policy_drops_a_target_the_key_may_not_use(client: TestClient, master_key_header: dict[str, str]) -> None:
    headers = _key_for(client, master_key_header, "typesafe-only", ["typesafe:jev-latest"])
    refused = DecisionProviderError("typesafe decisions returned HTTP 422", status_code=422)
    with _mock_decision(side_effect=refused) as mock:
        resp = client.post(f"{API_ROOT}/decisions", json=POLICY_PAYLOAD, headers=headers)

    # The one target left was the only one called, so it answers as naming it directly would.
    assert resp.status_code == 400, resp.text
    assert mock.await_count == 1


def test_a_policy_with_no_permitted_target_is_refused_without_naming_its_targets(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    headers = _key_for(client, master_key_header, "chat-only", ["openai:gpt-4o"])
    with _mock_decision() as mock:
        resp = client.post(f"{API_ROOT}/decisions", json=POLICY_PAYLOAD, headers=headers)

    assert resp.status_code == 403, resp.text
    detail = resp.json()["detail"]
    assert "tagger" in detail
    assert "typesafe" not in detail
    assert "openrouter" not in detail
    mock.assert_not_awaited()


CHAT_REPLY: dict[str, Any] = {"urgency": 0.9, "team": {"billing": 0.2, "technical": 0.8}}


def _chat_completion(content: str, *, finish_reason: str = "stop") -> ChatCompletion:
    return ChatCompletion(
        id="cmpl-1",
        choices=[
            Choice(
                finish_reason=finish_reason,
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
            )
        ],
        created=0,
        model="claude-haiku-5-5-20261001",
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=400, completion_tokens=20, total_tokens=420),
    )


def _mock_chat(*, side_effect: Any = None) -> Any:
    mock = AsyncMock(return_value=_chat_completion(json.dumps(CHAT_REPLY)), side_effect=side_effect)
    return patch("gateway.api.routes._passthrough.acompletion", mock)


def _credit_balance_too_low() -> anthropic.BadRequestError:
    message = "Your credit balance is too low to access the Anthropic API."
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.BadRequestError(
        f"Error code: 400 - {body}", response=httpx2.Response(400, request=request, json=body), body=body
    )


def test_a_chat_model_answers_a_decision_with_the_probabilities_it_states(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    client.post(
        f"{API_ROOT}/pricing",
        json={
            "model_key": "claude-api:claude-haiku-5-5",
            "input_price_per_million": 1.0,
            "output_price_per_million": 5.0,
        },
        headers=master_key_header,
    )
    images = ["data:image/png;base64,iVBORw0KGgo="]
    payload = {**PAYLOAD, "model": "claude-api:claude-haiku-5-5", "images": images}
    with _mock_chat() as chat, _mock_decision() as decision:
        resp = client.post(f"{API_ROOT}/decisions", json=payload, headers=api_key_header)

    assert resp.status_code == 200, resp.text
    decision.assert_not_awaited()
    sent = chat.await_args.kwargs
    assert sent["model"] == "anthropic:claude-haiku-5-5"
    assert sent["api_key"] == "sk-ant"
    assert sent["response_format"]["json_schema"]["schema"]["required"] == ["urgency", "team"]
    user_content = sent["messages"][1]["content"]
    assert user_content[1:] == [{"type": "image_url", "image_url": {"url": images[0]}}]

    answers = resp.json()["answers"]
    assert (answers["urgency"]["noul"], answers["urgency"]["confidence"]) == (0.9, 0.9)
    assert answers["team"]["choice"] == "technical"
    assert answers["team"]["probabilities"] == pytest.approx({"billing": 0.2, "technical": 0.8})

    (row,) = _decision_rows(client, master_key_header, api_key_obj["user_id"])
    assert (row["status"], row["provider"], row["model"]) == ("success", "claude-api", "claude-haiku-5-5")
    assert (row["prompt_tokens"], row["completion_tokens"]) == (400, 20)
    # 400 input tokens at $1/M plus 20 output tokens at $5/M.
    assert float(row["cost"]) == pytest.approx(0.0005)


def test_a_chat_model_answer_the_questions_cannot_be_read_from_is_502(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    payload = {**PAYLOAD, "model": "claude-api:claude-haiku-5-5"}
    with _mock_chat(side_effect=[_chat_completion('{"urgency": 0.9}')]):
        resp = client.post(f"{API_ROOT}/decisions", json=payload, headers=api_key_header)
    assert resp.status_code == 502
    with _mock_chat(side_effect=[_chat_completion(json.dumps(CHAT_REPLY), finish_reason="length")]):
        resp = client.post(f"{API_ROOT}/decisions", json=payload, headers=api_key_header)
    assert resp.status_code == 502


def test_a_policy_falls_over_from_a_chat_model_out_of_credit_to_a_decision_model(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    with _mock_chat(side_effect=_credit_balance_too_low()) as chat, _mock_decision() as decision:
        resp = client.post(f"{API_ROOT}/decisions", json={**PAYLOAD, "model": "chat-first"}, headers=api_key_header)

    assert resp.status_code == 200, resp.text
    assert chat.await_count == 1
    assert decision.await_count == 1
    assert resp.json()["model"] == "chat-first"
    assert (resp.headers[PROVIDER_HEADER], resp.headers[FALLBACK_HEADER]) == ("typesafe", "true")

    rows = _decision_rows(client, master_key_header, api_key_obj["user_id"])
    assert sorted((row["status"], row["provider"], row["status_code"]) for row in rows) == [
        ("absorbed", "claude-api", 400),
        ("success", "typesafe", None),
    ]
