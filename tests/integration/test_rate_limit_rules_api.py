"""``/api/v1/rate-limits``: the rules an operator manages from the dashboard, and their enforcement."""

from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig, RateLimitRule

from .conftest import build_test_client

_MASTER = {API_KEY_HEADER: "Bearer test-master-key"}
_RULES = f"{API_ROOT}/rate-limits"


@pytest.fixture
def config(postgres_url: str) -> GatewayConfig:
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        rate_limits=[RateLimitRule(name="from-file", per="deployment", max_concurrent=100)],
    )


@pytest.fixture
def client(config: GatewayConfig) -> Generator[TestClient]:
    yield from build_test_client(config)


async def _completion(**kwargs: Any) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=1700000000,
        model="gpt-4o-mini",
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _key(client: TestClient, user_id: str) -> dict[str, str]:
    assert client.post(f"{API_ROOT}/users", json={"user_id": user_id}, headers=_MASTER).status_code == 200
    response = client.post(f"{API_ROOT}/keys", json={"key_name": user_id, "user_id": user_id}, headers=_MASTER)
    assert response.status_code == 200
    return {API_KEY_HEADER: f"Bearer {response.json()['key']}"}


def _chats(client: TestClient, headers: dict[str, str], count: int) -> list[int]:
    body = {"model": "openai:gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
    with patch("gateway.api.routes.chat.acompletion", new=_completion):
        return [
            client.post(f"{API_ROOT}/chat/completions", json=body, headers=headers).status_code for _ in range(count)
        ]


def test_the_list_names_where_each_rule_is_defined(client: TestClient) -> None:
    assert client.post(_RULES, json={"name": "keys", "per": "key", "rpm": 5}, headers=_MASTER).status_code == 201

    rules = client.get(_RULES, headers=_MASTER).json()["rules"]

    assert [(rule["name"], rule["source"]) for rule in rules] == [("from-file", "config"), ("keys", "dashboard")]
    assert rules[1]["rpm"] == 5
    assert rules[1]["updated_at"] is not None


def test_a_stored_rule_is_enforced_from_the_next_request(client: TestClient) -> None:
    headers = _key(client, "alice")
    assert _chats(client, headers, 2) == [200, 200]

    client.post(_RULES, json={"name": "keys", "per": "key", "rpm": 1}, headers=_MASTER)

    assert _chats(client, headers, 2) == [200, 429]


def test_a_changed_rule_applies_its_new_limit(client: TestClient) -> None:
    headers = _key(client, "bob")
    client.post(_RULES, json={"name": "keys", "per": "key", "rpm": 1}, headers=_MASTER)
    assert _chats(client, headers, 2) == [200, 429]

    response = client.patch(f"{_RULES}/keys", json={"rpm": 10}, headers=_MASTER)

    assert response.status_code == 200
    assert response.json()["rpm"] == 10
    assert _chats(client, headers, 1) == [200]


def test_a_deleted_rule_stops_applying(client: TestClient) -> None:
    headers = _key(client, "carol")
    client.post(_RULES, json={"name": "keys", "per": "key", "rpm": 1}, headers=_MASTER)
    assert _chats(client, headers, 2) == [200, 429]

    assert client.delete(f"{_RULES}/keys", headers=_MASTER).status_code == 204

    assert _chats(client, headers, 1) == [200]
    assert [rule["name"] for rule in client.get(_RULES, headers=_MASTER).json()["rules"]] == ["from-file"]


def test_a_config_rule_is_read_only(client: TestClient) -> None:
    created = client.post(_RULES, json={"name": "from-file", "per": "key", "rpm": 1}, headers=_MASTER)
    changed = client.patch(f"{_RULES}/from-file", json={"rpm": 1}, headers=_MASTER)
    deleted = client.delete(f"{_RULES}/from-file", headers=_MASTER)

    assert [created.status_code, changed.status_code, deleted.status_code] == [409, 409, 409]
    assert "config.yml" in deleted.json()["detail"]


def test_a_name_is_stored_once(client: TestClient) -> None:
    client.post(_RULES, json={"name": "keys", "per": "key", "rpm": 1}, headers=_MASTER)

    response = client.post(_RULES, json={"name": "keys", "per": "user", "rpm": 2}, headers=_MASTER)

    assert response.status_code == 409


def test_an_update_that_leaves_no_limit_is_refused(client: TestClient) -> None:
    client.post(_RULES, json={"name": "keys", "per": "key", "rpm": 1}, headers=_MASTER)

    response = client.patch(f"{_RULES}/keys", json={"rpm": None}, headers=_MASTER)

    assert response.status_code == 422
    assert client.get(_RULES, headers=_MASTER).json()["rules"][1]["rpm"] == 1


def test_a_rule_without_a_limit_is_refused(client: TestClient) -> None:
    response = client.post(_RULES, json={"name": "keys", "per": "key"}, headers=_MASTER)

    assert response.status_code == 422


def test_an_unknown_rule_is_not_found(client: TestClient) -> None:
    assert client.patch(f"{_RULES}/nope", json={"rpm": 1}, headers=_MASTER).status_code == 404
    assert client.delete(f"{_RULES}/nope", headers=_MASTER).status_code == 404


def test_the_rules_need_operator_standing(client: TestClient) -> None:
    headers = _key(client, "dave")

    assert client.get(_RULES, headers=headers).status_code in {401, 403}


@pytest.fixture
def model_rule_client(config: GatewayConfig) -> Generator[TestClient]:
    model_rule = RateLimitRule(name="model-cap", per="model", models=["openai:gpt-4o"], rpm=100)
    yield from build_test_client(config.model_copy(update={"rate_limits": [model_rule]}))


def test_a_per_model_rule_is_listed_with_its_models(model_rule_client: TestClient) -> None:
    rules = model_rule_client.get(_RULES, headers=_MASTER).json()["rules"]

    assert [(rule["name"], rule["source"], rule["models"]) for rule in rules] == [
        ("model-cap", "config", ["openai:gpt-4o"])
    ]


def test_a_stored_per_model_rule_limits_its_model(client: TestClient) -> None:
    body = {"name": "cap", "per": "model", "models": ["openai/gpt-4o-mini"], "rpm": 1}
    response = client.post(_RULES, json=body, headers=_MASTER)
    assert response.status_code == 201
    assert response.json()["models"] == ["openai:gpt-4o-mini"]

    assert _chats(client, _key(client, "mallory"), 2) == [200, 429]


def test_a_rule_moved_off_per_model_drops_its_models(client: TestClient) -> None:
    body = {"name": "cap", "per": "model", "models": ["openai:gpt-4o"], "rpm": 5}
    assert client.post(_RULES, json=body, headers=_MASTER).status_code == 201

    response = client.patch(f"{_RULES}/cap", json={"per": "key"}, headers=_MASTER)

    assert response.status_code == 200
    assert response.json()["models"] is None
    assert client.patch(f"{_RULES}/cap", json={"per": "model"}, headers=_MASTER).status_code == 422
