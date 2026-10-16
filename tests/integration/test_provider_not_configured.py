"""A request for a provider the gateway holds no credential for.

any-llm refuses such a call before anything leaves the gateway, so the failure is
the deployment's configuration rather than the provider's. It is answered as a
client error that names the provider and the remedy, with a stable code, and
never as a 5xx: an SDK retries a 5xx, and no retry can supply a key.
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.core.error_codes import ERROR_CODE_HEADER, PROVIDER_NOT_CONFIGURED
from gateway.models.routing import RoutingConfig
from gateway.models.usage import UsageLog

from .conftest import build_test_client

# any-llm's own sentence, which the response must not copy.
_ANY_LLM_WORDING = "Please provide it in the config"


@pytest.fixture(autouse=True)
def _no_provider_keys_in_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The providers named below must be unconfigured whatever the developer's shell holds."""
    for name in ("ANTHROPIC_API_KEY", "COHERE_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _assert_provider_not_configured(response: Any, provider: str, env_var: str) -> None:
    assert response.status_code == 424, response.text
    assert response.headers[ERROR_CODE_HEADER] == PROVIDER_NOT_CONFIGURED
    body = response.json()
    assert body["code"] == PROVIDER_NOT_CONFIGURED
    assert f"'{provider}'" in body["detail"]
    assert env_var in body["detail"]
    assert _ANY_LLM_WORDING not in body["detail"]


def test_chat_names_the_provider_and_the_variable(
    client: TestClient,
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    test_user: dict[str, Any],
    db_session: Session,
) -> None:
    response = client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": "anthropic:claude-sonnet-4-5", "messages": [{"role": "user", "content": "Hello"}]},
        headers=api_key_header,
    )
    _assert_provider_not_configured(response, "anthropic", "ANTHROPIC_API_KEY")

    # Recorded as the refusal it is, so the error rate can tell it from a provider outage.
    row = db_session.query(UsageLog).filter(UsageLog.api_key_id == api_key_obj["id"]).one()
    assert row.status == "error"
    assert row.status_code == 424


def test_a_streaming_request_is_refused_before_the_stream_opens(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    response = client.post(
        f"{API_ROOT}/chat/completions",
        json={
            "model": "anthropic:claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
        headers=api_key_header,
    )
    _assert_provider_not_configured(response, "anthropic", "ANTHROPIC_API_KEY")
    assert response.headers["content-type"].startswith("application/json")


def test_a_passthrough_route_answers_the_same_way(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    response = client.post(
        f"{API_ROOT}/embeddings",
        json={"model": "cohere:embed-english-v3.0", "input": "hello"},
        headers=api_key_header,
    )
    _assert_provider_not_configured(response, "cohere", "COHERE_API_KEY")


def test_the_messages_route_answers_in_its_own_envelope(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    response = client.post(
        f"{API_ROOT}/messages",
        json={
            "model": "anthropic:claude-sonnet-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Hello"}],
        },
        headers=api_key_header,
    )
    assert response.status_code == 424, response.text
    assert response.headers[ERROR_CODE_HEADER] == PROVIDER_NOT_CONFIGURED
    message = response.json()["detail"]["error"]["message"]
    assert "'anthropic'" in message
    assert "ANTHROPIC_API_KEY" in message
    assert _ANY_LLM_WORDING not in message


# ---------------------------------------------------------------------------
# A routing policy whose head has no credential falls over to one that has
# ---------------------------------------------------------------------------

_MASTER = {API_KEY_HEADER: "Bearer test-master-key"}


@pytest.fixture
def failover_config(postgres_url: str) -> GatewayConfig:
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        providers={"openai": {"api_key": "sk-openai"}},
        routing=RoutingConfig.model_validate(
            {
                "policies": {
                    "fast": {
                        "select": [{"default": "mistral:mistral-small-latest"}],
                        "on_failure": ["openai:gpt-5-mini"],
                    },
                    "solo": {"select": [{"default": "mistral:mistral-small-latest"}]},
                    "unconfigured_pair": {
                        "select": [{"default": "mistral:mistral-small-latest"}],
                        "on_failure": ["cohere:command-r"],
                    },
                }
            }
        ),
    )


@pytest.fixture
def failover_client(failover_config: GatewayConfig) -> Generator[TestClient]:
    yield from build_test_client(failover_config)


def _completion(model: str) -> ChatCompletion:
    return ChatCompletion(
        id="cmpl-1",
        choices=[Choice(finish_reason="stop", index=0, message=ChatCompletionMessage(role="assistant", content="hi"))],
        created=0,
        model=model,
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


async def _provider_with_only_an_openai_key(**kwargs: Any) -> ChatCompletion:
    """What any-llm does with the candidates: refuses mistral for want of a key, serves openai."""
    if kwargs["model"].startswith("mistral"):
        raise MissingApiKeyError("mistral", "MISTRAL_API_KEY")
    if kwargs["model"].startswith("cohere"):
        raise MissingApiKeyError("cohere", "COHERE_API_KEY")
    return _completion(kwargs["model"])


def _chat(client: TestClient, model: str) -> Any:
    return client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "hi"}], "user": "test-user"},
        headers=_MASTER,
    )


def test_a_candidate_without_a_credential_is_passed_over(failover_client: TestClient) -> None:
    assert failover_client.post(f"{API_ROOT}/users", json={"user_id": "test-user"}, headers=_MASTER).status_code == 200
    with patch(
        "gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=_provider_with_only_an_openai_key)
    ) as mock:
        response = _chat(failover_client, "fast")
    assert response.status_code == 200, response.text
    dispatched = [call.kwargs["model"] for call in mock.await_args_list]
    assert [model.split(":")[0] for model in dispatched] == ["mistral", "openai"]


def test_a_policy_with_no_credentialed_candidate_is_refused_as_unconfigured(failover_client: TestClient) -> None:
    assert failover_client.post(f"{API_ROOT}/users", json={"user_id": "test-user"}, headers=_MASTER).status_code == 200
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=_provider_with_only_an_openai_key)):
        response = _chat(failover_client, "solo")
    _assert_provider_not_configured(response, "mistral", "MISTRAL_API_KEY")


def test_a_policy_whose_candidates_all_lack_a_credential_is_refused_as_unconfigured(
    failover_client: TestClient,
) -> None:
    assert failover_client.post(f"{API_ROOT}/users", json={"user_id": "test-user"}, headers=_MASTER).status_code == 200
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=_provider_with_only_an_openai_key)):
        response = _chat(failover_client, "unconfigured_pair")
    assert response.status_code == 424, response.text
    assert response.headers[ERROR_CODE_HEADER] == PROVIDER_NOT_CONFIGURED
    body = response.json()
    assert body["code"] == PROVIDER_NOT_CONFIGURED
    assert "cohere, mistral" in body["detail"]
    assert _ANY_LLM_WORDING not in body["detail"]
