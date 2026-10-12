"""A provider that refuses the credential the gateway holds for it.

The provider answered 401 or 403 to the gateway's own key, so the failure is the
deployment's configuration, like a missing credential, and not the caller's key.
It is answered as a 424 with a stable code and a fixed detail, never as a 5xx
(an SDK retries a 5xx, and no retry makes the key valid) and never as a 401 or
403 (which would read as the caller's own key being wrong).
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from any_llm.utils.exception_handler import convert_exception
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.api.routes._pipeline import PROVIDER_CREDENTIALS_DETAIL
from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.core.error_codes import ERROR_CODE_HEADER, PROVIDER_CREDENTIAL_REJECTED
from gateway.models.routing import RoutingConfig
from gateway.models.usage import UsageLog

from .conftest import build_test_client

_LEAKED_KEY = "sk-proj-AAAAAAAAAAAAAAAAAAAA"


def _rejected(status_code: int = 401) -> Exception:
    """The provider's refusal of the gateway's key, as any-llm hands it back."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    error_type = openai.AuthenticationError if status_code == 401 else openai.PermissionDeniedError
    upstream = error_type(
        f"Error code: {status_code} - Incorrect API key provided: {_LEAKED_KEY}",
        response=httpx.Response(status_code, request=request),
        body=None,
    )
    return convert_exception(upstream, "openai")


def _assert_credential_rejected(response: Any) -> None:
    assert response.status_code == 424, response.text
    assert response.headers[ERROR_CODE_HEADER] == PROVIDER_CREDENTIAL_REJECTED
    assert _LEAKED_KEY not in response.text


def _assert_credential_rejected_body(response: Any) -> None:
    _assert_credential_rejected(response)
    body = response.json()
    assert body["code"] == PROVIDER_CREDENTIAL_REJECTED
    assert body["detail"] == PROVIDER_CREDENTIALS_DETAIL


@pytest.mark.parametrize("status_code", [401, 403])
def test_chat_refuses_with_a_424_and_records_the_upstream_status(
    client: TestClient,
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    test_user: dict[str, Any],
    db_session: Session,
    status_code: int,
) -> None:
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=_rejected(status_code))):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": "openai:gpt-5-mini", "messages": [{"role": "user", "content": "Hello"}]},
            headers=api_key_header,
        )
    _assert_credential_rejected_body(response)

    # The row keeps the provider's own status, so misconfiguration stays countable.
    row = db_session.query(UsageLog).filter(UsageLog.api_key_id == api_key_obj["id"]).one()
    assert row.status == "error"
    assert row.status_code == status_code


def test_a_streaming_request_is_refused_before_the_stream_opens(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=_rejected())):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": "openai:gpt-5-mini", "messages": [{"role": "user", "content": "Hello"}], "stream": True},
            headers=api_key_header,
        )
    _assert_credential_rejected_body(response)


def test_the_responses_route_answers_the_same_way(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    with patch("gateway.api.routes.responses.aresponses", new=AsyncMock(side_effect=_rejected())):
        response = client.post(
            f"{API_ROOT}/responses",
            json={"model": "openai:gpt-5-mini", "input": "Hello"},
            headers=api_key_header,
        )
    _assert_credential_rejected(response)
    assert PROVIDER_CREDENTIALS_DETAIL in response.text


def test_the_messages_route_answers_in_its_own_envelope(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    with patch("gateway.api.routes.messages.amessages", new=AsyncMock(side_effect=_rejected())):
        response = client.post(
            f"{API_ROOT}/messages",
            json={"model": "openai:gpt-5-mini", "max_tokens": 16, "messages": [{"role": "user", "content": "Hello"}]},
            headers=api_key_header,
        )
    _assert_credential_rejected(response)
    assert response.json()["detail"]["error"]["message"] == PROVIDER_CREDENTIALS_DETAIL


@pytest.mark.parametrize(
    ("path", "target", "body"),
    [
        (
            "embeddings",
            "gateway.api.routes.embeddings.aembedding",
            {"model": "openai:text-embedding-3-small", "input": "hello"},
        ),
        (
            "rerank",
            "gateway.api.routes.rerank.arerank",
            {"model": "cohere:rerank-v3.5", "query": "q", "documents": ["a"]},
        ),
    ],
)
def test_a_passthrough_route_answers_the_same_way(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
    path: str,
    target: str,
    body: dict[str, Any],
) -> None:
    with patch(target, new=AsyncMock(side_effect=_rejected())):
        response = client.post(f"{API_ROOT}/{path}", json=body, headers=api_key_header)
    _assert_credential_rejected_body(response)


# ---------------------------------------------------------------------------
# A routing policy whose head's credential is rejected falls over to the next
# ---------------------------------------------------------------------------

_MASTER = {API_KEY_HEADER: "Bearer test-master-key"}


@pytest.fixture
def failover_client(postgres_url: str) -> Generator[TestClient]:
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        providers={"openai": {"api_key": "sk-openai"}, "mistral": {"api_key": "revoked"}},
        routing=RoutingConfig.model_validate(
            {
                "policies": {
                    "fast": {
                        "select": [{"default": "mistral:mistral-small-latest"}],
                        "on_failure": ["openai:gpt-5-mini"],
                    },
                    "unusable_pair": {
                        "select": [{"default": "mistral:mistral-small-latest"}],
                        "on_failure": ["cohere:command-r"],
                    },
                }
            }
        ),
    )
    yield from build_test_client(config)


async def _mistral_rejects_its_key(**kwargs: Any) -> ChatCompletion:
    if kwargs["model"].startswith("mistral"):
        raise _rejected()
    return ChatCompletion(
        id="cmpl-1",
        choices=[Choice(finish_reason="stop", index=0, message=ChatCompletionMessage(role="assistant", content="hi"))],
        created=0,
        model=kwargs["model"],
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def test_a_candidate_whose_credential_is_rejected_is_passed_over(failover_client: TestClient) -> None:
    assert failover_client.post(f"{API_ROOT}/users", json={"user_id": "test-user"}, headers=_MASTER).status_code == 200
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=_mistral_rejects_its_key)) as mock:
        response = failover_client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": "fast", "messages": [{"role": "user", "content": "hi"}], "user": "test-user"},
            headers=_MASTER,
        )
    assert response.status_code == 200, response.text
    dispatched = [call.kwargs["model"] for call in mock.await_args_list]
    assert [model.split(":")[0] for model in dispatched] == ["mistral", "openai"]


async def _no_usable_credential(**kwargs: Any) -> ChatCompletion:
    """Mistral rejects its key and Cohere has none, so no candidate can be served."""
    if kwargs["model"].startswith("cohere"):
        raise MissingApiKeyError("cohere", "COHERE_API_KEY")
    raise _rejected()


def test_a_policy_with_no_usable_credential_is_refused_with_a_424(failover_client: TestClient) -> None:
    assert failover_client.post(f"{API_ROOT}/users", json={"user_id": "test-user"}, headers=_MASTER).status_code == 200
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=_no_usable_credential)):
        response = failover_client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": "unusable_pair", "messages": [{"role": "user", "content": "hi"}], "user": "test-user"},
            headers=_MASTER,
        )
    _assert_credential_rejected(response)
    assert response.json()["code"] == PROVIDER_CREDENTIAL_REJECTED
    assert "2 attempts" in response.json()["detail"]


# ---------------------------------------------------------------------------
# A 403 that refuses the request is the caller's 400, not a rejected credential
# ---------------------------------------------------------------------------

_LOGPROBS_MSG = "You are not allowed to request logprobs from this model"


def _request_fault_403() -> Exception:
    """OpenAI's 403 for logprobs on a model without them, sent with a valid key."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    error = {"message": _LOGPROBS_MSG, "type": "invalid_request_error", "param": None, "code": None}
    upstream = openai.PermissionDeniedError(
        f"Error code: 403 - {{'error': {error}}}",
        response=httpx.Response(403, request=request, json={"error": error}),
        body=error,
    )
    return convert_exception(upstream, "openai")


def _assert_request_fault(response: Any) -> None:
    assert response.status_code == 400, response.text
    assert response.headers.get(ERROR_CODE_HEADER) != PROVIDER_CREDENTIAL_REJECTED
    assert response.json()["detail"] == _LOGPROBS_MSG


@pytest.mark.parametrize("stream", [False, True])
def test_chat_answers_a_request_fault_403_with_a_400_carrying_the_providers_sentence(
    client: TestClient,
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    test_user: dict[str, Any],
    db_session: Session,
    stream: bool,
) -> None:
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=_request_fault_403())):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": "openai:gpt-5-mini",
                "messages": [{"role": "user", "content": "Hello"}],
                "logprobs": True,
                "stream": stream,
            },
            headers=api_key_header,
        )
    _assert_request_fault(response)
    row = db_session.query(UsageLog).filter(UsageLog.api_key_id == api_key_obj["id"]).one()
    assert row.status_code == 403


def test_a_passthrough_route_answers_a_request_fault_403_with_a_400(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    with patch("gateway.api.routes.embeddings.aembedding", new=AsyncMock(side_effect=_request_fault_403())):
        response = client.post(
            f"{API_ROOT}/embeddings", json={"model": "openai:gpt-5-mini", "input": "hello"}, headers=api_key_header
        )
    _assert_request_fault(response)


def test_a_policy_whose_attempts_all_refuse_the_request_is_not_a_credential_failure(
    failover_client: TestClient,
) -> None:
    assert failover_client.post(f"{API_ROOT}/users", json={"user_id": "test-user"}, headers=_MASTER).status_code == 200
    with patch("gateway.api.routes.chat.acompletion", new=AsyncMock(side_effect=_request_fault_403())):
        response = failover_client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": "fast", "messages": [{"role": "user", "content": "hi"}], "user": "test-user"},
            headers=_MASTER,
        )
    assert response.status_code == 400, response.text
    assert response.headers.get(ERROR_CODE_HEADER) != PROVIDER_CREDENTIAL_REJECTED
