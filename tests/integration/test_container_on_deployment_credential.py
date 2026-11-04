"""A caller's provider container never rides on a credential the deployment owns, end to end.

A provider container id names an execution environment and its files on the
provider account that minted it. The deployment's own account is shared by every
organization it serves, so a candidate served on it is refused a container the
caller picked, and a plan falls through to a candidate on the caller's own key.
A probe overlay serves ``openai`` on the deployment's key; the configured
``anthropic`` key is the caller's own.
"""

import itertools
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.messages import MessageResponse, MessageUsage, TextBlock
from fastapi import status
from fastapi.testclient import TestClient
from openai.types.responses import Response, ResponseUsage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.models.routing import RoutingConfig

from .conftest import build_test_client

MODULE = "probe_container_fleet"
PROBE_BOOTSTRAP = """
from gateway.container import Container
from gateway.ports.model_provider_port import HostedCredential, ModelProviderPort


class Fleet:
    def __init__(self, session):
        del session

    async def resolve_hosted_credential(self, *, organization_id, workspace_id, provider, model):
        if provider != "openai":
            return None
        return HostedCredential(api_key="fleet-key", api_base=None, response_provider="openai")

    async def get_hosted_models(self, *, organization_id):
        return {"openai": None}


def register(container: Container) -> None:
    container.bind(ModelProviderPort, Fleet)
"""
MASTER = {API_KEY_HEADER: "Bearer test-master-key"}
FLEET = "openai:gpt-4o"
OWN = "anthropic:claude-opus-4"
REFUSED = "container cannot be used on this route"

_NAMES = (f"{MODULE}_{n}" for n in itertools.count())


@pytest.fixture(autouse=True)
def _no_ambient_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's own shell key would credential the ladder and skip the fleet."""
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def client(
    postgres_url: str, clean_database: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient]:
    name = next(_NAMES)
    (tmp_path / f"{name}.py").write_text(PROBE_BOOTSTRAP)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(name, None)
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        providers={"anthropic": {"api_key": "sk-own"}},
        routing=RoutingConfig.model_validate(
            {"policies": {"fleet_first": {"select": [{"default": FLEET}], "on_failure": [OWN]}}}
        ),
        bootstrap=f"{name}:register",
    )
    try:
        yield from build_test_client(config)
    finally:
        sys.modules.pop(name, None)


def _message() -> MessageResponse:
    return MessageResponse(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-opus-4",
        content=[TextBlock(type="text", text="ok", citations=None)],
        stop_reason=cast(Any, "end_turn"),
        stop_sequence=None,
        usage=MessageUsage(input_tokens=5, output_tokens=2),
    )


def _response() -> Response:
    return Response(
        id="resp_test",
        created_at=0.0,
        model="gpt-4o",
        object="response",
        status=cast(Any, "completed"),
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage=ResponseUsage(
            input_tokens=5,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens=2,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            total_tokens=7,
        ),
        error=None,
        incomplete_details=None,
        instructions=None,
        metadata=None,
        temperature=None,
        top_p=None,
    )


def _messages(client: TestClient, model: str, container: str | None, *, stream: bool = False) -> tuple[Any, AsyncMock]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 16,
        "stream": stream,
    }
    if container is not None:
        body["container"] = container
    provider = AsyncMock(return_value=_message())
    with patch("gateway.api.routes.messages.amessages", new=provider):
        response = client.post(f"{API_ROOT}/messages", json={**body, "metadata": {"user_id": "u1"}}, headers=MASTER)
    return response, provider


def _responses(client: TestClient, container: object) -> tuple[Any, AsyncMock]:
    provider = AsyncMock(return_value=_response())
    with patch("gateway.api.routes.responses.aresponses", new=provider):
        response = client.post(
            f"{API_ROOT}/responses",
            json={
                "model": "openai:gpt-4o",
                "input": "Hi",
                "tools": [{"type": "code_interpreter", "container": container}],
                "user": "u1",
            },
            headers=MASTER,
        )
    return response, provider


@pytest.fixture(autouse=True)
def _user(client: TestClient) -> None:
    assert client.post(f"{API_ROOT}/users", json={"user_id": "u1"}, headers=MASTER).status_code == 200


def test_a_container_id_is_refused_on_the_deployments_credential(client: TestClient) -> None:
    response, provider = _messages(client, FLEET, "container_abc")

    assert response.status_code == status.HTTP_400_BAD_REQUEST, response.text
    assert REFUSED in response.text
    provider.assert_not_called()


def test_a_streamed_container_id_is_refused_on_the_deployments_credential(client: TestClient) -> None:
    response, provider = _messages(client, FLEET, "container_abc", stream=True)

    assert response.status_code == status.HTTP_400_BAD_REQUEST, response.text
    assert REFUSED in response.text
    provider.assert_not_called()


def test_a_request_with_no_container_is_served_on_the_deployments_credential(client: TestClient) -> None:
    response, provider = _messages(client, FLEET, None)

    assert response.status_code == status.HTTP_200_OK, response.text
    assert provider.call_args.kwargs["api_key"] == "fleet-key"


def test_a_container_id_reaches_the_callers_own_credential(client: TestClient) -> None:
    response, provider = _messages(client, OWN, "container_abc")

    assert response.status_code == status.HTTP_200_OK, response.text
    assert provider.call_args.kwargs["api_key"] == "sk-own"
    assert provider.call_args.kwargs["container"] == "container_abc"


def test_a_plan_passes_over_the_deployments_credential_for_the_callers_own(client: TestClient) -> None:
    response, provider = _messages(client, "fleet_first", "container_abc")

    assert response.status_code == status.HTTP_200_OK, response.text
    assert [call.kwargs["api_key"] for call in provider.call_args_list] == ["sk-own"]
    assert provider.call_args.kwargs["container"] == "container_abc"


def test_a_code_interpreter_container_id_is_refused_on_the_deployments_credential(client: TestClient) -> None:
    response, provider = _responses(client, "cntr_abc")

    assert response.status_code == status.HTTP_400_BAD_REQUEST, response.text
    assert REFUSED in response.text
    provider.assert_not_called()


def test_a_fresh_code_interpreter_container_is_served_on_the_deployments_credential(client: TestClient) -> None:
    response, provider = _responses(client, {"type": "auto"})

    assert response.status_code == status.HTTP_200_OK, response.text
    assert provider.call_args.kwargs["api_key"] == "fleet-key"
