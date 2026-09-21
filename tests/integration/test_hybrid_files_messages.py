"""File ownership gates dispatch and output failures preserve inference accounting."""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from any_llm.types.messages import MessageDelta, MessageDeltaEvent, MessageDeltaUsage
from fastapi.testclient import TestClient
from pydantic import SecretStr

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.services.provider_files.anthropic_inference import AnthropicFileOutputBinder
from gateway.services.provider_files.client import PlatformFilesClient
from gateway.services.provider_files.contracts import FileAccount, FilesError, Operation

from .conftest import app_for
from .test_hybrid_mode_messages import _attempt, _message_response, _resolve_payload


@pytest.mark.parametrize(
    "outcome, status_code, error_type",
    [
        ("success", 200, None),
        ("foreign", 404, "not_found_error"),
        ("wrong_generation", 403, "permission_error"),
        ("wrong_provider", 502, "api_error"),
        ("wrong_key", 409, "api_error"),
        ("wrong_base", 409, "api_error"),
        ("registration_failure", 502, "api_error"),
        ("reference_failure", 400, "invalid_request_error"),
        ("reference_failure", 401, "authentication_error"),
        ("reference_failure", 403, "permission_error"),
        ("reference_failure", 429, "rate_limit_error"),
        ("registration_failure", 400, "invalid_request_error"),
        ("registration_failure", 401, "authentication_error"),
        ("registration_failure", 403, "permission_error"),
        ("registration_failure", 404, "not_found_error"),
        ("registration_failure", 429, "rate_limit_error"),
    ],
)
def test_file_reference_dispatch_and_accounting(
    monkeypatch: pytest.MonkeyPatch, outcome: str, status_code: int, error_type: str | None
) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gateway-token")
    generation = uuid.uuid4()
    account = FileAccount(
        generation_id=generation,
        api_key=SecretStr("different-key" if outcome == "wrong_key" else "owned-key"),
        api_base="https://different.example" if outcome == "wrong_base" else None,
        provider="openai" if outcome == "wrong_provider" else "anthropic",
    )
    attempts = [
        _attempt(0, str(uuid.uuid4()), "other-model", "other-key"),
        _attempt(1, str(uuid.uuid4()), "owned-model", "owned-key"),
    ]
    for attempt in attempts:
        attempt["managed"] = False
    attempts[1]["provider_account_generation_id"] = str(uuid.uuid4() if outcome == "wrong_generation" else generation)
    events: list[str] = []

    async def platform(url: str, **kwargs: Any) -> httpx.Response:
        if url.endswith("/resolve"):
            return httpx.Response(200, json=_resolve_payload(attempts))
        events.append("usage")
        body = kwargs["body"]
        assert body["status"] == "success"
        return httpx.Response(
            200,
            json={
                "correlation_id": body["correlation_id"],
                "status": "completed",
                "outcome": "success",
                "cost_usd": "0.01",
                "currency": "USD",
                "usage_status": "reported",
                "pricing": {"source": "managed"},
            },
        )

    async def files(self: Any, path: str, body: dict[str, Any], result_type: Any) -> Any:
        if path == "cleanup/claim":
            return {"lease": None}
        events.append(path)
        if path == "references/resolve":
            assert body == {"ids": ["file_history"], "provider": "anthropic"}
            if outcome == "foreign":
                raise FilesError(404, "File not found")
            if outcome == "reference_failure":
                raise FilesError(status_code, "File reference rejected", headers={"Retry-After": "30"})
            return account
        if path == "outputs/prepare":
            assert body["attempt_id"] == attempts[1]["attempt_id"]
            return Operation(
                id=uuid.uuid4(),
                cleanup_token=SecretStr("cleanup"),
                deadline=datetime.now(UTC) + timedelta(minutes=5),
                account=account,
                max_bytes=1024,
                expires_in_seconds=3600,
            )
        return None

    async def provider(**kwargs: Any) -> Any:
        events.append("provider")
        assert kwargs["api_key"] == "owned-key"
        assert kwargs["model"] == "anthropic:owned-model"
        return _message_response()

    async def register(self: Any, value: Any) -> None:
        events.append("register")
        assert "usage" in events
        if outcome == "registration_failure":
            raise FilesError(status_code, "Unable to register generated files", headers={"Retry-After": "30"})

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", platform)
    monkeypatch.setattr("gateway.api.routes.messages.amessages", provider)
    monkeypatch.setattr(PlatformFilesClient, "post", files)
    monkeypatch.setattr(AnthropicFileOutputBinder, "register", register)
    app = app_for(
        GatewayConfig(
            mode="hybrid",
            platform={"base_url": "http://platform.test/api/v1"},
            files_provider_native_enabled=True,
        )
    )
    with TestClient(app) as client:
        response = client.post(
            f"{API_ROOT}/messages",
            headers={"Authorization": "Bearer user-token"},
            json={
                "model": "routed-model",
                "max_tokens": 100,
                **({"container": "container_01ABC"} if outcome == "registration_failure" else {}),
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "document", "source": {"type": "file", "file_id": "file_history"}}],
                    },
                    {"role": "assistant", "content": "Read it."},
                    {"role": "user", "content": "Summarize the earlier document."},
                ],
            },
        )
    assert response.status_code == status_code, response.text
    if error_type is not None:
        assert response.json()["detail"]["error"]["type"] == error_type
    if outcome in {"reference_failure", "registration_failure"}:
        assert response.headers["Retry-After"] == "30"
    if outcome in {"foreign", "wrong_generation", "wrong_provider", "wrong_key", "wrong_base", "reference_failure"}:
        assert "provider" not in events
        assert "outputs/prepare" not in events
    else:
        assert events.count("provider") == 1
        if outcome == "registration_failure":
            assert events.index("usage") < events.index("register")
        else:
            assert events == ["references/resolve", "provider", "usage"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool_loop", [False, True])
def test_input_only_dispatch_skips_full_output_quota(
    monkeypatch: pytest.MonkeyPatch, stream: bool, tool_loop: bool
) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gateway-token")
    generation = uuid.uuid4()
    account = FileAccount(generation_id=generation, api_key=SecretStr("owned-key"), workspace="trusted-workspace")
    attempt = _attempt(0, str(uuid.uuid4()), "owned-model", "owned-key")
    attempt["provider_account_generation_id"] = str(generation)
    events: list[str] = []

    async def platform(url: str, **kwargs: Any) -> httpx.Response:
        if url.endswith("/resolve"):
            return httpx.Response(200, json=_resolve_payload([attempt]))
        events.append("usage")
        return httpx.Response(200, json={"correlation_id": kwargs["body"]["correlation_id"], "status": "completed"})

    async def files(self: Any, path: str, body: dict[str, Any], result_type: Any) -> Any:
        if path == "cleanup/claim":
            return {"lease": None}
        events.append(path)
        if path == "references/resolve":
            return account
        if path == "outputs/prepare":
            raise FilesError(429, "File capacity exceeded")
        raise AssertionError(f"Unexpected Files call: {path}")

    def check_dispatch(kwargs: dict[str, Any]) -> None:
        events.append("provider")
        assert kwargs["api_key"] == "owned-key"
        assert kwargs["model"] == "anthropic:owned-model"
        assert kwargs["client_args"] == {
            "max_retries": 0,
            "default_headers": {"anthropic-workspace-id": "trusted-workspace"},
        }
        assert "_file_attempt" not in kwargs
        assert not {"anthropic-workspace-id", "x-api-key"} & kwargs.get("extra_headers", {}).keys()

    async def chunks() -> AsyncIterator[MessageDeltaEvent]:
        yield MessageDeltaEvent(
            type="message_delta",
            delta=MessageDelta(stop_reason="end_turn", stop_sequence=None),
            usage=MessageDeltaUsage(input_tokens=3, output_tokens=5),
        )

    async def provider(**kwargs: Any) -> Any:
        check_dispatch(kwargs)
        return chunks() if stream else _message_response()

    async def loop(**kwargs: Any) -> Any:
        check_dispatch(kwargs["completion_kwargs"])
        return _message_response()

    async def loop_stream(**kwargs: Any) -> AsyncIterator[MessageDeltaEvent]:
        check_dispatch(kwargs["completion_kwargs"])
        async for event in chunks():
            yield event

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", platform)
    monkeypatch.setattr(PlatformFilesClient, "post", files)
    monkeypatch.setattr("gateway.api.routes.messages.amessages", provider)
    monkeypatch.setattr("gateway.api.routes.messages.anthropic_tool_loop", loop)
    monkeypatch.setattr("gateway.api.routes.messages.anthropic_tool_loop_stream", loop_stream)
    monkeypatch.setattr(
        "gateway.services.mcp_client.MCPClientPool.__aenter__",
        AsyncMock(return_value=AsyncMock(purpose_hints=lambda: [])),
    )
    monkeypatch.setattr("gateway.services.mcp_client.MCPClientPool.__aexit__", AsyncMock(return_value=None))
    app = app_for(
        GatewayConfig(
            mode="hybrid",
            platform={"base_url": "http://platform.test/api/v1"},
            files_provider_native_enabled=True,
        )
    )
    with TestClient(app) as client:
        response = client.post(
            f"{API_ROOT}/messages",
            headers={"Authorization": "Bearer user-token"},
            json={
                "model": "routed-model",
                "max_tokens": 100,
                "stream": stream,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "document", "source": {"type": "file", "file_id": "file_history"}}],
                    }
                ],
                "extra_headers": {"anthropic-workspace-id": "foreign", "x-api-key": "foreign"},
                **({"mcp_servers": [{"name": "test", "url": "http://127.0.0.1:18080/mcp"}]} if tool_loop else {}),
            },
        )
    assert response.status_code == 200, response.text
    assert events == ["references/resolve", "provider", "usage"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("attempt_count", [1, 2])
@pytest.mark.parametrize(
    "status_code, error_type",
    [(403, "permission_error"), (409, "api_error"), (429, "rate_limit_error")],
)
def test_output_preparation_failure_keeps_its_status(
    monkeypatch: pytest.MonkeyPatch, stream: bool, attempt_count: int, status_code: int, error_type: str
) -> None:
    """A Files refusal before dispatch reaches the caller as itself, not as a provider failure."""
    monkeypatch.setenv("OTARI_AI_TOKEN", "gateway-token")
    attempts = [_attempt(index, str(uuid.uuid4()), "owned-model", "owned-key") for index in range(attempt_count)]
    for attempt in attempts:
        attempt["managed"] = False
        attempt["provider_account_generation_id"] = str(uuid.uuid4())
    events: list[str] = []

    async def platform(url: str, **kwargs: Any) -> httpx.Response:
        if url.endswith("/resolve"):
            return httpx.Response(200, json=_resolve_payload(attempts))
        events.append("usage")
        return httpx.Response(200, json={"correlation_id": kwargs["body"]["correlation_id"], "status": "completed"})

    async def files(self: Any, path: str, body: dict[str, Any], result_type: Any) -> Any:
        events.append(path)
        assert path == "outputs/prepare"
        raise FilesError(status_code, "Provider file outputs unavailable", headers={"Retry-After": "30"})

    async def provider(**kwargs: Any) -> Any:
        raise AssertionError("the provider must not be called when output preparation fails")

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", platform)
    monkeypatch.setattr("gateway.api.routes.messages.amessages", provider)
    monkeypatch.setattr(PlatformFilesClient, "post", files)
    app = app_for(
        GatewayConfig(
            mode="hybrid",
            platform={"base_url": "http://platform.test/api/v1"},
            files_provider_native_enabled=True,
        )
    )
    with TestClient(app) as client:
        response = client.post(
            f"{API_ROOT}/messages",
            headers={"Authorization": "Bearer user-token"},
            json={
                "model": "routed-model",
                "max_tokens": 100,
                "stream": stream,
                "container": "container_01ABC",
                "messages": [{"role": "user", "content": "Run it."}],
            },
        )
    assert response.status_code == status_code, response.text
    body = response.json()["detail"]["error"]
    assert body == {"type": error_type, "message": "Provider file outputs unavailable"}
    assert response.headers["Retry-After"] == "30"
    assert events.count("outputs/prepare") == (attempt_count if stream else 1)
