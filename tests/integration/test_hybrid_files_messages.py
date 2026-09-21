"""File ownership gates dispatch and output failures preserve inference accounting."""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
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
        api_key=SecretStr("owned-key"),
        provider="openai" if outcome == "wrong_provider" else "anthropic",
    )
    attempts = [
        _attempt(0, str(uuid.uuid4()), "other-model", "other-key"),
        _attempt(1, str(uuid.uuid4()), "owned-model", "owned-key"),
    ]
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
    if outcome in {"foreign", "wrong_generation", "wrong_provider", "reference_failure"}:
        assert "provider" not in events
    else:
        assert events.count("provider") == 1
        assert events.index("usage") < events.index("register")
