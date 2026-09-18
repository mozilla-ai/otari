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
    "outcome", ["success", "foreign", "wrong_generation", "wrong_provider", "registration_failure"]
)
def test_file_reference_dispatch_and_accounting(monkeypatch: pytest.MonkeyPatch, outcome: str) -> None:
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
            raise FilesError(502, "Unable to register generated files")

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
    assert (
        response.status_code
        == {
            "success": 200,
            "foreign": 404,
            "wrong_generation": 403,
            "wrong_provider": 502,
            "registration_failure": 502,
        }[outcome]
    ), response.text
    if outcome in {"foreign", "wrong_generation", "wrong_provider"}:
        assert "provider" not in events
    else:
        assert events.count("provider") == 1
        assert events.index("usage") < events.index("register")
