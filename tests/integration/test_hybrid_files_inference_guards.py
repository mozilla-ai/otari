"""OpenAI file state cannot bypass ownership through unsupported inference envelopes."""

from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT, GatewayConfig

from .conftest import app_for


@pytest.mark.parametrize(
    "fields",
    [
        {"tools": [{"type": "code_interpreter", "container": {"type": "auto", "file_ids": ["file_foreign"]}}]},
        {"tools": [{"type": "code_interpreter", "container": "cntr_foreign"}]},
        {"tools": [{"type": "code_interpreter", "container": {"type": "auto"}}]},
        {"tools": [{"type": "file_search", "vector_store_ids": ["vs_foreign"]}]},
        {"previous_response_id": "resp_foreign"},
        {"conversation": "conv_foreign"},
        {"tools": [{"type": "shell", "environment": {"type": "container_reference", "container_id": "cntr_foreign"}}]},
        {"tools": [{"type": "image_generation", "input_image_mask": {"file_id": "file_foreign"}}]},
        {"input": [{"type": "compaction", "encrypted_content": "opaque-state"}]},
        {
            "input": [
                {
                    "type": "computer_call_output",
                    "call_id": "call_1",
                    "output": {"type": "computer_screenshot", "file_id": "file_foreign"},
                }
            ]
        },
        {"input": [{"type": "item_reference", "id": "item_foreign"}]},
        {"input": [{"role": "user", "content": [{"type": "input_file", "file_id": "file_foreign"}]}]},
    ],
)
def test_openai_state_rejected_before_resolution(monkeypatch: pytest.MonkeyPatch, fields: dict[str, Any]) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gateway-token")
    calls: list[str] = []

    async def platform(url: str, **kwargs: Any) -> httpx.Response:
        calls.append(url)
        return httpx.Response(502)

    monkeypatch.setattr("gateway.api.routes._platform._post_platform", platform)
    app = app_for(
        GatewayConfig(
            mode="hybrid", platform={"base_url": "https://authority.test"}, files_provider_native_enabled=True
        )
    )
    with TestClient(app) as client:
        response = client.post(
            f"{API_ROOT}/responses",
            headers={"Authorization": "Bearer user-token"},
            json={"model": "openai:gpt-4o", "input": "Read it", **fields},
        )
    assert response.status_code == 400, response.text
    assert calls == []
