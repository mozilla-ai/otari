from collections.abc import Generator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import InstallControlPlane
from gateway.api.deps import reset_config
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.database import reset_db

from .conftest import app_for


@pytest.fixture
def platform_client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient]:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
    app = app_for(GatewayConfig(mode="hybrid", platform={"base_url": "http://platform.test/api/v1"}))

    with TestClient(app) as client:
        yield client

    reset_config()
    reset_db()


def test_hybrid_models_requires_credentials(platform_client: TestClient) -> None:
    response = platform_client.get(f"{API_ROOT}/models")

    assert response.status_code == 401


def test_hybrid_models_relays_the_platform_list(
    platform_client: TestClient, control_plane_transport: InstallControlPlane
) -> None:
    seen: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        seen.append((url, headers, body))
        return httpx.Response(
            200,
            json={
                "models": [
                    {"id": "openai:gpt-4o", "created": 7, "owned_by": "openai"},
                    {"id": "anthropic:claude-sonnet-4-5"},
                ]
            },
        )

    control_plane_transport(fake_post_platform)

    response = platform_client.get(f"{API_ROOT}/models", headers={"Authorization": "Bearer user_test_token"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert [(m["id"], m["created"], m["owned_by"]) for m in payload["data"]] == [
        ("anthropic:claude-sonnet-4-5", 0, "anthropic"),
        ("openai:gpt-4o", 7, "openai"),
    ]
    url, headers, body = seen[0]
    assert url == "http://platform.test/api/v1/gateway/models/resolve"
    assert headers["X-User-Token"] == "user_test_token"
    assert headers["X-Gateway-Token"] == "gw_test_token"
    assert body == {}


def test_hybrid_models_forwards_platform_refusal(
    platform_client: TestClient, control_plane_transport: InstallControlPlane
) -> None:
    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Invalid user token"})

    control_plane_transport(fake_post_platform)

    response = platform_client.get(f"{API_ROOT}/models", headers={"Otari-Key": "bad"})

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid user token"}


@pytest.mark.parametrize("payload", [{}, {"models": "nope"}, {"models": [{"created": 1}]}, ["x"]])
def test_hybrid_models_maps_unreadable_answer_to_bad_gateway(
    platform_client: TestClient, control_plane_transport: InstallControlPlane, payload: Any
) -> None:
    async def fake_post_platform(
        url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        return httpx.Response(200, json=payload)

    control_plane_transport(fake_post_platform)

    response = platform_client.get(f"{API_ROOT}/models", headers={"Otari-Key": "k"})

    assert response.status_code == 502
