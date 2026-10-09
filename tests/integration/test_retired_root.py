"""``/v1``, the API root before otari#1026, answers with a 404 naming the address to use."""

from collections.abc import Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway.api.deps import reset_config
from gateway.core.config import API_ROOT, PLATFORM_TOKEN_ENV_VAR, GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app

from .conftest import build_test_client

MOVED = ("/models", "/chat/completions", "/messages")


def _detail(target: str) -> dict[str, str]:
    return {"detail": f"Otari serves its API under {API_ROOT}, not /v1. Use {target} instead."}


def _assert_points_to_api_root(client: TestClient) -> None:
    for resource in MOVED:
        for method in ("GET", "POST"):
            response = client.request(method, f"/v1{resource}")

            assert response.status_code == 404, f"{method} /v1{resource}: {response.status_code}"
            assert response.json() == _detail(f"{API_ROOT}{resource}"), f"{method} /v1{resource}"


def test_the_old_root_points_to_the_api_root(client: TestClient) -> None:
    _assert_points_to_api_root(client)


def test_the_old_root_itself_and_its_other_methods_point_to_the_api_root(client: TestClient) -> None:
    assert client.get("/v1").json() == _detail(API_ROOT)
    assert client.delete("/v1/files/file-1").json() == _detail(f"{API_ROOT}/files/file-1")
    # 404 rather than 405 on the verbs a probe sends, which would read as "served here".
    assert client.head("/v1/models").status_code == 404
    assert client.options("/v1/models").status_code == 404


def test_the_api_root_and_the_dashboard_are_unaffected(client: TestClient, master_key_header: dict[str, str]) -> None:
    assert client.get(f"{API_ROOT}/health").status_code == 200
    assert client.get(f"{API_ROOT}/models", headers=master_key_header).status_code == 200
    assert client.get("/").status_code == 200


def test_an_unrelated_unknown_path_keeps_the_bare_404(client: TestClient) -> None:
    for path in ("/nope", "/v1beta/models", f"{API_ROOT}/nope"):
        response = client.get(path)

        assert response.status_code == 404, path
        assert response.json() == {"detail": "Not Found"}, path


def test_the_old_root_is_not_published(client: TestClient) -> None:
    assert isinstance(client.app, FastAPI)
    paths = client.app.openapi()["paths"]

    assert paths
    assert not [p for p in paths if p == "/v1" or p.startswith("/v1/")]


def _config(postgres_url: str, mode: str) -> GatewayConfig:
    return GatewayConfig(
        mode=mode,
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        bootstrap_api_key=False,
    )


@pytest.fixture
def hosted_client(postgres_url: str) -> Generator[TestClient]:
    yield from build_test_client(_config(postgres_url, "hosted"))


def test_a_hosted_deployment_points_to_the_api_root(hosted_client: TestClient) -> None:
    _assert_points_to_api_root(hosted_client)


def test_a_hybrid_gateway_points_to_the_api_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PLATFORM_TOKEN_ENV_VAR, "gw_test_token")
    config = GatewayConfig(mode="hybrid", platform={"base_url": "http://localhost:8100/api/v1"})

    with TestClient(create_app(config)) as client:
        _assert_points_to_api_root(client)

    reset_config()
    reset_db()
