"""Endpoint tests for GET /v1/tools (gateway-run tool discovery)."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gateway.core.config import GatewayConfig
from gateway.main import create_app

AUTH = {"Authorization": "Bearer sk-test-master"}


@pytest.fixture(autouse=True)
def _no_tool_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep the ambient environment from deciding what the endpoint reports."""
    for name in ("OTARI_WEB_SEARCH_URL", "OTARI_SANDBOX_URL", "OTARI_WEB_SEARCH_INTERCEPT"):
        monkeypatch.delenv(name, raising=False)
    yield


def _client(tmp_path: Path, **overrides: Any) -> TestClient:
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'tools-test.db'}",
        master_key="sk-test-master",
        **overrides,
    )
    return TestClient(create_app(config))


def _tools(client: TestClient) -> dict[str, Any]:
    body = client.get("/v1/tools", headers=AUTH).json()
    assert body["object"] == "list"
    return {tool["id"]: tool for tool in body["data"]}


def test_requires_auth(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        assert client.get("/v1/tools").status_code == 401


def test_lists_both_gateway_tools_with_schemas_and_examples(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        tools = _tools(client)

    assert set(tools) == {"otari_web_search", "otari_code_execution"}
    web_search = tools["otari_web_search"]
    assert web_search["object"] == "tool"
    assert web_search["example"] == {"type": "otari_web_search"}
    # The schema advertised is the one the model is actually given.
    assert web_search["input_schema"]["required"] == ["query"]
    assert "query" in web_search["input_schema"]["properties"]
    assert tools["otari_code_execution"]["input_schema"]["required"] == ["code"]


def test_unconfigured_tools_are_listed_as_unavailable(tmp_path: Path) -> None:
    """Listed but unavailable is the actionable answer: the tool exists, the
    operator has not wired up a backend."""
    with _client(tmp_path) as client:
        tools = _tools(client)

    assert tools["otari_web_search"]["available"] is False
    assert tools["otari_code_execution"]["available"] is False


def test_configured_backends_report_available(tmp_path: Path) -> None:
    with _client(tmp_path, web_search_url="http://searxng:8080", sandbox_url="http://sandbox:8000") as client:
        tools = _tools(client)

    assert tools["otari_web_search"]["available"] is True
    assert tools["otari_code_execution"]["available"] is True


def test_accepted_types_are_canonical_only_by_default(tmp_path: Path) -> None:
    with _client(tmp_path, web_search_url="http://searxng:8080") as client:
        tools = _tools(client)

    assert tools["otari_web_search"]["accepted_types"] == ["otari_web_search"]
    assert tools["otari_code_execution"]["accepted_types"] == ["otari_code_execution"]


def test_interception_advertises_the_provider_named_keywords(tmp_path: Path) -> None:
    with _client(tmp_path, web_search_url="http://searxng:8080", web_search_intercept=True) as client:
        tools = _tools(client)

    assert tools["otari_web_search"]["accepted_types"] == [
        "otari_web_search",
        "web_search",
        "web_search_<date>",
    ]
    # Code execution has no interception mode.
    assert tools["otari_code_execution"]["accepted_types"] == ["otari_code_execution"]


def test_interception_from_env_is_reflected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_WEB_SEARCH_INTERCEPT", "true")
    with _client(tmp_path, web_search_url="http://searxng:8080") as client:
        tools = _tools(client)

    assert "web_search" in tools["otari_web_search"]["accepted_types"]


def test_available_reflects_the_env_url_for_a_pure_env_deployment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", "http://searxng:8080")
    with _client(tmp_path) as client:
        tools = _tools(client)

    assert tools["otari_web_search"]["available"] is True
