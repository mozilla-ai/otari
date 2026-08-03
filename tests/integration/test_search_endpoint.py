"""Integration tests for the POST /v1/search endpoint.

The provider adapter is stubbed at ``run_search`` so these exercise the route's
own job: auth, tool selection, budget reservation and settlement, and the usage
row that makes a search visible in the Activity and Usage views.
"""

import logging
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from gateway.api.deps import reset_config
from gateway.core.config import API_KEY_HEADER, GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app
from gateway.services.search_backend import SearchHit, SearchOutcome, SearchProviderError

SEARCH_PAYLOAD: dict[str, Any] = {"query": "what is otari"}

_HITS = [
    SearchHit(
        url="https://example.com/otari",
        title="Otari",
        snippet="An OpenAI-compatible LLM gateway.",
        date="2026-01-02T00:00:00.000Z",
    )
]


@pytest.fixture
def test_config(postgres_url: str) -> GatewayConfig:
    """Override the shared config with two configured search tools."""
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        search_tools={
            "exa-search": {"provider": "exa", "api_key": "exa-secret"},
            "exa-fast": {"provider": "exa", "api_key": "exa-secret", "options": {"type": "fast"}},
        },
    )


def _mock_search(outcome: SearchOutcome | None = None, *, side_effect: Exception | None = None) -> Any:
    """Stub the provider adapter with a fixed outcome or a failure."""
    mock = AsyncMock(
        return_value=outcome if outcome is not None else SearchOutcome(results=_HITS, cost_usd=0.007),
        side_effect=side_effect,
    )
    return patch("gateway.api.routes.search.run_search", mock)


def test_search_requires_auth(client: TestClient) -> None:
    """POST /v1/search requires authentication."""
    resp = client.post("/v1/search", json={**SEARCH_PAYLOAD, "search_tool_name": "exa-search"})
    assert resp.status_code == 401


def test_search_with_api_key(client: TestClient, api_key_header: dict[str, str]) -> None:
    """POST /v1/search returns normalized results for an authenticated key."""
    with _mock_search():
        resp = client.post(
            "/v1/search",
            json={**SEARCH_PAYLOAD, "search_tool_name": "exa-search"},
            headers=api_key_header,
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "search"
    assert data["search_tool"] == "exa-search"
    assert data["results"] == [
        {
            "title": "Otari",
            "url": "https://example.com/otari",
            "snippet": "An OpenAI-compatible LLM gateway.",
            "date": "2026-01-02T00:00:00.000Z",
        }
    ]


def test_search_by_path_selects_the_tool(client: TestClient, api_key_header: dict[str, str]) -> None:
    """POST /v1/search/{tool} runs against the tool named in the path."""
    mock = AsyncMock(return_value=SearchOutcome(results=_HITS))
    with patch("gateway.api.routes.search.run_search", mock):
        resp = client.post("/v1/search/exa-fast", json=SEARCH_PAYLOAD, headers=api_key_header)
    assert resp.status_code == 200
    assert resp.json()["search_tool"] == "exa-fast"
    assert mock.call_args.args[0].name == "exa-fast"


def test_search_path_tool_wins_over_the_body(client: TestClient, api_key_header: dict[str, str]) -> None:
    """A search_tool_name in the body does not override the path segment."""
    mock = AsyncMock(return_value=SearchOutcome(results=_HITS))
    with patch("gateway.api.routes.search.run_search", mock):
        resp = client.post(
            "/v1/search/exa-fast",
            json={**SEARCH_PAYLOAD, "search_tool_name": "exa-search"},
            headers=api_key_header,
        )
    assert resp.status_code == 200
    assert resp.json()["search_tool"] == "exa-fast"


def test_search_forwards_request_fields(client: TestClient, api_key_header: dict[str, str]) -> None:
    """Request knobs reach the backend as a neutral SearchQuery."""
    mock = AsyncMock(return_value=SearchOutcome(results=_HITS))
    with patch("gateway.api.routes.search.run_search", mock):
        resp = client.post(
            "/v1/search/exa-search",
            json={
                **SEARCH_PAYLOAD,
                "max_results": 3,
                "search_domain_filter": ["arxiv.org"],
                "country": "US",
                "max_tokens_per_page": 256,
            },
            headers=api_key_header,
        )
    assert resp.status_code == 200
    query = mock.call_args.args[1]
    assert query.query == "what is otari"
    assert query.max_results == 3
    assert query.domain_filter == ("arxiv.org",)
    assert query.country == "US"
    assert query.max_tokens_per_page == 256


def test_search_ambiguous_tool_is_400(client: TestClient, api_key_header: dict[str, str]) -> None:
    """Omitting the tool name with several configured is a client error."""
    resp = client.post("/v1/search", json=SEARCH_PAYLOAD, headers=api_key_header)
    assert resp.status_code == 400
    assert "search tools are configured" in resp.json()["detail"]


def test_search_unknown_tool_is_400(client: TestClient, api_key_header: dict[str, str]) -> None:
    """Naming a tool that is not configured is a client error."""
    resp = client.post("/v1/search/nope", json=SEARCH_PAYLOAD, headers=api_key_header)
    assert resp.status_code == 400
    assert "Unknown search tool" in resp.json()["detail"]


def test_search_master_key_requires_user(client: TestClient, master_key_header: dict[str, str]) -> None:
    """POST /v1/search with the master key requires a 'user' field."""
    with _mock_search():
        resp = client.post("/v1/search/exa-search", json=SEARCH_PAYLOAD, headers=master_key_header)
    assert resp.status_code == 400
    assert "user" in resp.json()["detail"].lower()


def test_search_master_key_with_user(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """POST /v1/search with the master key plus a user field succeeds."""
    payload = {**SEARCH_PAYLOAD, "user": test_user["user_id"]}
    with _mock_search():
        resp = client.post("/v1/search/exa-search", json=payload, headers=master_key_header)
    assert resp.status_code == 200


def test_search_provider_error_is_502(client: TestClient, api_key_header: dict[str, str]) -> None:
    """An upstream failure surfaces as 502 without leaking the provider message."""
    with _mock_search(side_effect=SearchProviderError("exa search returned HTTP 401: bad key exa-secret")):
        resp = client.post("/v1/search/exa-search", json=SEARCH_PAYLOAD, headers=api_key_header)
    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert "exa-secret" not in detail
    assert "provider" in detail.lower()


def test_search_empty_query_is_422(client: TestClient, api_key_header: dict[str, str]) -> None:
    resp = client.post("/v1/search/exa-search", json={"query": ""}, headers=api_key_header)
    assert resp.status_code == 422


def test_search_max_results_above_cap_is_422(client: TestClient, api_key_header: dict[str, str]) -> None:
    payload = {**SEARCH_PAYLOAD, "max_results": 50}
    resp = client.post("/v1/search/exa-search", json=payload, headers=api_key_header)
    assert resp.status_code == 422


def test_search_rejects_a_country_that_is_not_a_two_letter_code(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    payload = {**SEARCH_PAYLOAD, "country": "United States"}
    resp = client.post("/v1/search/exa-search", json=payload, headers=api_key_header)
    assert resp.status_code == 422


def test_search_honors_the_keys_model_allowlist(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """A key restricted to models it may call cannot spend on an unlisted search tool."""
    client.post("/v1/users", json={"user_id": "narrow-user"}, headers=master_key_header)
    key = client.post(
        "/v1/keys",
        json={"key_name": "narrow-key", "user_id": "narrow-user", "allowed_models": ["openai:gpt-4o"]},
        headers=master_key_header,
    ).json()
    headers = {API_KEY_HEADER: f"Bearer {key['key']}"}

    with _mock_search():
        denied = client.post("/v1/search/exa-search", json=SEARCH_PAYLOAD, headers=headers)
    assert denied.status_code == 403

    # Naming the tool as <provider>:<tool> is what grants it.
    client.patch(
        f"/v1/keys/{key['id']}",
        json={"allowed_models": ["openai:gpt-4o", "exa:exa-search"]},
        headers=master_key_header,
    )
    with _mock_search():
        allowed = client.post("/v1/search/exa-search", json=SEARCH_PAYLOAD, headers=headers)
    assert allowed.status_code == 200


def test_search_logs_an_unknown_tool_refusal(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """A 400 for a tool that is not configured is still visible as dropped traffic."""
    user_id = api_key_obj["user_id"]

    resp = client.post("/v1/search/nope", json=SEARCH_PAYLOAD, headers=api_key_header)
    assert resp.status_code == 400

    logs = client.get(f"/v1/users/{user_id}/usage", headers=master_key_header).json()
    search_logs = [log for log in logs if log["endpoint"] == "/v1/search"]
    assert len(search_logs) == 1
    entry = search_logs[0]
    assert entry["status"] == "error"
    assert entry["model"] == "nope"
    assert entry["cost"] is None
    assert "Unknown search tool" in entry["error_message"]


def test_search_logs_an_allowlist_refusal(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """A 403 from the key's allowed-models list writes a usage row, with no cost."""
    client.post("/v1/users", json={"user_id": "denied-user"}, headers=master_key_header)
    key = client.post(
        "/v1/keys",
        json={"key_name": "denied-key", "user_id": "denied-user", "allowed_models": ["openai:gpt-4o"]},
        headers=master_key_header,
    ).json()

    with _mock_search():
        resp = client.post(
            "/v1/search/exa-search",
            json=SEARCH_PAYLOAD,
            headers={API_KEY_HEADER: f"Bearer {key['key']}"},
        )
    assert resp.status_code == 403

    logs = client.get("/v1/users/denied-user/usage", headers=master_key_header).json()
    search_logs = [log for log in logs if log["endpoint"] == "/v1/search"]
    assert len(search_logs) == 1
    entry = search_logs[0]
    assert entry["status"] == "error"
    assert entry["model"] == "exa-search"
    assert entry["provider"] == "exa"
    assert entry["cost"] is None


def test_search_does_not_warn_about_provider_pricing(
    client: TestClient,
    api_key_header: dict[str, str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The tool key must not reach reserve_budget's any-llm model split.

    It is not an any-llm selector, so the split fails and is swallowed into a
    warning on every request. Worse, the shortcut behind it runs a
    default-pricing lookup that the route's own use_defaults=False guard is
    there to prevent, and can skip the reservation outright.
    """
    with caplog.at_level(logging.WARNING), _mock_search():
        resp = client.post("/v1/search/exa-search", json=SEARCH_PAYLOAD, headers=api_key_header)
    assert resp.status_code == 200
    assert "Failed to determine provider pricing" not in caplog.text


def test_search_logs_usage_with_provider_reported_cost(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """A search writes a zero-token usage row billed at the provider's own charge."""
    user_id = api_key_obj["user_id"]

    with _mock_search():
        resp = client.post("/v1/search/exa-search", json=SEARCH_PAYLOAD, headers=api_key_header)
    assert resp.status_code == 200

    logs = client.get(f"/v1/users/{user_id}/usage", headers=master_key_header).json()
    search_logs = [log for log in logs if log["endpoint"] == "/v1/search"]
    assert len(search_logs) == 1
    entry = search_logs[0]
    assert entry["status"] == "success"
    assert entry["model"] == "exa-search"
    assert entry["provider"] == "exa"
    assert entry["prompt_tokens"] == 0
    assert entry["completion_tokens"] == 0
    assert entry["total_tokens"] == 0
    assert entry["cost"] == pytest.approx(0.007)


def test_search_falls_back_to_configured_flat_pricing(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """With no provider-reported cost, the configured per-request rate is billed."""
    # Flat per-request convention: the stored rate is USD per million requests.
    client.post(
        "/v1/pricing",
        json={
            "model_key": "exa:exa-search",
            "input_price_per_million": 5000.0,
            "output_price_per_million": 0.0,
        },
        headers=master_key_header,
    )
    user_id = api_key_obj["user_id"]

    with _mock_search(SearchOutcome(results=_HITS)):
        resp = client.post("/v1/search/exa-search", json=SEARCH_PAYLOAD, headers=api_key_header)
    assert resp.status_code == 200

    logs = client.get(f"/v1/users/{user_id}/usage", headers=master_key_header).json()
    search_logs = [log for log in logs if log["endpoint"] == "/v1/search"]
    assert search_logs[0]["cost"] == pytest.approx(0.005)


def test_search_logs_error_and_refunds_on_failure(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """A failed search is logged as an error and leaves no reservation behind."""
    user_id = api_key_obj["user_id"]

    with _mock_search(side_effect=SearchProviderError("exa is down")):
        resp = client.post("/v1/search/exa-search", json=SEARCH_PAYLOAD, headers=api_key_header)
    assert resp.status_code == 502

    logs = client.get(f"/v1/users/{user_id}/usage", headers=master_key_header).json()
    error_logs = [log for log in logs if log["endpoint"] == "/v1/search" and log["status"] == "error"]
    assert len(error_logs) == 1
    assert "exa is down" in error_logs[0]["error_message"]

    user = client.get(f"/v1/users/{user_id}", headers=master_key_header).json()
    assert user["reserved"] == pytest.approx(0.0)


def test_search_is_budget_enforced(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """A user already at their budget cap cannot run a search."""
    client.post(
        "/v1/pricing",
        json={
            "model_key": "exa:exa-search",
            "input_price_per_million": 5000.0,
            "output_price_per_million": 0.0,
        },
        headers=master_key_header,
    )
    budget = client.post("/v1/budgets", json={"max_budget": 0.001}, headers=master_key_header).json()
    client.post(
        "/v1/users",
        json={"user_id": "broke-user", "budget_id": budget["budget_id"]},
        headers=master_key_header,
    )
    key = client.post(
        "/v1/keys",
        json={"key_name": "broke-key", "user_id": "broke-user"},
        headers=master_key_header,
    ).json()

    with _mock_search():
        resp = client.post(
            "/v1/search/exa-search",
            json=SEARCH_PAYLOAD,
            headers={API_KEY_HEADER: f"Bearer {key['key']}"},
        )
    assert resp.status_code == 403


def test_search_is_not_registered_in_hybrid_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Search is a standalone-mode surface: hybrid has no local search config."""
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
    app = create_app(GatewayConfig(mode="hybrid", platform={"base_url": "http://localhost:8100/api/v1"}))
    try:
        with TestClient(app) as hybrid_client:
            resp = hybrid_client.post("/v1/search/exa-search", json=SEARCH_PAYLOAD)
        assert resp.status_code == 404
    finally:
        reset_config()
        reset_db()
