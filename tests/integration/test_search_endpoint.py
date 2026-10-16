"""Integration tests for the POST /api/v1/search endpoint.

The search tools are any-search's ``fake`` provider, which answers from its
options and echoes the request it got in ``raw``, so these exercise the route
and its rules without a network: auth, which tool and which key, the
workspace's policy, budget reservation and settlement, the LiteLLM mapping on a
provider without native equivalents, and the usage row that makes a search
visible in the Activity and Usage views. A Tavily organization key, which runs
on the old client until any-search has its adapter, is stubbed at
``run_keyed_search``.
"""

import logging
from collections.abc import Generator, Sequence
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from gateway.api.deps import get_web_search_policy_port, reset_config
from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig, RateLimitRule
from gateway.core.database import reset_db
from gateway.core.settings.pricing import PricingConfig
from gateway.exceptions.tools_exceptions import SearchToolNotFoundError
from gateway.inflight import InFlightRegistry
from gateway.main import create_app
from gateway.models.tools import ResolvedWebSearchConfig, WebSearchCredential, WebTool, WebToolCredentialSource
from gateway.ports.web_search_policy_port import WebSearchPolicyScope
from gateway.schemas.tools import SearchResponse
from gateway.services.search_backend import SearchHit, SearchOutcome, SearchProviderError
from gateway.services.secret_box import generate_secret_key
from gateway.services.tools import DirectSearch, DirectSearchResult

from .conftest import build_test_client

SEARCH_USAGE_LABEL = "/v1/search"

SEARCH_PAYLOAD: dict[str, Any] = {"query": "what is otari"}

# The fake provider's hits: one with a snippet and no page text, and one with both.
_HIT = {
    "url": "https://example.com/otari",
    "title": "Otari",
    "snippet": "An OpenAI-compatible LLM gateway.",
    "published": "2026-01-02T00:00:00Z",
}
_PAGE_HIT = {"url": "https://example.org/page", "title": "Page", "snippet": "short", "text": "The page text."}

_OLD_CLIENT_HITS = [SearchHit(url="https://example.com/otari", title="Otari", snippet="From the old client.")]


def _search_tools() -> dict[str, dict[str, Any]]:
    return {
        "fake-search": {"provider": "fake", "timeout": 7, "options": {"hits": [_HIT], "cost": 0.007}},
        "fake-fast": {"provider": "fake", "options": {"hits": [_HIT, _PAGE_HIT]}},
    }


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
        search_tools=_search_tools(),
    )


def _client_on(test_config: GatewayConfig, **updates: Any) -> Generator[TestClient]:
    yield from build_test_client(test_config.model_copy(update=updates))


def _keyed(client: TestClient, master_key_header: dict[str, str], user_id: str) -> dict[str, str]:
    assert client.post(f"{API_ROOT}/users", json={"user_id": user_id}, headers=master_key_header).status_code == 200
    key = client.post(f"{API_ROOT}/keys", json={"key_name": user_id, "user_id": user_id}, headers=master_key_header)
    assert key.status_code == 200, key.text
    return {API_KEY_HEADER: f"Bearer {key.json()['key']}"}


def _search_rows(client: TestClient, headers: dict[str, str], user_id: str) -> list[dict[str, Any]]:
    """The user's search usage rows, from the admin view that exposes every column.

    ``/api/v1/users/{id}/usage`` omits ``counts_toward_budget``, which the refusal
    rows below have to assert on.
    """
    resp = client.get(
        f"{API_ROOT}/usage",
        params={"user_id": user_id, "endpoint": SEARCH_USAGE_LABEL},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    return [dict(row) for row in resp.json()]


def _reached_the_provider(resp: Any) -> dict[str, Any]:
    """The request the fake provider got, which it echoes in ``raw``."""
    assert resp.status_code == 200, resp.text
    request: dict[str, Any] = resp.json()["raw"]["request"]
    return request


def test_search_requires_auth(client: TestClient) -> None:
    """POST /api/v1/search requires authentication."""
    resp = client.post(f"{API_ROOT}/search", json={**SEARCH_PAYLOAD, "search_tool_name": "fake-search"})
    assert resp.status_code == 401


def test_search_with_api_key(client: TestClient, api_key_header: dict[str, str]) -> None:
    """POST /api/v1/search returns LiteLLM's results for an authenticated key, without ``raw`` unless asked."""
    resp = client.post(
        f"{API_ROOT}/search",
        json={**SEARCH_PAYLOAD, "search_tool_name": "fake-search"},
        headers=api_key_header,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "search"
    assert data["search_tool"] == "fake-search"
    assert data["results"] == [
        {
            "title": "Otari",
            "url": "https://example.com/otari",
            # A hit with no page text comes back with its snippet.
            "snippet": "An OpenAI-compatible LLM gateway.",
            # The fake gives no date string of its own, so the parsed date in ISO form.
            "date": "2026-01-02T00:00:00+00:00",
            "last_updated": None,
        }
    ]
    assert "raw" not in data


def test_include_raw_returns_the_providers_whole_response(client: TestClient, api_key_header: dict[str, str]) -> None:
    resp = client.post(
        f"{API_ROOT}/search/fake-search", json={**SEARCH_PAYLOAD, "include_raw": True}, headers=api_key_header
    )
    assert resp.status_code == 200, resp.text
    raw = resp.json()["raw"]
    assert raw["results"] == [_HIT]
    assert raw["request"]["max_results"] == 10


def test_search_by_path_selects_the_tool(client: TestClient, api_key_header: dict[str, str]) -> None:
    """POST /api/v1/search/{tool} runs against the tool named in the path."""
    resp = client.post(f"{API_ROOT}/search/fake-fast", json=SEARCH_PAYLOAD, headers=api_key_header)
    assert resp.status_code == 200
    assert resp.json()["search_tool"] == "fake-fast"
    assert [result["url"] for result in resp.json()["results"]] == [_HIT["url"], _PAGE_HIT["url"]]


def test_search_path_tool_wins_over_the_body(client: TestClient, api_key_header: dict[str, str]) -> None:
    """A search_tool_name in the body does not override the path segment."""
    resp = client.post(
        f"{API_ROOT}/search/fake-fast",
        json={**SEARCH_PAYLOAD, "search_tool_name": "fake-search"},
        headers=api_key_header,
    )
    assert resp.status_code == 200
    assert resp.json()["search_tool"] == "fake-fast"


def test_the_litellm_fields_on_a_provider_without_native_equivalents(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    """The count reaches the provider; the domain filter and the page budget are applied to the hits."""
    resp = client.post(
        f"{API_ROOT}/search/fake-fast",
        json={
            **SEARCH_PAYLOAD,
            "max_results": 3,
            "search_domain_filter": ["example.org"],
            "country": "US",
            "max_tokens_per_page": 1,
            "include_raw": True,
        },
        headers=api_key_header,
    )
    request = _reached_the_provider(resp)
    assert request["max_results"] == 3
    # No native equivalents, so nothing of the domain filter or the country is sent.
    assert not {"includeDomains", "excludeDomains", "userLocation"} & set(request["options"])
    (result,) = resp.json()["results"]
    assert result["url"] == _PAGE_HIT["url"]
    # The page text wins over the snippet, cut to one token's four characters.
    assert result["snippet"] == "The "


def test_a_count_above_20_reaches_the_provider(client: TestClient, api_key_header: dict[str, str]) -> None:
    """The endpoint sets no ceiling of its own (decision 65): this was a 422."""
    payload = {**SEARCH_PAYLOAD, "max_results": 50, "include_raw": True}
    resp = client.post(f"{API_ROOT}/search/fake-search", json=payload, headers=api_key_header)
    assert _reached_the_provider(resp)["max_results"] == 50


def test_search_ambiguous_tool_is_400(client: TestClient, api_key_header: dict[str, str]) -> None:
    """Omitting the tool name with several configured and no default is a client error."""
    resp = client.post(f"{API_ROOT}/search", json=SEARCH_PAYLOAD, headers=api_key_header)
    assert resp.status_code == 400
    assert resp.json()["detail"] == (
        "Several search tools are configured; name one in 'search_tool_name' or use "
        f"POST {API_ROOT}/search/{{tool}}. Available: fake-fast, fake-search."
    )


def test_search_unknown_tool_is_400(client: TestClient, api_key_header: dict[str, str]) -> None:
    """Naming a tool that is not configured is a client error."""
    resp = client.post(f"{API_ROOT}/search/nope", json=SEARCH_PAYLOAD, headers=api_key_header)
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Unknown search tool 'nope'. Available: fake-fast, fake-search."


@pytest.fixture
def no_tools_client(test_config: GatewayConfig) -> Generator[TestClient]:
    yield from _client_on(test_config, search_tools={})


def test_no_configured_tools_names_every_route_in(
    no_tools_client: TestClient, master_key_header: dict[str, str]
) -> None:
    """The message named only config.yml, which a file-less deployment cannot follow (issue #601)."""
    headers = _keyed(no_tools_client, master_key_header, "no-tools")
    resp = no_tools_client.post(f"{API_ROOT}/search", json=SEARCH_PAYLOAD, headers=headers)
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail.startswith("No search tools are configured.")
    assert "dashboard" in detail
    assert f"{API_ROOT}/search-tools" in detail
    assert "OTARI_CONFIG_YAML" in detail


@pytest.fixture(params=[None, "none", "gone"])
def one_tool_client(request: pytest.FixtureRequest, test_config: GatewayConfig) -> Generator[TestClient]:
    """One tool, with no default, ``none``, or a default that names no instance."""
    tools = {"fake-fast": _search_tools()["fake-fast"]}
    yield from _client_on(test_config, search_tools=tools, web_search_default_tool=request.param)


def test_an_unnamed_call_runs_on_the_only_tool(one_tool_client: TestClient, master_key_header: dict[str, str]) -> None:
    """``none`` concerns the in-loop tool only, and a default naming no instance counts as unset."""
    headers = _keyed(one_tool_client, master_key_header, "one-tool")
    resp = one_tool_client.post(f"{API_ROOT}/search", json=SEARCH_PAYLOAD, headers=headers)
    assert resp.status_code == 200, resp.text
    assert resp.json()["search_tool"] == "fake-fast"


@pytest.fixture
def default_tool_client(test_config: GatewayConfig, _secret_key: None) -> Generator[TestClient]:
    yield from _client_on(test_config, web_search_default_tool="fake-search")


def test_an_unnamed_call_runs_on_the_deployments_default(
    default_tool_client: TestClient, master_key_header: dict[str, str]
) -> None:
    headers = _keyed(default_tool_client, master_key_header, "default-tool")
    resp = default_tool_client.post(f"{API_ROOT}/search", json=SEARCH_PAYLOAD, headers=headers)
    assert resp.status_code == 200, resp.text
    assert resp.json()["search_tool"] == "fake-search"


def test_search_master_key_requires_user(client: TestClient, master_key_header: dict[str, str]) -> None:
    """POST /api/v1/search with the master key requires a 'user' field."""
    resp = client.post(f"{API_ROOT}/search/fake-search", json=SEARCH_PAYLOAD, headers=master_key_header)
    assert resp.status_code == 400
    assert "user" in resp.json()["detail"].lower()


def test_search_master_key_with_user(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """POST /api/v1/search with the master key plus a user field succeeds."""
    payload = {**SEARCH_PAYLOAD, "user": test_user["user_id"]}
    resp = client.post(f"{API_ROOT}/search/fake-search", json=payload, headers=master_key_header)
    assert resp.status_code == 200


def test_search_provider_error_is_502(client: TestClient, api_key_header: dict[str, str]) -> None:
    """An upstream failure surfaces as 502 without the provider's own words."""
    payload = {**SEARCH_PAYLOAD, "provider_options": {"error": "INVALID_API_KEY", "error_status": 401}}
    resp = client.post(f"{API_ROOT}/search/fake-search", json=payload, headers=api_key_header)
    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert "INVALID_API_KEY" not in detail
    assert "provider" in detail.lower()


_FAILURE_HEADERS = ("Otari-Error-Provider", "Otari-Error-Status", "Otari-Error-Tag", "Otari-Error-Retryable")


@pytest.mark.parametrize(
    ("options", "fields"),
    [
        pytest.param({"error": "http_error", "error_status": 503}, ("fake", "503", "http_error", "true"), id="5xx"),
        pytest.param({"error": "rate_limited", "error_status": 429}, ("fake", "429", "rate_limited", "true"), id="429"),
        pytest.param(
            {"error": "INVALID_API_KEY", "error_status": 401}, ("fake", "401", "INVALID_API_KEY", "false"), id="401"
        ),
        pytest.param({"error": "timeout"}, ("fake", None, "timeout", "true"), id="timeout"),
        pytest.param({"in_body_error": "quota"}, ("fake", None, "quota", "false"), id="inside-a-200"),
        pytest.param(
            {"error": "unexpected_response", "error_status": 200},
            ("fake", None, "unexpected_response", "false"),
            id="an-unreadable-200",
        ),
    ],
)
def test_a_provider_failure_carries_its_safe_fields_as_headers(
    client: TestClient, api_key_header: dict[str, str], options: dict[str, Any], fields: tuple[str | None, ...]
) -> None:
    """Decision 67: the provider, its HTTP status, the tag and whether a retry may help; the detail stays generic."""
    payload = {**SEARCH_PAYLOAD, "provider_options": options}
    resp = client.post(f"{API_ROOT}/search/fake-search", json=payload, headers=api_key_header)
    assert resp.status_code == 502
    assert tuple(resp.headers.get(name) for name in _FAILURE_HEADERS) == fields
    assert set(resp.json()) == {"detail"}


def test_a_refusal_before_the_provider_carries_no_failure_headers(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    resp = client.post(f"{API_ROOT}/search/nope", json=SEARCH_PAYLOAD, headers=api_key_header)
    assert resp.status_code == 400
    assert not {name.lower() for name in _FAILURE_HEADERS} & set(resp.headers)


def test_an_error_inside_a_successful_answer_is_a_502(client: TestClient, api_key_header: dict[str, str]) -> None:
    payload = {**SEARCH_PAYLOAD, "provider_options": {"in_body_error": "quota"}}
    resp = client.post(f"{API_ROOT}/search/fake-search", json=payload, headers=api_key_header)
    assert resp.status_code == 502


def test_a_provider_exceptions_text_never_reaches_the_caller_or_the_usage_row(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A provider that raises with the query in its message, as a careless adapter might (design section 7)."""
    sentinel = "sentinel-query-7f3a"
    payload = {"query": sentinel, "provider_options": {"leak_query": True}}
    with caplog.at_level(logging.ERROR):
        resp = client.post(f"{API_ROOT}/search/fake-search", json=payload, headers=api_key_header)
    assert resp.status_code == 502
    assert sentinel not in resp.text
    (row,) = _search_rows(client, master_key_header, api_key_obj["user_id"])
    assert row["error_message"] == "RuntimeError"
    assert sentinel not in caplog.text


def test_search_empty_query_is_422(client: TestClient, api_key_header: dict[str, str]) -> None:
    resp = client.post(f"{API_ROOT}/search/fake-search", json={"query": ""}, headers=api_key_header)
    assert resp.status_code == 422


def test_search_rejects_a_country_that_is_not_a_two_letter_code(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    payload = {**SEARCH_PAYLOAD, "country": "United States"}
    resp = client.post(f"{API_ROOT}/search/fake-search", json=payload, headers=api_key_header)
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "field", [{"include_raw": ["not", "a", "boolean"]}, {"provider_options": ["not", "an", "object"]}]
)
def test_a_malformed_new_field_is_a_request_validation_422(
    client: TestClient, api_key_header: dict[str, str], field: dict[str, Any]
) -> None:
    resp = client.post(f"{API_ROOT}/search/fake-search", json={**SEARCH_PAYLOAD, **field}, headers=api_key_header)
    assert resp.status_code == 422
    assert resp.headers["Otari-Error-Code"] == "invalid_request"


def test_an_option_the_provider_does_not_take_is_a_400_naming_it(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    payload = {**SEARCH_PAYLOAD, "provider_options": {"numResultz": 3}}
    resp = client.post(f"{API_ROOT}/search/fake-search", json=payload, headers=api_key_header)
    assert resp.status_code == 400
    assert "numResultz" in resp.json()["detail"]
    (row,) = _search_rows(client, master_key_header, api_key_obj["user_id"])
    assert (row["model"], row["status_code"]) == ("fake-search", 400)


def test_an_operator_only_option_is_refused_from_a_request(client: TestClient, api_key_header: dict[str, str]) -> None:
    payload = {**SEARCH_PAYLOAD, "provider_options": {"account": "someone-else"}}
    resp = client.post(f"{API_ROOT}/search/fake-search", json=payload, headers=api_key_header)
    assert resp.status_code == 400
    assert "account" in resp.json()["detail"]


def test_provider_options_reach_the_provider_over_the_tools_own(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    payload = {**SEARCH_PAYLOAD, "provider_options": {"cost": 0.5}, "include_raw": True}
    resp = client.post(f"{API_ROOT}/search/fake-search", json=payload, headers=api_key_header)
    assert _reached_the_provider(resp)["options"]["cost"] == 0.5


def _default_workspace_id(client: TestClient, master_key_header: dict[str, str]) -> str:
    listed = client.get(f"{API_ROOT}/workspaces", headers=master_key_header)
    assert listed.status_code == 200
    workspace_id: str = listed.json()["data"][0]["id"]
    return workspace_id


def test_the_workspaces_provider_options_do_not_apply(
    client: TestClient, master_key_header: dict[str, str], api_key_header: dict[str, str]
) -> None:
    """Of the workspace's policy, only the switch and the result ceiling apply here (design G)."""
    workspace_id = _default_workspace_id(client, master_key_header)
    stored = client.put(
        f"{API_ROOT}/workspaces/{workspace_id}/web-search",
        json={"enabled": True, "provider_options": {"delay": 0.01, "cost": 9}},
        headers=master_key_header,
    )
    assert stored.status_code == 200, stored.text
    resp = client.post(
        f"{API_ROOT}/search/fake-search", json={**SEARCH_PAYLOAD, "include_raw": True}, headers=api_key_header
    )
    options = _reached_the_provider(resp)["options"]
    assert "delay" not in options
    assert options["cost"] == 0.007


def test_the_workspaces_result_ceiling_lowers_the_count(
    client: TestClient, master_key_header: dict[str, str], api_key_header: dict[str, str]
) -> None:
    workspace_id = _default_workspace_id(client, master_key_header)
    stored = client.put(
        f"{API_ROOT}/workspaces/{workspace_id}/web-search",
        json={"enabled": True, "max_results": 30},
        headers=master_key_header,
    )
    assert stored.status_code == 200, stored.text
    payload = {**SEARCH_PAYLOAD, "max_results": 50, "include_raw": True}
    resp = client.post(f"{API_ROOT}/search/fake-search", json=payload, headers=api_key_header)
    assert _reached_the_provider(resp)["max_results"] == 30


def test_search_honors_the_keys_model_allowlist(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """A key restricted to models it may call cannot spend on an unlisted search tool."""
    client.post(f"{API_ROOT}/users", json={"user_id": "narrow-user"}, headers=master_key_header)
    key = client.post(
        f"{API_ROOT}/keys",
        json={"key_name": "narrow-key", "user_id": "narrow-user", "allowed_models": ["openai:gpt-4o"]},
        headers=master_key_header,
    ).json()
    headers = {API_KEY_HEADER: f"Bearer {key['key']}"}

    denied = client.post(f"{API_ROOT}/search/fake-search", json=SEARCH_PAYLOAD, headers=headers)
    assert denied.status_code == 403

    # Naming the tool as <provider>:<tool> is what grants it.
    client.patch(
        f"{API_ROOT}/keys/{key['id']}",
        json={"allowed_models": ["openai:gpt-4o", "fake:fake-search"]},
        headers=master_key_header,
    )
    allowed = client.post(f"{API_ROOT}/search/fake-search", json=SEARCH_PAYLOAD, headers=headers)
    assert allowed.status_code == 200


def test_search_logs_an_unknown_tool_refusal(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """A 400 for a tool that is not configured is still visible as dropped traffic."""
    user_id = api_key_obj["user_id"]

    resp = client.post(f"{API_ROOT}/search/nope", json=SEARCH_PAYLOAD, headers=api_key_header)
    assert resp.status_code == 400

    search_logs = _search_rows(client, master_key_header, user_id)
    assert len(search_logs) == 1
    entry = search_logs[0]
    assert entry["status"] == "error"
    assert entry["model"] == "nope"
    assert entry["cost"] is None
    # Never an imported-looking row: the dashboard offers those for bulk delete
    # and set-price, which must not reach a row the gateway wrote itself.
    assert entry["counts_toward_budget"] is True
    assert "Unknown search tool" in entry["error_message"]
    # The status this refusal returned, so it groups as a client error in the
    # failure taxonomy instead of as an unclassified row (#433).
    assert entry["status_code"] == 400


def test_search_logs_an_allowlist_refusal(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """A 403 from the key's allowed-models list writes a usage row, with no cost."""
    client.post(f"{API_ROOT}/users", json={"user_id": "denied-user"}, headers=master_key_header)
    key = client.post(
        f"{API_ROOT}/keys",
        json={"key_name": "denied-key", "user_id": "denied-user", "allowed_models": ["openai:gpt-4o"]},
        headers=master_key_header,
    ).json()

    resp = client.post(
        f"{API_ROOT}/search/fake-search",
        json=SEARCH_PAYLOAD,
        headers={API_KEY_HEADER: f"Bearer {key['key']}"},
    )
    assert resp.status_code == 403

    search_logs = _search_rows(client, master_key_header, "denied-user")
    assert len(search_logs) == 1
    entry = search_logs[0]
    assert entry["status"] == "error"
    assert entry["model"] == "fake-search"
    assert entry["provider"] == "fake"
    assert entry["cost"] is None
    assert entry["counts_toward_budget"] is True
    assert entry["status_code"] == 403


def test_a_budget_exempt_keys_search_refusal_still_counts_toward_budget(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """An exempt key's refusal row must not look like imported usage.

    A key flagged ``exclude_from_budget`` writes ``counts_toward_budget=False``
    on its served rows, and it can still be refused by the allow-list gate. The
    flag is pinned True for every gateway-written refusal (see
    ``log_gateway_rejection``), so the dashboard never offers one for bulk delete.
    """
    client.post(f"{API_ROOT}/users", json={"user_id": "exempt-search-user"}, headers=master_key_header)
    key = client.post(
        f"{API_ROOT}/keys",
        json={
            "key_name": "exempt-search-key",
            "user_id": "exempt-search-user",
            "allowed_models": ["openai:gpt-4o"],
            "exclude_from_budget": True,
        },
        headers=master_key_header,
    ).json()

    resp = client.post(
        f"{API_ROOT}/search/fake-search",
        json=SEARCH_PAYLOAD,
        headers={API_KEY_HEADER: f"Bearer {key['key']}"},
    )
    assert resp.status_code == 403

    search_logs = _search_rows(client, master_key_header, "exempt-search-user")
    assert len(search_logs) == 1
    assert search_logs[0]["counts_toward_budget"] is True


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
    with caplog.at_level(logging.WARNING):
        resp = client.post(f"{API_ROOT}/search/fake-search", json=SEARCH_PAYLOAD, headers=api_key_header)
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

    resp = client.post(f"{API_ROOT}/search/fake-search", json=SEARCH_PAYLOAD, headers=api_key_header)
    assert resp.status_code == 200

    logs = client.get(f"{API_ROOT}/users/{user_id}/usage", headers=master_key_header).json()
    search_logs = [log for log in logs if log["endpoint"] == "/v1/search"]
    assert len(search_logs) == 1
    entry = search_logs[0]
    assert entry["status"] == "success"
    assert entry["model"] == "fake-search"
    assert entry["provider"] == "fake"
    assert entry["prompt_tokens"] == 0
    assert entry["completion_tokens"] == 0
    assert entry["total_tokens"] == 0
    assert entry["cost"] == pytest.approx(0.007)


def _price(client: TestClient, master_key_header: dict[str, str], model_key: str, per_million: float) -> None:
    """A flat per-request rate: the stored rate is USD per million requests."""
    resp = client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": model_key, "input_price_per_million": per_million, "output_price_per_million": 0.0},
        headers=master_key_header,
    )
    assert resp.status_code in (200, 201), resp.text


def test_search_falls_back_to_configured_flat_pricing(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """With no provider-reported cost, the configured per-request rate is billed."""
    _price(client, master_key_header, "fake:fake-fast", 5000.0)
    user_id = api_key_obj["user_id"]

    resp = client.post(f"{API_ROOT}/search/fake-fast", json=SEARCH_PAYLOAD, headers=api_key_header)
    assert resp.status_code == 200

    logs = client.get(f"{API_ROOT}/users/{user_id}/usage", headers=master_key_header).json()
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

    payload = {**SEARCH_PAYLOAD, "provider_options": {"error": "http_error", "error_status": 503}}
    resp = client.post(f"{API_ROOT}/search/fake-search", json=payload, headers=api_key_header)
    assert resp.status_code == 502

    logs = client.get(f"{API_ROOT}/users/{user_id}/usage", headers=master_key_header).json()
    error_logs = [log for log in logs if log["endpoint"] == "/v1/search" and log["status"] == "error"]
    assert len(error_logs) == 1
    assert error_logs[0]["error_message"] == "fake search failed (http_error, HTTP 503)"
    # The provider's own status, as the other routes' rows keep it.
    assert _search_rows(client, master_key_header, user_id)[0]["status_code"] == 503

    user = client.get(f"{API_ROOT}/users/{user_id}", headers=master_key_header).json()
    assert user["reserved"] == pytest.approx(0.0)


def test_search_is_budget_enforced(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """A user already at their budget cap cannot run a search."""
    _price(client, master_key_header, "fake:fake-search", 5000.0)
    budget = client.post(f"{API_ROOT}/budgets", json={"max_budget": 0.001}, headers=master_key_header).json()
    client.post(
        f"{API_ROOT}/users",
        json={"user_id": "broke-user", "budget_id": budget["budget_id"]},
        headers=master_key_header,
    )
    key = client.post(
        f"{API_ROOT}/keys",
        json={"key_name": "broke-key", "user_id": "broke-user"},
        headers=master_key_header,
    ).json()

    resp = client.post(
        f"{API_ROOT}/search/fake-search",
        json=SEARCH_PAYLOAD,
        headers={API_KEY_HEADER: f"Bearer {key['key']}"},
    )
    assert resp.status_code == 403


@pytest.fixture
def config_priced_client(test_config: GatewayConfig) -> Generator[TestClient]:
    """A client whose search rate comes from the config file's ``pricing`` section."""
    pricing = {"fake:fake-search": PricingConfig(input_price_per_million=5000.0, output_price_per_million=0.0)}
    yield from _client_on(test_config, pricing=pricing)


def test_a_config_priced_search_is_reserved_before_it_runs(
    config_priced_client: TestClient, master_key_header: dict[str, str]
) -> None:
    """The config rate is held up front, so a search that would overshoot the cap never reaches the provider."""
    client = config_priced_client
    budget = client.post(f"{API_ROOT}/budgets", json={"max_budget": 0.001}, headers=master_key_header).json()
    client.post(
        f"{API_ROOT}/users",
        json={"user_id": "near-cap-user", "budget_id": budget["budget_id"]},
        headers=master_key_header,
    )
    key = client.post(
        f"{API_ROOT}/keys",
        json={"key_name": "near-cap-key", "user_id": "near-cap-user"},
        headers=master_key_header,
    ).json()

    with patch("gateway.api.routes.search.run_direct_search", AsyncMock()) as searched:
        resp = client.post(
            f"{API_ROOT}/search/fake-search",
            json=SEARCH_PAYLOAD,
            headers={API_KEY_HEADER: f"Bearer {key['key']}"},
        )
    assert resp.status_code == 403, resp.text
    searched.assert_not_awaited()


def test_search_is_not_registered_in_hybrid_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Search is a standalone-mode surface: hybrid has no local search config."""
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
    app = create_app(GatewayConfig(mode="hybrid", platform={"base_url": "http://localhost:8100/api/v1"}))
    try:
        with TestClient(app) as hybrid_client:
            resp = hybrid_client.post(f"{API_ROOT}/search/fake-search", json=SEARCH_PAYLOAD)
        assert resp.status_code == 404
    finally:
        reset_config()
        reset_db()


def test_search_is_tracked_while_its_provider_call_runs(client: TestClient, api_key_header: dict[str, str]) -> None:
    """A search is registered in flight during its provider call and gone after.

    A search provider can take as long as a local model, so the live view has to
    cover it or the Activity panel silently misses one whole surface. Mirrors the
    pass-through in-flight test: the registry is read from inside the stubbed
    provider call, which is the window an operator would be watching.
    """
    registry: InFlightRegistry = client.app.state.inflight  # type: ignore[attr-defined]
    seen: list[Any] = []

    async def slow_search(search: DirectSearch, *_args: Any) -> DirectSearchResult:
        seen.append(registry.snapshot())
        return DirectSearchResult(response=SearchResponse(search_tool=search.instance.name, results=[]), cost=None)

    with patch("gateway.api.routes.search.run_direct_search", slow_search):
        resp = client.post(
            f"{API_ROOT}/search",
            json={**SEARCH_PAYLOAD, "search_tool_name": "fake-search"},
            headers=api_key_header,
        )

    assert resp.status_code == 200
    assert len(seen) == 1
    (entry,) = seen[0]
    assert entry.endpoint == SEARCH_USAGE_LABEL
    assert entry.model == "fake-search"
    assert entry.provider == "fake"
    assert entry.user_id == "default"
    assert entry.api_key_id
    assert entry.policy_name is None
    # And it is gone afterwards, so the panel cannot accumulate a ghost.
    assert len(registry) == 0


def test_a_service_key_bills_search_to_the_end_user_it_names(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """``user`` on a service key's search names an end user, created with the key's end-user budget."""
    budget = client.post(
        f"{API_ROOT}/budgets", json={"request_limit": 1, "budget_duration_sec": 86400}, headers=master_key_header
    ).json()
    key = client.post(
        f"{API_ROOT}/keys",
        json={
            "key_name": "mlpa-search",
            "user_id": "mlpa-search",
            "is_service_key": True,
            "end_user_budget_id": budget["budget_id"],
        },
        headers=master_key_header,
    ).json()
    headers = {API_KEY_HEADER: f"Bearer {key['key']}"}
    body = {**SEARCH_PAYLOAD, "user": "fxa-123:search"}

    first = client.post(f"{API_ROOT}/search/fake-search", json=body, headers=headers)
    second = client.post(f"{API_ROOT}/search/fake-search", json=body, headers=headers)
    other = client.post(
        f"{API_ROOT}/search/fake-search", json={**SEARCH_PAYLOAD, "user": "fxa-456:search"}, headers=headers
    )

    assert first.status_code == 200, first.text
    assert second.status_code == 403, second.text
    assert other.status_code == 200, other.text
    users = client.get(f"{API_ROOT}/users", params={"parent_user_id": "mlpa-search"}, headers=master_key_header).json()
    end_user = next(user for user in users if user["external_id"] == "fxa-123:search")
    assert end_user["budget_id"] == budget["budget_id"]
    assert len(_search_rows(client, master_key_header, end_user["user_id"])) >= 1


def _rules_client(test_config: GatewayConfig, rules: list[dict[str, Any]]) -> Generator[TestClient]:
    yield from _client_on(test_config, rate_limits=[RateLimitRule(**rule) for rule in rules])


@pytest.fixture
def per_key_client(test_config: GatewayConfig) -> Generator[TestClient]:
    yield from _rules_client(test_config, [{"name": "keys", "per": "key", "rpm": 1}])


@pytest.fixture
def per_tool_client(test_config: GatewayConfig) -> Generator[TestClient]:
    yield from _rules_client(
        test_config, [{"name": "fake-cap", "per": "model", "models": ["fake:fake-search"], "rpm": 1}]
    )


def test_a_search_refused_before_the_reservation_uses_no_rate_limit_slot(
    per_key_client: TestClient, master_key_header: dict[str, str]
) -> None:
    headers = _keyed(per_key_client, master_key_header, "rules-search")

    unknown = per_key_client.post(f"{API_ROOT}/search/no-such-tool", json=SEARCH_PAYLOAD, headers=headers)
    first = per_key_client.post(f"{API_ROOT}/search/fake-search", json=SEARCH_PAYLOAD, headers=headers)
    second = per_key_client.post(f"{API_ROOT}/search/fake-search", json=SEARCH_PAYLOAD, headers=headers)

    assert unknown.status_code == 400, unknown.text
    assert first.status_code == 200, first.text
    assert second.status_code == 429, second.text


def test_a_per_model_rule_limits_the_search_tool_it_names(
    per_tool_client: TestClient, master_key_header: dict[str, str]
) -> None:
    headers = _keyed(per_tool_client, master_key_header, "rules-tool")

    first = per_tool_client.post(f"{API_ROOT}/search/fake-search", json=SEARCH_PAYLOAD, headers=headers)
    second = per_tool_client.post(f"{API_ROOT}/search/fake-search", json=SEARCH_PAYLOAD, headers=headers)
    other = per_tool_client.post(f"{API_ROOT}/search/fake-fast", json=SEARCH_PAYLOAD, headers=headers)

    assert first.status_code == 200, first.text
    assert second.status_code == 429, second.text
    assert "fake-cap" in second.json()["detail"]
    assert other.status_code == 200, other.text


def test_a_service_key_starts_a_search_end_user_on_the_budget_it_names(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """``Otari-End-User-Budget`` picks the new end user's budget on search as it does on chat."""
    for budget_id in ("search-default", "search-memories"):
        put = client.put(f"{API_ROOT}/budgets/{budget_id}", json={"request_limit": 5}, headers=master_key_header)
        assert put.status_code == 201, put.text
    key = client.post(
        f"{API_ROOT}/keys",
        json={
            "user_id": "mlpa-multi",
            "is_service_key": True,
            "end_user_budget_ids": ["search-default", "search-memories"],
            "end_user_budget_id": "search-default",
        },
        headers=master_key_header,
    ).json()
    headers = {API_KEY_HEADER: f"Bearer {key['key']}", "Otari-End-User-Budget": "search-memories"}

    named = client.post(
        f"{API_ROOT}/search/fake-search", json={**SEARCH_PAYLOAD, "user": "fxa-1:memories"}, headers=headers
    )
    refused = client.post(
        f"{API_ROOT}/search/fake-search",
        json={**SEARCH_PAYLOAD, "user": "fxa-2:memories"},
        headers={**headers, "Otari-End-User-Budget": "not-listed"},
    )

    assert named.status_code == 200, named.text
    assert named.headers["Otari-End-User-Budget"] == "search-memories"
    assert refused.status_code == 403
    assert refused.json()["code"] == "end_user_budget_not_allowed"
    assert refused.headers["Otari-Error-Code"] == "end_user_budget_not_allowed"


# --------------------------------------------------------------------------- #
# The workspace's policy and organization key
# --------------------------------------------------------------------------- #


def _add_org_search_key(client: TestClient, headers: dict[str, str], api_key: str) -> str:
    created = client.post(
        f"{API_ROOT}/organizations/me/web-search-keys",
        json={"provider": "tavily", "name": "ours", "api_key": api_key},
        headers=headers,
    )
    assert created.status_code == 201, created.text
    key_id: str = created.json()["id"]
    return key_id


@pytest.fixture
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())


def test_an_unnamed_call_runs_on_the_workspaces_own_key_priced_as_web_search(
    default_tool_client: TestClient, master_key_header: dict[str, str]
) -> None:
    """The key serves in the default tool's place, and is priced under otari:web_search (decisions 33, 51)."""
    client = default_tool_client
    headers = _keyed(client, master_key_header, "org-key-user")
    _add_org_search_key(client, master_key_header, "tvly-org-secret")
    _price(client, master_key_header, "otari:web_search", 5000.0)
    _price(client, master_key_header, "fake:fake-search", 9000.0)
    keyed = AsyncMock(return_value=SearchOutcome(results=_OLD_CLIENT_HITS))
    with patch("gateway.services.search_backend.run_keyed_search", keyed):
        resp = client.post(f"{API_ROOT}/search", json=SEARCH_PAYLOAD, headers=headers)

    assert resp.status_code == 200, resp.text
    assert resp.json()["search_tool"] == "fake-search"
    assert resp.json()["results"][0]["snippet"] == "From the old client."
    credential = keyed.call_args.args[0]
    assert (credential.provider, credential.api_key) == ("tavily", "tvly-org-secret")
    # The key changes who pays, not how long the default tool may take.
    assert keyed.call_args.kwargs["timeout_s"] == 7
    (row,) = _search_rows(client, master_key_header, "org-key-user")
    # The row keeps the tool's names; the cost is the otari:web_search rate, not the tool's.
    assert (row["model"], row["provider"]) == ("fake-search", "fake")
    assert row["cost"] == pytest.approx(0.005)


@pytest.mark.parametrize("field", [{"include_raw": True}, {"provider_options": {"search_depth": "advanced"}}])
def test_the_new_fields_are_refused_on_a_call_a_tavily_key_serves(
    default_tool_client: TestClient, master_key_header: dict[str, str], field: dict[str, Any]
) -> None:
    client = default_tool_client
    headers = _keyed(client, master_key_header, "org-key-fields")
    _add_org_search_key(client, master_key_header, "tvly-org-secret")
    keyed = AsyncMock(return_value=SearchOutcome(results=_OLD_CLIENT_HITS))
    with patch("gateway.services.search_backend.run_keyed_search", keyed):
        refused = client.post(f"{API_ROOT}/search", json={**SEARCH_PAYLOAD, **field}, headers=headers)
        defaults = client.post(
            f"{API_ROOT}/search", json={**SEARCH_PAYLOAD, "include_raw": False, "provider_options": {}}, headers=headers
        )

    assert refused.status_code == 400
    assert "include_raw and provider_options" in refused.json()["detail"]
    assert defaults.status_code == 200, defaults.text
    keyed.assert_awaited_once()


@pytest.mark.usefixtures("_secret_key")
def test_search_without_a_usable_key_uses_the_named_tool(
    client: TestClient, master_key_header: dict[str, str], api_key_header: dict[str, str]
) -> None:
    """A key the workspace turned off leaves its searches on the deployment's tool."""
    key_id = _add_org_search_key(client, master_key_header, "tvly-org-secret")
    workspace_id = _default_workspace_id(client, master_key_header)
    turned_off = client.patch(
        f"{API_ROOT}/workspaces/{workspace_id}/web-search-keys/{key_id}",
        json={"disabled": True},
        headers=master_key_header,
    )
    assert turned_off.status_code == 200, turned_off.text
    keyed = AsyncMock(return_value=SearchOutcome(results=_OLD_CLIENT_HITS))
    with patch("gateway.services.search_backend.run_keyed_search", keyed):
        resp = client.post(f"{API_ROOT}/search/fake-search", json=SEARCH_PAYLOAD, headers=api_key_header)

    assert resp.status_code == 200, resp.text
    keyed.assert_not_called()
    assert resp.json()["results"][0]["snippet"] == _HIT["snippet"]


def test_a_named_call_runs_on_the_tools_own_key_though_the_workspace_has_one(
    default_tool_client: TestClient, master_key_header: dict[str, str]
) -> None:
    """Decision 51: naming a tool means that tool, and its own key; #2015 let the organization's key replace it."""
    client = default_tool_client
    headers = _keyed(client, master_key_header, "named-call")
    _add_org_search_key(client, master_key_header, "tvly-org-secret")
    keyed = AsyncMock(return_value=SearchOutcome(results=_OLD_CLIENT_HITS))
    with patch("gateway.services.search_backend.run_keyed_search", keyed):
        by_path = client.post(f"{API_ROOT}/search/fake-search", json=SEARCH_PAYLOAD, headers=headers)
        by_body = client.post(
            f"{API_ROOT}/search", json={**SEARCH_PAYLOAD, "search_tool_name": "fake-search"}, headers=headers
        )

    for resp in (by_path, by_body):
        assert resp.status_code == 200, resp.text
        assert resp.json()["results"][0]["snippet"] == _HIT["snippet"]
    keyed.assert_not_called()
    # Billed as the tool's own search, at the provider's reported cost.
    assert [row["cost"] for row in _search_rows(client, master_key_header, "named-call")] == [
        pytest.approx(0.007),
        pytest.approx(0.007),
    ]


def test_a_tool_removed_after_it_was_picked_is_refused_and_logged(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """Another writer can delete a stored tool between the pick and the resolve."""
    with patch("gateway.api.routes.search.resolve_direct_search", side_effect=SearchToolNotFoundError("fake-search")):
        resp = client.post(f"{API_ROOT}/search/fake-search", json=SEARCH_PAYLOAD, headers=api_key_header)

    assert resp.status_code == 404
    (row,) = _search_rows(client, master_key_header, api_key_obj["user_id"])
    assert row["status_code"] == 404


def test_a_workspace_key_its_provider_cannot_be_called_with_is_a_generic_502(
    default_tool_client: TestClient, master_key_header: dict[str, str]
) -> None:
    """The refusal's own words describe the workspace's credential, so they stay on the usage row."""
    client = default_tool_client
    headers = _keyed(client, master_key_header, "refused-key")
    key = WebSearchCredential(
        provider="tavily", source=WebToolCredentialSource.ORG, api_key="tvly-k", api_base="https://tavily.example"
    )
    _stub_policy(client, _policy(credential=key))
    resp = client.post(f"{API_ROOT}/search", json=SEARCH_PAYLOAD, headers=headers)

    assert resp.status_code == 502
    assert "base URL" not in resp.json()["detail"]
    assert not {name.lower() for name in _FAILURE_HEADERS} & set(resp.headers)
    (row,) = _search_rows(client, master_key_header, "refused-key")
    assert "base URL" in row["error_message"]


@pytest.fixture
def keyless_client(test_config: GatewayConfig) -> Generator[TestClient]:
    """An Exa tool with no key, as a config built in-process or a stored key that cannot be decrypted leaves it."""
    yield from _client_on(test_config, search_tools={"exa": {"provider": "exa"}})


def test_a_library_tool_without_its_key_is_refused_before_the_provider(
    keyless_client: TestClient, master_key_header: dict[str, str]
) -> None:
    headers = _keyed(keyless_client, master_key_header, "keyless")
    with patch("gateway.api.routes.search.run_direct_search", AsyncMock()) as searched:
        resp = keyless_client.post(f"{API_ROOT}/search/exa", json=SEARCH_PAYLOAD, headers=headers)

    assert resp.status_code == 400
    assert resp.json()["detail"] == "Search tool 'exa' is not configured correctly."
    searched.assert_not_awaited()


class _StubPolicy:
    """A web search policy port answering a fixed policy, recording what it was asked."""

    def __init__(self, answer: ResolvedWebSearchConfig | None) -> None:
        self.answer = answer
        self.asked: list[tuple[WebSearchPolicyScope, Sequence[WebTool]]] = []

    async def resolve(
        self, scope: WebSearchPolicyScope, requested_tools: Sequence[WebTool]
    ) -> ResolvedWebSearchConfig | None:
        self.asked.append((scope, tuple(requested_tools)))
        return self.answer


def _policy(*, enabled: bool = True, max_results: int | None = None, credential: Any = None) -> ResolvedWebSearchConfig:
    return ResolvedWebSearchConfig(
        enabled=enabled,
        max_results=max_results,
        purpose_hint=None,
        allowed_domains=None,
        blocked_domains=None,
        provider_options=None,
        authorized_tools=None,
        credential=credential,
    )


def _stub_policy(client: TestClient, answer: ResolvedWebSearchConfig | None) -> _StubPolicy:
    stub = _StubPolicy(answer)
    client.app.dependency_overrides[get_web_search_policy_port] = lambda: stub  # type: ignore[attr-defined]
    return stub


def test_the_route_reads_the_policy_through_the_port(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
) -> None:
    """A disabled policy the port answers, with no stored row behind it, is today's 403."""
    stub = _stub_policy(client, _policy(enabled=False))
    resp = client.post(f"{API_ROOT}/search/fake-search", json=SEARCH_PAYLOAD, headers=api_key_header)

    assert resp.status_code == 403
    ((scope, requested),) = stub.asked
    assert scope.workspace_id is not None
    assert scope.user_token is None
    assert requested == (WebTool.SEARCH,)
    (row,) = _search_rows(client, master_key_header, api_key_obj["user_id"])
    assert (row["model"], row["status_code"]) == ("fake-search", 403)


def test_an_unnamed_call_runs_on_the_credential_the_port_answers(
    default_tool_client: TestClient, master_key_header: dict[str, str]
) -> None:
    client = default_tool_client
    headers = _keyed(client, master_key_header, "port-key")
    credential = WebSearchCredential(provider="brave", source=WebToolCredentialSource.ORG, api_key="brave-key")
    _stub_policy(client, _policy(credential=credential))
    keyed = AsyncMock(return_value=SearchOutcome(results=_OLD_CLIENT_HITS))
    with patch("gateway.services.search_backend.run_keyed_search", keyed):
        resp = client.post(f"{API_ROOT}/search", json=SEARCH_PAYLOAD, headers=headers)

    assert resp.status_code == 200, resp.text
    assert keyed.call_args.args[0] == credential


def test_a_failure_on_the_organization_key_names_the_keys_provider_in_the_headers(
    default_tool_client: TestClient, master_key_header: dict[str, str]
) -> None:
    client = default_tool_client
    headers = _keyed(client, master_key_header, "port-key-failure")
    credential = WebSearchCredential(provider="brave", source=WebToolCredentialSource.ORG, api_key="brave-key")
    _stub_policy(client, _policy(credential=credential))
    keyed = AsyncMock(side_effect=SearchProviderError("brave answered 503", tag="http_error", status=503))
    with patch("gateway.services.search_backend.run_keyed_search", keyed):
        resp = client.post(f"{API_ROOT}/search", json=SEARCH_PAYLOAD, headers=headers)

    assert resp.status_code == 502
    # Brave, not the default tool's fake provider.
    assert tuple(resp.headers.get(name) for name in _FAILURE_HEADERS) == ("brave", "503", "http_error", "true")


def test_the_ceiling_the_port_answers_lowers_the_count(client: TestClient, api_key_header: dict[str, str]) -> None:
    _stub_policy(client, _policy(max_results=3))
    payload = {**SEARCH_PAYLOAD, "max_results": 50, "include_raw": True}
    resp = client.post(f"{API_ROOT}/search/fake-search", json=payload, headers=api_key_header)
    assert _reached_the_provider(resp)["max_results"] == 3


def test_no_policy_from_the_port_sets_no_ceiling(client: TestClient, api_key_header: dict[str, str]) -> None:
    _stub_policy(client, None)
    payload = {**SEARCH_PAYLOAD, "max_results": 50, "include_raw": True}
    resp = client.post(f"{API_ROOT}/search/fake-search", json=payload, headers=api_key_header)
    assert _reached_the_provider(resp)["max_results"] == 50


# --------------------------------------------------------------------------- #
# A searxng tool, on the old client until any-search has its adapter
# --------------------------------------------------------------------------- #


@pytest.fixture
def searxng_client(test_config: GatewayConfig) -> Generator[TestClient]:
    tools = {"local": {"provider": "searxng", "api_base": "http://searxng:8080"}}
    yield from _client_on(test_config, search_tools=tools)


@pytest.mark.parametrize("field", [{"include_raw": True}, {"provider_options": {"engines": "wikipedia"}}])
def test_the_new_fields_are_refused_on_a_searxng_tool(
    searxng_client: TestClient, master_key_header: dict[str, str], field: dict[str, Any]
) -> None:
    headers = _keyed(searxng_client, master_key_header, "searxng-user")
    old = AsyncMock(return_value=SearchOutcome(results=_OLD_CLIENT_HITS, cost_usd=None))
    with patch("gateway.services.search_backend.run_search", old):
        refused = searxng_client.post(f"{API_ROOT}/search/local", json={**SEARCH_PAYLOAD, **field}, headers=headers)
        defaults = searxng_client.post(
            f"{API_ROOT}/search/local",
            json={**SEARCH_PAYLOAD, "include_raw": False, "provider_options": {}, "max_results": 4},
            headers=headers,
        )

    assert refused.status_code == 400
    assert refused.json()["detail"] == (
        "include_raw and provider_options are not available yet for a searxng search: leave them unset."
    )
    assert defaults.status_code == 200, defaults.text
    tool, query = old.call_args.args
    assert (tool.name, tool.provider, tool.api_base) == ("local", "searxng", "http://searxng:8080")
    assert query.max_results == 4
    assert old.call_args.kwargs["client"] is searxng_client.app.state.web_provider_client  # type: ignore[attr-defined]


def test_an_old_clients_failure_carries_its_safe_fields_and_never_its_body(
    searxng_client: TestClient, master_key_header: dict[str, str]
) -> None:
    headers = _keyed(searxng_client, master_key_header, "searxng-failed")
    failed = SearchProviderError(
        "searxng search returned HTTP 503: upstream-body-sentinel", tag="http_error", status=503
    )
    with patch("gateway.services.search_backend.run_search", AsyncMock(side_effect=failed)):
        resp = searxng_client.post(f"{API_ROOT}/search/local", json=SEARCH_PAYLOAD, headers=headers)

    assert resp.status_code == 502
    assert tuple(resp.headers.get(name) for name in _FAILURE_HEADERS) == ("searxng", "503", "http_error", "true")
    assert "upstream-body-sentinel" not in resp.text
    assert all("upstream-body-sentinel" not in value for value in resp.headers.values())
    # Nor on the usage row, which keeps the failure's safe fields and the provider's status.
    (row,) = _search_rows(searxng_client, master_key_header, "searxng-failed")
    assert (row["error_message"], row["status_code"]) == ("searxng search failed (http_error, HTTP 503)", 503)


def test_a_searxng_tools_reported_cost_is_none_so_its_rate_bills(
    searxng_client: TestClient, master_key_header: dict[str, str]
) -> None:
    headers = _keyed(searxng_client, master_key_header, "searxng-billed")
    _price(searxng_client, master_key_header, "searxng:local", 2000.0)
    old = AsyncMock(return_value=SearchOutcome(results=_OLD_CLIENT_HITS))
    with patch("gateway.services.search_backend.run_search", old):
        assert searxng_client.post(f"{API_ROOT}/search/local", json=SEARCH_PAYLOAD, headers=headers).status_code == 200
    (row,) = _search_rows(searxng_client, master_key_header, "searxng-billed")
    assert Decimal(str(row["cost"])) == Decimal("0.002")
