"""Integration tests for the POST /api/v1/fetch endpoint.

The fetch tools are any-fetch's ``fake`` provider, which answers a canned page
from its options and echoes the request it got in ``raw``, so these exercise
the route and its rules without a network: auth, which tool and which key, the
workspace's policy and domain lists, the gateway's own URL checks, pricing, the
provider's failure and the usage row that makes a fetch visible in Activity.
The built-in fetcher, which would reach the network, is exercised only up to
the checks that refuse a URL before it is called.
"""

import uuid
from collections.abc import AsyncIterator, Generator, Sequence
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.api.deps import get_web_provider_client, get_workspace_web_search_policies
from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.models.tools import ResolvedWebSearchConfig, WebFetchCredential, WebTool
from gateway.services.secret_box import generate_secret_key

from .conftest import build_test_client

FETCH_USAGE_LABEL = "/v1/fetch"
URL = "https://example.com/article"
PAYLOAD: dict[str, Any] = {"url": URL}
PROVIDER_ERROR_DETAIL = "The request could not be completed by the provider"


def _fetch_tools() -> dict[str, dict[str, Any]]:
    return {
        "fake-fetch": {
            "provider": "fake",
            "timeout": 7,
            "options": {"text": "The page's text.", "title": "A page", "cost": 0.002},
        },
        "fake-plain": {"provider": "fake", "options": {"text": "Plain text."}},
    }


@pytest.fixture
def test_config(postgres_url: str) -> GatewayConfig:
    """The shared config with fetch on, two fake fetch tools and the first as the default."""
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        web_fetch_enabled=True,
        fetch_tools=_fetch_tools(),
        web_fetch_default_tool="fake-fetch",
    )


def _client_on(test_config: GatewayConfig, **updates: Any) -> Generator[TestClient]:
    yield from build_test_client(test_config.model_copy(update=updates))


@pytest.fixture
def builtin_client(test_config: GatewayConfig) -> Generator[TestClient]:
    """No default named, so ``builtin_fetch`` serves an unnamed call."""
    yield from _client_on(test_config, web_fetch_default_tool=None)


@pytest.fixture
def disabled_client(test_config: GatewayConfig) -> Generator[TestClient]:
    yield from _client_on(test_config, web_fetch_enabled=False)


@pytest.fixture
def strict_client(test_config: GatewayConfig) -> Generator[TestClient]:
    """``require_pricing`` on, with no fetch tool priced."""
    yield from _client_on(test_config, require_pricing=True)


def _keyed(client: TestClient, master_key_header: dict[str, str], user_id: str, **lists: Any) -> dict[str, str]:
    assert client.post(f"{API_ROOT}/users", json={"user_id": user_id}, headers=master_key_header).status_code == 200
    key = client.post(
        f"{API_ROOT}/keys", json={"key_name": user_id, "user_id": user_id, **lists}, headers=master_key_header
    )
    assert key.status_code == 200, key.text
    return {API_KEY_HEADER: f"Bearer {key.json()['key']}"}


def _fetch_rows(client: TestClient, headers: dict[str, str], user_id: str) -> list[dict[str, Any]]:
    resp = client.get(f"{API_ROOT}/usage", params={"user_id": user_id, "endpoint": FETCH_USAGE_LABEL}, headers=headers)
    assert resp.status_code == 200, resp.text
    return [dict(row) for row in resp.json()]


def _price(
    client: TestClient, master_key_header: dict[str, str], model_key: str, per_million: float, **row: Any
) -> None:
    resp = client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": model_key, "input_price_per_million": per_million, "output_price_per_million": 0.0, **row},
        headers=master_key_header,
    )
    assert resp.status_code in (200, 201), resp.text


def test_fetch_requires_auth(client: TestClient) -> None:
    assert client.post(f"{API_ROOT}/fetch", json=PAYLOAD).status_code == 401


def test_an_unnamed_call_runs_on_the_default_tool(client: TestClient, api_key_header: dict[str, str]) -> None:
    resp = client.post(f"{API_ROOT}/fetch", json=PAYLOAD, headers=api_key_header)

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "object": "fetch",
        "fetch_tool": "fake-fetch",
        "url": URL,
        "final_url": URL,
        "title": "A page",
        "content_type": "text/html",
        "text": "The page's text.",
        "published": None,
        "truncated": False,
    }


def test_the_path_names_the_tool_and_wins_over_the_body(client: TestClient, api_key_header: dict[str, str]) -> None:
    resp = client.post(
        f"{API_ROOT}/fetch/fake-plain", json={**PAYLOAD, "fetch_tool_name": "fake-fetch"}, headers=api_key_header
    )

    assert resp.status_code == 200, resp.text
    assert (resp.json()["fetch_tool"], resp.json()["text"]) == ("fake-plain", "Plain text.")


def test_the_body_names_the_tool(client: TestClient, api_key_header: dict[str, str]) -> None:
    resp = client.post(f"{API_ROOT}/fetch", json={**PAYLOAD, "fetch_tool_name": "fake-plain"}, headers=api_key_header)

    assert resp.status_code == 200, resp.text
    assert resp.json()["fetch_tool"] == "fake-plain"


def test_include_raw_returns_the_providers_whole_response(client: TestClient, api_key_header: dict[str, str]) -> None:
    resp = client.post(f"{API_ROOT}/fetch", json={**PAYLOAD, "include_raw": True}, headers=api_key_header)

    assert resp.status_code == 200, resp.text
    assert resp.json()["raw"]["request"] == {
        "max_chars": None,
        "options": {"text": "The page's text.", "title": "A page", "cost": 0.002},
    }


def test_max_chars_cuts_the_text_and_wins_over_the_tools_option(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch, test_config: GatewayConfig
) -> None:
    monkeypatch.setitem(test_config.fetch_tools["fake-plain"]["options"], "max_chars", 5)

    cut_by_the_tool = client.post(f"{API_ROOT}/fetch/fake-plain", json=PAYLOAD, headers=api_key_header)
    assert cut_by_the_tool.status_code == 200, cut_by_the_tool.text
    assert (cut_by_the_tool.json()["text"], cut_by_the_tool.json()["truncated"]) == ("Plain", True)

    cut_by_the_request = client.post(
        f"{API_ROOT}/fetch/fake-plain", json={**PAYLOAD, "max_chars": 8}, headers=api_key_header
    )
    assert (cut_by_the_request.json()["text"], cut_by_the_request.json()["truncated"]) == ("Plain te", True)


def test_provider_options_reach_the_provider_over_the_tools_own(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    resp = client.post(
        f"{API_ROOT}/fetch",
        json={**PAYLOAD, "provider_options": {"title": "Mine"}, "include_raw": True},
        headers=api_key_header,
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["title"] == "Mine"
    assert resp.json()["raw"]["request"]["options"]["title"] == "Mine"


@pytest.mark.parametrize(
    ("options", "problem"),
    [({"nope": 1}, "nope"), ({"account": "ours"}, "account")],
    ids=["an option the provider does not take", "an operator-only option"],
)
def test_a_request_option_the_provider_refuses_is_a_400_naming_it(
    client: TestClient, api_key_header: dict[str, str], options: dict[str, Any], problem: str
) -> None:
    resp = client.post(f"{API_ROOT}/fetch", json={**PAYLOAD, "provider_options": options}, headers=api_key_header)

    assert resp.status_code == 400, resp.text
    assert f"'{problem}'" in resp.json()["detail"]


def test_a_disabled_fetch_is_a_400_before_any_tool_is_picked(
    disabled_client: TestClient, master_key_header: dict[str, str]
) -> None:
    headers = _keyed(disabled_client, master_key_header, "disabled-user")
    resp = disabled_client.post(f"{API_ROOT}/fetch/fake-fetch", json=PAYLOAD, headers=headers)

    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"].startswith("Web fetch is disabled on this gateway")
    (row,) = _fetch_rows(disabled_client, master_key_header, "disabled-user")
    assert (row["model"], row["provider"], row["status"], row["status_code"]) == ("fake-fetch", None, "error", 400)


def test_an_unknown_tool_is_a_400_naming_the_available_ones(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    headers = _keyed(client, master_key_header, "unknown-tool-user")
    resp = client.post(f"{API_ROOT}/fetch/nope", json=PAYLOAD, headers=headers)

    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == "Unknown fetch tool 'nope'. Available: builtin_fetch, fake-fetch, fake-plain."
    (row,) = _fetch_rows(client, master_key_header, "unknown-tool-user")
    assert (row["model"], row["status_code"]) == ("nope", 400)


def test_master_key_requires_user(client: TestClient, master_key_header: dict[str, str]) -> None:
    resp = client.post(f"{API_ROOT}/fetch", json=PAYLOAD, headers=master_key_header)
    assert resp.status_code == 400
    assert "user" in resp.json()["detail"].lower()


def test_an_empty_url_is_a_422(client: TestClient, api_key_header: dict[str, str]) -> None:
    assert client.post(f"{API_ROOT}/fetch", json={"url": ""}, headers=api_key_header).status_code == 422
    assert client.post(f"{API_ROOT}/fetch", json={**PAYLOAD, "max_chars": 0}, headers=api_key_header).status_code == 422


@pytest.mark.parametrize(
    "url", ["not a url", "ftp://example.com/file", "http://127.0.0.1/admin", "http://localhost/x", "http://10.0.0.1/"]
)
def test_a_url_the_gateway_refuses_is_a_400_before_the_tool_runs(
    builtin_client: TestClient, master_key_header: dict[str, str], url: str
) -> None:
    """The refusal reaches the built-in fetcher's row too, since the check comes before the fetcher is called."""
    headers = _keyed(builtin_client, master_key_header, "bad-url-user")
    resp = builtin_client.post(f"{API_ROOT}/fetch", json={"url": url}, headers=headers)

    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"].startswith("The URL cannot be fetched")
    assert url not in resp.text
    (row,) = _fetch_rows(builtin_client, master_key_header, "bad-url-user")
    assert (row["model"], row["provider"], row["status_code"]) == ("builtin_fetch", "builtin", 400)
    assert url not in (row["error_message"] or "")


class _StubPolicy:
    """A web search policy answering a fixed policy, recording what it was asked."""

    def __init__(self, answer: ResolvedWebSearchConfig | None) -> None:
        self.answer = answer
        self.asked: list[tuple[uuid.UUID, Sequence[WebTool]]] = []

    async def resolve(
        self, workspace_id: uuid.UUID, requested_tools: Sequence[WebTool]
    ) -> ResolvedWebSearchConfig | None:
        self.asked.append((workspace_id, tuple(requested_tools)))
        return self.answer


def _policy(
    *,
    enabled: bool = True,
    allowed_domains: tuple[str, ...] | None = None,
    blocked_domains: tuple[str, ...] | None = None,
    fetch_credential: WebFetchCredential | None = None,
) -> ResolvedWebSearchConfig:
    return ResolvedWebSearchConfig(
        enabled=enabled,
        max_results=None,
        purpose_hint=None,
        allowed_domains=allowed_domains,
        blocked_domains=blocked_domains,
        provider_options=None,
        authorized_tools=None,
        fetch_credential=fetch_credential,
    )


def _stub_policy(client: TestClient, answer: ResolvedWebSearchConfig | None) -> _StubPolicy:
    stub = _StubPolicy(answer)
    client.app.dependency_overrides[get_workspace_web_search_policies] = lambda: stub  # type: ignore[attr-defined]
    return stub


def test_a_workspace_with_web_access_off_is_refused(
    client: TestClient, master_key_header: dict[str, str], api_key_header: dict[str, str], api_key_obj: dict[str, Any]
) -> None:
    stub = _stub_policy(client, _policy(enabled=False))
    resp = client.post(f"{API_ROOT}/fetch", json=PAYLOAD, headers=api_key_header)

    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"] == "web access is not enabled for this workspace"
    ((_, requested),) = stub.asked
    assert requested == (WebTool.FETCH,)
    (row,) = _fetch_rows(client, master_key_header, api_key_obj["user_id"])
    assert (row["model"], row["status_code"]) == ("fake-fetch", 403)


@pytest.mark.parametrize(
    "policy",
    [_policy(blocked_domains=("example.com",)), _policy(allowed_domains=("example.org",))],
    ids=["blocked", "not allowed"],
)
def test_the_workspaces_domain_lists_refuse_the_url_asked_for(
    client: TestClient, master_key_header: dict[str, str], policy: ResolvedWebSearchConfig
) -> None:
    _stub_policy(client, policy)
    headers = _keyed(client, master_key_header, "domains-user")
    resp = client.post(f"{API_ROOT}/fetch", json=PAYLOAD, headers=headers)

    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"] == "The URL is disallowed by this workspace's web access policy."
    (row,) = _fetch_rows(client, master_key_header, "domains-user")
    assert (row["model"], row["status_code"], row["cost"]) == ("fake-fetch", 403, None)


def test_a_url_inside_the_workspaces_allowed_domains_is_fetched(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    _stub_policy(client, _policy(allowed_domains=("example.com",)))
    resp = client.post(f"{API_ROOT}/fetch", json={"url": "https://docs.example.com/page"}, headers=api_key_header)
    assert resp.status_code == 200, resp.text


def test_a_redirect_the_provider_reports_outside_the_lists_is_refused_after_the_call(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """The page is not returned and not billed, and the refusal is logged as one."""
    _stub_policy(client, _policy(blocked_domains=("tracker.example.net",)))
    headers = _keyed(client, master_key_header, "redirect-user")
    body = {**PAYLOAD, "provider_options": {"final_url": "https://tracker.example.net/landing"}}
    resp = client.post(f"{API_ROOT}/fetch", json=body, headers=headers)

    assert resp.status_code == 403, resp.text
    assert "tracker" not in resp.text
    (row,) = _fetch_rows(client, master_key_header, "redirect-user")
    assert (row["model"], row["status"], row["status_code"], row["cost"]) == ("fake-fetch", "error", 403, None)


def test_a_redirect_to_a_private_address_is_a_400_after_the_call(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    body = {**PAYLOAD, "provider_options": {"final_url": "http://192.168.1.1/admin"}}
    resp = client.post(f"{API_ROOT}/fetch", json=body, headers=api_key_header)
    assert resp.status_code == 400, resp.text
    assert "192.168" not in resp.text


def test_allowed_tools_governs_fetch_tools(client: TestClient, master_key_header: dict[str, str]) -> None:
    granted = _keyed(client, master_key_header, "one-fetch-user", allowed_tools=["fake-plain", "builtin_fetch"])
    assert client.post(f"{API_ROOT}/fetch/fake-plain", json=PAYLOAD, headers=granted).status_code == 200

    refused = client.post(f"{API_ROOT}/fetch/fake-fetch", json=PAYLOAD, headers=granted)
    assert refused.status_code == 403, refused.text
    assert refused.json()["detail"].startswith("Fetch tool 'fake-fetch' is not permitted for this API key.")

    none = _keyed(client, master_key_header, "no-tool-user", allowed_tools=[])
    assert client.post(f"{API_ROOT}/fetch", json=PAYLOAD, headers=none).status_code == 403

    every = _keyed(client, master_key_header, "every-tool-user", allowed_tools=["*"])
    assert client.post(f"{API_ROOT}/fetch", json=PAYLOAD, headers=every).status_code == 200


def test_a_key_restricted_to_models_runs_no_fetch_tool_until_one_is_named(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    restricted = _keyed(client, master_key_header, "models-only-user", allowed_models=["openai:gpt-4o"])
    assert client.post(f"{API_ROOT}/fetch", json=PAYLOAD, headers=restricted).status_code == 403


def test_the_usage_row_carries_the_reported_cost_and_no_tokens(
    client: TestClient, master_key_header: dict[str, str], api_key_header: dict[str, str], api_key_obj: dict[str, Any]
) -> None:
    assert client.post(f"{API_ROOT}/fetch", json=PAYLOAD, headers=api_key_header).status_code == 200

    (row,) = _fetch_rows(client, master_key_header, api_key_obj["user_id"])
    assert (row["model"], row["provider"], row["endpoint"], row["status"]) == (
        "fake-fetch",
        "fake",
        FETCH_USAGE_LABEL,
        "success",
    )
    assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (0, 0, 0)
    assert row["cost"] == pytest.approx(0.002)


def test_a_tool_reporting_no_cost_bills_its_own_fetch_price(
    client: TestClient, master_key_header: dict[str, str], api_key_header: dict[str, str], api_key_obj: dict[str, Any]
) -> None:
    _price(client, master_key_header, "fake-plain", 3000.0, kind="fetch")
    assert client.post(f"{API_ROOT}/fetch/fake-plain", json=PAYLOAD, headers=api_key_header).status_code == 200

    (row,) = _fetch_rows(client, master_key_header, api_key_obj["user_id"])
    assert row["cost"] == pytest.approx(0.003)


def test_an_unpriced_fetch_is_refused_with_402_naming_its_row(
    strict_client: TestClient, master_key_header: dict[str, str]
) -> None:
    headers = _keyed(strict_client, master_key_header, "unpriced-user")
    resp = strict_client.post(f"{API_ROOT}/fetch/fake-plain", json=PAYLOAD, headers=headers)

    assert resp.status_code == 402, resp.text
    assert resp.headers["Otari-Error-Code"] == "pricing_required"
    assert "A fetch on 'fake-plain' has no price the deployment's operator set" in resp.json()["detail"]
    assert "using kind 'fetch' and the tool's name as model_key" in resp.json()["detail"]
    assert "price_per_call" in resp.json()["detail"]
    (row,) = _fetch_rows(strict_client, master_key_header, "unpriced-user")
    assert (row["status"], row["status_code"], row["cost"]) == ("error", 402, None)

    _price(strict_client, master_key_header, "fake-plain", 0.0, kind="fetch")
    assert strict_client.post(f"{API_ROOT}/fetch/fake-plain", json=PAYLOAD, headers=headers).status_code == 200


def test_a_provider_failure_is_a_502_carrying_its_safe_fields(
    client: TestClient, master_key_header: dict[str, str], api_key_header: dict[str, str], api_key_obj: dict[str, Any]
) -> None:
    body = {**PAYLOAD, "provider_options": {"error": "http_error", "error_status": 503}}
    resp = client.post(f"{API_ROOT}/fetch", json=body, headers=api_key_header)

    assert resp.status_code == 502, resp.text
    assert resp.json()["detail"] == PROVIDER_ERROR_DETAIL
    assert (
        resp.headers["Otari-Error-Provider"],
        resp.headers["Otari-Error-Status"],
        resp.headers["Otari-Error-Tag"],
        resp.headers["Otari-Error-Retryable"],
    ) == ("fake", "503", "http_error", "true")
    (row,) = _fetch_rows(client, master_key_header, api_key_obj["user_id"])
    assert (row["status"], row["status_code"], row["error_message"]) == (
        "error",
        503,
        "fake fetch failed (http_error, HTTP 503)",
    )


def test_an_error_inside_a_successful_answer_is_a_502(client: TestClient, api_key_header: dict[str, str]) -> None:
    body = {**PAYLOAD, "provider_options": {"in_body_error": "no_content"}}
    resp = client.post(f"{API_ROOT}/fetch", json=body, headers=api_key_header)

    assert resp.status_code == 502, resp.text
    assert (resp.headers["Otari-Error-Tag"], "Otari-Error-Status" in resp.headers) == ("no_content", False)


def test_a_providers_exception_text_never_reaches_the_caller_or_the_row(
    client: TestClient, master_key_header: dict[str, str], api_key_header: dict[str, str], api_key_obj: dict[str, Any]
) -> None:
    body = {**PAYLOAD, "provider_options": {"leak_url": True}}
    resp = client.post(f"{API_ROOT}/fetch", json=body, headers=api_key_header)

    assert resp.status_code == 502, resp.text
    assert resp.json()["detail"] == PROVIDER_ERROR_DETAIL
    assert "example.com" not in resp.text
    assert (resp.headers["Otari-Error-Provider"], resp.headers["Otari-Error-Tag"]) == ("fake", "internal_error")
    (row,) = _fetch_rows(client, master_key_header, api_key_obj["user_id"])
    assert row["error_message"] == "fake fetch failed (internal_error)"


@pytest.fixture
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())


@pytest.fixture
def keyed_client(test_config: GatewayConfig, _secret_key: None) -> Generator[TestClient]:
    yield from _client_on(test_config)


def _exa_answers(client: TestClient, page: dict[str, Any], cost: float | None) -> list[httpx.Request]:
    """Exa answering ``page`` on the app's web provider client, reporting ``cost`` when it is not ``None``."""
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body: dict[str, Any] = {"results": [page]}
        if cost is not None:
            body["costDollars"] = {"total": cost}
        return httpx.Response(200, json=body)

    async def web_provider_client() -> AsyncIterator[httpx.AsyncClient]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as answering:
            yield answering

    client.app.dependency_overrides[get_web_provider_client] = web_provider_client  # type: ignore[attr-defined]
    return seen


def _add_org_fetch_key(client: TestClient, headers: dict[str, str]) -> None:
    created = client.post(
        f"{API_ROOT}/organizations/me/web-search-keys",
        json={"provider": "exa", "kind": "fetch", "name": "ours", "api_key": "exa-org-secret"},
        headers=headers,
    )
    assert created.status_code == 201, created.text
    starred = client.post(
        f"{API_ROOT}/organizations/me/web-search-keys/{created.json()['id']}/default", headers=headers
    )
    assert starred.status_code == 200, starred.text


def test_an_unnamed_call_runs_on_the_organizations_fetch_key_priced_as_web_fetch(
    keyed_client: TestClient, master_key_header: dict[str, str]
) -> None:
    """The key serves in the default tool's place, under the tool's names, priced at otari:web_fetch."""
    client = keyed_client
    headers = _keyed(client, master_key_header, "org-fetch-user")
    _add_org_fetch_key(client, master_key_header)
    _price(client, master_key_header, "otari:web_fetch", 4000.0)
    seen = _exa_answers(client, {"url": URL, "title": "From Exa", "text": "Exa's text."}, cost=None)

    resp = client.post(f"{API_ROOT}/fetch", json=PAYLOAD, headers=headers)

    assert resp.status_code == 200, resp.text
    assert (resp.json()["fetch_tool"], resp.json()["title"], resp.json()["text"]) == (
        "fake-fetch",
        "From Exa",
        "Exa's text.",
    )
    [request] = seen
    assert (request.url.host, request.url.path, request.headers["x-api-key"]) == (
        "api.exa.ai",
        "/contents",
        "exa-org-secret",
    )
    (row,) = _fetch_rows(client, master_key_header, "org-fetch-user")
    assert (row["model"], row["provider"]) == ("fake-fetch", "fake")
    # No cost reported, so the key's rate, otari:web_fetch, bills the page.
    assert row["cost"] == pytest.approx(0.004)


def test_a_named_call_runs_on_the_tools_own_key_though_the_workspace_has_one(
    keyed_client: TestClient, master_key_header: dict[str, str]
) -> None:
    client = keyed_client
    headers = _keyed(client, master_key_header, "named-fetch-user")
    _add_org_fetch_key(client, master_key_header)
    seen = _exa_answers(client, {"url": URL, "text": "Exa's text."}, cost=None)

    resp = client.post(f"{API_ROOT}/fetch/fake-fetch", json=PAYLOAD, headers=headers)

    assert resp.status_code == 200, resp.text
    assert resp.json()["text"] == "The page's text."
    assert seen == []
