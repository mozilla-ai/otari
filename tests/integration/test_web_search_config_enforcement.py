"""What a workspace's web-search row does to a request that asks to search.

The management surface is covered in ``test_workspace_web_search.py``; this is
the other half of #656's Definition of Done, the request path honoring what the
row says. The in-loop cases go through ``/v1/messages`` with the search backend
and the tool loop patched out, so what is asserted is admission and the values
handed to the backend, not any search. The last few cover ``POST /v1/search``,
the other door into the same capability.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.messages import (
    MessageResponse,
    MessageStopEvent,
    MessageStreamEvent,
    MessageUsage,
    TextBlock,
)
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from gateway.core.config import API_KEY_HEADER

_SEARCH_URL = "http://127.0.0.1:9998/search"
_REQUEST = {
    "model": "anthropic:claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "what happened today"}],
    "max_tokens": 100,
    "tools": [{"type": "otari_web_search"}],
}


def _text_response(text: str = "ok") -> MessageResponse:
    return MessageResponse(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-3-5-sonnet-20241022",
        content=[TextBlock(type="text", text=text, citations=None)],
        stop_reason=cast(Any, "end_turn"),
        stop_sequence=None,
        usage=MessageUsage(input_tokens=5, output_tokens=2),
    )


def _default_workspace_id(client: TestClient, master_key_header: dict[str, str]) -> str:
    """The workspace an API-key request bills to on a fresh deployment."""
    listed = client.get("/v1/workspaces", headers=master_key_header)
    assert listed.status_code == 200
    workspace_id: str = listed.json()["data"][0]["id"]
    return workspace_id


def _set_config(
    client: TestClient,
    master_key_header: dict[str, str],
    workspace_id: str,
    **config: Any,
) -> dict[str, Any]:
    response = client.put(
        f"/v1/workspaces/{workspace_id}/web-search",
        json=config,
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    stored: dict[str, Any] = response.json()
    return stored


class _Dispatch:
    """What the patched search backend / loop saw for one request."""

    def __init__(self) -> None:
        self.backend_kwargs: dict[str, Any] = {}
        self.ran = False


def _post_with_search_patched(
    client: TestClient, headers: dict[str, str], body: dict[str, Any]
) -> tuple[Any, _Dispatch]:
    seen = _Dispatch()

    async def fake_loop(
        *, completion_kwargs: Any, pool: Any, max_iterations: int, emit_native_web_search: bool = False
    ) -> MessageResponse:
        seen.ran = True
        return _text_response("via-search-loop")

    def fake_backend(**kwargs: Any) -> Any:
        seen.backend_kwargs = kwargs
        backend = AsyncMock()
        backend.purpose_hints = lambda: []
        return AsyncMock(__aenter__=AsyncMock(return_value=backend), __aexit__=AsyncMock(return_value=None))

    with (
        patch("gateway.api.routes.messages.anthropic_tool_loop", new=fake_loop),
        patch("gateway.api.routes._tools.WebSearchBackend", new=fake_backend),
    ):
        response = client.post("/v1/messages", json=body, headers=headers)
    return response, seen


def test_no_row_leaves_the_request_exactly_as_it_was(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The zero-rows requirement: a deployment that configured nothing is unchanged."""
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", _SEARCH_URL)

    response, seen = _post_with_search_patched(client, api_key_header, _REQUEST)

    assert response.status_code == 200
    assert seen.ran is True
    assert seen.backend_kwargs["max_results"] == 5, "the backend's own default"
    assert "allowed_domains" not in seen.backend_kwargs
    assert "blocked_domains" not in seen.backend_kwargs


def test_a_disabled_workspace_is_refused_before_the_provider_is_called(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", _SEARCH_URL)
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_config(client, master_key_header, workspace_id, enabled=False)

    response, seen = _post_with_search_patched(client, api_key_header, _REQUEST)

    assert response.status_code == 403
    # Anthropic-shaped body, the same mapping every other refusal on this route gets.
    detail = response.json()["detail"]
    assert detail["error"]["message"] == "web search is not enabled for this workspace"
    assert seen.ran is False, "the tool loop must never have run"


def test_a_disabled_workspace_still_serves_a_request_that_asks_for_no_search(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The veto is over the tool, not over the workspace's traffic."""
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", _SEARCH_URL)
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_config(client, master_key_header, workspace_id, enabled=False)

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        return _text_response("plain")

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        response = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            },
            headers=api_key_header,
        )

    assert response.status_code == 200
    assert response.json()["content"][0]["text"] == "plain"


def test_the_workspace_ceiling_lowers_the_result_count(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", _SEARCH_URL)
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_config(client, master_key_header, workspace_id, enabled=True, max_results=2)

    response, seen = _post_with_search_patched(
        client,
        api_key_header,
        {**_REQUEST, "tools": [{"type": "otari_web_search", "max_results": 15}]},
    )

    assert response.status_code == 200
    assert seen.backend_kwargs["max_results"] == 2


def test_a_request_asking_for_less_than_the_workspace_allows_keeps_its_own_number(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ceiling is a floor operation, so the stricter of the two wins."""
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", _SEARCH_URL)
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_config(client, master_key_header, workspace_id, enabled=True, max_results=9)

    response, seen = _post_with_search_patched(
        client,
        api_key_header,
        {**_REQUEST, "tools": [{"type": "otari_web_search", "max_results": 3}]},
    )

    assert response.status_code == 200
    assert seen.backend_kwargs["max_results"] == 3


def test_a_workspace_ceiling_above_the_deployments_own_raises_nothing(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A row may only narrow, and the thing it narrows includes the operator's own number.

    Written into the tool entry rather than floored, a workspace ceiling of 9 on
    a deployment that caps at 3 would have *raised* the operator's cap for every
    request that named none of its own.
    """
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", _SEARCH_URL)
    monkeypatch.setenv("OTARI_WEB_SEARCH_MAX_RESULTS", "3")
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_config(client, master_key_header, workspace_id, enabled=True, max_results=9)

    response, seen = _post_with_search_patched(client, api_key_header, _REQUEST)

    assert response.status_code == 200
    assert seen.backend_kwargs["max_results"] == 3


def test_a_request_cannot_shed_the_workspace_block_list(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fail-open case the hybrid path's default-only precedence would allow."""
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", _SEARCH_URL)
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_config(client, master_key_header, workspace_id, enabled=True, blocked_domains=["banned.example"])

    response, seen = _post_with_search_patched(
        client,
        api_key_header,
        {**_REQUEST, "tools": [{"type": "otari_web_search", "blocked_domains": ["noise.example"]}]},
    )

    assert response.status_code == 200
    assert set(seen.backend_kwargs["blocked_domains"]) == {"noise.example", "banned.example"}


def test_a_request_allow_list_is_intersected_with_the_workspace_one(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", _SEARCH_URL)
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_config(
        client,
        master_key_header,
        workspace_id,
        enabled=True,
        allowed_domains=["arxiv.org", "wikipedia.org"],
    )

    response, seen = _post_with_search_patched(
        client,
        api_key_header,
        {**_REQUEST, "tools": [{"type": "otari_web_search", "allowed_domains": ["arxiv.org", "elsewhere.example"]}]},
    )

    assert response.status_code == 200
    assert seen.backend_kwargs["allowed_domains"] == ("arxiv.org",)


def test_a_request_scoped_to_a_subdomain_of_a_permitted_domain_is_served(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end for the suffix semantics the backend's own filter uses.

    A domain-list entry is a suffix, not a host, so a request scoped to
    `docs.example.com` under a workspace that permits `example.com` is asking for
    strictly less than the workspace allows. Compared as opaque strings the
    intersection is empty and the whole completion 403s, which is the shape of
    bug the unit tests alone would have let through to the request path.
    """
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", _SEARCH_URL)
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_config(client, master_key_header, workspace_id, enabled=True, allowed_domains=["example.com"])

    response, seen = _post_with_search_patched(
        client,
        api_key_header,
        {**_REQUEST, "tools": [{"type": "otari_web_search", "allowed_domains": ["docs.example.com"]}]},
    )

    assert response.status_code == 200, response.text
    assert seen.backend_kwargs["allowed_domains"] == ("docs.example.com",)


def test_a_request_naming_only_domains_the_workspace_forbids_is_refused(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty intersection cannot narrow to an empty list: that reads as no list at all."""
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", _SEARCH_URL)
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_config(client, master_key_header, workspace_id, enabled=True, allowed_domains=["arxiv.org"])

    response, seen = _post_with_search_patched(
        client,
        api_key_header,
        {**_REQUEST, "tools": [{"type": "otari_web_search", "allowed_domains": ["elsewhere.example"]}]},
    )

    assert response.status_code == 403
    assert seen.ran is False


def test_the_workspace_hint_fills_in_only_when_the_request_gave_none(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", _SEARCH_URL)
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_config(client, master_key_header, workspace_id, enabled=True, purpose_hint="workspace hint")

    _, defaulted = _post_with_search_patched(client, api_key_header, _REQUEST)
    assert defaulted.backend_kwargs["purpose_hint"] == "workspace hint"

    _, overridden = _post_with_search_patched(
        client,
        api_key_header,
        {**_REQUEST, "tools": [{"type": "otari_web_search", "purpose_hint": "request hint"}]},
    )
    assert overridden.backend_kwargs["purpose_hint"] == "request hint"


def test_clearing_the_row_puts_the_request_back_where_it_started(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", _SEARCH_URL)
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_config(client, master_key_header, workspace_id, enabled=False)

    cleared = client.delete(f"/v1/workspaces/{workspace_id}/web-search", headers=master_key_header)
    assert cleared.status_code == 200
    assert cleared.json()["configured"] is False

    response, seen = _post_with_search_patched(client, api_key_header, _REQUEST)

    assert response.status_code == 200
    assert seen.ran is True


def test_a_streaming_request_gets_the_same_narrowing(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The streaming dispatch builds its own backend, so it needs its own case.

    A ceiling that reached only the non-streaming path would leave the longest
    running requests, the streamed ones, unnarrowed.
    """
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", _SEARCH_URL)
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_config(
        client,
        master_key_header,
        workspace_id,
        enabled=True,
        max_results=2,
        blocked_domains=["banned.example"],
    )

    seen = _Dispatch()

    async def fake_loop_stream(
        *, completion_kwargs: Any, pool: Any, max_iterations: int, emit_native_web_search: bool = False
    ) -> AsyncIterator[MessageStreamEvent]:
        seen.ran = True
        yield MessageStopEvent(type="message_stop")

    def fake_backend(**kwargs: Any) -> Any:
        seen.backend_kwargs = kwargs
        backend = AsyncMock()
        backend.purpose_hints = lambda: []
        backend.__aenter__ = AsyncMock(return_value=backend)
        backend.__aexit__ = AsyncMock(return_value=None)
        return backend

    with (
        patch("gateway.api.routes.messages.anthropic_tool_loop_stream", new=fake_loop_stream),
        patch("gateway.api.routes._tools.WebSearchBackend", new=fake_backend),
    ):
        response = client.post("/v1/messages", json={**_REQUEST, "stream": True}, headers=api_key_header)

    assert response.status_code == 200, response.text
    assert seen.backend_kwargs["max_results"] == 2
    assert seen.backend_kwargs["blocked_domains"] == ("banned.example",)


def test_a_master_key_request_is_narrowed_by_the_default_workspace(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A master-key request has no key row, so it bills to the default workspace.

    That is where `services/workspace_scope.py` lands every deployment-wide
    write, so narrowing that workspace narrows the operator's own requests too.
    Worth pinning: it is the one case where the workspace is not read off a key.
    """
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", _SEARCH_URL)
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_config(client, master_key_header, workspace_id, enabled=False)

    response, seen = _post_with_search_patched(
        client,
        master_key_header,
        {**_REQUEST, "metadata": {"user_id": test_user["user_id"]}},
    )

    assert response.status_code == 403, response.text
    assert seen.ran is False


def test_the_config_surface_needs_the_master_key(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
) -> None:
    workspace_id = _default_workspace_id(client, master_key_header)

    unauthenticated = client.get(f"/v1/workspaces/{workspace_id}/web-search")
    assert unauthenticated.status_code == 401

    # A working API key is not the master key, which is what this router gates on.
    with_an_api_key = client.get(f"/v1/workspaces/{workspace_id}/web-search", headers=api_key_header)
    assert with_an_api_key.status_code == 401


def test_a_ceiling_the_backend_could_never_honor_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    workspace_id = _default_workspace_id(client, master_key_header)

    response = client.put(
        f"/v1/workspaces/{workspace_id}/web-search",
        json={"enabled": True, "max_results": 500},
        headers=master_key_header,
    )

    assert response.status_code == 422


def test_a_config_read_that_fails_releases_the_budget_reservation(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A database error in admission must not stay debited against the caller.

    `resolve_request_context` reserves the estimate before this runs, and the
    web-search read is one more await that can fail some way other than an
    `HTTPException`. Structured as an A/B like the code-execution version next
    door, because `reserved == 0.0` would also hold if the estimate were zero,
    which would make this pass while guarding nothing. The control puts the same
    request under a budget smaller than the estimate: it is refused at the budget
    gate, which is only possible if pricing matched and the estimate really was
    nonzero.
    """
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", _SEARCH_URL)
    priced = client.post(
        "/v1/pricing",
        json={
            "model_key": "anthropic:claude-3-5-sonnet-20241022",
            "input_price_per_million": 3.0,
            "output_price_per_million": 15.0,
        },
        headers=master_key_header,
    )
    assert priced.status_code == 200, priced.text

    def _user(name: str, max_budget: float) -> str:
        budget_id = client.post("/v1/budgets", json={"max_budget": max_budget}, headers=master_key_header).json()[
            "budget_id"
        ]
        created = client.post(
            "/v1/users",
            json={"user_id": name, "budget_id": budget_id},
            headers=master_key_header,
        )
        assert created.status_code == 200, created.text
        return name

    def _post(user: str) -> Any:
        return client.post(
            "/v1/messages",
            json={**_REQUEST, "metadata": {"user_id": user}},
            headers=master_key_header,
        )

    # Control: a budget below the estimate refuses at the budget gate, proving
    # the estimate this test is about is nonzero.
    assert _post(_user("web-search-tiny-budget", 0.000_001)).status_code == 403

    funded = _user("web-search-refund", 100.0)

    async def failing_resolve(*_args: Any, **_kwargs: Any) -> None:
        raise SQLAlchemyError("connection lost mid-admission")

    monkeypatch.setattr(
        "gateway.api.routes._pipeline.resolve_workspace_web_search_config",
        failing_resolve,
    )
    # TestClient re-raises a server exception rather than rendering a 500, so the
    # failure arrives here; what matters is the state it left behind.
    with pytest.raises(SQLAlchemyError):
        _post(funded)

    session = db_session_factory()
    try:
        reserved = session.execute(
            text("SELECT reserved FROM users WHERE user_id = :user"), {"user": funded}
        ).scalar_one()
    finally:
        session.close()
    assert float(reserved) == 0.0, "the admission failure left the estimate debited"


def test_an_unresolvable_workspace_refuses_rather_than_searching(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The veto's defensive arm fails closed.

    Unreachable today: a standalone request with no session is refused before
    this, and `resolve_workspace_id` always answers, falling back to the default
    workspace. It is pinned anyway because the arm guards a veto, so falling
    through would hand web search to a workspace whose row refuses it, with
    nothing to notice.
    """
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", _SEARCH_URL)

    async def no_workspace(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr("gateway.api.routes._pipeline.resolve_workspace_id", no_workspace)
    # The organization-guardrail resolve (otari#654) fails closed on the same
    # condition and runs first, so without this the request would be refused
    # before the arm under test. Neutralized rather than asserted around,
    # because this case is about the web-search veto; the guardrail gate has its
    # own coverage in `test_organization_guardrail_enforcement.py`.
    monkeypatch.setattr(
        "gateway.api.routes._pipeline._resolve_organization_guardrails",
        AsyncMock(return_value=[]),
    )

    response, seen = _post_with_search_patched(client, api_key_header, _REQUEST)

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["error"]["message"] == "Web search configuration could not be resolved for this request"
    assert seen.ran is False


def _stored_search_tool(client: TestClient, master_key_header: dict[str, str], name: str) -> None:
    """Add a runtime search tool, the way an operator without a config file does.

    ``searxng`` because it is the one provider that needs no API key, so the
    fixture needs no ``OTARI_SECRET_KEY``.
    """
    created = client.post(
        "/v1/search-tools",
        json={"name": name, "provider": "searxng", "api_base": _SEARCH_URL},
        headers=master_key_header,
    )
    assert created.status_code == 201, created.text


def test_the_direct_search_endpoint_honors_the_same_veto(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """`POST /v1/search` is the other door into web search, and it must not stay open.

    A workspace that has turned search off has turned it off; leaving this
    endpoint unguarded would make the switch bypassable by any key in that
    workspace. The refusal is logged like every other one this route makes.
    """
    _stored_search_tool(client, master_key_header, "stub-search")
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_config(client, master_key_header, workspace_id, enabled=False)

    client.post("/v1/users", json={"user_id": "direct-search-user"}, headers=master_key_header)
    key = client.post(
        "/v1/keys",
        json={"key_name": "direct-search-key", "user_id": "direct-search-user"},
        headers=master_key_header,
    ).json()

    with patch("gateway.api.routes.search.run_search", new=AsyncMock()) as ran:
        response = client.post(
            "/v1/search/stub-search",
            json={"query": "anything"},
            headers={API_KEY_HEADER: f"Bearer {key['key']}"},
        )

    assert response.status_code == 403, response.text
    assert response.json()["detail"] == "web search is not enabled for this workspace"
    assert ran.await_count == 0

    rows = client.get(
        "/v1/usage",
        params={"user_id": "direct-search-user", "endpoint": "/v1/search"},
        headers=master_key_header,
    ).json()
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["status_code"] == 403


def test_the_direct_search_endpoint_is_unchanged_for_a_workspace_with_no_row(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The zero-rows requirement again, on this endpoint."""
    _stored_search_tool(client, master_key_header, "open-stub-search")
    client.post("/v1/users", json={"user_id": "open-search-user"}, headers=master_key_header)
    key = client.post(
        "/v1/keys",
        json={"key_name": "open-search-key", "user_id": "open-search-user"},
        headers=master_key_header,
    ).json()

    from gateway.services.search_backend import SearchHit, SearchOutcome

    outcome = SearchOutcome(results=[SearchHit(url="https://example.com", title="t", snippet="s")], cost_usd=0.0)
    with patch("gateway.api.routes.search.run_search", new=AsyncMock(return_value=outcome)):
        response = client.post(
            "/v1/search/open-stub-search",
            json={"query": "anything"},
            headers={API_KEY_HEADER: f"Bearer {key['key']}"},
        )

    assert response.status_code == 200, response.text
