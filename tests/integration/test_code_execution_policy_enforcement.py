"""What a workspace's code-execution policy does to a request that asks for the sandbox.

The management surface is covered in
``test_workspace_code_execution_policy.py``; this is the other half of #657's
Definition of Done, the request path honoring what the policy says. Every case
goes through ``/v1/messages`` with the sandbox backend and the tool loop
patched out, so what is asserted is admission and the values handed to the
loop, not any execution.
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

_SANDBOX_URL = "http://127.0.0.1:9999/sandbox"
_REQUEST = {
    "model": "anthropic:claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "compute"}],
    "max_tokens": 100,
    "tools": [{"type": "otari_code_execution"}],
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


def _set_policy(
    client: TestClient,
    master_key_header: dict[str, str],
    workspace_id: str,
    **policy: Any,
) -> dict[str, Any]:
    response = client.put(
        f"/v1/workspaces/{workspace_id}/code-execution-policy",
        json=policy,
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    stored: dict[str, Any] = response.json()
    return stored


class _Dispatch:
    """What the patched sandbox/loop saw for one request."""

    def __init__(self) -> None:
        self.backend_kwargs: dict[str, Any] = {}
        self.max_iterations: int | None = None


def _post_with_sandbox_patched(
    client: TestClient, headers: dict[str, str], body: dict[str, Any]
) -> tuple[Any, _Dispatch]:
    seen = _Dispatch()

    async def fake_loop(
        *, completion_kwargs: Any, pool: Any, max_iterations: int, emit_native_web_search: bool = False
    ) -> MessageResponse:
        seen.max_iterations = max_iterations
        return _text_response("via-sandbox-loop")

    def fake_sandbox(**kwargs: Any) -> Any:
        seen.backend_kwargs = kwargs
        backend = AsyncMock()
        backend.purpose_hints = lambda: []
        return AsyncMock(__aenter__=AsyncMock(return_value=backend), __aexit__=AsyncMock(return_value=None))

    with (
        patch("gateway.api.routes.messages.anthropic_tool_loop", new=fake_loop),
        patch("gateway.api.routes._pipeline.SandboxBackend", new=fake_sandbox),
    ):
        response = client.post("/v1/messages", json=body, headers=headers)
    return response, seen


def test_no_policy_leaves_the_request_exactly_as_it_was(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The zero-rows requirement: a deployment that configured nothing is unchanged."""
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)

    response, seen = _post_with_sandbox_patched(client, api_key_header, _REQUEST)

    assert response.status_code == 200
    assert seen.max_iterations == 10, "the request's own default iteration count"
    assert seen.backend_kwargs["timeout_s"] == 60.0, "the deployment's own execution budget"


def test_a_disabled_workspace_is_refused_before_the_provider_is_called(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_policy(client, master_key_header, workspace_id, enabled=False)

    response, seen = _post_with_sandbox_patched(client, api_key_header, _REQUEST)

    assert response.status_code == 403
    # Anthropic-shaped body, the same mapping every other refusal on this route gets.
    detail = response.json()["detail"]
    assert detail["error"]["message"] == "code execution is not enabled for this workspace"
    assert seen.max_iterations is None, "the tool loop must never have run"


def test_a_disabled_workspace_still_serves_a_request_that_asks_for_no_sandbox(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The veto is over the tool, not over the workspace's traffic."""
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_policy(client, master_key_header, workspace_id, enabled=False)

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


def test_the_workspace_ceilings_lower_the_loop_and_the_execution_budget(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_policy(client, master_key_header, workspace_id, enabled=True, max_iterations=2, exec_timeout_s=7)

    response, seen = _post_with_sandbox_patched(client, api_key_header, _REQUEST)

    assert response.status_code == 200
    assert seen.max_iterations == 2
    assert seen.backend_kwargs["timeout_s"] == 7.0


def test_a_request_asking_for_less_than_the_workspace_allows_keeps_its_own_number(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ceiling is a floor operation, so the stricter of the two wins."""
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_policy(client, master_key_header, workspace_id, enabled=True, max_iterations=8)

    response, seen = _post_with_sandbox_patched(
        client,
        api_key_header,
        {**_REQUEST, "max_tool_iterations": 3},
    )

    assert response.status_code == 200
    assert seen.max_iterations == 3


def test_the_workspace_hint_fills_in_only_when_the_request_gave_none(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_policy(client, master_key_header, workspace_id, enabled=True, default_purpose_hint="workspace hint")

    _, defaulted = _post_with_sandbox_patched(client, api_key_header, _REQUEST)
    assert defaulted.backend_kwargs["purpose_hint"] == "workspace hint"

    _, overridden = _post_with_sandbox_patched(
        client,
        api_key_header,
        {**_REQUEST, "tools": [{"type": "otari_code_execution", "purpose_hint": "request hint"}]},
    )
    assert overridden.backend_kwargs["purpose_hint"] == "request hint"


def test_clearing_the_policy_puts_the_request_back_where_it_started(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_policy(client, master_key_header, workspace_id, enabled=False)

    cleared = client.delete(
        f"/v1/workspaces/{workspace_id}/code-execution-policy",
        headers=master_key_header,
    )
    assert cleared.status_code == 200
    assert cleared.json()["configured"] is False

    response, seen = _post_with_sandbox_patched(client, api_key_header, _REQUEST)

    assert response.status_code == 200
    assert seen.max_iterations == 10


def test_a_streaming_request_gets_the_same_ceilings(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The streaming dispatch builds its own backend, so it needs its own case.

    A ceiling that reached only the non-streaming path would leave the longest
    running requests, the streamed ones, unnarrowed.
    """
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_policy(client, master_key_header, workspace_id, enabled=True, max_iterations=2, exec_timeout_s=9)

    seen = _Dispatch()

    async def fake_loop_stream(
        *, completion_kwargs: Any, pool: Any, max_iterations: int, emit_native_web_search: bool = False
    ) -> AsyncIterator[MessageStreamEvent]:
        seen.max_iterations = max_iterations
        yield MessageStopEvent(type="message_stop")

    def fake_sandbox(**kwargs: Any) -> Any:
        seen.backend_kwargs = kwargs
        backend = AsyncMock()
        backend.purpose_hints = lambda: []
        backend.__aenter__ = AsyncMock(return_value=backend)
        backend.__aexit__ = AsyncMock(return_value=None)
        return backend

    with (
        patch("gateway.api.routes.messages.anthropic_tool_loop_stream", new=fake_loop_stream),
        patch("gateway.api.routes._pipeline.SandboxBackend", new=fake_sandbox),
    ):
        response = client.post("/v1/messages", json={**_REQUEST, "stream": True}, headers=api_key_header)

    assert response.status_code == 200, response.text
    assert seen.max_iterations == 2
    assert seen.backend_kwargs["timeout_s"] == 9.0


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
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_policy(client, master_key_header, workspace_id, enabled=False)

    # A master-key request names the user it bills to, which is the only extra
    # the route asks of it.
    response, seen = _post_with_sandbox_patched(
        client,
        master_key_header,
        {**_REQUEST, "metadata": {"user_id": test_user["user_id"]}},
    )

    assert response.status_code == 403, response.text
    assert seen.max_iterations is None


def test_the_policy_surface_needs_the_master_key(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
) -> None:
    workspace_id = _default_workspace_id(client, master_key_header)

    unauthenticated = client.get(f"/v1/workspaces/{workspace_id}/code-execution-policy")
    assert unauthenticated.status_code == 401

    # A working API key is not the master key, which is what this router gates on.
    with_an_api_key = client.get(
        f"/v1/workspaces/{workspace_id}/code-execution-policy",
        headers=api_key_header,
    )
    assert with_an_api_key.status_code == 401


def test_a_limit_the_deployment_could_never_honor_is_refused(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    workspace_id = _default_workspace_id(client, master_key_header)

    response = client.put(
        f"/v1/workspaces/{workspace_id}/code-execution-policy",
        json={"enabled": True, "exec_timeout_s": 600},
        headers=master_key_header,
    )

    assert response.status_code == 422


def test_a_policy_read_that_fails_releases_the_budget_reservation(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A database error in admission must not stay debited against the caller.

    `resolve_request_context` reserves the estimate before this runs, and the
    policy read is one more await that can fail some way other than an
    `HTTPException`. Before the broad release arm, such a failure left the hold
    on `users.reserved` until the next budget reset, and forever for a budget
    with no reset period.

    Structured as an A/B like `test_gateway_rejection_logging`'s own stranded
    reservation test, because `reserved == 0.0` would also hold if the estimate
    were zero, which would make this pass while guarding nothing. The control
    puts the same request under a budget smaller than the estimate: it is
    refused at the budget gate, which is only possible if pricing matched and
    the estimate really was nonzero.
    """
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
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
        budget_id = client.post(
            "/v1/budgets", json={"max_budget": max_budget}, headers=master_key_header
        ).json()["budget_id"]
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
    assert _post(_user("code-exec-tiny-budget", 0.000_001)).status_code == 403

    funded = _user("code-exec-refund", 100.0)

    async def failing_resolve(*_args: Any, **_kwargs: Any) -> None:
        raise SQLAlchemyError("connection lost mid-admission")

    monkeypatch.setattr(
        "gateway.api.routes._pipeline.resolve_workspace_code_execution_policy",
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


def test_an_unresolvable_workspace_refuses_rather_than_running_the_sandbox(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The veto's defensive arm fails closed.

    Unreachable today: a standalone request with no session is refused before
    this, and `resolve_workspace_id` always answers, falling back to the default
    workspace. It is pinned anyway because the arm guards a veto, so falling
    through would hand code execution to a workspace whose row refuses it, with
    nothing to notice. `_resolve_mcp_server_ids` refuses at the same condition.
    """
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)

    async def no_workspace(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr("gateway.api.routes._pipeline.resolve_workspace_id", no_workspace)
    # The organization-guardrail resolve (otari#654) fails closed on the same
    # condition and runs first, so without this the request would be refused
    # before the arm under test. Neutralized rather than asserted around,
    # because this case is about the sandbox veto; the guardrail gate has its
    # own coverage in `test_organization_guardrail_enforcement.py`.
    monkeypatch.setattr(
        "gateway.api.routes._pipeline._resolve_organization_guardrails",
        AsyncMock(return_value=[]),
    )

    response, seen = _post_with_sandbox_patched(client, api_key_header, _REQUEST)

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["error"]["message"] == "Code execution policy could not be resolved for this request"
    assert seen.max_iterations is None, "the tool loop must never have run"


# ---------------------------------------------------------------------------
# The sandbox image and the exposed tool set (#740)
# ---------------------------------------------------------------------------

_IMAGE = "mzdotai/otari-sandbox-container:latest"
_OTHER_IMAGE = "ghcr.io/acme/sandbox:2"


def test_no_policy_and_no_deployment_image_asks_the_backend_for_nothing(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)

    response, seen = _post_with_sandbox_patched(client, api_key_header, _REQUEST)

    assert response.status_code == 200
    assert seen.backend_kwargs["image"] is None
    assert seen.backend_kwargs["allowed_tools"] is None


def test_the_deployment_image_is_used_when_the_workspace_pins_none(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback half: a policy that says nothing about images changes nothing."""
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    monkeypatch.setenv("OTARI_SANDBOX_SESSION_IMAGE", _IMAGE)
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_policy(client, master_key_header, workspace_id, enabled=True)

    response, seen = _post_with_sandbox_patched(client, api_key_header, _REQUEST)

    assert response.status_code == 200
    assert seen.backend_kwargs["image"] == _IMAGE


def test_the_workspace_image_wins_over_the_deployments(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    monkeypatch.setenv("OTARI_SANDBOX_SESSION_IMAGE", _IMAGE)
    monkeypatch.setenv("OTARI_SANDBOX_ALLOWED_SESSION_IMAGES", _OTHER_IMAGE)
    workspace_id = _default_workspace_id(client, master_key_header)
    stored = _set_policy(client, master_key_header, workspace_id, enabled=True, image=_OTHER_IMAGE)
    assert stored["image"] == _OTHER_IMAGE

    response, seen = _post_with_sandbox_patched(client, api_key_header, _REQUEST)

    assert response.status_code == 200
    assert seen.backend_kwargs["image"] == _OTHER_IMAGE


def test_an_image_the_operator_no_longer_allows_refuses_the_request(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The write-time guard is not the only one: an operator can shrink the list later.

    Refusing beats quietly falling back to the deployment image, which would
    leave a workspace believing its pin was in force.
    """
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    monkeypatch.setenv("OTARI_SANDBOX_ALLOWED_SESSION_IMAGES", _OTHER_IMAGE)
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_policy(client, master_key_header, workspace_id, enabled=True, image=_OTHER_IMAGE)

    monkeypatch.delenv("OTARI_SANDBOX_ALLOWED_SESSION_IMAGES")
    response, seen = _post_with_sandbox_patched(client, api_key_header, _REQUEST)

    assert response.status_code == 403
    assert seen.max_iterations is None, "the tool loop must never have run"


def test_an_image_the_operator_never_allowed_cannot_be_stored(
    client: TestClient,
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    workspace_id = _default_workspace_id(client, master_key_header)

    response = client.put(
        f"/v1/workspaces/{workspace_id}/code-execution-policy",
        json={"enabled": True, "image": "ghcr.io/attacker/pwn:latest"},
        headers=master_key_header,
    )

    assert response.status_code == 400


def test_a_tool_list_that_keeps_code_execution_narrows_the_backend(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_policy(client, master_key_header, workspace_id, enabled=True, tools=["code_execution"])

    response, seen = _post_with_sandbox_patched(client, api_key_header, _REQUEST)

    assert response.status_code == 200
    assert seen.backend_kwargs["allowed_tools"] == frozenset({"code_execution"})


def test_a_tool_list_this_deployment_cannot_run_is_refused_at_the_write(
    client: TestClient,
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The management surface refuses it, so no request ever meets one through the API."""
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    workspace_id = _default_workspace_id(client, master_key_header)

    response = client.put(
        f"/v1/workspaces/{workspace_id}/code-execution-policy",
        json={"enabled": True, "tools": ["bash_code_execution"]},
        headers=master_key_header,
    )

    assert response.status_code == 400


def test_a_stored_tool_list_without_code_execution_refuses_rather_than_serving_nothing(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    db_session_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The admission backstop, reached by writing the row past the service.

    Unreachable through the API since the write refuses it, which is why the row
    is planted directly: what this pins is that the *request* path fails closed
    too, for the row that was valid when written and stopped being runnable when
    a deployment's backend changed under it. Handing the model a sandbox that
    advertises no tools would return a perfectly successful response that
    silently never ran any code, which is the failure a policy exists to make
    loud.
    """
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_policy(client, master_key_header, workspace_id, enabled=True)

    session = db_session_factory()
    try:
        session.execute(
            text("UPDATE workspace_code_execution_policies SET tools = :tools WHERE workspace_id = :ws"),
            {"tools": '["bash_code_execution"]', "ws": workspace_id},
        )
        session.commit()
    finally:
        session.close()

    response, seen = _post_with_sandbox_patched(client, api_key_header, _REQUEST)

    assert response.status_code == 403
    assert seen.max_iterations is None, "the tool loop must never have run"


def test_a_streaming_request_gets_the_same_image_and_tool_set(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The eager-open streaming path builds its own backend, so it needs its own case."""
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    monkeypatch.setenv("OTARI_SANDBOX_ALLOWED_SESSION_IMAGES", _IMAGE)
    workspace_id = _default_workspace_id(client, master_key_header)
    _set_policy(
        client,
        master_key_header,
        workspace_id,
        enabled=True,
        image=_IMAGE,
        tools=["code_execution"],
    )

    seen: dict[str, Any] = {}

    def fake_sandbox(**kwargs: Any) -> Any:
        seen.update(kwargs)
        backend = AsyncMock()
        backend.purpose_hints = lambda: []
        backend.__aenter__ = AsyncMock(return_value=backend)
        backend.__aexit__ = AsyncMock(return_value=None)
        return backend

    async def fake_stream(*_args: Any, **_kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
        yield cast(Any, MessageStopEvent(type="message_stop"))

    with (
        patch("gateway.api.routes._pipeline.SandboxBackend", new=fake_sandbox),
        patch("gateway.api.routes.messages.anthropic_tool_loop_stream", new=fake_stream),
    ):
        response = client.post("/v1/messages", json={**_REQUEST, "stream": True}, headers=api_key_header)

    assert response.status_code == 200
    assert seen["image"] == _IMAGE
    assert seen["allowed_tools"] == frozenset({"code_execution"})
