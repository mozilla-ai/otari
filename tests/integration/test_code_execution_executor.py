"""The executor decision on the request path: who runs a provider-native code declaration.

The pure logic is covered by ``tests/unit/test_code_executor.py``. These pin the
wiring in ``prepare_gateway_tools``: which declarations are claimed, how the
deployment default, the workspace pin and the request header compose, and what a
claimed request hands the tool loop.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.messages import MessageResponse, MessageUsage, TextBlock
from fastapi.testclient import TestClient
from openai.types.responses import Response, ResponseUsage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from gateway.core.config import API_ROOT

_SANDBOX_URL = "http://127.0.0.1:9999/sandbox"
_ANTHROPIC = "anthropic:claude-3-5-sonnet-20241022"
_OPENAI = "openai:gpt-4o-mini"
_DATED = {"type": "code_execution_20250825", "name": "code_execution"}
_BARE = {"type": "code_execution"}
_INTERPRETER = {"type": "code_interpreter"}
_HEADER = "X-Otari-Code-Execution"


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


def _responses_response() -> Response:
    return Response(
        id="resp_test",
        created_at=0.0,
        model="fake",
        object="response",
        status=cast(Any, "completed"),
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage=ResponseUsage(
            input_tokens=5,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens=2,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            total_tokens=7,
        ),
        error=None,
        incomplete_details=None,
        instructions=None,
        metadata=None,
        temperature=None,
        top_p=None,
    )


class _Seen:
    """What one request did: forwarded to the provider, or claimed by the sandbox loop."""

    def __init__(self) -> None:
        self.provider_kwargs: dict[str, Any] | None = None
        self.loop_kwargs: dict[str, Any] | None = None
        self.loop_extra: dict[str, Any] | None = None
        self.backend_kwargs: dict[str, Any] | None = None

    @property
    def forwarded_tool_types(self) -> set[str]:
        assert self.provider_kwargs is not None, "the provider was never called"
        return {tool["type"] for tool in self.provider_kwargs.get("tools") or []}


def _post_messages(client: TestClient, headers: dict[str, str], body: dict[str, Any]) -> tuple[Any, _Seen]:
    seen = _Seen()

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        seen.provider_kwargs = kwargs
        return _text_response("via-provider")

    async def fake_loop(*, completion_kwargs: Any, pool: Any, max_iterations: int, **extra: Any) -> MessageResponse:
        seen.loop_kwargs = completion_kwargs
        seen.loop_extra = extra
        return _text_response("via-sandbox-loop")

    def fake_sandbox(**kwargs: Any) -> Any:
        seen.backend_kwargs = kwargs
        backend = AsyncMock()
        backend.purpose_hints = lambda: []
        return AsyncMock(__aenter__=AsyncMock(return_value=backend), __aexit__=AsyncMock(return_value=None))

    with (
        patch("gateway.api.routes.messages.amessages", new=fake_amessages),
        patch("gateway.api.routes.messages.anthropic_tool_loop", new=fake_loop),
        patch("gateway.api.routes._pipeline.SandboxBackend", new=fake_sandbox),
    ):
        response = client.post(f"{API_ROOT}/messages", json=body, headers=headers)
    return response, seen


def _post_responses(client: TestClient, headers: dict[str, str], body: dict[str, Any]) -> tuple[Any, _Seen]:
    seen = _Seen()

    async def fake_aresponses(**kwargs: Any) -> Response:
        seen.provider_kwargs = kwargs
        return _responses_response()

    async def fake_loop(*, completion_kwargs: Any, pool: Any, max_iterations: int, **extra: Any) -> Response:
        seen.loop_kwargs = completion_kwargs
        seen.loop_extra = extra
        return _responses_response()

    def fake_sandbox(**kwargs: Any) -> Any:
        seen.backend_kwargs = kwargs
        backend = AsyncMock()
        backend.purpose_hints = lambda: []
        return AsyncMock(__aenter__=AsyncMock(return_value=backend), __aexit__=AsyncMock(return_value=None))

    with (
        patch("gateway.api.routes.responses.aresponses", new=fake_aresponses),
        patch("gateway.api.routes.responses.responses_tool_loop", new=fake_loop),
        patch("gateway.api.routes._pipeline.SandboxBackend", new=fake_sandbox),
    ):
        response = client.post(f"{API_ROOT}/responses", json=body, headers=headers)
    return response, seen


def _messages_body(model: str, *tools: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "compute"}],
        "max_tokens": 100,
        "tools": list(tools),
    }


def _default_workspace_id(client: TestClient, master_key_header: dict[str, str]) -> str:
    listed = client.get(f"{API_ROOT}/workspaces", headers=master_key_header)
    assert listed.status_code == 200
    workspace_id: str = listed.json()["data"][0]["id"]
    return workspace_id


def _pin_executor(client: TestClient, master_key_header: dict[str, str], executor: str) -> None:
    workspace_id = _default_workspace_id(client, master_key_header)
    response = client.put(
        f"{API_ROOT}/workspaces/{workspace_id}/code-execution-policy",
        json={"enabled": True, "executor": executor},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    assert response.json()["executor"] == executor


# --- auto: the default -----------------------------------------------------------------


def test_auto_leaves_anthropics_own_declaration_with_anthropic(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case an upgrade must not change: the provider runs what it runs natively."""
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)

    response, seen = _post_messages(client, api_key_header, _messages_body(_ANTHROPIC, _DATED))

    assert response.status_code == 200, response.text
    assert seen.forwarded_tool_types == {"code_execution_20250825"}
    assert seen.loop_kwargs is None


def test_auto_brings_anthropics_declaration_here_for_a_model_that_cannot_run_it(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The transparent model swap: same request, other model, the sandbox runs the code."""
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)

    response, seen = _post_messages(client, api_key_header, _messages_body(_OPENAI, _DATED))

    assert response.status_code == 200, response.text
    assert seen.provider_kwargs is None
    assert seen.loop_kwargs is not None
    assert "tools" not in seen.loop_kwargs or not seen.loop_kwargs["tools"], "the claimed declaration was forwarded"
    # The caller spoke Anthropic's vocabulary, so it is answered in it.
    assert seen.loop_extra is not None
    assert seen.loop_extra.get("emit_native_code_execution") is True


def test_auto_claims_the_bare_keyword_which_no_provider_owns(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)

    response, seen = _post_messages(client, api_key_header, _messages_body(_ANTHROPIC, _BARE))

    assert response.status_code == 200, response.text
    assert seen.loop_kwargs is not None
    # The bare form implies no native response shape, so the plain result is kept.
    assert seen.loop_extra is not None
    assert "emit_native_code_execution" not in seen.loop_extra


def test_without_a_sandbox_a_provider_declaration_is_forwarded_and_nothing_else_happens(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OTARI_SANDBOX_URL", raising=False)

    response, seen = _post_messages(client, api_key_header, _messages_body(_OPENAI, _BARE))

    assert response.status_code == 200, response.text
    assert seen.forwarded_tool_types == {"code_execution"}


# --- the request header ----------------------------------------------------------------


def test_the_header_can_bring_anthropics_declaration_here_even_for_anthropic(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)

    response, seen = _post_messages(
        client, {**api_key_header, _HEADER: "otari"}, _messages_body(_ANTHROPIC, _DATED)
    )

    assert response.status_code == 200, response.text
    assert seen.loop_kwargs is not None
    assert seen.loop_extra is not None
    assert seen.loop_extra.get("emit_native_code_execution") is True


def test_the_header_can_leave_a_claimed_keyword_with_the_provider(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)

    response, seen = _post_messages(
        client, {**api_key_header, _HEADER: "provider"}, _messages_body(_ANTHROPIC, _BARE)
    )

    assert response.status_code == 200, response.text
    assert seen.forwarded_tool_types == {"code_execution"}


def test_a_header_outside_the_vocabulary_is_refused(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)

    response, _ = _post_messages(client, {**api_key_header, _HEADER: "anthropic"}, _messages_body(_ANTHROPIC, _DATED))

    assert response.status_code == 400
    assert response.json()["detail"]["error"]["type"] == "invalid_request_error"
    assert _HEADER in response.json()["detail"]["error"]["message"]


def test_asking_for_otari_without_a_sandbox_is_refused(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OTARI_SANDBOX_URL", raising=False)

    response, _ = _post_messages(client, {**api_key_header, _HEADER: "otari"}, _messages_body(_ANTHROPIC, _DATED))

    assert response.status_code == 400
    assert "no sandbox is configured" in response.json()["detail"]["error"]["message"]


# --- the deployment default ---------------------------------------------------------------


def test_a_provider_default_forwards_every_provider_declaration(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    monkeypatch.setenv("OTARI_CODE_EXECUTION_EXECUTOR", "provider")

    response, seen = _post_messages(client, api_key_header, _messages_body(_OPENAI, _DATED))

    assert response.status_code == 200, response.text
    assert seen.forwarded_tool_types == {"code_execution_20250825"}


def test_the_explicit_type_is_always_the_gateways_whatever_the_default_says(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    monkeypatch.setenv("OTARI_CODE_EXECUTION_EXECUTOR", "provider")

    response, seen = _post_messages(
        client, api_key_header, _messages_body(_ANTHROPIC, {"type": "otari_code_execution"})
    )

    assert response.status_code == 200, response.text
    assert seen.loop_kwargs is not None
    assert seen.loop_extra is not None
    assert "emit_native_code_execution" not in seen.loop_extra


def test_an_otari_default_claims_anthropics_declaration_for_anthropic(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    monkeypatch.setenv("OTARI_CODE_EXECUTION_EXECUTOR", "otari")

    response, seen = _post_messages(client, api_key_header, _messages_body(_ANTHROPIC, _DATED))

    assert response.status_code == 200, response.text
    assert seen.loop_kwargs is not None
    assert seen.loop_extra is not None
    assert seen.loop_extra.get("emit_native_code_execution") is True


# --- the workspace pin ---------------------------------------------------------------------


def test_a_workspace_pin_overrides_the_deployment_default(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    _pin_executor(client, master_key_header, "otari")

    response, seen = _post_messages(client, api_key_header, _messages_body(_ANTHROPIC, _DATED))

    assert response.status_code == 200, response.text
    assert seen.loop_kwargs is not None


def test_a_header_that_disagrees_with_the_pin_is_refused(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    _pin_executor(client, master_key_header, "otari")

    response, _ = _post_messages(client, {**api_key_header, _HEADER: "provider"}, _messages_body(_ANTHROPIC, _DATED))

    assert response.status_code == 403
    assert "pins" in response.json()["detail"]["error"]["message"] or "decides" in (
        response.json()["detail"]["error"]["message"]
    )


def test_the_pin_does_not_refuse_a_header_over_the_explicit_otari_tool(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``otari_code_execution`` names no provider tool, so the pin has nothing to say.

    Without the header the same body runs on the sandbox whatever the pin is;
    refusing it only because the header spelled that out would 403 a request
    that is otherwise served.
    """
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    _pin_executor(client, master_key_header, "provider")
    body = _messages_body(_ANTHROPIC, {"type": "otari_code_execution"})

    without, _ = _post_messages(client, api_key_header, body)
    assert without.status_code == 200, without.text

    with_header, seen = _post_messages(client, {**api_key_header, _HEADER: "otari"}, body)

    assert with_header.status_code == 200, with_header.text
    assert seen.loop_kwargs is not None


def test_a_header_that_agrees_with_the_pin_is_fine(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    _pin_executor(client, master_key_header, "provider")

    response, seen = _post_messages(client, {**api_key_header, _HEADER: "provider"}, _messages_body(_OPENAI, _DATED))

    assert response.status_code == 200, response.text
    assert seen.forwarded_tool_types == {"code_execution_20250825"}


def test_a_provider_pin_keeps_a_disabled_veto_out_of_the_way(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``enabled=False`` vetoes the sandbox; a declaration the provider runs is not the sandbox."""
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)
    workspace_id = _default_workspace_id(client, master_key_header)
    stored = client.put(
        f"{API_ROOT}/workspaces/{workspace_id}/code-execution-policy",
        json={"enabled": False, "executor": "provider"},
        headers=master_key_header,
    )
    assert stored.status_code == 200, stored.text

    response, seen = _post_messages(client, api_key_header, _messages_body(_OPENAI, _DATED))

    assert response.status_code == 200, response.text
    assert seen.forwarded_tool_types == {"code_execution_20250825"}


def test_the_policy_refuses_an_executor_outside_the_vocabulary(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    workspace_id = _default_workspace_id(client, master_key_header)
    response = client.put(
        f"{API_ROOT}/workspaces/{workspace_id}/code-execution-policy",
        json={"enabled": True, "executor": "anthropic"},
        headers=master_key_header,
    )
    assert response.status_code == 422


# --- two declarations in one request ----------------------------------------------------------


def test_the_explicit_type_beside_a_claimed_keyword_is_folded_in(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same request said twice, so one sandbox and the hint the keyword could not carry."""
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)

    response, seen = _post_messages(
        client,
        api_key_header,
        _messages_body(_OPENAI, {"type": "otari_code_execution", "purpose_hint": "Show your working"}, _DATED),
    )

    assert response.status_code == 200, response.text
    assert seen.backend_kwargs is not None
    assert seen.backend_kwargs["purpose_hint"] == "Show your working"
    assert seen.loop_extra is not None
    assert seen.loop_extra.get("emit_native_code_execution") is True


def test_the_explicit_type_beside_a_keyword_the_provider_keeps_is_still_two_sandboxes(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)

    response, _ = _post_messages(
        client, api_key_header, _messages_body(_ANTHROPIC, {"type": "otari_code_execution"}, _DATED)
    )

    assert response.status_code == 400
    assert "cannot be combined with a provider-native" in response.json()["detail"]["error"]["message"]


# --- Responses ----------------------------------------------------------------------------------


def test_responses_leaves_openais_interpreter_with_openai(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)

    response, seen = _post_responses(
        client, api_key_header, {"model": _OPENAI, "input": "compute", "tools": [_INTERPRETER]}
    )

    assert response.status_code == 200, response.text
    assert seen.forwarded_tool_types == {"code_interpreter"}


def test_responses_answers_a_claimed_interpreter_in_openais_vocabulary(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)

    response, seen = _post_responses(
        client, {**api_key_header, _HEADER: "otari"}, {"model": _OPENAI, "input": "compute", "tools": [_INTERPRETER]}
    )

    assert response.status_code == 200, response.text
    assert seen.loop_kwargs is not None
    assert seen.loop_extra is not None
    assert seen.loop_extra.get("emit_native_code_execution") is True


def test_responses_brings_anthropics_words_here_but_answers_plainly(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anthropic's keyword is not native on Responses, and its blocks do not exist there."""
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)

    response, seen = _post_responses(client, api_key_header, {"model": _OPENAI, "input": "compute", "tools": [_DATED]})

    assert response.status_code == 200, response.text
    assert seen.loop_kwargs is not None
    assert seen.loop_extra is not None
    assert "emit_native_code_execution" not in seen.loop_extra


def test_responses_header_outside_the_vocabulary_is_refused_in_its_own_envelope(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)

    response, _ = _post_responses(
        client, {**api_key_header, _HEADER: "nobody"}, {"model": _OPENAI, "input": "compute", "tools": [_INTERPRETER]}
    )

    assert response.status_code == 400
    assert _HEADER in response.json()["detail"]
