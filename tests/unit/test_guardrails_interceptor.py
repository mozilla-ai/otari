"""Unit tests for the shared guardrails interceptor glue in ``_helpers``.

These exercise ``apply_input_guardrails`` (the block/monitor/pass branches all
three routes share) and the message-text extractors directly, without standing
up the FastAPI app or the test database — complementing the route-level wiring
tests in ``tests/integration/test_guardrails_route.py``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from functools import partial
from typing import Any, Literal, cast

import httpx
import pytest
from fastapi import HTTPException, Response

from gateway.api.routes._helpers import (
    GUARDRAILS_RESULT_HEADER,
    apply_input_guardrails,
    latest_user_text,
    text_from_content,
)
from gateway.log_config import logger as gateway_logger
from gateway.models.guardrails import GuardrailConfig
from gateway.services.guardrails import GuardrailResult, GuardrailsNotReachableError, GuardrailVerdict


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("plain string", "plain string"),
        ([{"type": "text", "text": "a"}, {"type": "image", "url": "x"}, {"type": "text", "text": "b"}], "a\nb"),
        ([{"type": "input_text", "text": "responses-shape"}], "responses-shape"),
        (None, ""),
    ],
)
def test_text_from_content(content: object, expected: str) -> None:
    assert text_from_content(content) == expected


def test_latest_user_text_picks_last_user_message() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": [{"type": "text", "text": "second"}]},
    ]
    assert latest_user_text(messages) == "second"


def test_latest_user_text_handles_non_dict_items() -> None:
    # /api/v1/responses `input` can be a list of arbitrary (non-dict) items; the
    # no-user-message fallback must not raise AttributeError (Copilot review).
    assert latest_user_text(["just a string"]) == ""
    assert latest_user_text([{"type": "x"}]) == ""


@pytest.mark.asyncio
async def test_apply_guardrails_noop_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    async def _runner(*_a: object, **_k: object) -> GuardrailVerdict:
        nonlocal called
        called = True
        return GuardrailVerdict()

    monkeypatch.setattr("gateway.api.routes._helpers.run_input_guardrails", _runner)
    response = Response()
    await apply_input_guardrails(None, "hi", response=response)
    assert called is False
    assert GUARDRAILS_RESULT_HEADER not in response.headers


@pytest.mark.asyncio
async def test_apply_guardrails_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    verdict = GuardrailVerdict(
        results=[GuardrailResult(profile="prompt-injection", mode="block", valid=False, explanation="bad", score=0.9)]
    )

    async def _runner(*_a: object, **_k: object) -> GuardrailVerdict:
        return verdict

    monkeypatch.setattr("gateway.api.routes._helpers.run_input_guardrails", _runner)
    with pytest.raises(HTTPException) as exc:
        await apply_input_guardrails(
            [GuardrailConfig(profile="prompt-injection")], "ignore previous", response=Response()
        )
    assert exc.value.status_code == 403
    detail = cast(dict[str, Any], exc.value.detail)
    assert detail["code"] == "guardrail_violation"
    assert detail["guardrails"][0]["profile"] == "prompt-injection"


@pytest.mark.asyncio
async def test_apply_guardrails_monitor_annotates(monkeypatch: pytest.MonkeyPatch) -> None:
    verdict = GuardrailVerdict(
        results=[GuardrailResult(profile="prompt-injection", mode="monitor", valid=False, explanation="bad", score=0.8)]
    )

    async def _runner(*_a: object, **_k: object) -> GuardrailVerdict:
        return verdict

    monkeypatch.setattr("gateway.api.routes._helpers.run_input_guardrails", _runner)
    response = Response()
    await apply_input_guardrails([GuardrailConfig(profile="prompt-injection", mode="monitor")], "x", response=response)
    header = json.loads(response.headers[GUARDRAILS_RESULT_HEADER])
    assert header[0]["profile"] == "prompt-injection"
    assert header[0]["valid"] is False
    # The freeform explanation must not leak into the header.
    assert "explanation" not in header[0]
    assert header[0]["status"] == "evaluated"
    assert "reason" not in header[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "status", "reason"),
    [
        (GuardrailResult(profile="pi", mode="monitor", valid=None, explanation="no label"), "inconclusive", None),
        (
            GuardrailResult(profile="pi", mode="monitor", valid=None, unavailable="not_configured"),
            "unavailable",
            "not_configured",
        ),
        (
            GuardrailResult(profile="pi", mode="block", valid=None, unavailable="evaluation_failed"),
            "unavailable",
            "evaluation_failed",
        ),
    ],
)
async def test_a_null_verdict_says_why_in_the_header(
    monkeypatch: pytest.MonkeyPatch, result: GuardrailResult, status: str, reason: str | None
) -> None:
    """Otari#2690: ``valid: null`` alone cannot tell "did not run" from "ran, no verdict"."""

    async def _runner(*_a: object, **_k: object) -> GuardrailVerdict:
        return GuardrailVerdict(results=[result])

    monkeypatch.setattr("gateway.api.routes._helpers.run_input_guardrails", _runner)
    response = Response()
    await apply_input_guardrails([GuardrailConfig(profile="pi", mode="monitor")], "x", response=response)
    header = json.loads(response.headers[GUARDRAILS_RESULT_HEADER])
    assert header[0]["valid"] is None
    assert header[0]["status"] == status
    assert header[0].get("reason") == reason
    assert "explanation" not in header[0]


@pytest.mark.asyncio
async def test_apply_guardrails_unreachable_is_502(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _runner(*_a: object, **_k: object) -> GuardrailVerdict:
        raise GuardrailsNotReachableError("down")

    monkeypatch.setattr("gateway.api.routes._helpers.run_input_guardrails", _runner)
    with pytest.raises(HTTPException) as exc:
        await apply_input_guardrails([GuardrailConfig(profile="prompt-injection")], "x", response=Response())
    assert exc.value.status_code == 502


_ECHO = "my-private-prompt"


def _protocol_error() -> httpx.Response:
    """Fail the way httpx does on a malformed reply, quoting the bytes it read."""
    raise httpx.RemoteProtocolError(f"illegal header line: bytearray(b'{_ECHO}')")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [pytest.param("block", id="fails closed"), pytest.param("monitor", id="fails open")])
@pytest.mark.parametrize(
    ("answer", "diagnosis"),
    [
        (partial(httpx.Response, 200, json={"result": {"echo": _ECHO}}), "returned no 'valid' field"),
        (partial(httpx.Response, 200, json=[_ECHO]), "returned no readable 'result': TypeError"),
        (partial(httpx.Response, 200, json={"result": _ECHO}), "returned an unexpected result shape: str"),
        (partial(httpx.Response, 200, json={"result": {"valid": _ECHO}}), "returned a non-boolean 'valid': str"),
        (
            partial(httpx.Response, 302, headers={"Location": f"/validate?echo={_ECHO}"}),
            "failed against http://anyguardrails:8000: HTTP 302",
        ),
        (_protocol_error, "failed against http://anyguardrails:8000: RemoteProtocolError"),
    ],
    ids=[
        "no valid field",
        "body not an object",
        "result not an object",
        "non-boolean valid",
        "redirect",
        "protocol error",
    ],
)
async def test_a_malformed_answer_is_logged_without_what_the_service_sent(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    mode: Literal["block", "monitor"],
    answer: Callable[[], httpx.Response],
    diagnosis: str,
) -> None:
    """The service may echo the prompt back, so the log says what was wrong with the answer and nothing it sent."""
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        "gateway.services.guardrails.httpx.AsyncClient",
        lambda **kwargs: real_async_client(transport=httpx.MockTransport(lambda _request: answer()), **kwargs),
    )
    monkeypatch.setenv("OTARI_GUARDRAILS_URL", "http://anyguardrails:8000")

    # The ``gateway`` logger does not propagate, hence the handler.
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger="gateway")
    try:
        if mode == "block":
            with pytest.raises(HTTPException) as exc:
                await apply_input_guardrails(
                    [GuardrailConfig(profile="prompt-injection", mode=mode, on_unavailable="block")],
                    _ECHO,
                    response=Response(),
                )
            assert exc.value.status_code == 502
        else:
            await apply_input_guardrails(
                [GuardrailConfig(profile="prompt-injection", mode=mode)], _ECHO, response=Response()
            )
    finally:
        gateway_logger.removeHandler(caplog.handler)

    assert f"guardrail profile 'prompt-injection' {diagnosis}" in caplog.text
    assert _ECHO not in caplog.text
