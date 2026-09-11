"""Unit tests for the shared guardrails interceptor glue in ``_helpers``.

These exercise ``apply_input_guardrails`` (the block/monitor/pass branches all
three routes share) and the message-text extractors directly, without standing
up the FastAPI app or the test database — complementing the route-level wiring
tests in ``tests/integration/test_guardrails_route.py``.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from fastapi import HTTPException, Response

from gateway.api.routes._helpers import (
    GUARDRAILS_RESULT_HEADER,
    apply_input_guardrails,
    latest_user_text,
    text_from_content,
)
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


@pytest.mark.asyncio
async def test_apply_guardrails_unreachable_is_502(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _runner(*_a: object, **_k: object) -> GuardrailVerdict:
        raise GuardrailsNotReachableError("down")

    monkeypatch.setattr("gateway.api.routes._helpers.run_input_guardrails", _runner)
    with pytest.raises(HTTPException) as exc:
        await apply_input_guardrails([GuardrailConfig(profile="prompt-injection")], "x", response=Response())
    assert exc.value.status_code == 502
