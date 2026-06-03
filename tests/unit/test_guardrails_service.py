"""Unit tests for the guardrails service client (``run_input_guardrails``).

Stubs the guardrails service ``POST /validate`` contract with an
``httpx.MockTransport`` so we test the verdict logic without a live container.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from gateway.models.guardrails import GuardrailConfig
from gateway.services.guardrails import GuardrailsNotReachableError, run_input_guardrails

_URL = "http://anyguardrails:8000"


def _patch_transport(monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]) -> None:
    """Replace the module's ``httpx.AsyncClient`` with one backed by ``handler``."""
    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient  # capture before patching to avoid recursion

    def factory(*_args: object, **_kwargs: object) -> httpx.AsyncClient:
        return real_async_client(transport=transport)

    monkeypatch.setattr("gateway.services.guardrails.httpx.AsyncClient", factory)


def _result_handler(result: dict[str, object]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/validate"
        return httpx.Response(200, json={"profile": "prompt-injection", "result": result})

    return handler


@pytest.mark.asyncio
async def test_flagged_input_blocks_in_block_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_transport(monkeypatch, _result_handler({"valid": False, "explanation": "injection", "score": 0.97}))
    verdict = await run_input_guardrails(
        [GuardrailConfig(profile="prompt-injection", mode="block")], "ignore previous", default_url=_URL
    )
    assert verdict.blocked is True
    assert verdict.flagged[0].score == 0.97


@pytest.mark.asyncio
async def test_flagged_input_does_not_block_in_monitor_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_transport(monkeypatch, _result_handler({"valid": False, "explanation": "injection"}))
    verdict = await run_input_guardrails(
        [GuardrailConfig(profile="prompt-injection", mode="monitor")], "ignore previous", default_url=_URL
    )
    assert verdict.blocked is False
    assert len(verdict.flagged) == 1


@pytest.mark.asyncio
async def test_valid_input_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_transport(monkeypatch, _result_handler({"valid": True, "score": 0.02}))
    verdict = await run_input_guardrails([GuardrailConfig(profile="prompt-injection")], "hello", default_url=_URL)
    assert verdict.blocked is False
    assert verdict.flagged == []
    assert verdict.results[0].valid is True


@pytest.mark.asyncio
async def test_list_result_is_unwrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """The service may return a list of results; we send one input, so unwrap it."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"profile": "prompt-injection", "result": [{"valid": False, "score": 0.9}]})

    _patch_transport(monkeypatch, handler)
    verdict = await run_input_guardrails(
        [GuardrailConfig(profile="prompt-injection", mode="block")], "x", default_url=_URL
    )
    assert verdict.blocked is True


@pytest.mark.asyncio
async def test_output_only_guardrail_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """v1 enforces only input-direction guardrails; an output-only entry makes
    no service call and never blocks."""
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"profile": "prompt-injection", "result": {"valid": False}})

    _patch_transport(monkeypatch, handler)
    verdict = await run_input_guardrails(
        [GuardrailConfig(profile="prompt-injection", on=["output"])], "x", default_url=_URL
    )
    assert verdict.results == []
    assert verdict.blocked is False
    assert called is False


@pytest.mark.asyncio
async def test_missing_url_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(GuardrailsNotReachableError):
        await run_input_guardrails(
            [GuardrailConfig(profile="prompt-injection", mode="block")], "x", default_url=None
        )


@pytest.mark.asyncio
async def test_block_mode_fails_closed_when_service_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A block guardrail that can't be evaluated raises (caller → 502)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    _patch_transport(monkeypatch, handler)
    with pytest.raises(GuardrailsNotReachableError):
        await run_input_guardrails([GuardrailConfig(profile="prompt-injection", mode="block")], "x", default_url=_URL)


@pytest.mark.asyncio
async def test_monitor_mode_fails_open_when_service_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A monitor guardrail that can't be evaluated does NOT raise/block: it
    records an inconclusive (valid=None) result and the request proceeds."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    _patch_transport(monkeypatch, handler)
    verdict = await run_input_guardrails(
        [GuardrailConfig(profile="prompt-injection", mode="monitor")], "x", default_url=_URL
    )
    assert verdict.blocked is False
    assert len(verdict.results) == 1
    assert verdict.results[0].valid is None
    assert verdict.flagged == []


@pytest.mark.asyncio
async def test_monitor_mode_fails_open_when_no_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """No configured URL is also a fail-open case for monitor guardrails."""
    verdict = await run_input_guardrails(
        [GuardrailConfig(profile="prompt-injection", mode="monitor")], "x", default_url=None
    )
    assert verdict.blocked is False
    assert verdict.results[0].valid is None


@pytest.mark.asyncio
async def test_malformed_result_block_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A result with no usable 'valid' must not silently pass a block guardrail."""
    _patch_transport(monkeypatch, _result_handler({}))  # no 'valid' field
    with pytest.raises(GuardrailsNotReachableError):
        await run_input_guardrails([GuardrailConfig(profile="prompt-injection", mode="block")], "x", default_url=_URL)


@pytest.mark.asyncio
async def test_malformed_result_monitor_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same malformed result fails open (not blocked) for a monitor guardrail."""
    _patch_transport(monkeypatch, _result_handler({}))
    verdict = await run_input_guardrails(
        [GuardrailConfig(profile="prompt-injection", mode="monitor")], "x", default_url=_URL
    )
    assert verdict.blocked is False
    assert verdict.results[0].valid is None


@pytest.mark.asyncio
async def test_non_boolean_valid_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-boolean 'valid' (e.g. a string) is treated as malformed -> fail closed."""
    _patch_transport(monkeypatch, _result_handler({"valid": "nope"}))
    with pytest.raises(GuardrailsNotReachableError):
        await run_input_guardrails([GuardrailConfig(profile="prompt-injection", mode="block")], "x", default_url=_URL)


@pytest.mark.asyncio
async def test_explicit_null_valid_is_inconclusive_not_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``valid: null`` is a legitimate inconclusive verdict, not malformed."""
    _patch_transport(monkeypatch, _result_handler({"valid": None, "score": 0.5}))
    verdict = await run_input_guardrails(
        [GuardrailConfig(profile="prompt-injection", mode="block")], "x", default_url=_URL
    )
    assert verdict.blocked is False
    assert verdict.results[0].valid is None
