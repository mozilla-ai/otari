"""Unit tests for classify_provider_error.

The classifier maps an upstream provider exception to a safe, client-facing
(status, detail). It must never echo the raw provider message, and must return
None for failures it cannot safely classify so callers keep the generic 502.
"""

import asyncio

import httpx
import pytest
from anthropic import APITimeoutError as AnthropicAPITimeoutError
from openai import APITimeoutError as OpenAIAPITimeoutError

from gateway.api.routes._pipeline import (
    PROVIDER_BAD_REQUEST_DETAIL,
    PROVIDER_BILLING_DETAIL,
    PROVIDER_CREDENTIALS_DETAIL,
    PROVIDER_MODEL_NOT_FOUND_DETAIL,
    PROVIDER_RATE_LIMITED_DETAIL,
    PROVIDER_REASONING_TOOLS_UNSUPPORTED_DETAIL,
    PROVIDER_TIMEOUT_DETAIL,
    classify_provider_error,
    failure_status_code,
)
from gateway.api.routes._platform import _provider_failure_http_exc
from gateway.services.mcp_loop import MaxToolIterationsExceeded

_RAW = "raw provider detail SECRET token=abc123"

# The exact upstream OpenAI message for the tools + reasoning_effort rejection.
_REASONING_TOOLS_MSG = (
    "Function tools with reasoning_effort are not supported for gpt-5.6-sol in "
    "/v1/chat/completions. To use function tools, use /v1/responses or set "
    "reasoning_effort to 'none'."
)


class _StatusError(Exception):
    """Upstream error exposing a top-level status_code, like any-llm surfaces."""

    def __init__(self, status_code: int) -> None:
        super().__init__(_RAW)
        self.status_code = status_code


class _ParamError(Exception):
    """Upstream error carrying an OpenAI-style ``param`` + ``message``, like the
    raw ``openai.BadRequestError`` any-llm currently passes through."""

    def __init__(self, status_code: int, param: str | None, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.param = param
        self.message = message


class _WrappedError(Exception):
    """Upstream error whose signal lives only on ``original_exception``, matching
    any-llm's unified-exception layer once it becomes the default."""

    def __init__(self, status_code: int, original: BaseException) -> None:
        super().__init__("Invalid request")
        self.status_code = status_code
        self.original_exception = original


class _ResponseStatusError(Exception):
    """Upstream error exposing status via an attached response object."""

    def __init__(self, status_code: int) -> None:
        super().__init__(_RAW)
        self.response = httpx.Response(status_code)


def test_timeout_maps_to_504() -> None:
    for exc in (asyncio.TimeoutError(), TimeoutError(), httpx.TimeoutException("slow")):
        mapping = classify_provider_error(exc)
        assert mapping == (504, PROVIDER_TIMEOUT_DETAIL)


def test_sdk_wrapped_timeout_maps_to_504() -> None:
    """The OpenAI/Anthropic SDKs wrap httpx timeouts into their own
    ``APITimeoutError`` (no ``status_code``, not an httpx exception instance).
    any-llm surfaces that wrapped type directly, so it must still classify
    as a 504, not fall through to the generic 502."""
    request = httpx.Request("POST", "http://upstream")
    for exc in (OpenAIAPITimeoutError(request=request), AnthropicAPITimeoutError(request=request)):
        assert classify_provider_error(exc) == (504, PROVIDER_TIMEOUT_DETAIL)


def test_unified_any_llm_wrapped_timeout_maps_to_504() -> None:
    """Once otari enables ``ANY_LLM_UNIFIED_EXCEPTIONS=1``, a raw SDK timeout
    error arrives wrapped in a generic ``AnyLLMError`` subclass (no
    ``status_code``, a class name the duck-typed fallback won't recognize)
    rather than the SDK type directly. ``classify_provider_error`` must still
    resolve it to 504 via the shared ``original_exception`` unwrap, not fall
    back to the generic 502."""

    class _WrappedByAnyLLM(Exception):
        def __init__(self, original_exception: BaseException) -> None:
            super().__init__(str(original_exception))
            self.original_exception = original_exception

    request = httpx.Request("POST", "http://upstream")
    wrapped = _WrappedByAnyLLM(OpenAIAPITimeoutError(request=request))
    assert classify_provider_error(wrapped) == (504, PROVIDER_TIMEOUT_DETAIL)


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (400, (400, PROVIDER_BAD_REQUEST_DETAIL)),
        (422, (400, PROVIDER_BAD_REQUEST_DETAIL)),
        (404, (404, PROVIDER_MODEL_NOT_FOUND_DETAIL)),
        (401, (502, PROVIDER_CREDENTIALS_DETAIL)),
        (403, (502, PROVIDER_CREDENTIALS_DETAIL)),
        (429, (429, PROVIDER_RATE_LIMITED_DETAIL)),
    ],
)
def test_known_status_codes_map_to_safe_pairs(status_code: int, expected: tuple[int, str]) -> None:
    assert classify_provider_error(_StatusError(status_code)) == expected


def test_status_read_from_attached_response() -> None:
    assert classify_provider_error(_ResponseStatusError(404)) == (404, PROVIDER_MODEL_NOT_FOUND_DETAIL)


@pytest.mark.parametrize("exc", [_StatusError(500), _StatusError(503), Exception(_RAW), ValueError(_RAW)])
def test_unclassifiable_returns_none(exc: BaseException) -> None:
    assert classify_provider_error(exc) is None


def test_no_classified_detail_leaks_raw_message() -> None:
    for status_code in (400, 401, 403, 404, 422, 429):
        mapping = classify_provider_error(_StatusError(status_code))
        assert mapping is not None
        assert "SECRET" not in mapping.detail
        assert "abc123" not in mapping.detail


def test_platform_terminal_exc_uses_classified_status() -> None:
    """Platform-mode terminal failures get the same classified status as the
    standalone adapters, so the production path is not stuck on a generic 502."""
    exc = _provider_failure_http_exc(_StatusError(404), fallback_detail="LLM provider error")
    assert exc.status_code == 404
    assert exc.detail == PROVIDER_MODEL_NOT_FOUND_DETAIL


def test_platform_terminal_exc_falls_back_to_generic() -> None:
    exc = _provider_failure_http_exc(Exception(_RAW), fallback_detail="LLM provider error")
    assert exc.status_code == 502
    assert exc.detail == "LLM provider error"
    assert "SECRET" not in str(exc.detail)


# ---------------------------------------------------------------------------
# failure_status_code: what lands on usage_logs.status_code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422, 429, 500, 503, 529])
def test_failure_status_code_keeps_the_upstream_status(status_code: int) -> None:
    """The upstream status is recorded verbatim, including the codes the caller
    never sees: an upstream 401/403 surfaces as a generic 502 (a provider
    rejecting the gateway's credentials must not be echoed as a client error),
    but the log has to keep the 401 or the misconfiguration is unattributable.
    """
    assert failure_status_code(_StatusError(status_code)) == status_code


def test_failure_status_code_reads_an_attached_response() -> None:
    assert failure_status_code(_ResponseStatusError(429)) == 429


def test_failure_status_code_records_504_for_a_timeout() -> None:
    """A timeout carries no upstream status, so the gateway's own classification
    is recorded rather than leaving the row unclassifiable."""
    request = httpx.Request("POST", "http://upstream")
    for exc in (asyncio.TimeoutError(), httpx.TimeoutException("slow"), OpenAIAPITimeoutError(request=request)):
        assert failure_status_code(exc) == 504


@pytest.mark.parametrize("exc", [Exception(_RAW), ValueError(_RAW), httpx.ConnectError("refused")])
def test_failure_status_code_falls_back_to_502(exc: BaseException) -> None:
    """An unreachable or otherwise unclassifiable provider still gets a code, so
    every error row is groupable."""
    assert failure_status_code(exc) == 502


def test_failure_status_code_unwraps_original_exception() -> None:
    """An any-llm-wrapped error surfaces the inner status, matching how
    ``classify_provider_error`` reads the same chain."""

    class _WrappedByAnyLLM(Exception):
        def __init__(self, original_exception: BaseException) -> None:
            super().__init__(str(original_exception))
            self.original_exception = original_exception

    assert failure_status_code(_WrappedByAnyLLM(_StatusError(429))) == 429


def test_failure_status_code_records_422_for_the_tool_loop_cap() -> None:
    """The gateway's own tool-loop cap records 422, not the generic 502.

    The non-streaming path stamps 422 at its own ``except`` clause, but a
    streaming request raises the cap while the SSE body is already in flight, so
    it settles through ``on_error`` and lands here instead. Without this branch
    the cap carries no status, falls through to 502, and shows up in the error
    taxonomy as a provider outage: precisely the confusion the 422 exists to
    prevent, and only on streaming traffic, so the two halves of the same failure
    would classify differently.
    """
    assert failure_status_code(MaxToolIterationsExceeded("Exceeded max_tool_iterations=8")) == 422


@pytest.mark.parametrize("status_code", [400, 422])
def test_reasoning_effort_tools_conflict_maps_to_actionable_detail(status_code: int) -> None:
    """A 400/422 flagged param=reasoning_effort whose message names function tools
    gets the actionable remedy instead of the generic bad-request detail."""
    exc = _ParamError(status_code, "reasoning_effort", _REASONING_TOOLS_MSG)
    assert classify_provider_error(exc) == (400, PROVIDER_REASONING_TOOLS_UNSUPPORTED_DETAIL)


def test_reasoning_effort_tools_conflict_read_from_original_exception() -> None:
    """The signal is picked up when it lives only on ``original_exception`` (the
    any-llm unified-exception shape)."""
    original = _ParamError(400, "reasoning_effort", _REASONING_TOOLS_MSG)
    exc = _WrappedError(400, original)
    assert classify_provider_error(exc) == (400, PROVIDER_REASONING_TOOLS_UNSUPPORTED_DETAIL)


def test_reasoning_effort_without_function_tools_stays_generic() -> None:
    """A reasoning_effort rejection that is not the tools conflict (message does
    not mention function tools) keeps the generic bad-request detail."""
    exc = _ParamError(400, "reasoning_effort", "Unsupported value 'ultra' for reasoning_effort.")
    assert classify_provider_error(exc) == (400, PROVIDER_BAD_REQUEST_DETAIL)


def test_function_tools_message_on_other_param_stays_generic() -> None:
    """A different offending param does not get the reasoning-effort remedy even
    if the message happens to mention function tools."""
    exc = _ParamError(400, "tools", "Function tools are malformed for this request.")
    assert classify_provider_error(exc) == (400, PROVIDER_BAD_REQUEST_DETAIL)


def test_reasoning_effort_tools_conflict_does_not_leak_raw_message() -> None:
    """The actionable detail is a fixed string; the model name and endpoint from
    the raw upstream message are never echoed to the caller."""
    exc = _ParamError(400, "reasoning_effort", _REASONING_TOOLS_MSG + " SECRET token=abc123")
    mapping = classify_provider_error(exc)
    assert mapping is not None
    assert mapping.detail == PROVIDER_REASONING_TOOLS_UNSUPPORTED_DETAIL
    assert "gpt-5.6-sol" not in mapping.detail
    assert "SECRET" not in mapping.detail


# The exact upstream Anthropic message for an out-of-credit account. Anthropic
# returns this as a 400 invalid_request_error, not a 402.
_ANTHROPIC_BILLING_MSG = (
    "Your credit balance is too low to access the Anthropic API. "
    "Please go to Plans & Billing to upgrade or purchase credits."
)


@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (400, _ANTHROPIC_BILLING_MSG),
        (402, "Insufficient Balance"),
        (400, "Billing hard limit has been reached"),
        (422, "insufficient credits for this request"),
    ],
)
def test_billing_exhaustion_maps_to_502_billing_detail(status_code: int, message: str) -> None:
    """A provider saying "this account is out of money" is a gateway-side account
    fault, so it surfaces as a 502 with the billing remedy rather than a 400
    telling the caller to check the model name. Reported by clawbolt: an Anthropic
    account ran dry and every agent turn failed with "check the model name and
    parameters", sending the operator to debug a model alias for 10 hours."""
    exc = _ParamError(status_code, None, message)
    assert classify_provider_error(exc) == (502, PROVIDER_BILLING_DETAIL)


def test_billing_exhaustion_read_from_original_exception() -> None:
    """The signal is picked up when it lives only on ``original_exception`` (the
    any-llm unified-exception shape)."""
    original = _ParamError(400, None, _ANTHROPIC_BILLING_MSG)
    assert classify_provider_error(_WrappedError(400, original)) == (502, PROVIDER_BILLING_DETAIL)


def test_billing_probe_is_gated_on_the_status_code() -> None:
    """A billing-sounding phrase on a status the probe does not cover keeps its
    existing classification, so the probe can only ever reinterpret a 400/402/422
    dead end. 500 stays unclassifiable (generic 502); 429 stays a rate limit,
    which is still an actionable signal for the caller."""
    assert classify_provider_error(_ParamError(500, None, _ANTHROPIC_BILLING_MSG)) is None
    assert classify_provider_error(_ParamError(429, None, "insufficient_quota")) == (
        429,
        PROVIDER_RATE_LIMITED_DETAIL,
    )


def test_unrecognized_400_message_stays_generic_bad_request() -> None:
    """A genuinely malformed request is not swept up by the billing probe: an
    unrecognized phrasing degrades to the existing generic detail rather than
    being misclassified into wasted failover attempts."""
    exc = _ParamError(400, None, "messages.3: tool_use ids were found without tool_result blocks")
    assert classify_provider_error(exc) == (400, PROVIDER_BAD_REQUEST_DETAIL)


def test_billing_detail_does_not_leak_raw_message() -> None:
    """The billing detail is a fixed string; nothing from the upstream body is
    echoed to the caller."""
    exc = _ParamError(400, None, _ANTHROPIC_BILLING_MSG + " SECRET token=abc123")
    mapping = classify_provider_error(exc)
    assert mapping is not None
    assert mapping.detail == PROVIDER_BILLING_DETAIL
    assert "SECRET" not in mapping.detail
    assert "Anthropic" not in mapping.detail


def test_failure_status_code_keeps_the_upstream_status_for_billing() -> None:
    """The usage log keeps the status the provider actually returned, so "how much
    of my error rate is an empty wallet" stays answerable even though the caller
    saw a 502."""
    assert failure_status_code(_ParamError(400, None, _ANTHROPIC_BILLING_MSG)) == 400
