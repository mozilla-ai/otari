"""Unit tests for classify_provider_error.

The classifier maps an upstream provider exception to a client-facing
(status, detail), and must return None for failures it cannot safely classify
so callers keep the generic 502.

Detail text splits on fault. A rejection of the caller's request (400/422/404)
carries the provider's own message, redacted and length-capped, because only the
provider knows what it objected to. A failure that is the gateway's own (rejected
credentials, an exhausted account, a 5xx) keeps a fixed string and never echoes
upstream text.
"""

import asyncio

import httpx
import pytest
from anthropic import APITimeoutError as AnthropicAPITimeoutError
from openai import APITimeoutError as OpenAIAPITimeoutError

from gateway.api.routes._pipeline import (
    _FORWARDED_PARAMS,
    PROVIDER_BAD_REQUEST_DETAIL,
    PROVIDER_BILLING_DETAIL,
    PROVIDER_CREDENTIALS_DETAIL,
    PROVIDER_MODEL_NOT_FOUND_DETAIL,
    PROVIDER_RATE_LIMITED_DETAIL,
    PROVIDER_TIMEOUT_DETAIL,
    classify_provider_error,
    failure_status_code,
)
from gateway.api.routes._platform import (
    MAX_EXPOSED_DETAIL_CHARS,
    _provider_failure_http_exc,
    redact_upstream_message,
)
from gateway.api.routes._schema_derive import SENSITIVE_PARAM_FIELDS
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
    ("status_code", "expected_status"),
    [(400, 400), (422, 400), (404, 404)],
)
def test_caller_fault_statuses_carry_the_upstream_message(status_code: int, expected_status: int) -> None:
    """A provider rejecting the caller's request returns the provider's own
    message. Only the provider knows what it objected to, and every fixed string
    we could write in its place discards that."""
    exc = _ParamError(status_code, None, "max_tokens must be less than or equal to 8192")
    assert classify_provider_error(exc) == (expected_status, "max_tokens must be less than or equal to 8192")


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (401, (502, PROVIDER_CREDENTIALS_DETAIL)),
        (403, (502, PROVIDER_CREDENTIALS_DETAIL)),
        (429, (429, PROVIDER_RATE_LIMITED_DETAIL)),
    ],
)
def test_gateway_fault_statuses_keep_a_fixed_detail(status_code: int, expected: tuple[int, str]) -> None:
    """A rejected credential or a rate limit is not the caller's request to fix,
    so the detail stays fixed and the upstream text is never echoed."""
    assert classify_provider_error(_StatusError(status_code)) == expected


@pytest.mark.parametrize("status_code", [400, 404, 422])
def test_caller_fault_falls_back_when_the_provider_said_nothing(status_code: int) -> None:
    """An exception carrying no usable text still gets a usable detail rather
    than an empty string."""
    mapping = classify_provider_error(_ParamError(status_code, None, ""))
    assert mapping is not None
    assert mapping.detail in (PROVIDER_BAD_REQUEST_DETAIL, PROVIDER_MODEL_NOT_FOUND_DETAIL)


def test_status_read_from_attached_response() -> None:
    mapping = classify_provider_error(_ResponseStatusError(404))
    assert mapping is not None
    assert mapping.status_code == 404


@pytest.mark.parametrize("exc", [_StatusError(500), _StatusError(503), Exception(_RAW), ValueError(_RAW)])
def test_unclassifiable_returns_none(exc: BaseException) -> None:
    assert classify_provider_error(exc) is None


def test_gateway_fault_details_never_echo_the_raw_message() -> None:
    """The statuses where the gateway's own credentials and topology concentrate
    keep a fixed detail, whatever the provider put in the body."""
    for status_code in (401, 403, 429):
        mapping = classify_provider_error(_StatusError(status_code))
        assert mapping is not None
        assert "SECRET" not in mapping.detail
        assert "abc123" not in mapping.detail


def test_platform_terminal_exc_uses_classified_status() -> None:
    """Platform-mode terminal failures get the same classified status as the
    standalone adapters, so the production path is not stuck on a generic 502."""
    exc = _provider_failure_http_exc(_StatusError(404), fallback_detail="LLM provider error")
    assert exc.status_code == 404


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


def test_reasoning_effort_tools_conflict_reaches_the_caller_verbatim() -> None:
    """#331's case, with no probe behind it. OpenAI's message already names the
    remedy, so passing it through does the job the hand-written
    PROVIDER_REASONING_TOOLS_UNSUPPORTED_DETAIL used to, for every rewording of
    it we would otherwise have to chase."""
    exc = _ParamError(400, "reasoning_effort", _REASONING_TOOLS_MSG)
    assert classify_provider_error(exc) == (400, _REASONING_TOOLS_MSG)


def test_upstream_message_read_from_original_exception() -> None:
    """The message is found when it lives only on ``original_exception`` (the
    any-llm unified-exception shape)."""
    original = _ParamError(400, None, "temperature must be between 0 and 2")
    mapping = classify_provider_error(_WrappedError(400, original))
    assert mapping is not None
    assert "temperature must be between 0 and 2" in mapping.detail


def test_upstream_message_and_unsupported_feature_read_from_nested_original_exceptions() -> None:
    """The shared chain walk handles arbitrarily nested any-llm wrappers."""
    nested = _WrappedError(400, _WrappedError(400, NotImplementedError(_CONTEXT_MANAGEMENT_MSG)))
    mapping = classify_provider_error(nested)
    assert mapping is not None
    assert mapping.status_code == 400
    assert "context_management" in mapping.detail


@pytest.mark.parametrize(
    ("raw", "leaked"),
    [
        ("Auth failed for Bearer sk-proj-AAAAAAAAAAAAAAAAAAAA", "sk-proj-AAAAAAAAAAAAAAAAAAAA"),
        ("Invalid key sk-ant-api03-BBBBBBBBBBBBBBBB", "sk-ant-api03-BBBBBBBBBBBBBBBB"),
        ("Rejected by AIzaSyCCCCCCCCCCCCCCCCCC", "AIzaSyCCCCCCCCCCCCCCCCCC"),
        ("Cannot reach http://10.0.0.4:8000/v1/chat", "10.0.0.4"),
        ("Bad request (api_key=hunter2hunter2)", "hunter2hunter2"),
        ("Model unavailable for org-DDDDDDDDDD", "org-DDDDDDDDDD"),
        ("Rejected: " + "E" * 48, "E" * 48),
    ],
)
def test_redaction_strips_credential_shapes(raw: str, leaked: str) -> None:
    """Passing the provider's message through does not mean passing its
    credentials through. The gateway calls providers with the operator's key, so
    a message naming that key, a self-hosted api_base, or the account serving a
    managed model is not the caller's to read."""
    assert leaked in raw
    assert leaked not in redact_upstream_message(raw)


_CONTEXT_MANAGEMENT_MSG = "context_management and betas require a provider with a native Anthropic Messages API"


def test_unsupported_feature_maps_to_400_with_the_reason() -> None:
    """#530: any-llm rejects an unsupported request feature with a bare
    NotImplementedError carrying no HTTP status, so it used to fall through to a
    generic 500 telling the caller to retry something that can never succeed.
    The exception type carries the whole signal, so this needs no probe into
    any-llm's wording, and the message names the feature."""
    mapping = classify_provider_error(NotImplementedError(_CONTEXT_MANAGEMENT_MSG))
    assert mapping is not None
    assert mapping.status_code == 400
    assert "context_management" in mapping.detail
    assert "retry" not in mapping.detail.lower()


def test_unsupported_feature_survives_the_unified_exception_wrapper() -> None:
    """The type check still fires once ANY_LLM_UNIFIED_EXCEPTIONS=1 wraps the
    raw error, because the wrapper keeps it on ``original_exception``."""
    mapping = classify_provider_error(_WrappedError(500, NotImplementedError(_CONTEXT_MANAGEMENT_MSG)))
    assert mapping is not None
    assert mapping.status_code == 400
    assert "context_management" in mapping.detail


def test_unsupported_feature_is_recorded_as_400_on_the_usage_log() -> None:
    """The usage-log status follows the classification rather than the generic 502."""
    assert failure_status_code(NotImplementedError(_CONTEXT_MANAGEMENT_MSG)) == 400


def test_rejected_param_maps_to_400_naming_the_param() -> None:
    """#1062: any-llm forwards every param its ``CompletionParams`` declares, so
    an OpenAI-only param against a provider that never grew one reaches the SDK
    and comes back as a TypeError with no HTTP status. It used to read as a
    generic 502, which reports a permanent mismatch as an upstream outage."""
    exc = TypeError("AsyncMessages.create() got an unexpected keyword argument 'seed'")
    mapping = classify_provider_error(exc)
    assert mapping is not None
    assert mapping.status_code == 400
    assert "'seed'" in mapping.detail


def test_rejected_param_detail_does_not_name_the_sdk_internals() -> None:
    """Only the param name crosses the boundary: the SDK's class and method are
    the gateway's implementation, not something a caller can act on."""
    exc = TypeError("AsyncMessages.create() got an unexpected keyword argument 'logit_bias'")
    mapping = classify_provider_error(exc)
    assert mapping is not None
    assert "AsyncMessages" not in mapping.detail
    assert "create()" not in mapping.detail


def test_rejected_param_survives_the_unified_exception_wrapper() -> None:
    """Same unwrapping the other permanent-failure branches get, so the mapping
    keeps working once ANY_LLM_UNIFIED_EXCEPTIONS=1 wraps the raw SDK error."""
    wrapped = _WrappedError(500, TypeError("create() got an unexpected keyword argument 'n'"))
    mapping = classify_provider_error(wrapped)
    assert mapping is not None
    assert mapping.status_code == 400
    assert "'n'" in mapping.detail


def test_rejected_param_is_recorded_as_400_on_the_usage_log() -> None:
    """The usage-log status follows the classification rather than the generic 502."""
    assert failure_status_code(TypeError("create() got an unexpected keyword argument 'seed'")) == 400


def test_operator_client_args_typo_is_not_the_callers_fault() -> None:
    """#769 review: ``client_args`` is operator-owned (config.yml or a stored
    provider credential) and reaches the provider *client's* constructor, so a
    typo there raises the same wording. Returning it as a 400 would blame the
    caller for a parameter they cannot remove, and would drop a gateway
    misconfiguration off the error-rate panel as a 4xx."""
    exc = TypeError("AsyncOpenAI.__init__() got an unexpected keyword argument 'timeoutt'")
    assert classify_provider_error(exc) is None
    assert failure_status_code(exc) == 502


def test_gateway_internal_signature_drift_is_not_the_callers_fault() -> None:
    """The same wording arrives from gateway-internal code, because this
    classifier is reached from ``except Exception`` arms that wrap the tool
    backends and the stream opener, not only the provider call. A backend whose
    constructor drifted (mozilla-ai/otari#766) must not surface as a 400 naming
    an internal parameter."""
    exc = TypeError("_FakeSandboxBackend.__init__() got an unexpected keyword argument 'image'")
    assert classify_provider_error(exc) is None
    assert failure_status_code(exc) == 502


def test_client_args_key_colliding_with_a_real_param_is_still_not_a_400() -> None:
    """The case the name gate alone cannot see: an operator ``client_args`` key
    that happens to be a real request param. It is a constructor rejection, so
    the second gate keeps it a 502."""
    exc = TypeError("AsyncOpenAI.__init__() got an unexpected keyword argument 'seed'")
    assert classify_provider_error(exc) is None


def test_param_the_gateway_never_forwards_stays_unclassified() -> None:
    """A name outside the request-body surface cannot have come from the caller,
    whatever raised it."""
    assert classify_provider_error(TypeError("post() got an unexpected keyword argument 'proxies'")) is None


def test_the_caller_fault_gate_excludes_the_params_the_schema_refuses() -> None:
    """The gate's name set is the request schema's, carve-out included.

    ``derive_request_base`` skips ``SENSITIVE_PARAM_FIELDS`` so a credential or
    provider-selection field any-llm adds to a typed ``*Params`` can never become
    caller-settable (mozilla-ai/otari#160). A name the schema will not accept
    cannot have come from a caller, so it must not pass this gate either. No
    ``*Params`` declares one today, which is exactly why this is pinned: the two
    sets agree by luck right now and by construction after this.
    """
    assert not _FORWARDED_PARAMS & SENSITIVE_PARAM_FIELDS

    sensitive = min(SENSITIVE_PARAM_FIELDS)
    exc = TypeError(f"acompletion() got an unexpected keyword argument '{sensitive}'")
    assert classify_provider_error(exc) is None
    assert failure_status_code(exc) == 502


def test_unrelated_type_error_stays_unclassified() -> None:
    """A TypeError that is not a keyword-argument rejection carries no signal a
    caller could act on, so it keeps the generic 502 rather than becoming a 400
    that blames the caller for a gateway-side fault."""
    assert classify_provider_error(TypeError("unsupported operand type(s) for +: 'int' and 'str'")) is None


def test_redaction_keeps_the_part_that_explains_the_rejection() -> None:
    """Redaction targets secret shapes, not meaning: the text that tells the
    caller what to change has to survive."""
    redacted = redact_upstream_message("max_tokens: must be <= 8192 for claude-opus-5, got 200000")
    assert "max_tokens" in redacted
    assert "8192" in redacted
    assert "claude-opus-5" in redacted


def test_exposed_detail_is_length_capped() -> None:
    """A provider that echoes the offending request back does not get to put all
    of it in a response detail."""
    mapping = classify_provider_error(_ParamError(400, None, "problem: " + "word " * 500))
    assert mapping is not None
    assert len(mapping.detail) <= MAX_EXPOSED_DETAIL_CHARS
    assert mapping.detail.endswith("...")


def test_caller_fault_falls_back_after_the_message_is_completely_redacted() -> None:
    mapping = classify_provider_error(_ParamError(400, None, "token=supersecretvalue123"))
    assert mapping == (400, PROVIDER_BAD_REQUEST_DETAIL)


def test_redaction_strips_prefixless_32_character_api_keys() -> None:
    key = "0123456789abcdef0123456789abcdef"
    assert key not in redact_upstream_message(f"Incorrect api-key provided: {key}")


@pytest.mark.parametrize(
    "raw",
    [
        "messages.0.content: input_value='private prompt'",
        "content=private prompt",
        "input_value=private prompt",
    ],
)
def test_redaction_rejects_validation_payload_echoes(raw: str) -> None:
    assert redact_upstream_message(raw) == ""


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


def test_unrecognized_400_message_stays_a_caller_fault_400() -> None:
    """A genuinely malformed request is not swept up by the billing probe: it
    stays a 400 and now says what was actually wrong, which is the case that
    motivated passing the message through at all."""
    exc = _ParamError(400, None, "messages.3: tool_use ids were found without tool_result blocks")
    assert classify_provider_error(exc) == (400, "messages.3: tool_use ids were found without tool_result blocks")


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
