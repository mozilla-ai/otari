"""Unit tests for classify_provider_error.

The classifier maps an upstream provider exception to a client-facing
(status, detail), and must return None for failures it cannot safely classify
so callers keep the generic 502.

Detail text splits on whether the caller can act on the failure. A rejection of
the caller's request (400/422/404) and a rate limit (429) carry the provider's
own message, redacted and length-capped, because only the provider knows what it
objected to or which quota ran out. A failure that is the gateway's own (rejected
credentials, an exhausted account, a 5xx) keeps a fixed string and never echoes
upstream text.
"""

import asyncio
import json
from typing import Any

import anthropic
import httpx
import openai
import pytest
from anthropic import APITimeoutError as AnthropicAPITimeoutError
from any_llm.exceptions import (
    ContextLengthExceededError,
    InvalidRequestError,
    MissingApiKeyError,
    ProviderError,
    UnsupportedParameterError,
)
from any_llm.utils.exception_handler import convert_exception
from botocore.exceptions import ClientError
from cohere.errors import BadRequestError as CohereBadRequestError
from fastapi import HTTPException
from openai import APITimeoutError as OpenAIAPITimeoutError

from gateway.api.routes._pipeline import (
    _FORWARDED_PARAMS,
    PROVIDER_ACCOUNT_QUOTA_DETAIL,
    PROVIDER_BAD_REQUEST_DETAIL,
    PROVIDER_BILLING_DETAIL,
    PROVIDER_CREDENTIALS_DETAIL,
    PROVIDER_ERROR_DETAIL,
    PROVIDER_MODEL_NOT_FOUND_DETAIL,
    PROVIDER_RATE_LIMITED_DETAIL,
    PROVIDER_TIMEOUT_DETAIL,
    classify_provider_error,
    failure_error_message,
    failure_status_code,
    provider_error_headers,
    provider_failure_detail,
    refusal_code,
    stream_error_code,
    stream_error_message,
)
from gateway.api.routes._platform import (
    SHOULD_RETRY_HEADER,
    _provider_failure_http_exc,
    get_shared_rejection,
    is_provider_credential_rejection,
    with_retry_hint,
)
from gateway.core.error_codes import PROVIDER_CREDENTIAL_REJECTED, PROVIDER_NOT_CONFIGURED
from gateway.core.provider_params import SENSITIVE_PARAM_FIELDS
from gateway.services.inference import (
    is_retry_futile,
    is_upstream_request_fault,
    upstream_retry_after,
)
from gateway.services.mcp_loop import MaxToolIterationsExceeded
from gateway.services.provider_kwargs import MissingCredential
from gateway.services.upstream_redaction import MAX_EXPOSED_DETAIL_CHARS, redact_upstream_message
from gateway.streaming import OPENAI_STREAM_FORMAT, IncompleteStreamError, openai_error_event

_RAW = "raw provider detail SECRET token=abc123"

# The exact upstream OpenAI message for the tools + reasoning_effort rejection.
_REASONING_TOOLS_MSG = (
    "Function tools with reasoning_effort are not supported for gpt-5.6-sol in "
    "/v1/chat/completions. To use function tools, use /v1/responses or set "
    "reasoning_effort to 'none'."
)


def _openai_connection_error(cause: Exception) -> openai.APIConnectionError:
    """The OpenAI SDK's connection error, raised from the httpx error as the SDK does."""
    request = httpx.Request("POST", "https://internal-proxy.example:8443/v1/chat/completions")
    try:
        raise openai.APIConnectionError(request=request) from cause
    except openai.APIConnectionError as exc:
        return exc


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


class _CodeError(Exception):
    """Upstream error carrying its status on ``code``, as google-genai's
    ``APIError`` does. It never sets ``status_code``."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"{code} {message}")
        self.code = code
        self.message = message


def test_timeout_maps_to_504() -> None:
    for exc in (asyncio.TimeoutError(), TimeoutError(), httpx.TimeoutException("slow")):
        mapping = classify_provider_error(exc)
        assert mapping == (504, PROVIDER_TIMEOUT_DETAIL)


def test_sdk_wrapped_timeout_maps_to_504() -> None:
    """The OpenAI/Anthropic SDKs wrap httpx timeouts into their own
    ``APITimeoutError`` (no ``status_code``, not an httpx exception instance).
    any-llm surfaces that wrapped type directly, so it must still classify
    as a 504, not fall through to the generic 502."""
    for exc in (
        OpenAIAPITimeoutError(request=httpx.Request("POST", "http://upstream")),
        AnthropicAPITimeoutError(request=httpx.Request("POST", "http://upstream")),
    ):
        assert classify_provider_error(exc) == (504, PROVIDER_TIMEOUT_DETAIL)
        assert failure_status_code(exc) == 504


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


def test_rate_limit_carries_the_upstream_message() -> None:
    """A 429 is not the caller's request to fix, but it is theirs to act on, and
    only the provider's text says which quota ran out and how long it lasts. A
    fixed "you were rate-limited" discards the retry window for something the
    status already said."""
    exc = _ParamError(429, None, "Quota exceeded for generate_content_requests. Please retry in 34.6s.")
    assert classify_provider_error(exc) == (
        429,
        "Quota exceeded for generate_content_requests. Please retry in 34.6s.",
    )


def test_rate_limit_message_is_still_redacted() -> None:
    """Passing a 429's text through does not exempt it from redaction."""
    exc = _ParamError(429, None, "Quota exceeded on project proj-a1b2c3d4 via https://internal.upstream/v1")
    mapping = classify_provider_error(exc)
    assert mapping is not None
    assert mapping.status_code == 429
    assert "Quota exceeded" in mapping.detail
    assert "proj-a1b2c3d4" not in mapping.detail
    assert "internal.upstream" not in mapping.detail


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (401, (424, PROVIDER_CREDENTIALS_DETAIL)),
        (403, (424, PROVIDER_CREDENTIALS_DETAIL)),
    ],
)
def test_gateway_fault_statuses_keep_a_fixed_detail(status_code: int, expected: tuple[int, str]) -> None:
    """A rejected credential is not something the caller can act on, so the
    detail stays fixed and the upstream text is never echoed. It is a 424, not a
    5xx: retrying cannot make the gateway's key valid."""
    assert classify_provider_error(_StatusError(status_code)) == expected


@pytest.mark.parametrize("status_code", [400, 404, 422, 429])
def test_caller_actionable_falls_back_when_the_provider_said_nothing(status_code: int) -> None:
    """An exception carrying no usable text still gets a usable detail rather
    than an empty string."""
    mapping = classify_provider_error(_ParamError(status_code, None, ""))
    assert mapping is not None
    assert mapping.detail in (
        PROVIDER_BAD_REQUEST_DETAIL,
        PROVIDER_MODEL_NOT_FOUND_DETAIL,
        PROVIDER_RATE_LIMITED_DETAIL,
    )


def test_status_read_from_attached_response() -> None:
    mapping = classify_provider_error(_ResponseStatusError(404))
    assert mapping is not None
    assert mapping.status_code == 404


def test_status_read_from_an_int_code() -> None:
    """google-genai's ``APIError`` puts the status on ``code`` and never sets
    ``status_code``, so a Gemini rejection is unclassifiable without it."""
    mapping = classify_provider_error(_CodeError(404, "models/gemini-9 is not found"))
    assert mapping is not None
    assert mapping.status_code == 404


def test_a_string_code_is_not_read_as_a_status() -> None:
    """OpenAI-family SDKs use ``code`` for an error slug, not a status, so it
    must not be guessed at and the exception stays unclassifiable."""
    exc = Exception(_RAW)
    exc.code = "invalid_api_key"  # type: ignore[attr-defined]
    assert classify_provider_error(exc) is None


def test_a_non_http_int_code_is_not_read_as_a_status() -> None:
    """google-genai's live API raises through the same ``APIError`` with a
    websocket close code. It is an ``int`` but not a status."""
    assert classify_provider_error(_CodeError(1008, "policy violation")) is None


def test_a_non_error_response_status_does_not_shadow_the_real_status() -> None:
    """Gemini reports a streaming failure in a body chunk, against a stream
    wrapper whose ``status_code`` is a hardcoded 200: the SSE response opened
    fine and the 429 arrived later. Reading that 200 as the failure's status is
    what turned a rate limit into an opaque 502 and recorded 200 as the
    request's outcome."""
    exc = _CodeError(429, "You exceeded your current quota. Please retry in 34.6s.")
    exc.response = httpx.Response(200)  # type: ignore[attr-defined]
    mapping = classify_provider_error(exc)
    assert mapping is not None
    assert mapping.status_code == 429
    assert failure_status_code(exc) == 429
    # google-genai's ``str()`` is "<code> <status>. <whole response body>", so
    # joining it to ``message`` would hand the caller the sentence twice, the
    # second time inside a JSON dump.
    assert mapping.detail == "You exceeded your current quota. Please retry in 34.6s."


# ---------------------------------------------------------------------------
# Retry-After: the one upstream header a rate-limited caller can act on
# ---------------------------------------------------------------------------


def _rate_limited_with(retry_after: str) -> Exception:
    """A 429 whose attached response carries ``retry_after``."""
    exc = _StatusError(429)
    exc.response = httpx.Response(429, headers={"Retry-After": retry_after})  # type: ignore[attr-defined]
    return exc


def test_retry_after_is_forwarded_on_a_429() -> None:
    assert provider_error_headers(_rate_limited_with("34"), 429) == {
        "Otari-Error-Code": "upstream_rate_limited",
        "Retry-After": "34",
    }


def test_retry_after_rounds_a_fraction_up() -> None:
    """A client honoring the header must not retry before the window the
    provider named, so 0.4s becomes 1s rather than 0s."""
    assert provider_error_headers(_rate_limited_with("0.4"), 429) == {
        "Otari-Error-Code": "upstream_rate_limited",
        "Retry-After": "1",
    }


def test_retry_after_is_clamped() -> None:
    """A provider does not get to tell this gateway's callers to sleep for a
    year."""
    assert provider_error_headers(_rate_limited_with("99999999"), 429) == {
        "Otari-Error-Code": "upstream_rate_limited",
        "Retry-After": "86400",
    }


@pytest.mark.parametrize(
    "raw",
    [
        "Wed, 21 Oct 2015 07:28:00 GMT",  # the HTTP-date form, deliberately dropped
        "soon",
        "-5",
        "",
        "12\r\nX-Injected: 1",
        # float() accepts these and math.ceil raises OverflowError on them.
        # This runs inside error handling, so a raise would turn the rate limit
        # into a 500.
        "inf",
        "1e400",
        "nan",
    ],
)
def test_retry_after_that_is_not_a_number_is_dropped(raw: str) -> None:
    """The value is re-serialized from a parsed number, never relayed as
    received: a header value is not a body, and CRLF in one is not a formatting
    problem."""
    assert provider_error_headers(_rate_limited_with(raw), 429) == {"Otari-Error-Code": "upstream_rate_limited"}


def test_retry_after_is_not_forwarded_on_a_gateway_fault() -> None:
    """A 401 surfaces as a fixed-detail 424. Its Retry-After would describe the
    gateway's own upstream account, which is not the caller's to read."""
    exc = _StatusError(401)
    exc.response = httpx.Response(401, headers={"Retry-After": "34"})  # type: ignore[attr-defined]
    assert provider_error_headers(exc, 424) == {"Otari-Error-Code": PROVIDER_CREDENTIAL_REJECTED}


def test_retry_after_absent_sends_only_the_error_code() -> None:
    assert provider_error_headers(_StatusError(429), 429) == {"Otari-Error-Code": "upstream_rate_limited"}


def test_retry_after_read_through_the_exception_chain() -> None:
    """Read through ``original_exception`` like the other upstream readers, so
    it survives any-llm's unified-exception wrapping."""
    assert upstream_retry_after(_WrappedError(429, _rate_limited_with("7"))) == "7"


def test_platform_terminal_exc_forwards_retry_after() -> None:
    exc = _provider_failure_http_exc(_rate_limited_with("34"), fallback_detail="LLM provider error")
    assert exc.status_code == 429
    assert exc.headers == {"Otari-Error-Code": "upstream_rate_limited", "Retry-After": "34"}


@pytest.mark.parametrize("exc", [_StatusError(500), _StatusError(503), Exception(_RAW), ValueError(_RAW)])
def test_unclassifiable_returns_none(exc: BaseException) -> None:
    assert classify_provider_error(exc) is None


def test_gateway_fault_details_never_echo_the_raw_message() -> None:
    """The statuses where the gateway's own credentials and topology concentrate
    keep a fixed detail, whatever the provider put in the body."""
    for status_code in (401, 403):
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
# A provider the gateway holds no credential for
# ---------------------------------------------------------------------------


def test_missing_credential_maps_to_424_naming_the_provider_and_its_variable() -> None:
    """Nothing went upstream, so the answer is the deployment's to fix: a client
    error that names the provider and the variable, in the gateway's own words."""
    mapping = classify_provider_error(MissingApiKeyError("anthropic", "ANTHROPIC_API_KEY"))
    assert mapping is not None
    assert mapping.status_code == 424
    assert mapping.detail == MissingCredential("anthropic", "ANTHROPIC_API_KEY").detail
    assert "'anthropic'" in mapping.detail
    assert "ANTHROPIC_API_KEY" in mapping.detail
    assert "Please provide it in the config" not in mapping.detail


def test_missing_credential_is_read_through_the_exception_chain() -> None:
    wrapped = _WrappedError(500, MissingApiKeyError("mistral", "MISTRAL_API_KEY"))
    mapping = classify_provider_error(wrapped)
    assert mapping is not None
    assert mapping.status_code == 424
    assert "'mistral'" in mapping.detail
    # The usage row records what the caller was answered, not the wrapper's 500.
    assert failure_status_code(wrapped) == 424


def test_attempts_that_all_lack_a_credential_share_one_424() -> None:
    rejection = get_shared_rejection(
        [MissingApiKeyError("mistral", "MISTRAL_API_KEY"), MissingApiKeyError("cohere", "COHERE_API_KEY")]
    )
    assert rejection is not None
    assert rejection.status_code == 424
    assert rejection.headers == {"Otari-Error-Code": PROVIDER_NOT_CONFIGURED}
    assert "cohere, mistral" in rejection.detail


def test_attempts_that_fail_differently_share_no_rejection_for_a_missing_credential() -> None:
    assert get_shared_rejection([MissingApiKeyError("mistral", "MISTRAL_API_KEY"), _StatusError(404)]) is None


def test_missing_credential_carries_its_error_code() -> None:
    exc = MissingApiKeyError("anthropic", "ANTHROPIC_API_KEY")
    assert provider_error_headers(exc, 424) == {"Otari-Error-Code": PROVIDER_NOT_CONFIGURED}
    assert refusal_code(exc) == PROVIDER_NOT_CONFIGURED
    assert failure_status_code(exc) == 424


@pytest.mark.parametrize("status_code", [401, 403])
def test_rejected_credential_carries_its_error_code(status_code: int) -> None:
    exc = _StatusError(status_code)
    assert provider_error_headers(exc, 424) == {"Otari-Error-Code": PROVIDER_CREDENTIAL_REJECTED}
    assert refusal_code(exc) == PROVIDER_CREDENTIAL_REJECTED
    assert stream_error_code(exc) == PROVIDER_CREDENTIAL_REJECTED
    # The row keeps what the provider said, so misconfiguration stays countable.
    assert failure_status_code(exc) == status_code


def test_attempts_whose_credentials_were_all_rejected_or_missing_share_one_424() -> None:
    rejection = get_shared_rejection([_StatusError(401), MissingApiKeyError("cohere", "COHERE_API_KEY")])
    assert rejection is not None
    assert rejection.status_code == 424
    assert rejection.headers == {"Otari-Error-Code": PROVIDER_CREDENTIAL_REJECTED}
    assert "2 attempts" in rejection.detail
    assert _RAW not in rejection.detail


def test_a_rejected_credential_beside_another_failure_shares_no_rejection() -> None:
    assert get_shared_rejection([_StatusError(401), _StatusError(503)]) is None


def test_missing_credential_error_code_is_not_sent_for_an_upstream_424() -> None:
    """A provider's own 424 is a different failure, so it keeps no code of ours."""
    assert provider_error_headers(_StatusError(424), 424) is None


def test_platform_terminal_exc_for_a_missing_credential_is_a_424() -> None:
    exc = _provider_failure_http_exc(MissingApiKeyError("anthropic", "ANTHROPIC_API_KEY"), fallback_detail="x")
    assert exc.status_code == 424
    assert (exc.headers or {}).get("Otari-Error-Code") == PROVIDER_NOT_CONFIGURED


def test_missing_credential_detail_without_a_variable() -> None:
    assert MissingCredential("vertexai", None).detail.endswith("through the dashboard.")


# ---------------------------------------------------------------------------
# failure_status_code: what lands on usage_logs.status_code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422, 429, 500, 503, 529])
def test_failure_status_code_keeps_the_upstream_status(status_code: int) -> None:
    """The upstream status is recorded verbatim, including the codes the caller
    never sees: an upstream 401/403 surfaces as a fixed-detail 424 (a provider
    rejecting the gateway's credentials must not be echoed as a 401 or 403),
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
        ("Profile arn:aws:bedrock:eu-west-1:123456789012:inference-profile/p1 is not ready", "123456789012"),
        (
            "Profile arn:aws-us-gov:bedrock:us-gov-west-1:123456789012:inference-profile/p1 failed",
            "inference-profile/p1",
        ),
        ("Account 123456789012 has no access to this model", "123456789012"),
        ("Incorrect key ab123456789012cdef0123456789abcd provided", "cdef0123456789abcd"),
        ("Account 1234-5678-9012 has no access to this model", "1234-5678-9012"),
        ("Denied for account123456789012", "123456789012"),
        ("Profile ARN:AWS:bedrock:eu-west-1::inference-profile/p1 is not ready", "inference-profile/p1"),
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


_CONTAINER_MSG = "container requires a provider with a native Anthropic Messages API"


def test_unsupported_container_maps_to_400_naming_the_param() -> None:
    """A Messages request carrying ``container`` against a provider any-llm bridges
    through Chat Completions is refused before the bridge, because only a native
    Anthropic account can resolve the id. The rejection is permanent and the caller
    is the one who can fix it, so it owes a 400 naming the param rather than the
    generic 502 an unclassified failure would report as an upstream outage."""
    mapping = classify_provider_error(NotImplementedError(_CONTAINER_MSG))
    assert mapping is not None
    assert mapping.status_code == 400
    assert "container" in mapping.detail


def test_unsupported_parameter_maps_to_400_with_the_reason() -> None:
    """Typed any-llm capability failures are permanent caller errors, not 502s."""
    exc = UnsupportedParameterError("prompt_cache_key", "anthropic")
    mapping = classify_provider_error(exc)
    assert mapping is not None
    assert mapping.status_code == 400
    assert mapping.detail == "'prompt_cache_key' is not supported for anthropic"
    assert failure_status_code(exc) == 400


def test_unsupported_parameter_payload_echo_uses_fallback_detail() -> None:
    """A capability error that echoes the payload never returns an empty detail."""
    exc = UnsupportedParameterError(
        "prompt_cache_key",
        "bedrock",
        'request_body={"messages":[{"content":"secret prompt"}]}',
    )
    mapping = classify_provider_error(exc)
    assert mapping is not None
    assert mapping.status_code == 400
    assert mapping.detail == PROVIDER_BAD_REQUEST_DETAIL


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


def test_rejected_param_names_a_field_the_route_says_the_caller_set() -> None:
    """A pass-through route's fields are not completion params, so it names the
    ones the caller set, and only those can read as the caller's 400."""
    exc = TypeError("AsyncV2Client.embed() got an unexpected keyword argument 'encoding_format'")
    assert classify_provider_error(exc) is None

    mapping = classify_provider_error(exc, request_params={"encoding_format"})
    assert mapping is not None
    assert mapping.status_code == 400
    assert "'encoding_format'" in mapping.detail
    assert "AsyncV2Client" not in mapping.detail
    assert failure_status_code(exc, request_params={"encoding_format"}) == 400


def test_rejected_param_the_caller_did_not_set_stays_unclassified() -> None:
    """A keyword the route added itself is the gateway's mismatch, not the caller's."""
    exc = TypeError("embed() got an unexpected keyword argument 'inputs'")
    assert classify_provider_error(exc, request_params={"input", "encoding_format"}) is None
    assert failure_status_code(exc, request_params={"input", "encoding_format"}) == 502


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
        "request: {'model': 'command-a', 'query': 'private'}",
        'response="private completion"',
        "request: private prompt",
        "response: private completion",
        "Request = private prompt",
        "invalid response: private completion",
        "invalid request: {'query': 'private'}",
        "invalid request: 'private prompt'",
    ],
)
def test_redaction_rejects_validation_payload_echoes(raw: str) -> None:
    assert redact_upstream_message(raw) == ""


@pytest.mark.parametrize(
    "raw",
    ["invalid request: query must be a non-empty string", "Invalid request: documents must not be empty"],
)
def test_redaction_keeps_a_rejection_phrased_as_invalid_request(raw: str) -> None:
    assert redact_upstream_message(raw) == raw


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
    assert classify_provider_error(_ParamError(429, None, "insufficient_quota")) == (429, "insufficient_quota")


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


# ---------------------------------------------------------------------------
# InvalidRequestError with no HTTP status maps to 400 (Fixes #989)
# ---------------------------------------------------------------------------


def test_status_less_invalid_request_error_maps_to_400() -> None:
    """A bare InvalidRequestError with no HTTP status (as raised by gemini,
    bedrock, and anthropic for bad request shapes) must map to HTTP 400 with
    the provider's own explanation, not fall through to the generic 502."""
    exc = InvalidRequestError("max_tokens exceeds context window for this model")
    mapping = classify_provider_error(exc)
    assert mapping is not None
    assert mapping.status_code == 400
    assert "max_tokens" in mapping.detail


class _StatusLessWrapper(Exception):
    """A wrapper carrying no status of its own, holding the real failure on
    ``original_exception``. ``_WrappedError`` cannot model this: it requires a
    status, and a failure that carries one is classified by that status."""

    def __init__(self, original: BaseException) -> None:
        super().__init__("Invalid request")
        self.status_code = None
        self.original_exception = original


def test_status_less_invalid_request_error_survives_wrapped_error() -> None:
    """The InvalidRequestError type check fires when it lives on
    ``original_exception`` rather than on the failure itself."""
    original = InvalidRequestError("invalid message role: 'system'")
    mapping = classify_provider_error(_StatusLessWrapper(original))
    assert mapping is not None
    assert mapping.status_code == 400
    assert "invalid message role" in mapping.detail


@pytest.mark.parametrize(
    ("status_code", "expected_status"),
    [(401, 424), (403, 424), (429, 429)],
)
def test_a_carried_status_wins_over_an_invalid_request_error_in_the_chain(
    status_code: int, expected_status: int
) -> None:
    """A failure that carries its own status keeps it, even when a status-less
    InvalidRequestError sits anywhere on ``original_exception``.

    Guarding per-link rather than on the failure's own status turned all of
    these into a client 400, which is how a rejected credential's message would
    have reached the caller."""
    wrapped = _WrappedError(status_code, InvalidRequestError("upstream text"))
    mapping = classify_provider_error(wrapped)
    assert mapping is not None
    assert mapping.status_code == expected_status


def test_a_credential_failure_keeps_its_fixed_detail_over_a_chained_invalid_request() -> None:
    """The 401 case specifically: its detail is fixed precisely so an upstream
    message never reaches the caller, which is the invariant
    ``redact_upstream_message`` documents about itself."""
    wrapped = _WrappedError(401, InvalidRequestError("sk-ant-would-leak-here"))
    mapping = classify_provider_error(wrapped)
    assert mapping is not None
    assert mapping.detail == PROVIDER_CREDENTIALS_DETAIL
    assert "sk-ant" not in mapping.detail


def test_status_less_invalid_request_error_is_recorded_as_400() -> None:
    """failure_status_code records 400 for a status-less InvalidRequestError so
    the usage log reflects the real classification, not the generic 502."""
    exc = InvalidRequestError("unknown parameter 'response_format'")
    assert failure_status_code(exc) == 400


def test_invalid_request_error_with_status_is_classified_by_status() -> None:
    """An InvalidRequestError that already carries an HTTP status must not be
    rerouted by the type-based branch; the status branch handles it as usual."""
    exc = InvalidRequestError("bad request shape")
    exc.status_code = 400
    mapping = classify_provider_error(exc)
    assert mapping is not None
    assert mapping.status_code == 400


def test_a_prompt_too_long_for_the_model_has_its_own_code() -> None:
    exc = ContextLengthExceededError("prompt is too long", status_code=400)

    mapping = classify_provider_error(exc)

    assert mapping is not None
    assert mapping.status_code == 400
    assert provider_error_headers(exc, 400) == {"Otari-Error-Code": "context_length_exceeded"}
    assert refusal_code(exc) == "context_length_exceeded"


def test_a_stream_error_event_carries_the_code_that_ended_it() -> None:
    event = openai_error_event(OPENAI_STREAM_FORMAT, refusal_code(_rate_limited_with("3")))

    payload = json.loads(event.removeprefix("data: "))
    assert payload["error"]["code"] == "upstream_rate_limited"
    assert openai_error_event(OPENAI_STREAM_FORMAT, None) == OPENAI_STREAM_FORMAT.error_payload


def test_a_stream_error_event_names_a_cause_the_unstreamed_error_would() -> None:
    exc = TypeError("AsyncClientV2.chat_stream() got an unexpected keyword argument 'n'")
    event = openai_error_event(OPENAI_STREAM_FORMAT, stream_error_code(exc), stream_error_message(exc))

    payload = json.loads(event.removeprefix("data: "))
    assert payload["error"] == {
        "message": "The provider serving this model does not accept the 'n' parameter",
        "type": "server_error",
        "code": "provider_error",
    }


def test_a_stream_error_event_keeps_the_fixed_message_for_a_failure_that_names_no_cause() -> None:
    assert stream_error_message(_StatusError(503)) == "The provider failed the request with HTTP 503"
    assert stream_error_message(RuntimeError("bug")) is None
    assert stream_error_message(HTTPException(status_code=400, detail="internal")) is None


def test_a_stream_ended_by_a_more_specific_refusal_keeps_its_code() -> None:
    assert stream_error_code(_rate_limited_with("3")) == "upstream_rate_limited"
    assert stream_error_code(ContextLengthExceededError("prompt is too long", status_code=400)) == (
        "context_length_exceeded"
    )


@pytest.mark.parametrize(
    "exc",
    [
        _StatusError(400),
        _StatusError(503),
        httpx.ConnectError("refused"),
        ProviderError("upstream broke"),
        TypeError("AsyncClientV2.chat_stream() got an unexpected keyword argument 'n'"),
    ],
    ids=["rejected", "unavailable", "unreachable", "any-llm", "rejected-param"],
)
def test_a_stream_the_provider_failed_is_coded_as_a_provider_error(exc: Exception) -> None:
    assert stream_error_code(exc) == "provider_error"


@pytest.mark.parametrize(
    "exc",
    [
        IncompleteStreamError(3),
        httpx.RemoteProtocolError("peer closed connection without sending complete message body"),
        httpx.ReadError("connection reset"),
        _openai_connection_error(httpx.RemoteProtocolError("Server disconnected")),
    ],
    ids=["no-finish", "chunked-body-cut", "reset", "sdk-wrapped"],
)
def test_a_stream_the_provider_ended_early_has_its_own_code_and_message(exc: Exception) -> None:
    assert stream_error_code(exc) == "provider_stream_interrupted"
    assert stream_error_message(exc) == "The provider's stream ended before it finished"


@pytest.mark.parametrize(
    "exc",
    [MaxToolIterationsExceeded("cap"), RuntimeError("bug"), HTTPException(status_code=502, detail="x")],
    ids=["tool-cap", "gateway-bug", "uncoded-refusal"],
)
def test_a_stream_the_gateway_failed_is_not_blamed_on_the_provider(exc: Exception) -> None:
    assert stream_error_code(exc) is None


# ---------------------------------------------------------------------------
# Bedrock: botocore's ClientError carries its status in a response dict
# ---------------------------------------------------------------------------

_BEDROCK_INVALID_MODEL = "The provided model identifier is invalid."


def _bedrock_error(code: str, message: str, status_code: int | None) -> ClientError:
    """A botocore ``ClientError`` as Bedrock raises it from a Converse call."""
    response: dict[str, Any] = {"Error": {"Code": code, "Message": message}}
    if status_code is not None:
        response["ResponseMetadata"] = {"HTTPStatusCode": status_code, "HTTPHeaders": {}}
    return ClientError(response, "Converse")  # type: ignore[arg-type]


def test_bedrock_validation_error_is_a_caller_fault_400() -> None:
    exc = _bedrock_error("ValidationException", _BEDROCK_INVALID_MODEL, 400)

    mapping = classify_provider_error(exc)

    assert mapping is not None
    assert mapping.status_code == 400
    assert mapping.detail == _BEDROCK_INVALID_MODEL
    assert failure_status_code(exc) == 400


def test_bedrock_caller_fault_detail_is_still_redacted() -> None:
    exc = _bedrock_error("ValidationException", f"Rejected {_RAW}", 400)

    mapping = classify_provider_error(exc)

    assert mapping is not None
    assert "abc123" not in mapping.detail
    assert "Converse" not in mapping.detail


def test_bedrock_caller_fault_detail_hides_the_aws_account() -> None:
    arn = "arn:aws:bedrock:us-east-1:123456789012:provisioned-model/abc1"
    exc = _bedrock_error("ValidationException", f"The provisioned model {arn} is not ready.", 400)

    mapping = classify_provider_error(exc)

    assert mapping is not None
    assert mapping.status_code == 400
    assert "123456789012" not in mapping.detail
    assert "provisioned-model/abc1" not in mapping.detail
    assert "is not ready" in mapping.detail


@pytest.mark.parametrize(
    "raw",
    [
        '{"resource":"arn:aws:bedrock:us-east-1:123456789012:provisioned-model/abc1","reason":"not ready"}',
        "Model (arn:aws:bedrock:us-east-1:123456789012:provisioned-model/abc1), reason: not ready",
    ],
)
def test_redaction_masks_only_the_arn(raw: str) -> None:
    redacted = redact_upstream_message(raw)

    assert "provisioned-model/abc1" not in redacted
    assert "not ready" in redacted


def test_redaction_keeps_a_decimal_number() -> None:
    assert "0.123456789012" in redact_upstream_message("temperature 0.123456789012 is out of range")


def test_bedrock_account_quota_is_the_gateways_fault() -> None:
    exc = _bedrock_error(
        "ServiceQuotaExceededException", "Your request exceeded the service quota for your account.", 400
    )

    assert classify_provider_error(exc) == (502, PROVIDER_ACCOUNT_QUOTA_DETAIL)


def test_bedrock_throttling_forwards_its_retry_after() -> None:
    exc = _bedrock_error("ThrottlingException", "Too many requests, please wait before trying again.", 429)
    exc.response["ResponseMetadata"]["HTTPHeaders"] = {"retry-after": "7"}

    assert upstream_retry_after(exc) == "7"


def test_bedrock_access_denied_keeps_a_fixed_detail() -> None:
    exc = _bedrock_error("AccessDeniedException", f"User arn:aws:iam::123456789012:user/otari {_RAW}", 403)

    assert classify_provider_error(exc) == (424, PROVIDER_CREDENTIALS_DETAIL)
    assert failure_status_code(exc) == 403


def test_bedrock_throttling_is_an_upstream_rate_limit() -> None:
    exc = _bedrock_error("ThrottlingException", "Too many requests, please wait before trying again.", 429)

    mapping = classify_provider_error(exc)

    assert mapping is not None
    assert mapping.status_code == 429
    assert refusal_code(exc) == "upstream_rate_limited"


def test_bedrock_service_failure_stays_the_generic_502() -> None:
    exc = _bedrock_error("ServiceUnavailableException", _RAW, 503)

    assert classify_provider_error(exc) is None
    assert failure_status_code(exc) == 503


def test_bedrock_error_without_a_status_stays_unclassified() -> None:
    exc = _bedrock_error("ValidationException", _BEDROCK_INVALID_MODEL, None)

    assert classify_provider_error(exc) is None
    assert failure_status_code(exc) == 502


def test_bedrock_status_is_read_through_the_unified_exception_wrapper() -> None:
    wrapper = convert_exception(_bedrock_error("ValidationException", _BEDROCK_INVALID_MODEL, 400), "bedrock")

    mapping = classify_provider_error(wrapper)

    assert mapping is not None
    assert mapping.status_code == 400
    assert mapping.detail == _BEDROCK_INVALID_MODEL


# ---------------------------------------------------------------------------
# Whether a client's SDK should retry a provider failure
# ---------------------------------------------------------------------------

_NO_RETRY = {SHOULD_RETRY_HEADER: "false"}


def _key_error_during(cause: BaseException) -> KeyError:
    """A ``KeyError`` raised while handling ``cause``, as in an SDK's own error path."""
    exc = KeyError("content")
    exc.__context__ = cause
    return exc


@pytest.mark.parametrize(
    "exc",
    [
        KeyError("content"),
        IndexError("list index out of range"),
        AttributeError("'NoneType' object has no attribute 'text'"),
        TypeError("unsupported operand type(s)"),
        convert_exception(KeyError("content"), "cohere"),
    ],
    ids=["key", "index", "attribute", "type", "wrapped-by-any-llm"],
)
def test_a_local_fault_is_not_worth_retrying(exc: BaseException) -> None:
    """An adapter that fails before the request leaves fails the same way on every resend."""
    assert is_retry_futile(exc)
    assert with_retry_hint(exc, 502, None) == _NO_RETRY
    assert with_retry_hint(exc, 500, None) == _NO_RETRY


@pytest.mark.parametrize(
    "exc",
    [
        _StatusError(500),
        _StatusError(503),
        _StatusError(429),
        OpenAIAPITimeoutError(request=httpx.Request("POST", "https://api.example.com")),
        httpx.ConnectError("connection refused"),
        httpx.RemoteProtocolError("server disconnected"),
        ConnectionResetError("reset by peer"),
        IncompleteStreamError(0),
        Exception(_RAW),
        ValueError(_RAW),
        _key_error_during(httpx.ReadError("read failed")),
        _key_error_during(_StatusError(503)),
    ],
    ids=[
        "upstream-500",
        "upstream-503",
        "upstream-429",
        "sdk-timeout",
        "connect-error",
        "remote-protocol-error",
        "connection-reset",
        "incomplete-stream",
        "unknown",
        "value-error",
        "local-fault-while-handling-a-network-error",
        "local-fault-while-handling-an-upstream-status",
    ],
)
def test_a_transient_or_unplaced_failure_stays_retryable(exc: BaseException) -> None:
    assert not is_retry_futile(exc)
    assert with_retry_hint(exc, 502, None) is None


def test_an_exhausted_provider_account_is_not_worth_retrying() -> None:
    billing = _ParamError(400, None, "Your credit balance is too low to access the Anthropic API.")
    mapping = classify_provider_error(billing)
    assert mapping is not None
    assert mapping.status_code == 502
    assert is_retry_futile(billing)
    assert with_retry_hint(billing, mapping.status_code, provider_error_headers(billing, 502)) == _NO_RETRY


def test_an_account_quota_stays_retryable() -> None:
    """A service quota can be a rate quota that frees up in seconds, so the client keeps retrying it."""
    quota = _bedrock_error("ServiceQuotaExceededException", "Too many tokens", 400)
    assert not is_retry_futile(quota)


def test_the_retry_hint_keeps_the_refusal_headers() -> None:
    headers = {"Otari-Error-Code": "provider_error"}
    assert with_retry_hint(KeyError("content"), 502, headers) == {**headers, **_NO_RETRY}


def test_no_retry_is_not_sent_below_500() -> None:
    """A client SDK does not retry a 4xx it answers, so the header would say nothing there."""
    rejected = TypeError("create() got an unexpected keyword argument 'seed'")
    mapping = classify_provider_error(rejected)
    assert mapping is not None
    assert mapping.status_code == 400
    assert with_retry_hint(rejected, mapping.status_code, None) is None


def test_platform_terminal_exc_tells_a_client_not_to_retry_a_local_fault() -> None:
    exc = _provider_failure_http_exc(KeyError("content"), fallback_detail="LLM provider error")
    assert exc.status_code == 502
    assert exc.detail == "LLM provider error"
    assert exc.headers == _NO_RETRY


def test_platform_terminal_exc_keeps_retries_for_an_upstream_outage() -> None:
    exc = _provider_failure_http_exc(_StatusError(503), fallback_detail="LLM provider error")
    assert exc.status_code == 502
    assert exc.headers is None


# Provider SDKs that parse the error body onto ``exc.body`` frame it differently
# in their string form: OpenAI and Anthropic as ``Error code: 400 - {...}`` with a
# Python dict repr, Cohere with the response headers ahead of the body. The
# caller gets the sentence inside the body, and never that framing.

_OPENAI_TEMPERATURE_MSG = (
    "Unsupported value: 'temperature' does not support 0.5 with this model. Only the default (1) value is supported."
)
_OPENAI_CONTEXT_MSG = (
    "This model's maximum context length is 128000 tokens. However, your messages resulted in 130000 tokens. "
    "Please reduce the length of the messages."
)
_OPENAI_EMPTY_INPUT_MSG = "Invalid 'input[0]': input cannot be an empty string."
_COHERE_SCHEMA_MSG = "invalid validation: invalid 'json_schema': missing required field 'type'"
_COHERE_HEADERS = {"x-debug-trace-id": "f3a9c2", "x-endpoint-monthly-call-limit": "1000", "server": "envoy"}


def _openai_status_error(status_code: int, error: dict[str, Any]) -> openai.APIStatusError:
    """An OpenAI SDK status error built the way the SDK builds it from a response."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code, request=request, json={"error": error})
    error_class = openai.BadRequestError if status_code == 400 else openai.APIStatusError
    return error_class(f"Error code: {status_code} - {{'error': {error}}}", response=response, body=error)


def _anthropic_bad_request(message: str) -> anthropic.BadRequestError:
    """An Anthropic SDK 400, whose body keeps the ``{"type": "error", "error": {...}}`` envelope."""
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.BadRequestError(
        f"Error code: 400 - {body}", response=httpx.Response(400, request=request, json=body), body=body
    )


def _openai_error(message: str, *, param: str | None = None, code: str | None = None) -> dict[str, Any]:
    return {"message": message, "type": "invalid_request_error", "param": param, "code": code}


@pytest.mark.parametrize("unified", [False, True], ids=["raw", "unified"])
@pytest.mark.parametrize(
    ("upstream", "expected"),
    [
        (
            _openai_status_error(
                400, _openai_error(_OPENAI_TEMPERATURE_MSG, param="temperature", code="unsupported_value")
            ),
            _OPENAI_TEMPERATURE_MSG,
        ),
        (_openai_status_error(400, _openai_error(_OPENAI_EMPTY_INPUT_MSG, param="input")), _OPENAI_EMPTY_INPUT_MSG),
        (_anthropic_bad_request("max_tokens: Field required"), "max_tokens: Field required"),
        (
            CohereBadRequestError(body={"id": "a1b2", "message": _COHERE_SCHEMA_MSG}, headers=_COHERE_HEADERS),
            _COHERE_SCHEMA_MSG,
        ),
    ],
    ids=["openai-unsupported-value", "openai-empty-embedding-input", "anthropic", "cohere"],
)
def test_a_provider_rejection_reaches_the_caller_as_the_providers_own_sentence(
    upstream: Exception, expected: str, unified: bool
) -> None:
    exc = convert_exception(upstream, "provider") if unified else upstream

    assert classify_provider_error(exc) == (400, expected)
    assert stream_error_message(exc) == expected


def test_a_prompt_too_long_reaches_the_caller_unwrapped_and_keeps_its_code() -> None:
    upstream = _openai_status_error(
        400, _openai_error(_OPENAI_CONTEXT_MSG, param="messages", code="context_length_exceeded")
    )
    exc = convert_exception(upstream, "openai")

    assert classify_provider_error(exc) == (400, _OPENAI_CONTEXT_MSG)
    assert provider_error_headers(exc, 400) == {"Otari-Error-Code": "context_length_exceeded"}


def test_a_provider_404_and_429_read_the_body_too() -> None:
    not_found = _openai_status_error(404, _openai_error("The model `gpt-9` does not exist", code="model_not_found"))
    rate_limited = _openai_status_error(429, _openai_error("Rate limit reached for gpt-4o on tokens per min"))

    assert classify_provider_error(not_found) == (404, "The model `gpt-9` does not exist")
    assert classify_provider_error(rate_limited) == (429, "Rate limit reached for gpt-4o on tokens per min")


def test_cohere_response_headers_never_reach_the_caller() -> None:
    """Cohere's string form leads with the response headers; a body with no message falls back."""
    for upstream in (
        CohereBadRequestError(body={"id": "a1b2", "message": _COHERE_SCHEMA_MSG}, headers=_COHERE_HEADERS),
        CohereBadRequestError(body={"id": "a1b2"}, headers=_COHERE_HEADERS),
    ):
        for exc in (upstream, convert_exception(upstream, "cohere")):
            mapping = classify_provider_error(exc)
            assert mapping is not None
            assert "x-debug-trace-id" not in mapping.detail
            assert "f3a9c2" not in mapping.detail
            assert "headers" not in mapping.detail

    assert classify_provider_error(CohereBadRequestError(body={"id": "a1b2"}, headers=_COHERE_HEADERS)) == (
        400,
        PROVIDER_BAD_REQUEST_DETAIL,
    )


def test_a_body_message_is_still_redacted() -> None:
    upstream = _openai_status_error(
        400, _openai_error("Invalid key sk-proj-ABCDEFGHIJKLMNOP for https://proxy.internal/v1")
    )

    mapping = classify_provider_error(upstream)

    assert mapping is not None
    assert "sk-proj-ABCDEFGHIJKLMNOP" not in mapping.detail
    assert "proxy.internal" not in mapping.detail
    assert mapping.detail.startswith("Invalid key [redacted]")


def test_a_body_message_echoing_the_request_falls_back_without_reading_the_sdk_string() -> None:
    """When the body's message is refused as a payload echo, the dict repr around it is not tried instead."""
    upstream = _openai_status_error(400, _openai_error("messages: [{'role': 'user', 'content': 'my secret prompt'}]"))

    assert classify_provider_error(upstream) == (400, PROVIDER_BAD_REQUEST_DETAIL)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('{"object": "error", "message": "Invalid model: foo", "type": "invalid_model"}', "Invalid model: foo"),
        ("Unknown field 'foo'", "Unknown field 'foo'"),
        ("<html><body><h1>400 Bad Request</h1></body></html>", PROVIDER_BAD_REQUEST_DETAIL),
        ("{not json", PROVIDER_BAD_REQUEST_DETAIL),
        ({"error": "model is required"}, "model is required"),
        ({"error": "invalid_request_error", "message": "model is required"}, "model is required"),
        ({"error": {"code": 400}}, PROVIDER_BAD_REQUEST_DETAIL),
        (["not", "a", "mapping"], PROVIDER_BAD_REQUEST_DETAIL),
    ],
    ids=[
        "json-text",
        "plain-text",
        "html",
        "broken-json",
        "error-string",
        "error-type-beside-message",
        "no-message",
        "list",
    ],
)
def test_body_shapes_other_sdks_attach(body: object, expected: str) -> None:
    exc = _StatusError(400)
    exc.body = body  # type: ignore[attr-defined]

    assert classify_provider_error(exc) == (400, expected)


def test_a_billing_message_in_the_body_is_still_the_gateways_fault() -> None:
    upstream = _anthropic_bad_request("Your credit balance is too low to access the Anthropic API.")

    assert classify_provider_error(upstream) == (502, PROVIDER_BILLING_DETAIL)


def test_a_credential_rejection_never_echoes_its_body() -> None:
    upstream = _openai_status_error(401, _openai_error("Incorrect API key provided: sk-proj-ABCDEFGHIJKLMNOP"))

    assert classify_provider_error(upstream) == (424, PROVIDER_CREDENTIALS_DETAIL)


# ---------------------------------------------------------------------------
# An upstream 403 that refuses the request, not the gateway's key
# ---------------------------------------------------------------------------

_OPENAI_LOGPROBS_MSG = "You are not allowed to request logprobs from this model"


def _anthropic_forbidden() -> anthropic.PermissionDeniedError:
    body = {"type": "error", "error": {"type": "permission_error", "message": "Your API key does not have permission"}}
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.PermissionDeniedError(
        f"Error code: 403 - {body}", response=httpx.Response(403, request=request, json=body), body=body
    )


@pytest.mark.parametrize("unified", [False, True], ids=["raw", "unified"])
def test_an_openai_403_invalid_request_is_the_callers_400_with_the_providers_sentence(unified: bool) -> None:
    upstream = _openai_status_error(403, _openai_error(_OPENAI_LOGPROBS_MSG))
    exc = convert_exception(upstream, "openai") if unified else upstream

    assert is_upstream_request_fault(exc)
    assert not is_provider_credential_rejection(exc)
    assert classify_provider_error(exc) == (400, _OPENAI_LOGPROBS_MSG)
    assert provider_error_headers(exc, 400) is None
    assert stream_error_code(exc) != PROVIDER_CREDENTIAL_REJECTED
    # The row keeps the provider's own status.
    assert failure_status_code(exc) == 403


@pytest.mark.parametrize(
    "upstream",
    [
        _openai_status_error(403, _openai_error("Project `proj_abc` does not have access to model `gpt-5`")),
        _openai_status_error(403, {"message": "Country not supported", "type": "request_forbidden", "code": None}),
        _anthropic_forbidden(),
        _StatusError(403),
    ],
    ids=["openai-project", "openai-request-forbidden", "anthropic-permission", "no-body"],
)
def test_a_403_about_the_key_stays_a_rejected_credential(upstream: Exception) -> None:
    assert not is_upstream_request_fault(upstream)
    assert is_provider_credential_rejection(upstream)
    assert classify_provider_error(upstream) == (424, PROVIDER_CREDENTIALS_DETAIL)


def test_a_401_is_a_rejected_credential_whatever_its_body_says() -> None:
    upstream = _openai_status_error(401, _openai_error("Incorrect API key provided"))
    assert not is_upstream_request_fault(upstream)
    assert classify_provider_error(upstream) == (424, PROVIDER_CREDENTIALS_DETAIL)


def test_attempts_that_all_hit_a_request_fault_403_share_the_callers_400() -> None:
    errors = [_openai_status_error(403, _openai_error(_OPENAI_LOGPROBS_MSG)) for _ in range(2)]
    rejection = get_shared_rejection(errors)
    assert rejection is not None
    assert rejection.status_code == 400
    assert rejection.headers != {"Otari-Error-Code": PROVIDER_CREDENTIAL_REJECTED}


# ---------------------------------------------------------------------------
# provider_failure_detail: what the 502 says happened upstream
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unified", [False, True], ids=["raw", "unified"])
@pytest.mark.parametrize(
    ("upstream", "expected"),
    [
        (_StatusError(500), "The provider failed the request with HTTP 500"),
        (_StatusError(503), "The provider failed the request with HTTP 503"),
        (
            _openai_connection_error(httpx.RemoteProtocolError("Server disconnected without sending a response.")),
            "The provider closed the connection before answering",
        ),
        (
            _openai_connection_error(httpx.ConnectError("[Errno 61] Connection refused")),
            "The gateway could not connect to the provider",
        ),
    ],
    ids=["500", "503", "closed", "unreachable"],
)
def test_a_provider_failure_left_to_the_502_says_what_happened_upstream(
    upstream: Exception, expected: str, unified: bool
) -> None:
    exc = convert_exception(upstream, "openai") if unified else upstream

    assert classify_provider_error(exc) is None
    detail = provider_failure_detail(exc)
    assert detail == expected
    assert _RAW not in detail
    assert "internal-proxy" not in detail


def test_a_failure_inside_the_gateway_keeps_the_fixed_detail() -> None:
    assert provider_failure_detail(RuntimeError("bug at /srv/otari")) == PROVIDER_ERROR_DETAIL
    assert provider_failure_detail(RuntimeError("bug"), "fallback") == "fallback"


def test_the_502_keeps_its_retry_behavior() -> None:
    """Only the detail changed: a provider 5xx still carries no x-should-retry, so a client retries it."""
    answer = _provider_failure_http_exc(_StatusError(500), fallback_detail="LLM provider error")
    assert answer.status_code == 502
    assert answer.detail == "The provider failed the request with HTTP 500"
    assert SHOULD_RETRY_HEADER not in (answer.headers or {})


# What two providers answered, verbatim, to an image sent to a text-only model.
_FIREWORKS_IMAGE_MSG = "Unsupported ChatMessageContent type: image_url"
_COHERE_IMAGE_MSG = "invalid request: image content is not supported for this model"
_IMAGE_DETAIL_TAIL = (
    "does not accept image content. Remove the image parts, or send the request to a model that accepts images."
)


def _image_rejections() -> list[Exception]:
    fireworks = _openai_status_error(400, _openai_error(_FIREWORKS_IMAGE_MSG))
    cohere = CohereBadRequestError(body={"id": "a1b2", "message": _COHERE_IMAGE_MSG}, headers=_COHERE_HEADERS)
    return [fireworks, cohere, convert_exception(fireworks, "fireworks"), convert_exception(cohere, "cohere")]


@pytest.mark.parametrize("exc", _image_rejections(), ids=["fireworks", "cohere", "fireworks-unified", "cohere-unified"])
def test_an_image_the_model_cannot_take_is_one_400_with_one_code(exc: Exception) -> None:
    mapping = classify_provider_error(exc, model="cohere:command-r7b-12-2024")

    assert mapping == (400, f"The model 'cohere:command-r7b-12-2024' {_IMAGE_DETAIL_TAIL}")
    assert provider_error_headers(exc, 400) == {"Otari-Error-Code": "unsupported_content"}
    assert refusal_code(exc) == "unsupported_content"
    assert failure_status_code(exc) == 400


def test_an_image_rejection_without_a_model_names_the_serving_model() -> None:
    exc = _image_rejections()[0]

    assert classify_provider_error(exc) == (400, f"The model serving this request {_IMAGE_DETAIL_TAIL}")
    assert stream_error_code(exc) == "unsupported_content"
    assert stream_error_message(exc) == f"The model serving this request {_IMAGE_DETAIL_TAIL}"


def test_an_image_rejection_is_recorded_on_the_usage_row_in_the_gateways_words() -> None:
    cohere = _image_rejections()[1]

    # Cohere's string form redacts to nothing, which left the row reading "LLM provider error".
    assert redact_upstream_message(str(cohere)) == ""
    assert failure_error_message(cohere) == f"The model serving this request {_IMAGE_DETAIL_TAIL}"
    assert failure_error_message(_StatusError(400)) == _RAW


def test_an_image_mention_on_a_gateway_fault_is_not_an_image_rejection() -> None:
    exc = _openai_status_error(503, _openai_error(_COHERE_IMAGE_MSG))

    assert classify_provider_error(exc) is None
    assert provider_error_headers(exc, 503) is None


def test_an_image_refusal_in_wording_not_seen_yet_keeps_the_providers_sentence() -> None:
    """Only wording seen in a real response is matched; anything else stays the provider's 400."""
    message = "Invalid content type. image_url is only supported by certain models."
    exc = _openai_status_error(400, _openai_error(message))

    assert classify_provider_error(exc) == (400, message)
    assert provider_error_headers(exc, 400) is None


def test_attempts_that_all_refuse_the_image_share_one_400() -> None:
    fireworks, cohere = _image_rejections()[:2]

    rejection = get_shared_rejection([fireworks, cohere])

    assert rejection is not None
    assert rejection.status_code == 400
    assert rejection.detail == "None of the models tried accepts image content (2 attempts)."
    assert rejection.headers == {"Otari-Error-Code": "unsupported_content"}


def test_an_image_refusal_beside_another_rejection_shares_the_generic_400() -> None:
    other = _openai_status_error(400, _openai_error(_OPENAI_TEMPERATURE_MSG))

    rejection = get_shared_rejection([_image_rejections()[0], other])

    assert rejection is not None
    assert rejection.headers == {"Otari-Error-Code": "all_candidates_rejected"}
