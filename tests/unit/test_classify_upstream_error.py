"""Unit tests for ``_classify_upstream_error``, the function that decides
whether an upstream failure falls through to the next routing-policy attempt.
"""

import asyncio

import httpx
import pytest
from anthropic import APIConnectionError as AnthropicAPIConnectionError
from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic import APITimeoutError as AnthropicAPITimeoutError
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import APIStatusError as OpenAIAPIStatusError
from openai import APITimeoutError as OpenAIAPITimeoutError

from gateway.api.routes._platform import _classify_upstream_error


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://upstream")
    return httpx.HTTPStatusError(
        str(status),
        request=request,
        response=httpx.Response(status, request=request),
    )


@pytest.mark.parametrize(
    "status",
    [401, 403, 404, 405, 408, 409, 410, 429, 500, 502, 503, 504],
)
def test_retryable_status_codes_fall_through(status: int) -> None:
    retryable, error_class = _classify_upstream_error(_http_error(status))
    assert retryable is True
    assert error_class == f"http_{status}"


@pytest.mark.parametrize("status", [400, 422, 413, 414, 431])
def test_malformed_request_status_codes_are_terminal(status: int) -> None:
    # 400/422 are explicitly terminal; other "bad request" shapes (413/414/431)
    # are request-level problems that every provider would reject, so they must
    # not waste fallback attempts.
    retryable, error_class = _classify_upstream_error(_http_error(status))
    assert retryable is False
    assert error_class == f"http_{status}"


@pytest.mark.parametrize(
    "exc, expected_class",
    [
        (asyncio.TimeoutError(), "timeout"),
        (httpx.TimeoutException("slow"), "timeout"),
        (httpx.ConnectError("refused"), "conn_err"),
    ],
)
def test_network_and_timeout_errors_are_retryable(exc: BaseException, expected_class: str) -> None:
    retryable, error_class = _classify_upstream_error(exc)
    assert retryable is True
    assert error_class == expected_class


def test_error_without_status_is_unknown_and_terminal() -> None:
    retryable, error_class = _classify_upstream_error(ValueError("no status here"))
    assert retryable is False
    assert error_class == "unknown"


@pytest.mark.parametrize(
    "sdk_error, expected_class",
    [
        (OpenAIAPITimeoutError, "timeout"),
        (AnthropicAPITimeoutError, "timeout"),
        (OpenAIAPIConnectionError, "conn_err"),
        (AnthropicAPIConnectionError, "conn_err"),
    ],
)
def test_sdk_wrapped_timeout_and_connection_errors_are_retryable(sdk_error: type, expected_class: str) -> None:
    """any-llm calls the OpenAI/Anthropic SDKs directly, and both SDKs catch
    ``httpx.TimeoutException``/network errors internally and re-raise as their
    own ``APITimeoutError``/``APIConnectionError`` (``APITimeoutError``
    subclasses ``APIConnectionError`` in both SDKs). Neither wrapped type is an
    instance of any httpx exception or carries ``status_code``/``response``, so
    a real provider timeout or "provider unreachable" reaches the classifier in
    this shape in production, not as a raw httpx exception.
    """
    request = httpx.Request("POST", "http://upstream")
    exc = sdk_error(request=request)
    assert not hasattr(exc, "status_code")

    retryable, error_class = _classify_upstream_error(exc)
    assert retryable is True
    assert error_class == expected_class


@pytest.mark.parametrize(
    "class_name, expected_class",
    [
        ("SomeProviderAPITimeoutError", "timeout"),
        ("SomeProviderConnectionError", "conn_err"),
    ],
)
def test_duck_typed_fallback_for_unenumerated_sdks(class_name: str, expected_class: str) -> None:
    """Provider SDKs any-llm supports beyond OpenAI/Anthropic (cohere, mistral,
    groq, bedrock, ...) may raise their own timeout/connection exception types
    that this module doesn't import. A conservative class-name fallback still
    classifies them as retryable rather than falling into ``unknown``, as long
    as they carry no status code (so a real HTTP-status classification is
    never shadowed).
    """
    exc_type = type(class_name, (Exception,), {})
    retryable, error_class = _classify_upstream_error(exc_type("boom"))
    assert retryable is True
    assert error_class == expected_class


class _WrappedByAnyLLM(Exception):
    """Mimics ``any_llm.exceptions.AnyLLMError`` once ``ANY_LLM_UNIFIED_EXCEPTIONS=1``
    is enabled: the outer exception carries no ``status_code`` and a class name
    the duck-typed fallback doesn't recognize (``convert_exception`` maps a raw
    SDK timeout/connection error to the generic ``ProviderError`` bucket), but
    preserves the raw SDK exception on ``original_exception``.
    """

    def __init__(self, original_exception: BaseException) -> None:
        super().__init__(str(original_exception))
        self.original_exception = original_exception


@pytest.mark.parametrize(
    "sdk_error, expected_class",
    [
        (OpenAIAPITimeoutError, "timeout"),
        (AnthropicAPITimeoutError, "timeout"),
        (OpenAIAPIConnectionError, "conn_err"),
        (AnthropicAPIConnectionError, "conn_err"),
    ],
)
def test_unwraps_original_exception_for_unified_any_llm_errors(sdk_error: type, expected_class: str) -> None:
    """Once otari enables ``ANY_LLM_UNIFIED_EXCEPTIONS=1``, a raw SDK
    timeout/connection error arrives wrapped in a generic ``AnyLLMError``
    subclass rather than the SDK type directly. The classifier must still
    recognize it by unwrapping ``original_exception``, or this regresses back
    to the exact "unknown"/non-retryable bug this module fixes.
    """
    request = httpx.Request("POST", "http://upstream")
    wrapped = _WrappedByAnyLLM(sdk_error(request=request))

    retryable, error_class = _classify_upstream_error(wrapped)
    assert retryable is True
    assert error_class == expected_class


def test_unwrap_terminates_on_self_referential_original_exception() -> None:
    """Defensive: a (pathological, shouldn't-happen-in-practice) exception
    whose ``original_exception`` points back to itself must not loop forever;
    it should fall through to the same terminal ``unknown`` classification as
    any other unrecognized exception.
    """
    exc = _WrappedByAnyLLM(ValueError("placeholder"))
    exc.original_exception = exc  # self-reference, must not loop

    retryable, error_class = _classify_upstream_error(exc)
    assert retryable is False
    assert error_class == "unknown"


@pytest.mark.parametrize("sdk_error", [AnthropicAPIStatusError, OpenAIAPIStatusError])
@pytest.mark.parametrize("status, retryable", [(404, True), (410, True), (400, False), (422, False)])
def test_classifies_provider_sdk_status_errors(sdk_error: type, status: int, retryable: bool) -> None:
    # any_llm propagates the provider SDK's own ``APIStatusError`` (it does not
    # wrap upstream failures), and that exception exposes ``status_code``
    # directly on itself. The other tests here use ``httpx.HTTPStatusError``,
    # which only carries the code on ``.response`` (a shape that never reaches
    # the classifier in production). Pin the classifier against the exception
    # type providers actually raise so the direct-``status_code`` extraction
    # path stays covered.
    request = httpx.Request("POST", "http://upstream")
    exc = sdk_error(str(status), response=httpx.Response(status, request=request), body=None)
    assert getattr(exc, "status_code", None) == status  # guards the production exception shape

    classified_retryable, error_class = _classify_upstream_error(exc)
    assert classified_retryable is retryable
    assert error_class == f"http_{status}"


class _MessageStatusError(Exception):
    """Upstream error carrying both a status code and a provider message, the
    shape a provider SDK's ``BadRequestError`` reaches the classifier in."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


# Anthropic reports an out-of-credit account as a 400 invalid_request_error.
_ANTHROPIC_BILLING_MSG = (
    "Your credit balance is too low to access the Anthropic API. "
    "Please go to Plans & Billing to upgrade or purchase credits."
)


@pytest.mark.parametrize(
    "status, message",
    [
        (400, _ANTHROPIC_BILLING_MSG),
        (402, "Insufficient Balance"),
        (422, "insufficient credits for this request"),
    ],
)
def test_billing_exhaustion_falls_through_to_the_next_attempt(status: int, message: str) -> None:
    """An out-of-credit account is a condition on THIS provider, not a malformed
    request, so the route's next attempt is worth making. Reported by clawbolt:
    an Anthropic account ran dry, the 400 was treated as terminal, and traffic
    never failed over to the other providers configured on the same alias."""
    retryable, error_class = _classify_upstream_error(_MessageStatusError(status, message))
    assert retryable is True
    assert error_class == f"http_{status}_billing"


def test_billing_exhaustion_detected_through_original_exception() -> None:
    """The signal is found when it lives only on ``original_exception`` (the
    any-llm unified-exception shape)."""
    wrapped = _WrappedByAnyLLM(_MessageStatusError(400, _ANTHROPIC_BILLING_MSG))
    wrapped.status_code = 400  # type: ignore[attr-defined]
    retryable, error_class = _classify_upstream_error(wrapped)
    assert retryable is True
    assert error_class == "http_400_billing"


def test_malformed_400_is_still_terminal_after_the_billing_probe() -> None:
    """The billing probe must not sweep up a genuinely malformed request: an
    unrecognized 400 message stays terminal so it does not waste an attempt
    against every provider in the route."""
    exc = _MessageStatusError(400, "messages.3: tool_use ids were found without tool_result blocks")
    assert _classify_upstream_error(exc) == (False, "http_400")


def test_billing_probe_does_not_reinterpret_other_statuses() -> None:
    """A billing phrase on a status outside the candidate set keeps its normal
    classification and error_class, so the probe can only ever rescue a 400/402/422
    dead end."""
    assert _classify_upstream_error(_MessageStatusError(503, _ANTHROPIC_BILLING_MSG)) == (True, "http_503")
    assert _classify_upstream_error(_MessageStatusError(429, "insufficient_quota")) == (True, "http_429")


def test_payment_required_phrase_only_counts_on_a_402() -> None:
    """The words "payment required" are the 402 reason phrase, which httpx puts in
    every stringified 402, so they are a status signal rather than a provider
    phrase. On a 400/422 the same words are far more likely to be a
    caller-supplied value the provider echoed back, which must stay a terminal
    malformed request."""
    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    response = httpx.Response(402, request=request, json={"error": {"message": "out of funds"}})
    with pytest.raises(httpx.HTTPStatusError) as raised:
        response.raise_for_status()
    assert _classify_upstream_error(raised.value) == (True, "http_402_billing")

    echoed = _MessageStatusError(400, "Invalid value: 'payment required'. Supported values are: 'none', 'auto'.")
    assert _classify_upstream_error(echoed) == (False, "http_400")
