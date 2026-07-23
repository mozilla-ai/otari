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
