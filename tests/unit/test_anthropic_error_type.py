"""One table decides an Anthropic ``error.type``.

An error raised inside the gateway carries its kind.
One that reached the route already flattened into an ``HTTPException`` has only a status.
Both reach the same table, so one status cannot be classified two ways.
"""

from __future__ import annotations

import json
from typing import Any, cast

import anthropic
import httpx
import pytest
from fastapi import HTTPException, status

from gateway.api.routes._pipeline import ErrorKind
from gateway.api.routes.messages import _ADAPTER, _ERROR_KIND_TO_ANTHROPIC_TYPE, _ensure_anthropic_error
from gateway.streaming import ANTHROPIC_STREAM_FORMAT


def _rendered(exc: HTTPException) -> str:
    assert isinstance(exc.detail, dict)
    return str(exc.detail["error"]["type"])


def test_every_kind_renders() -> None:
    """A kind with no entry would raise ``KeyError`` at the moment of a refusal."""
    assert set(_ERROR_KIND_TO_ANTHROPIC_TYPE) == set(ErrorKind)


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (status.HTTP_400_BAD_REQUEST, "invalid_request_error"),
        (status.HTTP_401_UNAUTHORIZED, "authentication_error"),
        (status.HTTP_403_FORBIDDEN, "permission_error"),
        (status.HTTP_404_NOT_FOUND, "not_found_error"),
        (status.HTTP_429_TOO_MANY_REQUESTS, "rate_limit_error"),
        (status.HTTP_500_INTERNAL_SERVER_ERROR, "api_error"),
        (status.HTTP_502_BAD_GATEWAY, "api_error"),
    ],
)
def test_a_flattened_error_renders_the_type_its_status_has_always_given(status_code: int, expected: str) -> None:
    assert _rendered(_ensure_anthropic_error(HTTPException(status_code=status_code, detail="boom"))) == expected


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (ErrorKind.API, "api_error"),
        (ErrorKind.AUTHENTICATION, "authentication_error"),
        (ErrorKind.INVALID_REQUEST, "invalid_request_error"),
        (ErrorKind.NOT_FOUND, "not_found_error"),
        (ErrorKind.PERMISSION, "permission_error"),
        (ErrorKind.RATE_LIMIT, "rate_limit_error"),
    ],
)
def test_a_declared_kind_renders_its_own_type(kind: ErrorKind, expected: str) -> None:
    """The adapter renders it, so this fails if the presenter stops consulting the table."""
    assert _rendered(_ADAPTER.error(400, "boom", kind)) == expected


class _UpstreamStatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__("raw upstream detail")
        self.status_code = status_code


@pytest.mark.parametrize(
    ("upstream_status", "expected"),
    [
        (400, "invalid_request_error"),
        (404, "not_found_error"),
        (429, "rate_limit_error"),
    ],
)
def test_a_classified_provider_failure_renders_from_its_status(upstream_status: int, expected: str) -> None:
    assert _rendered(_ADAPTER.provider_error(_UpstreamStatusError(upstream_status))) == expected


def test_an_unclassifiable_provider_failure_says_nothing_about_the_provider() -> None:
    rendered = _ADAPTER.provider_error(RuntimeError("upstream stack trace"))

    assert rendered.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert _rendered(rendered) == "api_error"
    assert isinstance(rendered.detail, dict)
    assert "upstream" not in str(rendered.detail["error"]["message"])


def test_an_already_enveloped_error_is_left_alone() -> None:
    enveloped = HTTPException(status_code=404, detail={"type": "error", "error": {"type": "custom", "message": "x"}})

    assert _ensure_anthropic_error(enveloped) is enveloped


def test_the_retry_hint_survives_enveloping() -> None:
    limited = HTTPException(status_code=429, detail="slow down", headers={"Retry-After": "30"})

    assert _ensure_anthropic_error(limited).headers == {"Retry-After": "30"}


def _stream_error(exc: BaseException) -> dict[str, Any]:
    payload = _ADAPTER.stream_error_payload(exc)
    event_line, data_line, *_ = payload.split("\n")
    assert event_line == "event: error"
    return cast(dict[str, Any], json.loads(data_line.removeprefix("data: ")))


def _mid_stream_error(error_type: str) -> anthropic.APIStatusError:
    """What the Anthropic SDK raises for an SSE ``error`` event: the stream's 200, the event as body."""
    body = {"type": "error", "error": {"type": error_type, "message": "raw provider text"}, "request_id": "req_x"}
    response = httpx.Response(200, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    return anthropic.APIStatusError(str(body), response=response, body=body)


class _Wrapped(Exception):
    def __init__(self, original: BaseException) -> None:
        super().__init__("wrapped")
        self.original_exception = original


@pytest.mark.parametrize("error_type", ["overloaded_error", "rate_limit_error"])
def test_a_transient_mid_stream_failure_keeps_its_type(error_type: str) -> None:
    rendered = _stream_error(_mid_stream_error(error_type))

    assert rendered["error"]["type"] == error_type
    assert "raw provider text" not in rendered["error"]["message"]
    assert "req_x" not in json.dumps(rendered)


def test_the_type_survives_an_any_llm_wrapper() -> None:
    assert _stream_error(_Wrapped(_mid_stream_error("overloaded_error")))["error"]["type"] == "overloaded_error"


@pytest.mark.parametrize(("upstream_status", "expected"), [(529, "overloaded_error"), (429, "rate_limit_error")])
def test_a_transient_status_without_a_body_is_named_from_the_status(upstream_status: int, expected: str) -> None:
    assert _stream_error(_UpstreamStatusError(upstream_status))["error"]["type"] == expected


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("upstream stack trace"),
        _mid_stream_error("authentication_error"),
        _mid_stream_error("invalid_request_error"),
        _UpstreamStatusError(401),
    ],
)
def test_any_other_mid_stream_failure_stays_the_generic_api_error(exc: BaseException) -> None:
    assert _ADAPTER.stream_error_payload(exc) == ANTHROPIC_STREAM_FORMAT.error_payload
