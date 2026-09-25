"""One table decides an Anthropic ``error.type``.

An error raised inside the gateway carries its kind.
One that reached the route already flattened into an ``HTTPException`` has only a status.
Both reach the same table, so one status cannot be classified two ways.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException, status

from gateway.api.routes._pipeline import ErrorKind
from gateway.api.routes.messages import _ADAPTER, _ERROR_KIND_TO_ANTHROPIC_TYPE, _ensure_anthropic_error


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
