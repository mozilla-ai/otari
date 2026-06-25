"""Unit tests for classify_provider_error.

The classifier maps an upstream provider exception to a safe, client-facing
(status, detail). It must never echo the raw provider message, and must return
None for failures it cannot safely classify so callers keep the generic 502.
"""

import asyncio

import httpx
import pytest

from gateway.api.routes._pipeline import (
    PROVIDER_BAD_REQUEST_DETAIL,
    PROVIDER_CREDENTIALS_DETAIL,
    PROVIDER_MODEL_NOT_FOUND_DETAIL,
    PROVIDER_RATE_LIMITED_DETAIL,
    PROVIDER_TIMEOUT_DETAIL,
    classify_provider_error,
)
from gateway.api.routes._platform import _provider_failure_http_exc

_RAW = "raw provider detail SECRET token=abc123"


class _StatusError(Exception):
    """Upstream error exposing a top-level status_code, like any-llm surfaces."""

    def __init__(self, status_code: int) -> None:
        super().__init__(_RAW)
        self.status_code = status_code


class _ResponseStatusError(Exception):
    """Upstream error exposing status via an attached response object."""

    def __init__(self, status_code: int) -> None:
        super().__init__(_RAW)
        self.response = httpx.Response(status_code)


def test_timeout_maps_to_504() -> None:
    for exc in (asyncio.TimeoutError(), TimeoutError(), httpx.TimeoutException("slow")):
        mapping = classify_provider_error(exc)
        assert mapping == (504, PROVIDER_TIMEOUT_DETAIL)


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
