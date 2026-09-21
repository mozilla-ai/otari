"""Unit tests for the gateway credential-token extraction.

These tests exercise ``extract_credential_token`` directly — a pure function over
``Request.headers`` — so they cover all precedence and malformed-input branches
without needing the full FastAPI stack or a database.
"""

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from gateway.api.deps import extract_credential_token
from gateway.core.config import API_KEY_HEADER, X_API_KEY_HEADER
from gateway.metrics import REGISTRY


def _make_request(headers: dict[str, str]) -> Request:
    """Build a minimal ASGI ``Request`` with the provided headers."""
    scope: dict[str, object] = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "query_string": b"",
        "headers": [(name.lower().encode("latin-1"), value.encode("latin-1")) for name, value in headers.items()],
        "state": {},
    }
    return Request(scope)


def test_canonical_header_returns_token() -> None:
    request = _make_request({API_KEY_HEADER: "Bearer token-canonical"})

    assert extract_credential_token(request) == "token-canonical"


def test_authorization_header_returns_token() -> None:
    request = _make_request({"Authorization": "Bearer token-auth"})

    assert extract_credential_token(request) == "token-auth"


def test_canonical_takes_precedence_over_authorization() -> None:
    request = _make_request(
        {
            API_KEY_HEADER: "Bearer canonical-wins",
            "Authorization": "Bearer authorization-loses",
        }
    )

    assert extract_credential_token(request) == "canonical-wins"


@pytest.mark.parametrize("legacy_header", ["AnyLLM-Key", "X-AnyLLM-Key"])
def test_legacy_header_is_no_longer_honored(legacy_header: str) -> None:
    """The pre-rename AnyLLM-Key / X-AnyLLM-Key aliases were removed: a request
    carrying only a legacy header is treated as missing credentials."""
    request = _make_request({legacy_header: "Bearer token-legacy"})

    with pytest.raises(HTTPException) as exc_info:
        extract_credential_token(request)

    assert exc_info.value.status_code == 401
    assert API_KEY_HEADER in exc_info.value.detail


def test_canonical_header_accepts_raw_token() -> None:
    # The dashboard hands out `Otari-Key: gw-...` with no Bearer prefix; the raw
    # token is returned verbatim.
    request = _make_request({API_KEY_HEADER: "gw-raw-canonical"})

    assert extract_credential_token(request) == "gw-raw-canonical"


def test_malformed_authorization_header_raises_401() -> None:
    """A non-Bearer Authorization scheme is the ``invalid_format`` auth-failure trigger.

    It is the only one since the ``gw-`` shape check came off the key-verify path
    (issue #646), so the metric label is pinned here rather than left to lapse.
    """
    request = _make_request({"Authorization": "Basic abc123"})
    before = REGISTRY.get_sample_value("gateway_auth_failures_total", {"reason": "invalid_format"}) or 0.0

    with pytest.raises(HTTPException) as exc_info:
        extract_credential_token(request)

    assert exc_info.value.status_code == 401
    after = REGISTRY.get_sample_value("gateway_auth_failures_total", {"reason": "invalid_format"}) or 0.0
    assert after - before == 1.0


def test_missing_credentials_raises_401() -> None:
    request = _make_request({})

    with pytest.raises(HTTPException) as exc_info:
        extract_credential_token(request)

    assert exc_info.value.status_code == 401
    assert API_KEY_HEADER in exc_info.value.detail


# ---------------------------------------------------------------------------
# x-api-key (Anthropic-native clients)
# ---------------------------------------------------------------------------


def test_x_api_key_header_returns_raw_token() -> None:
    request = _make_request({X_API_KEY_HEADER: "test-raw-token"})

    assert extract_credential_token(request) == "test-raw-token"


def test_authorization_takes_precedence_over_x_api_key() -> None:
    request = _make_request(
        {
            "Authorization": "Bearer bearer-wins",
            X_API_KEY_HEADER: "x-api-key-loses",
        }
    )

    assert extract_credential_token(request) == "bearer-wins"


def test_canonical_takes_precedence_over_x_api_key() -> None:
    request = _make_request(
        {
            API_KEY_HEADER: "Bearer canonical-wins",
            X_API_KEY_HEADER: "x-api-key-loses",
        }
    )

    assert extract_credential_token(request) == "canonical-wins"


def test_x_api_key_without_bearer_prefix_succeeds() -> None:
    request = _make_request({X_API_KEY_HEADER: "test-raw-token-no-bearer-prefix"})

    token = extract_credential_token(request)
    assert token == "test-raw-token-no-bearer-prefix"


# ---------------------------------------------------------------------------
# Whitespace handling (shared by every mode)
# ---------------------------------------------------------------------------


def test_surrounding_whitespace_is_stripped() -> None:
    request = _make_request({API_KEY_HEADER: " tk_padded "})

    assert extract_credential_token(request) == "tk_padded"


@pytest.mark.parametrize(
    "headers",
    [
        {API_KEY_HEADER: "   "},
        {API_KEY_HEADER: "Bearer  "},
        {"Authorization": "   "},
        {"Authorization": "Bearer  "},
        {X_API_KEY_HEADER: "   "},
    ],
)
def test_whitespace_only_credential_is_missing(headers: dict[str, str]) -> None:
    """A credential of spaces is answered as missing, not sent on to fail
    verification as a token of whitespace."""
    request = _make_request(headers)

    with pytest.raises(HTTPException) as exc_info:
        extract_credential_token(request)

    assert exc_info.value.status_code == 401
    assert API_KEY_HEADER in exc_info.value.detail


@pytest.mark.parametrize(
    "headers",
    [
        {API_KEY_HEADER: " Bearer tk_padded "},
        {"Authorization": " Bearer tk_padded "},
    ],
)
def test_padded_bearer_prefix_is_recognized(headers: dict[str, str]) -> None:
    """Leading whitespace does not hide the Bearer scheme: the prefix comes off
    and the token is returned bare, in both headers that accept it."""
    request = _make_request(headers)

    assert extract_credential_token(request) == "tk_padded"
