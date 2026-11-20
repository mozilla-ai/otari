"""A tenant's endpoint credential is held to a closed method and a header name it may not choose freely."""

from datetime import UTC, datetime, timedelta

import pytest

from gateway.types.endpoint_auth import (
    EXPIRY_WARNING_WINDOW,
    EndpointAuth,
    EndpointAuthMethod,
    EndpointAuthState,
    refused_header_name_reason,
)

NOW = datetime(2026, 11, 20, tzinfo=UTC)
LATER = NOW + timedelta(days=30)


def _auth(**overrides: object) -> EndpointAuth:
    fields: dict[str, object] = {
        "method": EndpointAuthMethod.STATIC_HEADER,
        "header_name": "Fastly-Key",
        "encrypted_secret": "gAAAA-ciphertext",
        "expires_at": LATER,
    }
    return EndpointAuth(**{**fields, **overrides})  # type: ignore[arg-type]


def _refusal(*, has_api_base: bool = True, **overrides: object) -> str | None:
    return _auth(**overrides).refused_reason(has_api_base=has_api_base)


def _expiry_refusal(**overrides: object) -> str | None:
    return _auth(**overrides).refused_expiry_reason(now=NOW)


def test_a_static_header_on_a_tenant_endpoint_is_served() -> None:
    assert _refusal() is None


def test_a_key_declaring_no_method_is_served() -> None:
    assert _refusal(method=None, header_name=None, expires_at=None, has_api_base=False) is None


def test_a_key_declaring_no_method_carries_no_credential() -> None:
    assert EndpointAuth().declared is False
    assert _auth().declared is True


def test_an_endpoint_credential_is_refused_on_the_providers_own_endpoint() -> None:
    assert _refusal(has_api_base=False) is not None


def test_an_unknown_method_is_refused() -> None:
    assert _refusal(method="mtls") is not None


@pytest.mark.parametrize("header_name", [None, ""])
def test_a_static_header_with_no_header_name_is_refused(header_name: str | None) -> None:
    assert _refusal(header_name=header_name) is not None


def test_a_static_header_with_no_stored_value_is_refused() -> None:
    assert _refusal(encrypted_secret=None) is not None


def test_a_future_expiry_is_stored() -> None:
    assert _expiry_refusal() is None


def test_a_key_declaring_no_method_needs_no_expiry() -> None:
    assert _expiry_refusal(method=None, expires_at=None) is None


def test_a_static_header_with_no_expiry_is_refused_at_write() -> None:
    assert _expiry_refusal(expires_at=None) is not None


@pytest.mark.parametrize("expires_at", [NOW, NOW - timedelta(seconds=1)])
def test_an_expiry_that_is_not_in_the_future_is_refused_at_write(expires_at: datetime) -> None:
    assert _expiry_refusal(expires_at=expires_at) is not None


# A past expiry is a prediction about the proxy, not a fact about the
# credential, so it must not stop a stored key being served. The date is long
# past in absolute terms, so an implementation reading its own clock fails here.
def test_an_expiry_that_has_passed_does_not_refuse_a_stored_credential() -> None:
    expired = _auth(expires_at=datetime(2020, 1, 1, tzinfo=UTC))

    assert expired.refused_reason(has_api_base=True) is None
    assert expired.headers(secret="s3cret") == {"Fastly-Key": "s3cret"}


@pytest.mark.parametrize("name", ["Fastly-Key", "CF-Access-Client-Id", "a", "A" * 64])
def test_a_name_of_letters_digits_and_hyphens_is_served(name: str) -> None:
    assert refused_header_name_reason(name) is None


@pytest.mark.parametrize("name", ["", "A" * 65, "X Key", "x_key", ":authority", "Key:", "héader"])
def test_a_name_outside_the_allowed_characters_is_refused(name: str) -> None:
    assert refused_header_name_reason(name) is not None


# The provider's own credential header. A tenant that set one would send a value
# of its choosing to the provider in place of the organization's key.
@pytest.mark.parametrize(
    "name", ["authorization", "Authorization", "x-api-key", "X-Api-Key", "api-key", "x-goog-api-key"]
)
def test_a_providers_own_credential_header_is_refused(name: str) -> None:
    assert refused_header_name_reason(name) is not None


@pytest.mark.parametrize(
    "name",
    [
        "connection",
        "Content-Length",
        "keep-alive",
        "proxy-authenticate",
        "Proxy-Authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    ],
)
def test_a_header_reserved_for_the_connection_is_refused(name: str) -> None:
    assert refused_header_name_reason(name) is not None


@pytest.mark.parametrize("name", ["host", "Host", "HOST"])
def test_the_host_header_is_refused(name: str) -> None:
    assert refused_header_name_reason(name) is not None


@pytest.mark.parametrize("name", ["Otari-Key", "otari-region", "OTARI-Provider"])
def test_a_header_in_otaris_own_namespace_is_refused(name: str) -> None:
    assert refused_header_name_reason(name) is not None


def test_a_static_header_becomes_the_one_header_it_names() -> None:
    assert _auth().headers(secret="s3cret") == {"Fastly-Key": "s3cret"}


@pytest.mark.parametrize(
    ("overrides", "secret"),
    [({"method": None}, "s"), ({"header_name": None}, "s"), ({}, None)],
)
def test_a_credential_that_is_not_whole_adds_no_header(overrides: dict[str, object], secret: str | None) -> None:
    assert _auth(**overrides).headers(secret=secret) == {}


def test_a_credential_declaring_no_expiry_has_no_state() -> None:
    assert EndpointAuth().state(now=NOW) is None


@pytest.mark.parametrize(
    ("expires_at", "expected"),
    [
        (NOW - timedelta(seconds=1), EndpointAuthState.EXPIRED),
        (NOW, EndpointAuthState.EXPIRED),
        (NOW + timedelta(seconds=1), EndpointAuthState.EXPIRING_SOON),
        (NOW + EXPIRY_WARNING_WINDOW, EndpointAuthState.EXPIRING_SOON),
        (NOW + EXPIRY_WARNING_WINDOW + timedelta(seconds=1), EndpointAuthState.VALID),
    ],
)
def test_the_rotation_window_is_drawn_in_one_place(expires_at: datetime, expected: EndpointAuthState) -> None:
    assert _auth(expires_at=expires_at).state(now=NOW) == expected
