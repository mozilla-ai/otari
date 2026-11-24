"""A tenant's endpoint credential is held to a closed method and a header name it may not choose freely."""

from datetime import UTC, datetime, timedelta

import pytest

from gateway.types.endpoint_auth import (
    EXPIRY_WARNING_WINDOW,
    EndpointAuth,
    EndpointAuthMethod,
    EndpointAuthState,
    UnreadableEndpointAuthError,
    refused_header_name_reason,
    refused_header_value_reason,
)

NOW = datetime(2026, 11, 20, tzinfo=UTC)
LATER = NOW + timedelta(days=30)


def _columns(**overrides: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "method": EndpointAuthMethod.STATIC_HEADER,
        "header_name": "Fastly-Key",
        "encrypted_secret": "gAAAA-ciphertext",
        "last4": "text",
        "expires_at": LATER,
    }
    return {**fields, **overrides}


def _auth(**overrides: object) -> EndpointAuth:
    return EndpointAuth(**_columns(**overrides))  # type: ignore[arg-type]


def _refusal(*, has_api_base: bool = True, **overrides: object) -> str | None:
    return _auth(**overrides).refused_reason(has_api_base=has_api_base)


def _expiry_refusal(**overrides: object) -> str | None:
    return _auth(**overrides).refused_expiry_reason(now=NOW)


def test_a_static_header_on_a_tenant_endpoint_is_served() -> None:
    assert _refusal() is None


def test_a_row_holding_every_column_carries_the_credential_they_name() -> None:
    assert EndpointAuth.from_columns(**_columns()) == _auth()  # type: ignore[arg-type]


@pytest.mark.parametrize("missing", ["method", "header_name", "encrypted_secret", "last4", "expires_at"])
def test_a_row_holding_only_some_of_the_columns_cannot_be_read(missing: str) -> None:
    """Holding part of a credential is not holding none, so it is reported rather than read as absent."""
    with pytest.raises(UnreadableEndpointAuthError):
        EndpointAuth.from_columns(**_columns(**{missing: None}))  # type: ignore[arg-type]


def test_a_row_holding_none_of_the_columns_carries_no_credential() -> None:
    assert EndpointAuth.from_columns(**dict.fromkeys(_columns())) is None


def test_an_endpoint_credential_is_refused_on_the_providers_own_endpoint() -> None:
    assert _refusal(has_api_base=False) is not None


def test_a_method_this_build_does_not_know_cannot_be_read() -> None:
    """A later build may store one, and this build can neither produce it nor report the key as carrying none."""
    with pytest.raises(UnreadableEndpointAuthError):
        EndpointAuth.from_columns(**_columns(method="mtls"))  # type: ignore[arg-type]


def test_a_static_header_with_an_empty_header_name_is_refused() -> None:
    assert _refusal(header_name="") is not None


def test_a_future_expiry_is_stored() -> None:
    assert _expiry_refusal() is None


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


@pytest.mark.parametrize("value", ["edge-s3cret", "a", "Bearer abc.123", "one two\tthree", "~" * 512])
def test_a_value_of_printable_ascii_is_served(value: str) -> None:
    assert refused_header_value_reason(value) is None


@pytest.mark.parametrize(
    "value",
    ["", " lead", "trail ", "trail\n", "two\nlines", "two\rlines", "nul\x00byte", "h\u00e9ader", "\tlead"],
)
def test_a_value_a_header_field_cannot_carry_is_refused(value: str) -> None:
    """A control character ends the field, so the transport refuses the whole request rather than the value."""
    assert refused_header_value_reason(value) is not None


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


# Python's ``$`` also matches before a final newline, so a name that ends in one
# satisfies the character rule and then matches no reserved name below it.
@pytest.mark.parametrize("name", ["Authorization\n", "connection\n", "Fastly-Key\n", "Host\n", "Otari-Key\n"])
def test_a_name_with_a_trailing_newline_is_refused(name: str) -> None:
    assert refused_header_name_reason(name) is not None


def test_a_static_header_becomes_the_one_header_it_names() -> None:
    assert _auth().headers(secret="s3cret") == {"Fastly-Key": "s3cret"}


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
