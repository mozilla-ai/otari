"""The limits an organization's own connection request is held to at the boundary.

An issuer is named as a provider by prefixing it, so the room for one is what the provider column
holds less that prefix. Refusing a longer one here rather than after discovery costs no outbound
call and names the field at fault.
"""

import pytest
from pydantic import ValidationError

from gateway.models.identity import MAX_OIDC_ISSUER_LENGTH
from gateway.schemas.identity import OIDCConnectionRequest


def _request(issuer: str) -> OIDCConnectionRequest:
    return OIDCConnectionRequest(
        issuer=issuer,
        client_id="a-client-id",
        client_secret="a-client-secret",
        display_name="Example IdP",
        default_role="member",
    )


def _issuer_of(length: int) -> str:
    """An issuer of exactly ``length`` characters."""
    prefix = "https://"
    return prefix + "a" * (length - len(prefix))


def test_an_issuer_that_fits_as_a_provider_is_accepted() -> None:
    request = _request(_issuer_of(MAX_OIDC_ISSUER_LENGTH))

    assert len(request.issuer) == MAX_OIDC_ISSUER_LENGTH


def test_an_issuer_one_character_too_long_to_name_a_provider_is_refused() -> None:
    """Refused at the boundary: a longer one reaches the provider column's limit after discovery."""
    with pytest.raises(ValidationError):
        _request(_issuer_of(MAX_OIDC_ISSUER_LENGTH + 1))


def test_the_request_carries_no_sign_in_setting() -> None:
    """Whether a connection admits sign-in has one writer, and a field here would give it two."""
    assert "enabled" not in OIDCConnectionRequest.model_fields
