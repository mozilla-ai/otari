import hashlib
from unittest.mock import patch

import pytest

from gateway.auth.models import (
    API_KEY_PREFIX,
    API_KEY_PREFIXES,
    KEY_PREFIX_LENGTH,
    KEY_SUFFIX_LENGTH,
    MIN_API_KEY_LENGTH,
    generate_api_key,
    hash_key,
    key_prefix,
    key_suffix,
    validate_api_key_format,
)

BODY = "a" * MIN_API_KEY_LENGTH


@pytest.mark.parametrize("prefix", API_KEY_PREFIXES)
def test_validate_api_key_format_accepts_supported_prefixes(prefix: str) -> None:
    validate_api_key_format(prefix + BODY)


@pytest.mark.parametrize(
    "api_key",
    [
        API_KEY_PREFIX[:-1] + BODY,
        "tx-" + BODY,
        API_KEY_PREFIX[:-1] + "." + BODY,
        "gw-" + BODY,
    ],
)
def test_validate_api_key_format_rejects_invalid_prefixes(api_key: str) -> None:
    with pytest.raises(ValueError, match="prefix"):
        validate_api_key_format(api_key)


def test_hash_key_accepts_gw_underscore_prefix() -> None:
    digest = hash_key("gw_" + BODY)

    assert len(digest) == 64


def test_hash_key_hashes_a_key_that_is_not_gw_shaped() -> None:
    """``hash_key`` no longer validates format (issue #646).

    A key minted by another product (a migrated ``tk_`` platform key) hashes to
    the same unsalted SHA-256 digest as any other string, so a migrated row's
    hash still matches on the verify path.
    """
    api_key = "tk_" + "b" * MIN_API_KEY_LENGTH

    assert hash_key(api_key) == hashlib.sha256(api_key.encode()).hexdigest()


def test_generate_api_key_mints_a_user_key_not_a_gateway_token() -> None:
    """User keys carry the minted prefix; ``gw_`` is the gateway's own credential to the platform."""
    api_key = generate_api_key()

    assert api_key.startswith(API_KEY_PREFIX)
    assert not api_key.startswith("gw")


def test_generate_api_key_still_validates_at_mint_time() -> None:
    """Mint-time validation is unaffected by lifting the check out of ``hash_key``."""
    with (
        patch("gateway.auth.models.secrets.token_urlsafe", return_value="short"),
        pytest.raises(RuntimeError, match="failed validation"),
    ):
        generate_api_key()


def test_the_two_fingerprint_halves_do_not_overlap() -> None:
    """A minted key is long enough that the displayed halves never meet."""
    api_key = generate_api_key()

    assert key_prefix(api_key) == api_key[:KEY_PREFIX_LENGTH]
    assert key_suffix(api_key) == api_key[-KEY_SUFFIX_LENGTH:]
    assert len(api_key) > KEY_PREFIX_LENGTH + KEY_SUFFIX_LENGTH
