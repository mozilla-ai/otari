import hashlib
from unittest.mock import patch

import pytest

from gateway.auth.models import generate_api_key, hash_key, validate_api_key_format


@pytest.mark.parametrize(
    "api_key",
    [
        "gw-" + "a" * 48,
        "gw_" + "a" * 48,
    ],
)
def test_validate_api_key_format_accepts_supported_prefixes(api_key: str) -> None:
    validate_api_key_format(api_key)


@pytest.mark.parametrize(
    "api_key",
    [
        "gw" + "a" * 49,
        "gx-" + "a" * 48,
        "gw." + "a" * 48,
    ],
)
def test_validate_api_key_format_rejects_invalid_prefixes(api_key: str) -> None:
    with pytest.raises(ValueError, match="prefix"):
        validate_api_key_format(api_key)


def test_hash_key_accepts_gw_underscore_prefix() -> None:
    digest = hash_key("gw_" + "a" * 48)

    assert len(digest) == 64


def test_hash_key_hashes_a_key_that_is_not_gw_shaped() -> None:
    """``hash_key`` no longer validates format (issue #646).

    A key minted by another product (a migrated ``tk_`` platform key) hashes to
    the same unsalted SHA-256 digest as any other string, so a migrated row's
    hash still matches on the verify path.
    """
    api_key = "tk_" + "b" * 48

    assert hash_key(api_key) == hashlib.sha256(api_key.encode()).hexdigest()


def test_generate_api_key_still_validates_at_mint_time() -> None:
    """Mint-time validation is unaffected by lifting the check out of ``hash_key``."""
    with (
        patch("gateway.auth.models.secrets.token_urlsafe", return_value="short"),
        pytest.raises(RuntimeError, match="failed validation"),
    ):
        generate_api_key()
