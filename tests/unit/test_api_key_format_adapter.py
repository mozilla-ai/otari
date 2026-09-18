"""The open-source key format behaves as the gateway always has."""

from unittest.mock import patch

import pytest

from gateway.adapters.api_key_format_adapter import (
    FINGERPRINT_RANDOM_CHARS,
    DefaultApiKeyFormatAdapter,
    validate_api_key_format,
)
from gateway.auth.models import API_KEY_PREFIX, KEY_SUFFIX_LENGTH, MIN_API_KEY_LENGTH, key_suffix
from gateway.ports.api_key_format_port import Local

ADAPTER = DefaultApiKeyFormatAdapter(None)
BODY = "a" * MIN_API_KEY_LENGTH


def test_mint_produces_the_open_source_shape() -> None:
    api_key = ADAPTER.mint()

    assert api_key.startswith(API_KEY_PREFIX)
    assert not api_key.startswith("gw")
    # The prefix plus token_urlsafe(48), which is 64 characters.
    assert len(api_key) == len(API_KEY_PREFIX) + 64
    validate_api_key_format(api_key)


def test_two_mints_differ() -> None:
    assert ADAPTER.mint() != ADAPTER.mint()


def test_mint_validates_what_it_generates() -> None:
    with (
        patch("gateway.adapters.api_key_format_adapter.secrets.token_urlsafe", return_value="short"),
        pytest.raises(RuntimeError, match="failed validation"),
    ):
        ADAPTER.mint()


def test_fingerprint_is_the_prefix_plus_seven_random_characters() -> None:
    api_key = ADAPTER.mint()

    fingerprint = ADAPTER.fingerprint(api_key)

    assert fingerprint == api_key[: len(API_KEY_PREFIX) + FINGERPRINT_RANDOM_CHARS]
    assert api_key.startswith(fingerprint)
    # The two displayed halves never meet on a minted key.
    assert len(api_key) > len(fingerprint) + KEY_SUFFIX_LENGTH
    assert api_key.endswith(key_suffix(api_key))


@pytest.mark.parametrize(
    "presented",
    [
        API_KEY_PREFIX + BODY,
        "gw-" + BODY,
        "gw_" + BODY,
        "tk_" + BODY,
        "tk_live.migrated-platform-key-0123456789abcdefghij",
        "otr_tk_v1_eu_" + BODY,
        "",
        "not a key at all",
    ],
)
def test_every_presented_key_is_checked_locally(presented: str) -> None:
    """The open-source build has one deployment, so nothing is misdirected or malformed.

    A legacy key of any shape reaches the hash lookup, which is what keeps it
    authenticating (otari#646).
    """
    assert ADAPTER.route(presented) == Local()


def test_validate_accepts_the_minted_prefix() -> None:
    validate_api_key_format(API_KEY_PREFIX + BODY)


@pytest.mark.parametrize(
    "api_key",
    [
        API_KEY_PREFIX[:-1] + BODY,
        "tx-" + BODY,
        API_KEY_PREFIX[:-1] + "." + BODY,
        "gw-" + BODY,
        "tk_" + BODY,
    ],
)
def test_validate_rejects_other_prefixes(api_key: str) -> None:
    with pytest.raises(ValueError, match="prefix"):
        validate_api_key_format(api_key)


def test_validate_rejects_a_short_key() -> None:
    with pytest.raises(ValueError, match="too short"):
        validate_api_key_format(API_KEY_PREFIX + "a" * (MIN_API_KEY_LENGTH - len(API_KEY_PREFIX) - 1))


def test_validate_rejects_characters_token_urlsafe_never_emits() -> None:
    with pytest.raises(ValueError, match="invalid characters"):
        validate_api_key_format(API_KEY_PREFIX + BODY + ".")


def test_validate_rejects_a_non_string() -> None:
    with pytest.raises(ValueError, match="must be a string"):
        validate_api_key_format(123)  # type: ignore[arg-type]
