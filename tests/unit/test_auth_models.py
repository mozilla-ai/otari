import pytest

from gateway.auth.models import hash_key, validate_api_key_format


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
