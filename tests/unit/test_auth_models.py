import hashlib

from gateway.auth.models import KEY_SUFFIX_LENGTH, MIN_API_KEY_LENGTH, hash_key, key_suffix

BODY = "a" * MIN_API_KEY_LENGTH


def test_hash_key_is_the_unsalted_sha256_of_the_whole_key() -> None:
    api_key = "gw_" + BODY

    assert hash_key(api_key) == hashlib.sha256(api_key.encode()).hexdigest()


def test_hash_key_hashes_a_key_that_is_not_in_the_minted_shape() -> None:
    """``hash_key`` does not validate format (issue #646).

    A key minted by another product (a migrated ``tk_`` platform key) hashes to
    the same unsalted SHA-256 digest as any other string, so a migrated row's
    hash still matches on the verify path.
    """
    api_key = "tk_live." + "b" * MIN_API_KEY_LENGTH

    assert hash_key(api_key) == hashlib.sha256(api_key.encode()).hexdigest()


def test_key_suffix_is_the_trailing_characters() -> None:
    api_key = "tk-" + BODY + "wxyz"

    assert key_suffix(api_key) == "wxyz"
    assert len(key_suffix(api_key)) == KEY_SUFFIX_LENGTH
