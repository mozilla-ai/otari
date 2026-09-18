import hashlib

# Prefix stamped on every key the open-source build mints. ``gw_`` is the other
# credential, the gateway's own token to the platform, so a user key must not
# share it. The format itself lives behind ``ApiKeyFormatPort``; this constant
# is the default adapter's, kept here so tests and the dashboard can name it.
API_KEY_PREFIX = "tk-"

# Minimum length of a minted key; ``token_urlsafe(48)`` yields 64 characters.
MIN_API_KEY_LENGTH = 50

# Number of trailing plaintext characters kept alongside the fingerprint, so a
# key can be told apart from another sharing its prefix. Four, matching the
# ``last4`` that provider credentials have always stored; it takes the unexposed
# remainder from ~342 bits to ~318 and, like the fingerprint, never gates auth.
KEY_SUFFIX_LENGTH = 4


def key_suffix(api_key: str) -> str:
    """Return the display-only trailing characters of an API key.

    Called at every key-mint site, rotation included, so a rotated row never keeps
    the suffix of the secret it replaced. Like the fingerprint it is not a secret,
    is never used for authentication, and cannot be recovered from the stored hash.
    """
    return api_key[-KEY_SUFFIX_LENGTH:]


def hash_key(api_key: str) -> str:
    """Hash an API key using SHA-256.

    Deliberately does not validate the key's format. Format is a mint-time
    concern (the bound ``ApiKeyFormatPort`` validates what it mints); on the verify
    path a shape check only decides which error a wrong key gets, and it would
    reject a key minted elsewhere whose hash is legitimately on a row.

    Args:
        api_key: The API key to hash

    Returns:
        Hexadecimal string of the SHA-256 hash

    """
    return hashlib.sha256(api_key.encode()).hexdigest()
