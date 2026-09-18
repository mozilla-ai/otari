import hashlib
import re
import secrets

# Prefix stamped on every key this gateway mints. ``gw_`` is the other credential,
# the gateway's own token to the platform, so a user key must not share it.
API_KEY_PREFIX = "tk-"

# Prefixes accepted at mint time: the minted one plus the underscore form otari.ai
# issues, so a key from either product passes the same format check.
API_KEY_PREFIXES: tuple[str, ...] = (API_KEY_PREFIX, "tk_")

# Minimum length of a minted key; ``token_urlsafe(48)`` yields 64 characters.
MIN_API_KEY_LENGTH = 50

_API_KEY_CHARSET = "[A-Za-z0-9_-]+"
_API_KEY_PATTERN = re.compile(f"^(?:{'|'.join(re.escape(p) for p in API_KEY_PREFIXES)}){_API_KEY_CHARSET}$")

# Number of leading plaintext characters kept as a display-only fingerprint
# (``API_KEY_PREFIX`` plus 7 random chars). A minted key is the prefix plus
# token_urlsafe(48), 67 chars, so exposing 10 leaves ~57 secret chars; the prefix
# never gates auth and cannot be recovered from the stored SHA-256 hash.
KEY_PREFIX_LENGTH = 10

# Number of trailing plaintext characters kept alongside the prefix, so a key can be
# told apart from another sharing its prefix. Four, matching the ``last4`` that
# provider credentials have always stored; it takes the unexposed remainder from
# ~342 bits to ~318 and, like the prefix, never gates auth.
KEY_SUFFIX_LENGTH = 4


def generate_api_key() -> str:
    """Generate a new API key with prefix.

    Returns:
        A new API key: ``API_KEY_PREFIX`` followed by 64 URL-safe random characters

    Raises:
        RuntimeError: If generated key doesn't match expected format (should never happen)

    """
    api_key = f"{API_KEY_PREFIX}{secrets.token_urlsafe(48)}"

    try:
        validate_api_key_format(api_key)
    except ValueError as e:
        msg = f"Generated API key failed validation: {e}"
        raise RuntimeError(msg) from e

    return api_key


def key_prefix(api_key: str) -> str:
    """Return the display-only fingerprint (leading characters) of an API key.

    Called at every key-mint site so the stored prefix length cannot drift. The
    prefix is shown in the dashboard to recognize a key after its one-time reveal;
    it is not a secret and is never used for authentication.
    """
    return api_key[:KEY_PREFIX_LENGTH]


def key_suffix(api_key: str) -> str:
    """Return the display-only trailing characters of an API key.

    Called at every key-mint site, rotation included, so a rotated row never keeps
    the suffix of the secret it replaced. Like the prefix it is not a secret, is
    never used for authentication, and cannot be recovered from the stored hash.
    """
    return api_key[-KEY_SUFFIX_LENGTH:]


def validate_api_key_format(api_key: str) -> None:
    """Validate API key format.

    Args:
        api_key: The API key to validate

    Raises:
        ValueError: If the API key format is invalid

    """
    if not isinstance(api_key, str):
        msg = f"API key must be a string, got {type(api_key).__name__}"
        raise ValueError(msg)

    if not api_key.startswith(API_KEY_PREFIXES):
        accepted = " or ".join(f"'{p}'" for p in API_KEY_PREFIXES)
        msg = f"API key must start with {accepted} prefix"
        raise ValueError(msg)

    if len(api_key) < MIN_API_KEY_LENGTH:
        msg = f"API key is too short. Expected at least {MIN_API_KEY_LENGTH} characters, got {len(api_key)}"
        raise ValueError(msg)

    if not _API_KEY_PATTERN.match(api_key):
        msg = f"API key contains invalid characters. Must match pattern: {_API_KEY_PATTERN.pattern}"
        raise ValueError(msg)


def hash_key(api_key: str) -> str:
    """Hash an API key using SHA-256.

    Deliberately does not validate the key's format. Format is a mint-time
    concern (``generate_api_key`` validates what it generates); on the verify
    path a shape check only decides which error a wrong key gets, and it would
    reject a key minted elsewhere whose hash is legitimately on a row.

    Args:
        api_key: The API key to hash

    Returns:
        Hexadecimal string of the SHA-256 hash

    """
    return hashlib.sha256(api_key.encode()).hexdigest()
