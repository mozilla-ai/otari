"""The open-source API key format.

Satisfies :class:`gateway.ports.api_key_format_port.ApiKeyFormatPort` with the
format a standalone gateway has always minted: a short prefix and 64 URL-safe
random characters. Every presented key routes ``Local``, so verification runs
the same hash lookup for a key of any shape, which is what keeps a key minted
before this adapter existed working.
"""

import re
import secrets

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.auth.models import API_KEY_PREFIX, MIN_API_KEY_LENGTH
from gateway.ports.api_key_format_port import KeyRoute, Local

# Random characters kept after the prefix in the stored fingerprint. The prefix
# alone names every key alike; seven more tell them apart on the Keys page.
FINGERPRINT_RANDOM_CHARS = 7

_API_KEY_PATTERN = re.compile(f"^{re.escape(API_KEY_PREFIX)}[A-Za-z0-9_-]+$")


def validate_api_key_format(api_key: str) -> None:
    """Refuse a key that is not in the open-source format.

    A mint-time check only. The verify path never calls it: a presented key of
    any shape is hashed and looked up (otari#646).

    Raises:
        ValueError: If the key is not a string, lacks the prefix, is too short,
            or carries a character ``token_urlsafe`` never emits.

    """
    if not isinstance(api_key, str):
        msg = f"API key must be a string, got {type(api_key).__name__}"
        raise ValueError(msg)

    if not api_key.startswith(API_KEY_PREFIX):
        msg = f"API key must start with '{API_KEY_PREFIX}' prefix"
        raise ValueError(msg)

    if len(api_key) < MIN_API_KEY_LENGTH:
        msg = f"API key is too short. Expected at least {MIN_API_KEY_LENGTH} characters, got {len(api_key)}"
        raise ValueError(msg)

    if not _API_KEY_PATTERN.match(api_key):
        msg = f"API key contains invalid characters. Must match pattern: {_API_KEY_PATTERN.pattern}"
        raise ValueError(msg)


class DefaultApiKeyFormatAdapter:
    """Mints ``tk-`` keys and checks every presented key locally.

    Constructible with a session like every core adapter, because the container
    builds one per request whether or not the request will use it. Nothing here
    reads or writes it.
    """

    def __init__(self, session: AsyncSession | None) -> None:
        self._session = session

    def mint(self) -> str:
        """Return the prefix plus 64 URL-safe random characters (256 bits)."""
        api_key = f"{API_KEY_PREFIX}{secrets.token_urlsafe(48)}"
        try:
            validate_api_key_format(api_key)
        except ValueError as e:
            msg = f"Generated API key failed validation: {e}"
            raise RuntimeError(msg) from e
        return api_key

    def fingerprint(self, api_key: str) -> str:
        """Return the prefix and the seven random characters after it."""
        return api_key[: len(API_KEY_PREFIX) + FINGERPRINT_RANDOM_CHARS]

    def route(self, presented: str) -> KeyRoute:
        """Every key is checked here: the open-source build has one deployment."""
        return Local()
