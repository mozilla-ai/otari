"""Encryption at rest for provider credentials configured through the dashboard.

Provider API keys entered in the web UI are persisted encrypted with a Fernet
key supplied out of band as ``OTARI_SECRET_KEY``. The key lives outside the
database on purpose: a stolen database dump is useless without it. We never
auto-generate the key, never persist it next to the ciphertext, and never derive
it from the master key (which the gateway may rotate) because any of those would
make encryption-at-rest theatre against database theft.

``OTARI_SECRET_KEY`` may hold one or more Fernet keys separated by whitespace or
commas. The first is used to encrypt; all are tried on decrypt (``MultiFernet``),
so a key can be rotated by prepending a new one and re-encrypting stored rows
over time. Generate one with :func:`generate_secret_key`.
"""

import re

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

from gateway.core.env import otari_env

SECRET_KEY_ENV = "SECRET_KEY"


class SecretBoxUnavailableError(RuntimeError):
    """Raised when ``OTARI_SECRET_KEY`` is unset or not a valid Fernet key.

    Surfaced to the API as a "set OTARI_SECRET_KEY to store credentials" error;
    it must never contain key material.
    """


class SecretDecryptionError(ValueError):
    """Raised when a stored ciphertext cannot be decrypted with the current key.

    Wraps Fernet's ``InvalidToken`` with a message that never echoes the
    ciphertext, so a wrong or rotated-away key degrades one credential instead of
    leaking data or crashing the caller.
    """


def generate_secret_key() -> str:
    """Return a fresh url-safe Fernet key suitable for ``OTARI_SECRET_KEY``."""
    return Fernet.generate_key().decode()


def _load_keys() -> list[str]:
    raw = otari_env(SECRET_KEY_ENV)
    if not raw:
        return []
    return [key for key in re.split(r"[\s,]+", raw.strip()) if key]


def secret_box_configured() -> bool:
    """Whether at least one ``OTARI_SECRET_KEY`` value is set (not validated)."""
    return bool(_load_keys())


def get_secret_box() -> MultiFernet:
    """Build the ``MultiFernet`` from ``OTARI_SECRET_KEY``.

    Raises ``SecretBoxUnavailableError`` when the variable is unset or holds a
    value that is not a valid Fernet key.
    """
    keys = _load_keys()
    if not keys:
        raise SecretBoxUnavailableError(
            "OTARI_SECRET_KEY is not set; it is required to store provider credentials."
        )
    try:
        fernets = [Fernet(key.encode()) for key in keys]
    except (ValueError, TypeError):
        # Do not include the offending value: it is key material.
        raise SecretBoxUnavailableError("OTARI_SECRET_KEY is not a valid Fernet key.") from None
    return MultiFernet(fernets)


def validate_secret_key() -> None:
    """Fail fast when ``OTARI_SECRET_KEY`` is set but not a valid Fernet key.

    Unset is allowed: the provider store is simply unavailable until a key is
    configured. A set-but-invalid key would otherwise pass startup and only
    surface later when a credential is stored or read, so validating it here
    turns a latent runtime failure into a clear startup error. Raises
    ``SecretBoxUnavailableError``; the message never carries key material.
    """
    if not secret_box_configured():
        return
    get_secret_box()


def encrypt_secret(plaintext: str) -> str:
    """Encrypt ``plaintext`` with the primary key; return url-safe ciphertext."""
    return get_secret_box().encrypt(plaintext.encode()).decode()


def decrypt_secret(ciphertext: str) -> str:
    """Decrypt a stored ciphertext, trying every configured key.

    Raises ``SecretBoxUnavailableError`` when no key is configured and
    ``SecretDecryptionError`` when none of the configured keys can decrypt it.
    Neither error carries the ciphertext.
    """
    try:
        return get_secret_box().decrypt(ciphertext.encode()).decode()
    except InvalidToken:
        raise SecretDecryptionError(
            "A stored provider credential could not be decrypted with the configured OTARI_SECRET_KEY."
        ) from None
