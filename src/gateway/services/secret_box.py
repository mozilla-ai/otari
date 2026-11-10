"""Encryption at rest for provider credentials configured through the dashboard.

Provider API keys entered in the web UI are persisted encrypted with a Fernet
key supplied out of band as ``OTARI_SECRET_KEY``. The key lives outside the
database on purpose: a stolen database dump is useless without it. It is never
stored in the database and never derived from the master key (which the gateway
may rotate).

A SQLite deployment that sets no ``OTARI_SECRET_KEY`` keeps a generated key in a
file beside its database file instead (:func:`use_local_secret_key`), so a first
run can store a credential with nothing configured. That protects a leaked
database file, not a stolen directory or volume, which holds both. PostgreSQL
gets no such file: replicas cannot share one, so its operator sets the variable.

``OTARI_SECRET_KEY`` may hold one or more Fernet keys separated by whitespace or
commas. The first is used to encrypt; all are tried on decrypt (``MultiFernet``),
so a key can be rotated by prepending a new one and re-encrypting stored rows
over time. Generate one with :func:`generate_secret_key`.
"""

import contextlib
import functools
import os
import re
import secrets
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

from gateway.core.env import otari_env
from gateway.log_config import logger

SECRET_KEY_ENV = "SECRET_KEY"
_GENERATE_HINT = "Generate one with `otari gen-secret-key`."

# The key read from a SQLite deployment's key file, used while ``OTARI_SECRET_KEY`` is unset.
_local_keys: list[str] = []


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
        return list(_local_keys)
    return [key for key in re.split(r"[\s,]+", raw.strip()) if key]


def secret_key_file_for(database_file: Path) -> Path:
    """Where a SQLite deployment keeps its generated key: beside ``database_file``."""
    return database_file.with_name(f"{database_file.name}.secret-key")


def use_local_secret_key(key_file: Path, *, may_generate: bool) -> None:
    """Use the key in ``key_file`` while ``OTARI_SECRET_KEY`` is unset, generating the file on first run.

    A set ``OTARI_SECRET_KEY`` wins and the file is not touched. ``may_generate``
    is false when the database already holds values sealed with a key: a new key
    could not open them, so a missing file then raises
    ``SecretBoxUnavailableError`` rather than being replaced. So does a file that
    holds no valid key. A directory that cannot be written leaves the deployment
    without a key, as it was before this file existed.
    """
    if otari_env(SECRET_KEY_ENV):
        return
    try:
        key = key_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        if not may_generate:
            raise SecretBoxUnavailableError(
                f"This database holds encrypted credentials, but OTARI_SECRET_KEY is not set and the key file "
                f"{key_file} is missing. Restore that file or set OTARI_SECRET_KEY to the key they were stored "
                "with. Otari will not generate a new key, because it could not decrypt them."
            ) from None
        try:
            key = _create_key_file(key_file)
        except OSError as exc:
            logger.warning(
                "Could not write a secret key file at %s (%s), so provider credentials cannot be stored "
                "until OTARI_SECRET_KEY is set.",
                key_file,
                type(exc).__name__,
            )
            return
        logger.info("Generated a secret key for stored credentials at %s. Back it up with the database.", key_file)
    try:
        Fernet(key.encode())
    except (ValueError, TypeError):
        raise SecretBoxUnavailableError(
            f"The secret key file {key_file} does not hold a valid Fernet key. Restore it or set OTARI_SECRET_KEY."
        ) from None
    _local_keys[:] = [key]


def forget_local_secret_key() -> None:
    """Stop using a key loaded by :func:`use_local_secret_key`."""
    _local_keys.clear()


def _create_key_file(key_file: Path) -> str:
    """Write a new key to ``key_file`` with mode 0600 and return the key the file holds.

    The key is written to a temporary file and then linked into place, so the
    file appears complete or not at all, and a process that loses the race to
    create it adopts the winner's key instead of overwriting it.
    """
    temporary = key_file.with_name(f"{key_file.name}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(generate_secret_key() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        with contextlib.suppress(FileExistsError):
            os.link(temporary, key_file)
    finally:
        temporary.unlink(missing_ok=True)
    return key_file.read_text(encoding="utf-8").strip()


def configured_secret_keys() -> list[str]:
    """The configured ``OTARI_SECRET_KEY`` keys, the one new material is made with first. Empty when unset."""
    return _load_keys()


def shares_secret_key(value: str) -> bool:
    """Whether ``value`` is one of the configured ``OTARI_SECRET_KEY`` keys."""
    return value in _load_keys()


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
            f"OTARI_SECRET_KEY is not set or is empty; it is required to store provider credentials. {_GENERATE_HINT}"
        )
    return _box_for(tuple(keys))


# Keyed on the keys themselves, so a changed OTARI_SECRET_KEY builds a new box
# while a read that opens many rows builds one.
@functools.lru_cache(maxsize=1)
def _box_for(keys: tuple[str, ...]) -> MultiFernet:
    try:
        fernets = [Fernet(key.encode()) for key in keys]
    except (ValueError, TypeError):
        # Do not include the offending value: it is key material.
        raise SecretBoxUnavailableError(f"OTARI_SECRET_KEY is not a valid Fernet key. {_GENERATE_HINT}") from None
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
