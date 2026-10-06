"""How a hosted provider's secrets are stored and read back.

The key and the SDK client extras are both encrypted with the deployment's
secret box. They are read back here and nowhere else, into a value that never
crosses the API boundary.
"""

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from any_llm import LLMProvider

from gateway.core.config import PROVIDER_TYPE_ALIASES
from gateway.exceptions.providers_exceptions import HostedProviderUnknownProviderError
from gateway.exceptions.shared_exceptions import SecretBoxUnavailableTenancyError
from gateway.log_config import logger
from gateway.models.providers import HostedProvider
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
)

# What the secret-box error names as the thing that could not be stored.
STORED = "hosted provider credentials"


@dataclass(frozen=True)
class ResolvedHostedProvider:
    """A decrypted, usable upstream connection."""

    api_key: str
    api_base: str | None
    client_args: dict[str, Any] | None


def last4(api_key: str) -> str | None:
    """The tail of a key, or None for a key so short the tail would be the whole of it."""
    return api_key[-4:] if len(api_key) > 4 else None


def normalize_api_base(api_base: str | None) -> str | None:
    """Fold a blank endpoint to None, which means the provider's default."""
    if api_base is None:
        return None
    trimmed = api_base.strip()
    return trimmed or None


def host_of(api_base: str | None) -> str | None:
    """The host an endpoint points at, lower-cased, or None for the provider's default."""
    if api_base is None:
        return None
    host = urlsplit(api_base).hostname
    return host.lower() if host else None


def lookup_name(provider: str) -> str:
    """The stored spelling of a provider name, with the accepted aliases folded."""
    candidate = provider.strip()
    return PROVIDER_TYPE_ALIASES.get(candidate, candidate)


def normalize_provider(provider: str) -> str:
    """Resolve a provider name to the any-llm implementation that serves it.

    Raises:
        HostedProviderUnknownProviderError: no any-llm implementation goes by that name.
    """
    implementation = lookup_name(provider)
    if not implementation:
        raise HostedProviderUnknownProviderError(implementation)
    try:
        LLMProvider(implementation)
    except ValueError as exc:
        raise HostedProviderUnknownProviderError(provider) from exc
    return implementation


def encrypt_key(api_key: str) -> str:
    """Encrypt a key for storage.

    Raises:
        SecretBoxUnavailableTenancyError: no usable ``OTARI_SECRET_KEY``.
    """
    try:
        return encrypt_secret(api_key)
    except SecretBoxUnavailableError:
        raise SecretBoxUnavailableTenancyError(STORED) from None


def encrypt_client_args(client_args: dict[str, Any] | None) -> str | None:
    """Encrypt the SDK client extras for storage, as one JSON document.

    Raises:
        SecretBoxUnavailableTenancyError: no usable ``OTARI_SECRET_KEY``.
    """
    if client_args is None:
        return None
    try:
        return encrypt_secret(json.dumps(client_args, sort_keys=True))
    except SecretBoxUnavailableError:
        raise SecretBoxUnavailableTenancyError(STORED) from None


def _decrypt_client_args(ciphertext: str | None) -> dict[str, Any] | None:
    if ciphertext is None:
        return None
    decrypted: dict[str, Any] = json.loads(decrypt_secret(ciphertext))
    return decrypted


def credential_of(row: HostedProvider, *, quiet: bool = False) -> ResolvedHostedProvider | None:
    """Decrypt a row's key and extras, or None when no configured key reads them.

    Logged unless ``quiet``, because a rotated-away ``OTARI_SECRET_KEY``
    degrades one provider silently otherwise.
    """
    try:
        api_key = decrypt_secret(row.encrypted_api_key)
        client_args = _decrypt_client_args(row.encrypted_client_args)
    except (SecretDecryptionError, SecretBoxUnavailableError):
        if not quiet:
            logger.error(
                "Hosted provider %r could not be decrypted; treating it as unserved. "
                "Was OTARI_SECRET_KEY rotated? The credential must be re-entered.",
                row.provider,
            )
        return None
    return ResolvedHostedProvider(api_key=api_key, api_base=row.api_base, client_args=client_args)


def client_args_of(row: HostedProvider) -> dict[str, Any] | None:
    """The stored SDK client extras, or None when there are none or no configured key reads them."""
    try:
        return _decrypt_client_args(row.encrypted_client_args)
    except (SecretDecryptionError, SecretBoxUnavailableError):
        return None
