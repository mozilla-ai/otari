"""How a hosted provider's secrets are stored and read back.

The key and the SDK client extras are both encrypted with the deployment's
secret box. They are read back here and nowhere else, into a value that never
crosses the API boundary.
"""

import json
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from any_llm import LLMProvider

from gateway.core.config import PROVIDER_TYPE_ALIASES, hosted_provider_name_problem
from gateway.exceptions.providers_exceptions import (
    HostedProviderApiBaseMaskedError,
    HostedProviderInvalidNameError,
    HostedProviderUnknownProviderError,
)
from gateway.exceptions.shared_exceptions import SecretBoxUnavailableTenancyError
from gateway.log_config import logger
from gateway.models.providers import HostedProvider
from gateway.models.secret_fields import REDACTED_VALUE
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
)
from gateway.services.url_safety import redact_url_secrets

# What the secret-box error names as the thing that could not be stored.
STORED = "hosted provider credentials"


@dataclass(frozen=True)
class ResolvedHostedProvider:
    """A decrypted, usable upstream connection.

    ``api_key`` is None for a keyless backend, or a custom endpoint called
    with the placeholder key.
    """

    api_key: str | None = field(repr=False)
    api_base: str | None
    client_args: dict[str, Any] | None
    name: str | None = None
    """The hosted provider's name, an accepted alias folded, for a credential read back from a source."""
    implementation: str | None = None
    """The any-llm implementation serving it, for a credential read back from a source."""
    supported_countries: tuple[str, ...] | None = None
    """The countries whose requests it serves, or None for every country."""


def last4(api_key: str) -> str | None:
    """The tail of a key, or None for a key so short the tail would be the whole of it."""
    return api_key[-4:] if len(api_key) > 4 else None


def normalize_api_base(api_base: str | None) -> str | None:
    """Fold a blank endpoint to None, which means the provider's default."""
    if api_base is None:
        return None
    trimmed = api_base.strip()
    return trimmed or None


def public_api_base(api_base: str | None) -> str | None:
    """An endpoint as the API shows it, with a credential in its userinfo or query masked.

    An endpoint that will not parse is shown as None rather than whole,
    because nothing could find the credential in it to mask.
    """
    if api_base is None:
        return None
    try:
        urlsplit(api_base)
    except ValueError:
        return None
    return redact_url_secrets(api_base)


def unmask_api_base(incoming: str, stored: str | None) -> str:
    """The stored endpoint where the caller echoed back its masked form, else what was sent.

    The edit form sends the endpoint it was shown, so an unchanged base comes
    back masked and has to keep the credential it hides.

    Raises:
        HostedProviderApiBaseMaskedError: the endpoint carries the mask but is
            not the stored one echoed back, so the mask would be stored as the
            credential.
    """
    if stored is not None and incoming == public_api_base(stored):
        return stored
    if _carries_mask(incoming):
        raise HostedProviderApiBaseMaskedError
    return incoming


def _carries_mask(api_base: str) -> bool:
    try:
        parts = urlsplit(api_base)
        username, password = parts.username, parts.password
    except ValueError:
        return False
    if REDACTED_VALUE in (username, password):
        return True
    return any(value == REDACTED_VALUE for _, value in parse_qsl(parts.query, keep_blank_values=True))


_DEFAULT_PORTS = {"http": 80, "https": 443}


def origin_of(api_base: str | None) -> tuple[str, str, int | None] | None:
    """The origin an endpoint points at, as ``(scheme, host, port)``, or None for the provider's default.

    The scheme and the effective port are part of it: a key sent to the same
    host over plain HTTP, or to another port, is sent to another endpoint.
    """
    if api_base is None:
        return None
    parts = urlsplit(api_base)
    if not parts.hostname:
        return None
    scheme = parts.scheme.lower()
    return scheme, parts.hostname.lower(), parts.port if parts.port is not None else _DEFAULT_PORTS.get(scheme)


def lookup_name(provider: str) -> str:
    """The stored spelling of a provider name, with the accepted aliases folded."""
    candidate = provider.strip()
    return PROVIDER_TYPE_ALIASES.get(candidate, candidate)


def normalize_name(name: str | None, implementation: str) -> str:
    """The name a hosted provider is stored under: ``name`` when given, else its implementation.

    Raises:
        HostedProviderInvalidNameError: the name cannot prefix a selector, is
            reserved, or is another implementation's.
    """
    name = implementation if name is None else name
    if (problem := hosted_provider_name_problem(name, implementation)) is not None:
        raise HostedProviderInvalidNameError(name, problem)
    return name


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


def encrypt_key(api_key: str | None) -> str | None:
    """Encrypt a key for storage; None for a provider stored without one.

    Raises:
        SecretBoxUnavailableTenancyError: no usable ``OTARI_SECRET_KEY``.
    """
    if api_key is None:
        return None
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
        api_key = None if row.encrypted_api_key is None else decrypt_secret(row.encrypted_api_key)
        client_args = _decrypt_client_args(row.encrypted_client_args)
    except (SecretDecryptionError, SecretBoxUnavailableError):
        if not quiet:
            logger.error(
                "Hosted provider %r could not be decrypted; treating it as unserved. "
                "Was OTARI_SECRET_KEY rotated? The credential must be re-entered.",
                row.name,
            )
        return None
    return ResolvedHostedProvider(
        api_key=api_key,
        api_base=row.api_base,
        client_args=client_args,
        name=row.name,
        implementation=row.provider,
        supported_countries=tuple(row.supported_countries) if row.supported_countries is not None else None,
    )


def reencrypted(ciphertext: str | None) -> str | None:
    """``ciphertext`` decrypted and encrypted again under the primary key, or None for None.

    Raises:
        SecretDecryptionError: no configured key reads it.
        SecretBoxUnavailableError: no usable ``OTARI_SECRET_KEY``.
    """
    return None if ciphertext is None else encrypt_secret(decrypt_secret(ciphertext))


def client_args_of(row: HostedProvider) -> dict[str, Any] | None:
    """The stored SDK client extras, or None when there are none or no configured key reads them."""
    try:
        return _decrypt_client_args(row.encrypted_client_args)
    except (SecretDecryptionError, SecretBoxUnavailableError):
        return None
