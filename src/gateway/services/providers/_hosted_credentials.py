"""How a hosted provider's secrets are stored and read back.

The key and the SDK client extras are both encrypted with the deployment's
secret box. They are read back here and nowhere else, into a value that never
crosses the API boundary.
"""

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from any_llm import LLMProvider

from gateway.core.config import PROVIDER_TYPE_ALIASES, REGION_SEPARATOR, hosted_provider_name_problem
from gateway.exceptions.providers_exceptions import (
    HostedProviderApiBaseMaskedError,
    HostedProviderInvalidNameError,
    HostedProviderUnknownProviderError,
)
from gateway.exceptions.shared_exceptions import SecretBoxUnavailableTenancyError
from gateway.log_config import logger
from gateway.models.providers import HostedProvider, HostedProviderRegion
from gateway.models.secret_fields import REDACTED_VALUE
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_mapping,
    decrypt_secret,
    encrypt_mapping,
    encrypt_secret,
    readable_mapping,
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
    region: str | None = None
    """The slug of the region it serves from, or None for a provider with no regions."""
    data_location: str | None = None
    """Where the upstream keeps request data (a country code, ``EU`` or ``GLOBAL``), or None when undeclared."""
    routing_locality: str | None = None
    """Where the operator declares the inference runs (``local``, ``private``, ``cloud``), or None when undeclared.

    The provider's declaration: every region inherits it."""


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
    if REGION_SEPARATOR in name:
        # It would read as a region of another provider in a selector.
        raise HostedProviderInvalidNameError(name, f"must not contain '{REGION_SEPARATOR}'")
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
    """Encrypt the SDK client extras for storage.

    Raises:
        SecretBoxUnavailableTenancyError: no usable ``OTARI_SECRET_KEY``.
    """
    try:
        return encrypt_mapping(client_args)
    except SecretBoxUnavailableError:
        raise SecretBoxUnavailableTenancyError(STORED) from None


def credential_of(row: HostedProvider, *, quiet: bool = False) -> ResolvedHostedProvider | None:
    """Decrypt a row's key and extras, or None when no configured key reads them.

    Logged unless ``quiet``, because a rotated-away ``OTARI_SECRET_KEY``
    degrades one provider silently otherwise.
    """
    try:
        api_key = None if row.encrypted_api_key is None else decrypt_secret(row.encrypted_api_key)
        client_args = decrypt_mapping(row.encrypted_client_args)
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
        data_location=row.data_location,
        routing_locality=row.routing_locality,
    )


def _same_origin(api_base: str, other: str | None) -> bool:
    try:
        return origin_of(api_base) == origin_of(other)
    except ValueError:
        return False


def region_credential_of(
    provider: ResolvedHostedProvider, region: HostedProviderRegion, *, quiet: bool = False
) -> ResolvedHostedProvider | None:
    """``provider``'s connection as ``region`` changes it, or None when no configured key reads the region's.

    Every value the region leaves unset is the provider's, and the region's SDK
    extras are merged over the provider's, so a Bedrock region names only its
    ``region_name``. A region on another origin takes its own extras alone: the
    provider's may carry a credential (a header, an IAM secret) meant for the
    provider's endpoint.
    """
    try:
        api_key = None if region.encrypted_api_key is None else decrypt_secret(region.encrypted_api_key)
        client_args = decrypt_mapping(region.encrypted_client_args)
    except (SecretDecryptionError, SecretBoxUnavailableError):
        if not quiet:
            logger.error(
                "Region %r of hosted provider %r could not be decrypted; treating it as unserved. "
                "Was OTARI_SECRET_KEY rotated? The credential must be re-entered.",
                region.slug,
                provider.name,
            )
        return None
    same_origin = region.api_base is None or _same_origin(region.api_base, provider.api_base)
    if not same_origin and region.encrypted_api_key is None and provider.api_key is not None:
        # The provider's key was entered for the provider's endpoint, not this one.
        if not quiet:
            logger.error(
                "Region %r of hosted provider %r uses the provider's key on another endpoint; treating it as unserved.",
                region.slug,
                provider.name,
            )
        return None
    merged = (
        {**provider.client_args, **(client_args or {})}
        if same_origin and provider.client_args is not None
        else client_args
    )
    return ResolvedHostedProvider(
        api_key=api_key if region.encrypted_api_key is not None else provider.api_key,
        api_base=region.api_base if region.api_base is not None else provider.api_base,
        client_args=merged,
        name=provider.name,
        implementation=provider.implementation,
        supported_countries=tuple(region.supported_countries)
        if region.supported_countries is not None
        else provider.supported_countries,
        region=region.slug,
        # A region's own declaration, never the provider's: it is copied when the region is created.
        data_location=region.data_location,
        # The provider's: a region has no locality of its own.
        routing_locality=provider.routing_locality,
    )


def region_decryptable(region: HostedProviderRegion) -> bool:
    """Whether a configured key reads the region's own key and extras; true for a region holding neither."""
    try:
        if region.encrypted_api_key is not None:
            decrypt_secret(region.encrypted_api_key)
        decrypt_mapping(region.encrypted_client_args)
    except (SecretDecryptionError, SecretBoxUnavailableError):
        return False
    return True


def client_args_of(row: HostedProvider | HostedProviderRegion) -> dict[str, Any] | None:
    """The stored SDK client extras, or None when there are none or no configured key reads them."""
    return readable_mapping(row.encrypted_client_args)
