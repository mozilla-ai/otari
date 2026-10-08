"""The providers this package serves.

A literal table, not entry-point discovery, so what a host can call is what
this package ships (scripts/check_architecture.py bans the discovery modules).
"""

from typing import TYPE_CHECKING, Any

from any_search._errors import UnsupportedProviderError
from any_search._types import ProviderMetadata

if TYPE_CHECKING:
    from any_search._api import AnySearch


def _providers() -> dict[str, type["AnySearch"]]:
    # Imported on first use: each provider subclasses AnySearch, whose module imports this one.
    from any_search.providers.fake import FakeProvider

    return {provider.METADATA.name: provider for provider in (FakeProvider,)}


def get_supported_providers() -> list[str]:
    """Return the names of every provider this package serves."""
    return sorted(_providers())


def get_provider_class(provider: str) -> type["AnySearch"]:
    """Return a provider's class without building it."""
    providers = _providers()
    if provider not in providers:
        raise UnsupportedProviderError(provider, sorted(providers))
    return providers[provider]


def get_provider_metadata(provider: str) -> ProviderMetadata:
    """Return a provider's metadata."""
    return get_provider_class(provider).METADATA


def create(provider: str, **kwargs: Any) -> "AnySearch":
    """Build a provider by name, passing it the keyword arguments."""
    return get_provider_class(provider)(**kwargs)
