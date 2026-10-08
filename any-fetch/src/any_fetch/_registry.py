"""The providers this package serves.

A literal table, not entry-point discovery, so what a host can call is what
this package ships (scripts/check_architecture.py bans the discovery modules).
"""

from typing import TYPE_CHECKING, Any

from any_fetch._errors import UnsupportedProviderError
from any_fetch._types import BuiltinFactory, ProviderMetadata

if TYPE_CHECKING:
    from any_fetch._api import AnyFetch


def _providers() -> dict[str, type["AnyFetch"]]:
    # Imported on first use: each provider subclasses AnyFetch, whose module imports this one.
    from any_fetch.providers.builtin import BuiltinProvider
    from any_fetch.providers.exa import ExaProvider
    from any_fetch.providers.fake import FakeProvider

    return {provider.METADATA.name: provider for provider in (BuiltinProvider, ExaProvider, FakeProvider)}


def get_supported_providers() -> list[str]:
    """Return the names of every provider this package serves."""
    return sorted(_providers())


def get_provider_class(provider: str) -> type["AnyFetch"]:
    """Return a provider's class without building it."""
    providers = _providers()
    if provider not in providers:
        raise UnsupportedProviderError(provider, sorted(providers))
    return providers[provider]


def get_provider_metadata(provider: str) -> ProviderMetadata:
    """Return a provider's metadata."""
    return get_provider_class(provider).METADATA


def create(provider: str, **kwargs: Any) -> "AnyFetch":
    """Build a provider by name, passing it the keyword arguments."""
    return get_provider_class(provider)(**kwargs)


def register_builtin(factory: BuiltinFactory) -> None:
    """Register the host's implementation of ``builtin``."""
    from any_fetch.providers import builtin

    builtin.register(factory)
