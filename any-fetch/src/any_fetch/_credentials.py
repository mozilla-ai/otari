"""The API key and base URL a provider runs with: what the caller passed, else the environment."""

import os

from any_fetch._errors import MissingCredentialError
from any_fetch._types import ProviderMetadata


def resolve_api_key(metadata: ProviderMetadata, api_key: str | None) -> str | None:
    """Return the explicit key, else the one in the variable the metadata names.

    An explicit empty key counts as passed: it never falls back to the
    environment, so a host that hands over a tenant's empty credential cannot
    end up searching on its own key.
    """
    if api_key is None and metadata.env_key:
        api_key = os.environ.get(metadata.env_key) or None
    if not api_key:
        if metadata.requires_api_key:
            raise MissingCredentialError(metadata.name, "api_key", metadata.env_key)
        return None
    return api_key


def resolve_api_base(metadata: ProviderMetadata, api_base: str | None) -> str | None:
    """Return the explicit base URL, else the one in the environment, else the provider's default.

    An explicit empty base URL means the provider's default, not the environment's.
    """
    if api_base is None and metadata.env_api_base:
        api_base = os.environ.get(metadata.env_api_base) or None
    api_base = api_base or metadata.default_api_base
    if not api_base and metadata.requires_api_base:
        raise MissingCredentialError(metadata.name, "api_base", metadata.env_api_base)
    return api_base or None
