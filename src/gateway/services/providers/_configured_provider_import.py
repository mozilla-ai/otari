"""How a ``providers:`` entry of config.yml is copied onto a hosted provider of its name.

The entry is read once, its ``${VAR}`` references already resolved, and the
hosted provider serves the name from then on, with every control a hosted
provider has and every model it lists switched on. An entry a hosted provider
cannot hold keeps serving from config.yml, and this says why.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from gateway.core.config import hosted_key_optional, hosted_provider_name_problem
from gateway.exceptions import TenancyError
from gateway.schemas.providers import HostedProviderCreateRequest
from gateway.services.providers._hosted_credentials import normalize_provider

# What a hosted provider holds of an entry. Anything else configures the
# instance in a way a hosted provider has no field for.
_IMPORTED_KEYS = frozenset({"api_key", "api_base", "provider_type", "client_args"})


@dataclass(frozen=True)
class PlannedImport:
    """One entry's plan: the create that copies it, or why it stays in config.yml."""

    name: str
    create: HostedProviderCreateRequest | None = None
    reason: str | None = None


def plan_import(name: str, entry: Mapping[str, Any], *, hosted_names: set[str]) -> PlannedImport:
    """Plan the copy of one ``providers:`` entry; ``hosted_names`` are the stored hosted providers."""
    extra = sorted(set(entry) - _IMPORTED_KEYS)
    if extra:
        return PlannedImport(name, reason=f"it sets {', '.join(extra)}, which a hosted provider has no field for")
    try:
        implementation = normalize_provider(str(entry.get("provider_type") or name))
    except TenancyError as exc:
        return PlannedImport(name, reason=str(exc))
    if (problem := hosted_provider_name_problem(name, implementation)) is not None:
        return PlannedImport(name, reason=f"the name {problem}")
    if name in hosted_names:
        return PlannedImport(name, reason="a hosted provider already holds the name")
    api_key = entry.get("api_key") or None
    api_base = entry.get("api_base") or None
    if api_key is None and not hosted_key_optional(implementation, api_base):
        return PlannedImport(
            name, reason=f"it names no key, and a hosted '{implementation}' provider needs one at its default endpoint"
        )
    client_args = entry.get("client_args") or None
    if client_args is not None and not isinstance(client_args, Mapping):
        return PlannedImport(name, reason="its client_args is not a mapping")
    try:
        create = HostedProviderCreateRequest(
            provider=implementation,
            name=None if name == implementation else name,
            api_key=str(api_key) if api_key is not None else None,
            api_base=str(api_base) if api_base is not None else None,
            client_args=dict(client_args) if client_args is not None else None,
        )
    except ValidationError as exc:
        # Named by field, never by value: the input refused may be the key.
        fields = sorted({str(error["loc"][0]) for error in exc.errors(include_input=False) if error["loc"]})
        return PlannedImport(name, reason=f"a hosted provider cannot take its {', '.join(fields) or 'values'}")
    return PlannedImport(name, create=create)


__all__ = ["PlannedImport", "plan_import"]
