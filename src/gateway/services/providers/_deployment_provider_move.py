"""How a deployment provider stored through the dashboard moves onto a hosted provider.

A ``provider_credentials`` row becomes a hosted provider of the same name, so
an ``instance:model`` selector keeps reaching it, served by the implementation
the row declares. This module plans one row; the service creates the hosted
provider and removes the row.
"""

from dataclasses import dataclass, field
from typing import Any

from gateway.core.config import PROVIDER_TYPE_ALIASES, hosted_key_optional
from gateway.exceptions import TenancyError
from gateway.models.providers import ProviderCredential
from gateway.schemas.providers import DeploymentProviderMigrationRowPublic, HostedProviderCreateRequest
from gateway.services.providers._hosted_credentials import (
    lookup_name,
    normalize_name,
    normalize_provider,
    public_api_base,
)
from gateway.services.secret_box import SecretBoxUnavailableError, SecretDecryptionError, decrypt_secret


@dataclass
class PlannedMove:
    """One row's plan: what the preview shows, and the create that moves it when there is one."""

    row: DeploymentProviderMigrationRowPublic
    create: HostedProviderCreateRequest | None = field(default=None, repr=False)
    name: str | None = None
    updated_at: Any = None


def plan_move(credential: ProviderCredential, *, hosted_names: set[str], configured_instances: set[str]) -> PlannedMove:
    """Plan the move of one stored deployment provider.

    ``hosted_names`` are the stored hosted providers, whose names are taken;
    ``configured_instances`` the ``providers:`` instances config.yml declares,
    which keep being served ahead of every hosted provider.
    """
    instance = credential.instance
    declared = credential.provider_type or instance
    public = DeploymentProviderMigrationRowPublic(
        instance=instance,
        provider=lookup_name(declared),
        api_base=public_api_base(credential.api_base),
        outcome="migrate",
        updated_at=credential.updated_at,
    )
    try:
        implementation = normalize_provider(declared)
    except TenancyError as exc:
        public.outcome = "invalid"
        public.detail = str(exc)
        return PlannedMove(public)
    try:
        name = normalize_name(instance, implementation)
    except TenancyError as exc:
        public.outcome = "invalid"
        public.detail = (
            f"{exc}, so a hosted provider cannot keep it. Keep the deployment provider serving it; if its callers "
            f"can use '{implementation}:<model>', add a hosted provider of that name instead."
        )
        return PlannedMove(public)
    if name in hosted_names:
        public.outcome = "conflict"
        public.detail = f"A hosted provider named '{name}' already exists. Remove or rename one of them."
        return PlannedMove(public)

    api_key: str | None = None
    if credential.encrypted_api_key is not None:
        try:
            api_key = decrypt_secret(credential.encrypted_api_key)
        except (SecretDecryptionError, SecretBoxUnavailableError):
            public.outcome = "unreadable"
            public.detail = "Its key cannot be decrypted. Re-enter it on the deployment provider, then migrate."
            return PlannedMove(public)
    elif not hosted_key_optional(implementation, credential.api_base):
        public.outcome = "invalid"
        public.detail = f"It holds no key, and '{implementation}' needs one."
        return PlannedMove(public)

    if credential.provider_type in PROVIDER_TYPE_ALIASES:
        public.notes.append(
            f"Declared as {credential.provider_type}: its models are priced from {implementation}'s community "
            "defaults where one matches. Check the prices once it is migrated."
        )
    if instance in configured_instances:
        public.notes.append(
            "config.yml declares a provider of the same name, which is served ahead of every hosted provider."
        )
    create = HostedProviderCreateRequest(
        provider=implementation,
        name=None if name == implementation else name,
        api_key=api_key,
        api_base=credential.api_base,
        client_args=credential.client_args or None,
    )
    return PlannedMove(public, create=create, name=name, updated_at=credential.updated_at)


__all__ = ["PlannedMove", "plan_move"]
