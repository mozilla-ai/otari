"""The provider endpoints a workspace or one of its users owns.

An endpoint is somebody else's model server: the owner supplies the base URL
and the key, and pays the upstream, so three rules follow.

**Only a public address.** The URL comes from a tenant rather than the
operator, so it must resolve to public addresses only, whatever
``provider_allow_private_hosts`` says. It is checked here, and every dispatch
re-checks and pins it (``_owned_endpoint_network``).

**No shadowing.** A caller reaches an endpoint as ``<name>:<model>``, so a
name that is a provider's or a configured instance's would take over a
selector that already means something. Such names are refused.

**Defaults are request fields, nothing more.** ``default_params`` go into the
request body beneath the caller's own fields, so the credential and transport
fields otari never takes from a caller are refused, as are the fields that
shape the request itself.
"""

import re
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from any_llm import LLMProvider

from gateway.core.config import PROVIDER_TYPE_ALIASES, RESERVED_PROVIDER_INSTANCE_NAMES, GatewayConfig
from gateway.core.database import DATABASE_ERRORS
from gateway.core.provider_params import FORBIDDEN_ENDPOINT_DEFAULTS
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.providers_exceptions import (
    ProviderEndpointAlreadyExistsError,
    ProviderEndpointInvalidError,
    ProviderEndpointNotFoundError,
    ProviderEndpointOwnerNotFoundError,
    ProviderEndpointsDisabledError,
)
from gateway.log_config import logger
from gateway.models.secret_fields import restore_redacted_values
from gateway.repositories.providers import EndpointNameConflict, ProviderEndpointRepository
from gateway.schemas.providers import (
    ProviderEndpointCreateRequest,
    ProviderEndpointPublic,
    ProviderEndpointsPublic,
    ProviderEndpointUpdateRequest,
)
from gateway.services.providers._owned_endpoint_network import (
    OwnedEndpointAddressError,
    check_owned_endpoint_api_base,
)
from gateway.services.secret_box import SecretBoxUnavailableError, encrypt_secret
from gateway.services.tenancy.errors import SecretBoxUnavailableTenancyError

# Implementations that take nothing but a base URL and a key. A provider that
# can authenticate from the deployment's own environment (Bedrock's instance
# profile, Vertex AI's application default credentials) would hand those to a
# tenant's server, so it is not on this list.
OWNED_ENDPOINT_PROVIDERS: frozenset[str] = frozenset({LLMProvider.OPENAI.value, LLMProvider.ANTHROPIC.value})

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ProviderEndpointService:
    """Create, change and remove owned provider endpoints."""

    def __init__(
        self,
        uow: UnitOfWork,
        *,
        config: GatewayConfig,
        endpoints: ProviderEndpointRepository,
        resolve_workspace: Callable[[uuid.UUID | None], Awaitable[uuid.UUID | None]],
        user_is_active: Callable[[str], Awaitable[bool]],
        refresh_cache: Callable[[], Awaitable[None]],
    ) -> None:
        """Bind the unit of work and the lookups this service needs from other domains.

        ``resolve_workspace`` answers the deployment's default workspace for
        ``None`` and ``None`` for a workspace that does not exist.
        ``refresh_cache`` reloads the dispatch-path copy after a write has
        committed. All three are callables because they take the session, which
        a service may not name.
        """
        self.uow = uow
        self.config = config
        self.endpoints = endpoints
        self.resolve_workspace = resolve_workspace
        self.user_is_active = user_is_active
        self.refresh_cache = refresh_cache

    async def list_endpoints(
        self, *, workspace_id: uuid.UUID | None, user_id: str | None, skip: int, limit: int
    ) -> ProviderEndpointsPublic:
        """One page of endpoints, narrowed by owner when given."""
        self._require_enabled()
        async with self.uow:
            rows, count = await self.endpoints.list_page(
                workspace_id=workspace_id, user_id=user_id, skip=skip, limit=limit
            )
        return ProviderEndpointsPublic(data=[ProviderEndpointPublic.from_row(row) for row in rows], count=count)

    async def get_endpoint(self, endpoint_id: uuid.UUID) -> ProviderEndpointPublic:
        self._require_enabled()
        async with self.uow:
            row = await self.endpoints.get(endpoint_id)
        if row is None:
            raise ProviderEndpointNotFoundError(endpoint_id)
        return ProviderEndpointPublic.from_row(row)

    async def create_endpoint(self, request: ProviderEndpointCreateRequest) -> ProviderEndpointPublic:
        """Create an endpoint for a workspace, or for one user in it."""
        self._require_enabled()
        workspace_id = await self.resolve_workspace(request.workspace_id)
        if workspace_id is None:
            raise ProviderEndpointOwnerNotFoundError(f"Workspace '{request.workspace_id}' not found")
        if request.user_id is not None and not await self.user_is_active(request.user_id):
            raise ProviderEndpointOwnerNotFoundError(f"User '{request.user_id}' not found")
        name = self._validated_name(request.name)
        provider = _validated_provider(request.provider)
        await _check_api_base(request.api_base)
        default_params = _validated_default_params(request.default_params)
        encrypted_api_key, last4 = _encrypt_api_key(request.api_key)
        try:
            async with self.uow:
                row = await self.endpoints.insert(
                    workspace_id=workspace_id,
                    user_id=request.user_id,
                    name=name,
                    provider=provider,
                    api_base=request.api_base,
                    encrypted_api_key=encrypted_api_key,
                    last4=last4,
                    default_params=default_params,
                )
        except EndpointNameConflict:
            raise ProviderEndpointAlreadyExistsError(name) from None
        await self._refresh()
        return ProviderEndpointPublic.from_row(row)

    async def update_endpoint(
        self, endpoint_id: uuid.UUID, request: ProviderEndpointUpdateRequest
    ) -> ProviderEndpointPublic:
        """Apply the fields the request sets. The owner cannot change."""
        self._require_enabled()
        fields = request.model_dump(exclude_unset=True)
        async with self.uow:
            row = await self.endpoints.get(endpoint_id)
        if row is None:
            raise ProviderEndpointNotFoundError(endpoint_id)

        # Validated outside a block, so no transaction is open across the DNS lookup.
        changes: dict[str, Any] = {}
        if fields.get("name") is not None:
            changes["name"] = self._validated_name(fields["name"])
        if fields.get("provider") is not None:
            changes["provider"] = _validated_provider(fields["provider"])
        if fields.get("api_base") is not None:
            await _check_api_base(fields["api_base"])
            changes["api_base"] = fields["api_base"]
        if "default_params" in fields:
            # The read masks credential-shaped entries, so an editor resubmitting
            # what it was shown keeps the stored values.
            restored = restore_redacted_values(fields["default_params"], row.default_params)
            changes["default_params"] = _validated_default_params(restored)
        if "api_key" in fields:
            changes["encrypted_api_key"], changes["last4"] = _encrypt_api_key(fields["api_key"])

        try:
            async with self.uow:
                row = await self.endpoints.apply(row, changes)
        except EndpointNameConflict:
            raise ProviderEndpointAlreadyExistsError(changes["name"]) from None
        await self._refresh()
        return ProviderEndpointPublic.from_row(row)

    async def delete_endpoint(self, endpoint_id: uuid.UUID) -> None:
        self._require_enabled()
        async with self.uow:
            row = await self.endpoints.get(endpoint_id)
            if row is None:
                raise ProviderEndpointNotFoundError(endpoint_id)
            await self.endpoints.delete(row)
        await self._refresh()

    def _require_enabled(self) -> None:
        if not self.config.provider_endpoints_enabled:
            raise ProviderEndpointsDisabledError

    def _validated_name(self, name: str) -> str:
        trimmed = name.strip()
        if not _NAME_PATTERN.fullmatch(trimmed):
            raise ProviderEndpointInvalidError(
                "An endpoint name uses letters, digits, '.', '_' and '-', and starts with a letter or digit"
            )
        if (
            trimmed in RESERVED_PROVIDER_INSTANCE_NAMES
            or trimmed in PROVIDER_TYPE_ALIASES
            or trimmed in {provider.value for provider in LLMProvider}
        ):
            raise ProviderEndpointInvalidError(f"'{trimmed}' names a provider, so it cannot name an endpoint")
        if trimmed in self.config.providers:
            raise ProviderEndpointInvalidError(
                f"'{trimmed}' is a configured provider instance, so it cannot name an endpoint"
            )
        return trimmed

    async def _refresh(self) -> None:
        # The write has committed; a failed refresh must not turn it into a 500.
        # This worker catches up on its next tick, the others within the TTL.
        try:
            await self.refresh_cache()
        except DATABASE_ERRORS:
            logger.warning("Provider endpoint cache refresh failed after a write; converges within the TTL")


def _validated_provider(provider: str) -> str:
    trimmed = provider.strip()
    canonical = PROVIDER_TYPE_ALIASES.get(trimmed, trimmed)
    if canonical not in OWNED_ENDPOINT_PROVIDERS:
        allowed = ", ".join(sorted(OWNED_ENDPOINT_PROVIDERS))
        raise ProviderEndpointInvalidError(f"An endpoint's provider must be one of: {allowed}")
    return canonical


async def _check_api_base(api_base: str) -> None:
    try:
        await check_owned_endpoint_api_base(api_base)
    except OwnedEndpointAddressError as exc:
        raise ProviderEndpointInvalidError(str(exc)) from None


def _validated_default_params(default_params: dict[str, Any] | None) -> dict[str, Any] | None:
    if not default_params:
        return None
    forbidden = sorted(set(default_params) & FORBIDDEN_ENDPOINT_DEFAULTS)
    if forbidden:
        raise ProviderEndpointInvalidError(f"These fields cannot be endpoint defaults: {', '.join(forbidden)}")
    return default_params


def _encrypt_api_key(api_key: str | None) -> tuple[str | None, str | None]:
    """Encrypt a plaintext key for storage, or clear it. Never logs the plaintext."""
    if not api_key:
        return None, None
    try:
        return encrypt_secret(api_key), api_key[-4:]
    except SecretBoxUnavailableError:
        raise SecretBoxUnavailableTenancyError from None
