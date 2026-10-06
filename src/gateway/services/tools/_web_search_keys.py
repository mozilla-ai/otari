"""An organization's own web search keys, and which one each of its workspaces searches with.

A workspace whose organization holds a usable key searches with it, so the organization's
own search account pays and its own quota applies. Any other workspace uses the
deployment's search, which these keys never change: the two never merge.

Organization owners and admins manage the keys. A workspace's owners and admins pin one
key as the workspace's own or turn a key off for it, and any member can see which key
the workspace uses. :func:`resolve_web_search_key` decides that key.
"""

import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from gateway.core.config import WEB_SEARCH_PROVIDERS
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.shared_exceptions import SecretBoxUnavailableTenancyError
from gateway.exceptions.tools_exceptions import (
    WebSearchKeyAlreadyExistsError,
    WebSearchKeyArchivedError,
    WebSearchKeyDefaultConflictError,
    WebSearchKeyNameRequiredError,
    WebSearchKeyNotArchivedError,
    WebSearchKeyNotFoundError,
    WebSearchKeyUnknownProviderError,
    WorkspaceWebSearchKeyOverrideConflictError,
)
from gateway.log_config import logger
from gateway.models.tenancy import User
from gateway.models.tools import OrgWebSearchKey, WebSearchCredential
from gateway.repositories.tenancy import WorkspaceRepository
from gateway.repositories.tools import (
    OrgWebSearchKeyRepository,
    WebSearchKeyConflict,
    WorkspaceWebSearchKeyOverrideRepository,
    resolve_web_search_key,
)
from gateway.schemas.tools import (
    OrgWebSearchKeyCreateRequest,
    OrgWebSearchKeyPublic,
    OrgWebSearchKeysPublic,
    OrgWebSearchKeyUpdateRequest,
    WorkspaceWebSearchKeyOverrideRequest,
    WorkspaceWebSearchKeyPublic,
    WorkspaceWebSearchKeysPublic,
)
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
)
from gateway.services.tenancy import OrganizationService
from gateway.services.tenancy.authorization import WorkspaceAccess

# Takes the workspace's row lock for the rest of the transaction.
LockWorkspace = Callable[[uuid.UUID], Awaitable[None]]


def search_key_credential(key: OrgWebSearchKey) -> WebSearchCredential | None:
    """The key as a credential, or ``None`` where this deployment cannot decrypt it."""
    try:
        return WebSearchCredential(provider=key.provider, api_key=decrypt_secret(key.encrypted_api_key))
    except (SecretBoxUnavailableError, SecretDecryptionError):
        logger.warning("Web search key %s cannot be decrypted on this deployment", key.id)
        return None


def search_key_is_usable(key: OrgWebSearchKey) -> bool:
    return search_key_credential(key) is not None


async def workspace_search_credential(
    workspaces: WorkspaceRepository, overrides: WorkspaceWebSearchKeyOverrideRepository, workspace_id: uuid.UUID
) -> WebSearchCredential | None:
    """The key a workspace searches with, or ``None`` where it uses the deployment's search.

    Read by every search a workspace makes: the in-loop tool and direct search alike.
    Only the keys the choice reaches are decrypted.
    """
    workspace = await workspaces.get(workspace_id)
    if workspace is None:
        return None
    candidates = await overrides.candidates(organization_id=workspace.organization_id, workspace_id=workspace.id)
    credentials: dict[uuid.UUID, WebSearchCredential | None] = {}

    def credential(key: OrgWebSearchKey) -> WebSearchCredential | None:
        if key.id not in credentials:
            credentials[key.id] = search_key_credential(key)
        return credentials[key.id]

    chosen = resolve_web_search_key(candidates, lambda key: credential(key) is not None)
    return credential(chosen) if chosen is not None else None


class WebSearchKeyService:
    """Manage an organization's web search keys and its workspaces' choice among them."""

    def __init__(
        self,
        uow: UnitOfWork,
        *,
        keys: OrgWebSearchKeyRepository,
        overrides: WorkspaceWebSearchKeyOverrideRepository,
        organizations: OrganizationService,
        workspaces: WorkspaceAccess,
        lock_workspace: LockWorkspace,
    ) -> None:
        """Bind the unit of work, the repositories, and the tenancy checks this service asks.

        ``lock_workspace`` takes the workspace's row lock, which serializes two admins
        pinning different keys for one workspace; it is a callable because it needs the session.
        """
        self.uow = uow
        self.keys = keys
        self.overrides = overrides
        self.organizations = organizations
        self.workspaces = workspaces
        self.lock_workspace = lock_workspace

    # ------------------------------------------------------------------
    # The organization's keys: owners and admins only
    # ------------------------------------------------------------------

    async def list_keys(self, *, user: User, include_archived: bool, skip: int, limit: int) -> OrgWebSearchKeysPublic:
        organization_id = await self._managed_organization(user)
        async with self.uow:
            rows, count = await self.keys.list_for_organization(
                organization_id, include_archived=include_archived, skip=skip, limit=limit
            )
        return OrgWebSearchKeysPublic(data=[_public(row) for row in rows], count=count)

    async def create_key(self, *, user: User, request: OrgWebSearchKeyCreateRequest) -> OrgWebSearchKeyPublic:
        organization_id = await self._managed_organization(user)
        provider = _validated_provider(request.provider)
        name = _validated_name(request.name)
        encrypted, last4 = _encrypted(request.api_key)
        try:
            async with self.uow:
                row = await self.keys.create_key(
                    organization_id=organization_id,
                    provider=provider,
                    name=name,
                    encrypted_api_key=encrypted,
                    last4=last4,
                )
        except WebSearchKeyConflict:
            raise WebSearchKeyAlreadyExistsError(provider, name) from None
        return _public(row)

    async def update_key(
        self, *, user: User, key_id: uuid.UUID, request: OrgWebSearchKeyUpdateRequest
    ) -> OrgWebSearchKeyPublic:
        organization_id = await self._managed_organization(user)
        values: dict[str, object] = {}
        if request.name is not None:
            values["name"] = _validated_name(request.name)
        if request.api_key is not None:
            values["encrypted_api_key"], values["last4"] = _encrypted(request.api_key)
        provider = ""
        try:
            async with self.uow:
                key = await self._live_key(key_id, organization_id)
                provider = key.provider
                name = values.get("name")
                if (
                    name is not None
                    and name != key.name
                    and await self.keys.get_by_name(organization_id=organization_id, provider=provider, name=str(name))
                ):
                    raise WebSearchKeyAlreadyExistsError(provider, str(name))
                row = await self.keys.update_key(key, values) if values else key
        except WebSearchKeyConflict:
            # A rollback expires the loaded row, so the error is built from what was read before it.
            raise WebSearchKeyAlreadyExistsError(provider, str(values["name"])) from None
        return _public(row)

    async def archive_key(self, *, user: User, key_id: uuid.UUID) -> OrgWebSearchKeyPublic:
        """Take a key out of use; workspaces fall back as if it were never added."""
        organization_id = await self._managed_organization(user)
        async with self.uow:
            key = await self._key(key_id, organization_id)
            if key.archived_at is None:
                key = await self.keys.update_key(key, {"archived_at": datetime.now(UTC), "is_org_default": False})
        return _public(key)

    async def restore_key(self, *, user: User, key_id: uuid.UUID) -> OrgWebSearchKeyPublic:
        organization_id = await self._managed_organization(user)
        async with self.uow:
            key = await self._key(key_id, organization_id)
            key = await self.keys.update_key(key, {"archived_at": None})
        return _public(key)

    async def delete_key(self, *, user: User, key_id: uuid.UUID) -> None:
        """Delete an archived key and every workspace's override of it."""
        organization_id = await self._managed_organization(user)
        async with self.uow:
            key = await self._key(key_id, organization_id)
            if key.archived_at is None:
                raise WebSearchKeyNotArchivedError(key_id)
            await self.keys.delete_key(key)

    async def set_default(self, *, user: User, key_id: uuid.UUID) -> OrgWebSearchKeyPublic:
        """Make a key its provider's organization default."""
        organization_id = await self._managed_organization(user)
        provider = ""
        try:
            async with self.uow:
                key = await self._live_key(key_id, organization_id)
                provider = key.provider
                key = await self.keys.set_org_default(key)
        except WebSearchKeyConflict:
            raise WebSearchKeyDefaultConflictError(provider) from None
        return _public(key)

    # ------------------------------------------------------------------
    # One workspace's choice among its organization's keys
    # ------------------------------------------------------------------

    async def list_workspace_keys(self, *, user: User, workspace_id: uuid.UUID) -> WorkspaceWebSearchKeysPublic:
        """Every live key as the workspace sees it, and which one it searches with. Any member may read it."""
        workspace = await self.workspaces.resolve_visible_workspace(user=user, workspace_id=workspace_id)
        async with self.uow:
            candidates = await self.overrides.candidates(
                organization_id=workspace.organization_id, workspace_id=workspace.id
            )
        usable = {key.id: search_key_is_usable(key) for key, _ in candidates}
        effective = resolve_web_search_key(candidates, lambda key: usable[key.id])
        return WorkspaceWebSearchKeysPublic(
            data=[
                WorkspaceWebSearchKeyPublic(
                    org_web_search_key_id=key.id,
                    workspace_id=workspace.id,
                    provider=key.provider,
                    name=key.name,
                    last4=key.last4,
                    is_default=override.is_default if override else False,
                    disabled=override.disabled if override else False,
                    is_effective=effective is not None and key.id == effective.id,
                    usable=usable[key.id],
                )
                for key, override in candidates
            ]
        )

    async def set_workspace_override(
        self,
        *,
        user: User,
        workspace_id: uuid.UUID,
        key_id: uuid.UUID,
        request: WorkspaceWebSearchKeyOverrideRequest,
    ) -> WorkspaceWebSearchKeysPublic:
        """Pin a key as the workspace's own, or turn it off for the workspace.

        Tri-state: an omitted flag keeps its value. Pinning re-enables the key and unpins
        any other, turning it off unpins it, and both flags false removes the override.
        """
        if request.is_default and request.disabled:
            raise WorkspaceWebSearchKeyOverrideConflictError()
        workspace = await self.workspaces.resolve_visible_workspace(user=user, workspace_id=workspace_id)
        await self.workspaces.require_workspace_management_access(user=user, workspace=workspace)
        async with self.uow:
            key = await self._live_key(key_id, workspace.organization_id)
            await self.lock_workspace(workspace.id)
            existing = await self.overrides.get(workspace_id=workspace.id, key_id=key.id)
            is_default = existing.is_default if existing else False
            disabled = existing.disabled if existing else False
            if request.is_default is not None:
                is_default = request.is_default
                disabled = disabled and not is_default
            if request.disabled is not None:
                disabled = request.disabled
                is_default = is_default and not disabled
            if is_default:
                await self.overrides.clear_other_pins(workspace_id=workspace.id, except_key_id=key.id)
            if is_default or disabled:
                await self.overrides.save(
                    existing,
                    workspace_id=workspace.id,
                    organization_id=workspace.organization_id,
                    key_id=key.id,
                    is_default=is_default,
                    disabled=disabled,
                )
            elif existing is not None:
                await self.overrides.delete(existing)
        return await self.list_workspace_keys(user=user, workspace_id=workspace_id)

    async def reset_workspace_override(
        self, *, user: User, workspace_id: uuid.UUID, key_id: uuid.UUID
    ) -> WorkspaceWebSearchKeysPublic:
        """Return the workspace to inheriting the key. Idempotent."""
        workspace = await self.workspaces.resolve_visible_workspace(user=user, workspace_id=workspace_id)
        await self.workspaces.require_workspace_management_access(user=user, workspace=workspace)
        async with self.uow:
            key = await self._key(key_id, workspace.organization_id)
            existing = await self.overrides.get(workspace_id=workspace.id, key_id=key.id)
            if existing is not None:
                await self.overrides.delete(existing)
        return await self.list_workspace_keys(user=user, workspace_id=workspace_id)

    # ------------------------------------------------------------------

    async def _managed_organization(self, user: User) -> uuid.UUID:
        organization = await self.organizations.get_active_organization_for_user(user)
        await self.organizations.require_active_organization_management_access(user=user, organization=organization)
        return organization.id

    async def _key(self, key_id: uuid.UUID, organization_id: uuid.UUID) -> OrgWebSearchKey:
        key = await self.keys.get_in_organization(key_id, organization_id)
        if key is None:
            raise WebSearchKeyNotFoundError(key_id)
        return key

    async def _live_key(self, key_id: uuid.UUID, organization_id: uuid.UUID) -> OrgWebSearchKey:
        key = await self._key(key_id, organization_id)
        if key.archived_at is not None:
            raise WebSearchKeyArchivedError(key_id)
        return key


def _public(key: OrgWebSearchKey) -> OrgWebSearchKeyPublic:
    return OrgWebSearchKeyPublic.from_row(key, usable=search_key_is_usable(key))


def _validated_provider(provider: str) -> str:
    candidate = provider.strip().lower()
    if candidate not in WEB_SEARCH_PROVIDERS:
        raise WebSearchKeyUnknownProviderError(provider, WEB_SEARCH_PROVIDERS)
    return candidate


def _validated_name(name: str) -> str:
    stripped = name.strip()
    if not stripped:
        raise WebSearchKeyNameRequiredError()
    return stripped


def _encrypted(api_key: str) -> tuple[str, str]:
    """Encrypt a plaintext key for storage, with the last four characters an operator tells keys apart by."""
    try:
        return encrypt_secret(api_key), api_key[-4:]
    except SecretBoxUnavailableError:
        raise SecretBoxUnavailableTenancyError from None
