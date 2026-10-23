"""An organization's own web search and fetch keys, and which one each of its workspaces uses.

A workspace whose organization holds a usable key of a kind uses it for that kind's calls,
so the organization's own account pays and its own quota applies: a search key serves the
search tool, a fetch key the fetch tool and the enrichment of search hits. Any other
workspace uses the deployment's instance, which these keys never change: the two never merge.

Organization owners and admins manage the keys. A workspace's owners and admins pin one
key of each kind as the workspace's own or turn a key off for it, and any member can see
which keys the workspace uses. :func:`resolve_web_search_key` decides each.
"""

import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

import any_fetch
import any_search
from gateway.core.settings.tools import (
    WEB_SEARCH_PROVIDERS,
    ToolKind,
    ToolSettings,
    is_test_tier,
    option_problems,
    supported_fetch_providers,
)
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.shared_exceptions import SecretBoxUnavailableTenancyError
from gateway.exceptions.tools_exceptions import (
    WebSearchKeyAlreadyExistsError,
    WebSearchKeyArchivedError,
    WebSearchKeyCredentialRequiredError,
    WebSearchKeyDefaultConflictError,
    WebSearchKeyIncompleteError,
    WebSearchKeyMalformedError,
    WebSearchKeyNameRequiredError,
    WebSearchKeyNotArchivedError,
    WebSearchKeyNotFoundError,
    WebSearchKeySettingRefusedError,
    WebSearchKeyUnknownProviderError,
    WorkspaceWebSearchKeyOverrideConflictError,
)
from gateway.log_config import logger
from gateway.models.secret_fields import carries_redaction, holds_secret_like_values, restore_redacted_values
from gateway.models.tenancy import User
from gateway.models.tools import (
    OrgWebSearchKey,
    WebFetchCredential,
    WebSearchCredential,
    WebToolCredential,
    WebToolCredentialSource,
)
from gateway.repositories.tenancy import WorkspaceRepository
from gateway.repositories.tools import (
    OrgWebSearchKeyRepository,
    SearchKeyCandidate,
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
from gateway.services.tool_settings_service import validate_url
from gateway.services.tools._web_search import in_loop_default, resolve_fetch
from gateway.services.web_retrieval_network import RetrievalTargetError, validate_retrieval_target
from gateway.services.web_search_providers import is_sendable_api_key

# Takes the workspace's row lock for the rest of the transaction.
LockWorkspace = Callable[[uuid.UUID], Awaitable[None]]


def key_credential(key: OrgWebSearchKey) -> WebToolCredential | None:
    """The key as a credential of its kind, or ``None`` where this deployment cannot decrypt it."""
    try:
        api_key = decrypt_secret(key.encrypted_api_key) if key.encrypted_api_key is not None else None
    except (SecretBoxUnavailableError, SecretDecryptionError):
        logger.warning("Web search key %s cannot be decrypted on this deployment", key.id)
        return None
    credential = WebFetchCredential if key.kind == "fetch" else WebSearchCredential
    return credential(
        provider=key.provider,
        source=WebToolCredentialSource.ORG,
        api_key=api_key,
        api_base=key.api_base,
        options=MappingProxyType(dict(key.options or {})),
    )


class _Decrypted:
    """Each key decrypted at most once, and only when asked."""

    def __init__(self) -> None:
        self._credentials: dict[uuid.UUID, WebToolCredential | None] = {}

    def credential(self, key: OrgWebSearchKey) -> WebToolCredential | None:
        if key.id not in self._credentials:
            self._credentials[key.id] = key_credential(key)
        return self._credentials[key.id]

    def usable(self, key: OrgWebSearchKey) -> bool:
        return self.credential(key) is not None


@dataclass(frozen=True)
class WorkspaceKeys:
    """The keys a workspace uses, as credentials: ``None`` for a kind it uses the deployment's instance for."""

    search: WebSearchCredential | None = None
    fetch: WebFetchCredential | None = None


class WorkspaceSearchKeys:
    """The keys each workspace uses, read by every call it makes: the in-loop tools and direct search."""

    def __init__(self, *, workspaces: WorkspaceRepository, overrides: WorkspaceWebSearchKeyOverrideRepository) -> None:
        self.workspaces = workspaces
        self.overrides = overrides

    async def keys_for(self, workspace_id: uuid.UUID, *, search: bool, fetch: bool) -> WorkspaceKeys:
        """The workspace's search key where ``search``, and its fetch key where ``fetch``.

        One read covers both kinds, and only the keys the choice reaches are decrypted.
        """
        wanted: dict[ToolKind, bool] = {"search": search, "fetch": fetch}
        kinds = [kind for kind, asked in wanted.items() if asked]
        workspace = await self.workspaces.get(workspace_id) if kinds else None
        if workspace is None:
            return WorkspaceKeys()
        decrypted = _Decrypted()
        chosen = await _choose_keys(self.overrides, workspace.organization_id, workspace.id, kinds, decrypted)
        credentials = {kind: decrypted.credential(key) for kind, key in chosen.items() if key is not None}
        search_key, fetch_key = credentials.get("search"), credentials.get("fetch")
        return WorkspaceKeys(
            search=search_key if isinstance(search_key, WebSearchCredential) else None,
            fetch=fetch_key if isinstance(fetch_key, WebFetchCredential) else None,
        )

    async def has_search_key(self, workspace_id: uuid.UUID) -> bool:
        """Whether the workspace searches with a key of its organization's."""
        return (await self.keys_for(workspace_id, search=True, fetch=False)).search is not None


# How many of the oldest keys are read at a time when no pin or default is usable.
_OLDEST_BATCH = 20


async def _choose_keys(
    overrides: WorkspaceWebSearchKeyOverrideRepository,
    organization_id: uuid.UUID,
    workspace_id: uuid.UUID,
    kinds: Sequence[ToolKind],
    decrypted: _Decrypted,
) -> dict[ToolKind, OrgWebSearchKey | None]:
    """The key of each of ``kinds`` :func:`resolve_web_search_key` picks, reading only the rows it can reach.

    The pins and the defaults are read first, in one read for every kind. Only for a kind none
    of them serves are the oldest keys read, a batch at a time, so a large organization costs
    one small read on the common path.
    """
    preferred = await overrides.preferred_candidates(
        organization_id=organization_id, workspace_id=workspace_id, kind=kinds[0] if len(kinds) == 1 else None
    )
    chosen = {kind: resolve_web_search_key(_of_kind(preferred, kind), decrypted.usable) for kind in kinds}
    pending = [kind for kind in kinds if chosen[kind] is None]
    # One filter for every page, so the offsets stay those of one ordering.
    page_kind = pending[0] if len(pending) == 1 else None
    skip = 0
    while pending:
        batch = await overrides.candidate_page(
            organization_id=organization_id,
            workspace_id=workspace_id,
            skip=skip,
            limit=_OLDEST_BATCH,
            kind=page_kind,
            enabled_only=True,
        )
        for kind in list(pending):
            if (found := resolve_web_search_key(_of_kind(batch, kind), decrypted.usable)) is not None:
                chosen[kind] = found
                pending.remove(kind)
        if len(batch) < _OLDEST_BATCH:
            break
        skip += _OLDEST_BATCH
    return chosen


def _of_kind(candidates: Sequence[SearchKeyCandidate], kind: ToolKind) -> list[SearchKeyCandidate]:
    return [(key, override) for key, override in candidates if _kind(key) == kind]


class WebSearchKeyService:
    """Manage an organization's web search and fetch keys and its workspaces' choice among them."""

    def __init__(
        self,
        uow: UnitOfWork,
        *,
        keys: OrgWebSearchKeyRepository,
        overrides: WorkspaceWebSearchKeyOverrideRepository,
        organizations: OrganizationService,
        workspaces: WorkspaceAccess,
        lock_workspace: LockWorkspace,
        config: ToolSettings,
    ) -> None:
        """Bind the unit of work, the repositories, the tenancy checks this service asks, and the config.

        ``lock_workspace`` takes the workspace's row lock, which serializes two admins
        pinning different keys for one workspace; it is a callable because it needs the session.
        ``config`` says what a workspace without a key of a kind runs on.
        """
        self.uow = uow
        self.keys = keys
        self.overrides = overrides
        self.organizations = organizations
        self.workspaces = workspaces
        self.lock_workspace = lock_workspace
        self.config = config

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
        kind = request.kind
        provider = _validated_provider(kind, request.provider)
        name = _validated_name(request.name)
        encrypted, last4 = _encrypted(request.api_key) if request.api_key is not None else (None, None)
        api_base = _blank_to_none(request.api_base)
        options = request.options or None
        await _check_settings(kind, provider, has_key=encrypted is not None, api_base=api_base, options=options)
        try:
            async with self.uow:
                row = await self.keys.create_key(
                    organization_id=organization_id,
                    kind=kind,
                    provider=provider,
                    name=name,
                    encrypted_api_key=encrypted,
                    last4=last4,
                    api_base=api_base,
                    options=options,
                )
        except WebSearchKeyConflict:
            raise WebSearchKeyAlreadyExistsError(provider, name, kind=kind) from None
        return _public(row)

    async def update_key(
        self, *, user: User, key_id: uuid.UUID, request: OrgWebSearchKeyUpdateRequest
    ) -> OrgWebSearchKeyPublic:
        """Change what the request sets. Only that is checked, so renaming a key or replacing its secret
        never trips on an option a library upgrade left behind.
        """
        organization_id = await self._managed_organization(user)
        values: dict[str, object] = {}
        if request.name is not None:
            values["name"] = _validated_name(request.name)
        if request.api_key is not None:
            values["encrypted_api_key"], values["last4"] = _encrypted(request.api_key)
        sets_api_base = "api_base" in request.model_fields_set
        api_base = _blank_to_none(request.api_base)
        if sets_api_base and api_base is not None:
            # Before the transaction, since the gate may resolve the host.
            await _gate_api_base(api_base)
        provider, kind = "", "search"
        try:
            async with self.uow:
                key = await self._live_key(key_id, organization_id)
                provider, kind = key.provider, _kind(key)
                adds_a_key = request.api_key is not None and key.encrypted_api_key is None
                if sets_api_base and api_base is not None and api_base != key.api_base:
                    # The stored credential would go to the new host on the next call.
                    stored_options_follow = (
                        carries_redaction(request.options)
                        if "options" in request.model_fields_set
                        else holds_secret_like_values(key.options)
                    )
                    if (key.encrypted_api_key is not None and request.api_key is None) or stored_options_follow:
                        raise WebSearchKeyCredentialRequiredError
                if sets_api_base or adds_a_key:
                    _check_endpoint(
                        kind,
                        provider,
                        api_base if sets_api_base else key.api_base,
                        has_key=request.api_key is not None or key.encrypted_api_key is not None,
                    )
                if sets_api_base:
                    values["api_base"] = api_base
                if "options" in request.model_fields_set:
                    options = restore_redacted_values(request.options, key.options) or None
                    _check_options(kind, provider, options)
                    values["options"] = options
                name = values.get("name")
                if (
                    name is not None
                    and name != key.name
                    and await self.keys.get_by_name(
                        organization_id=organization_id, kind=kind, provider=provider, name=str(name)
                    )
                ):
                    raise WebSearchKeyAlreadyExistsError(provider, str(name), kind=kind)
                row = await self.keys.update_key(key, values) if values else key
        except WebSearchKeyConflict:
            # A rollback expires the loaded row, so the error is built from what was read before it.
            raise WebSearchKeyAlreadyExistsError(provider, str(values["name"]), kind=kind) from None
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
            if key.archived_at is not None:
                # Not a default again: another key may have become the default while it was archived.
                key = await self.keys.update_key(key, {"archived_at": None, "is_org_default": False})
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
        """Make a key its kind's organization default, in place of the previous one."""
        organization_id = await self._managed_organization(user)
        kind = "search"
        try:
            async with self.uow:
                key = await self._live_key(key_id, organization_id)
                kind = key.kind
                key = await self.keys.set_org_default(key)
        except WebSearchKeyConflict:
            raise WebSearchKeyDefaultConflictError(kind) from None
        return _public(key)

    # ------------------------------------------------------------------
    # One workspace's choice among its organization's keys
    # ------------------------------------------------------------------

    async def list_workspace_keys(
        self, *, user: User, workspace_id: uuid.UUID, skip: int = 0, limit: int = 100
    ) -> WorkspaceWebSearchKeysPublic:
        """A page of the live keys as the workspace sees them, and which of each kind it uses. Any member may read it.

        The effective keys are chosen across every live key, not the page, so paging never changes them,
        and the providers say what the workspace's searches and fetches run on.
        """
        workspace = await self.workspaces.resolve_visible_workspace(user=user, workspace_id=workspace_id)
        decrypted = _Decrypted()
        async with self.uow:
            page = await self.overrides.candidate_page(
                organization_id=workspace.organization_id, workspace_id=workspace.id, skip=skip, limit=limit
            )
            count = await self.overrides.count_candidates(organization_id=workspace.organization_id)
            chosen = await _choose_keys(
                self.overrides, workspace.organization_id, workspace.id, ("search", "fetch"), decrypted
            )
        search, fetch = chosen["search"], chosen["fetch"]
        effective = {key.id for key in (search, fetch) if key is not None}
        return WorkspaceWebSearchKeysPublic(
            data=[
                WorkspaceWebSearchKeyPublic(
                    org_web_search_key_id=key.id,
                    workspace_id=workspace.id,
                    kind=_kind(key),
                    provider=key.provider,
                    name=key.name,
                    last4=key.last4,
                    is_default=override.is_default if override else False,
                    disabled=override.disabled if override else False,
                    is_effective=key.id in effective,
                    usable=decrypted.usable(key),
                )
                for key, override in page
            ],
            count=count,
            search_provider=search.provider if search is not None else self._deployment_search_provider(),
            fetch_provider=fetch.provider if fetch is not None else self._deployment_fetch_provider(),
        )

    async def set_workspace_override(
        self,
        *,
        user: User,
        workspace_id: uuid.UUID,
        key_id: uuid.UUID,
        request: WorkspaceWebSearchKeyOverrideRequest,
    ) -> WorkspaceWebSearchKeysPublic:
        """Pin a key as the workspace's own for its kind, or turn it off for the workspace.

        Tri-state: an omitted flag keeps its value. Pinning re-enables the key and unpins
        the workspace's other key of its kind, turning it off unpins it, and both flags
        false removes the override.
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
                await self.overrides.clear_other_pins(workspace_id=workspace.id, except_key_id=key.id, kind=key.kind)
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

    def _deployment_search_provider(self) -> str | None:
        """The provider the deployment's in-loop default searches on, or ``None`` where it has none."""
        default = in_loop_default(self.config)
        return default.provider if default is not None else None

    def _deployment_fetch_provider(self) -> str:
        """The provider the deployment's default fetch instance runs on, ``builtin`` for the built-in fetcher."""
        return resolve_fetch(self.config, None, requested=None, for_search=None).provider

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
    return OrgWebSearchKeyPublic.from_row(key, usable=key_credential(key) is not None)


def _kind(key: OrgWebSearchKey) -> ToolKind:
    return "fetch" if key.kind == "fetch" else "search"


def _key_providers(kind: ToolKind) -> tuple[str, ...]:
    """The providers a key of ``kind`` may name, sorted.

    The libraries' providers for the kind, other than the ones that exist only for
    tests and, for fetch, ``builtin``; and, for search, ``tavily`` and ``brave``,
    which run on the old search code.
    """
    if kind == "fetch":
        served: tuple[str, ...] = supported_fetch_providers()
    else:
        served = (*any_search.AnySearch.get_supported_providers(), *WEB_SEARCH_PROVIDERS)
    offered = {provider for provider in served if provider in WEB_SEARCH_PROVIDERS or not is_test_tier(kind, provider)}
    return tuple(sorted(offered))


def _validated_provider(kind: ToolKind, provider: str) -> str:
    candidate = provider.strip().lower()
    known = _key_providers(kind)
    if candidate not in known:
        raise WebSearchKeyUnknownProviderError(provider, known, kind=kind)
    return candidate


def _requirements(kind: ToolKind, provider: str) -> tuple[bool, bool, bool]:
    """Whether the old search code runs the provider, and whether a key on it needs an API key and an endpoint.

    The old search code calls Tavily and Brave at their public endpoints, with no options of a key's.
    """
    if (
        kind == "search"
        and provider in WEB_SEARCH_PROVIDERS
        and provider not in any_search.AnySearch.get_supported_providers()
    ):
        return True, True, False
    metadata = (
        any_fetch.AnyFetch.get_provider_metadata(provider)
        if kind == "fetch"
        else any_search.AnySearch.get_provider_metadata(provider)
    )
    return False, metadata.requires_api_key, metadata.requires_api_base and not metadata.default_api_base


async def _check_settings(
    kind: ToolKind, provider: str, *, has_key: bool, api_base: str | None, options: Mapping[str, Any] | None
) -> None:
    """Refuse a new key a call could not be made with as written.

    Raises:
        WebSearchKeyIncompleteError: the provider needs a key or an endpoint the key does not carry.
        WebSearchKeySettingRefusedError: the endpoint or an option is one the provider or this deployment refuses.
    """
    _, requires_key, _ = _requirements(kind, provider)
    if requires_key and not has_key:
        raise WebSearchKeyIncompleteError(provider, "an API key")
    _check_endpoint(kind, provider, api_base, has_key=has_key)
    _check_options(kind, provider, options)
    if api_base is not None:
        await _gate_api_base(api_base)


def _check_endpoint(kind: ToolKind, provider: str, api_base: str | None, *, has_key: bool) -> None:
    """Refuse an endpoint the provider takes none of, that is no web address, or that would send a key in the clear."""
    on_old_code, _, requires_base = _requirements(kind, provider)
    if api_base is None:
        if requires_base:
            raise WebSearchKeyIncompleteError(provider, "an api_base")
        return
    if on_old_code:
        raise WebSearchKeySettingRefusedError("api_base", f"a {provider} key takes neither api_base nor options")
    try:
        validate_url(api_base)
    except ValueError as exc:
        raise WebSearchKeySettingRefusedError("api_base", str(exc)) from None
    if has_key and urlsplit(api_base).scheme != "https":
        raise WebSearchKeySettingRefusedError("api_base", "must use https when the key carries an API key")


def _check_options(kind: ToolKind, provider: str, options: Mapping[str, Any] | None) -> None:
    """Refuse options the provider's schema does not know or refuses the value of."""
    if not options:
        return
    if _requirements(kind, provider)[0]:
        raise WebSearchKeySettingRefusedError("options", f"a {provider} key takes neither api_base nor options")
    if problems := option_problems(kind, provider, options):
        raise WebSearchKeySettingRefusedError("options", next(iter(problems.values())))


async def _gate_api_base(api_base: str) -> None:
    """Refuse an endpoint that does not reach public addresses alone.

    A tenant chose it, so ``provider_allow_private_hosts`` does not apply; calls to it are pinned the same way.
    """
    try:
        await validate_retrieval_target(api_base)
    except RetrievalTargetError as exc:
        raise WebSearchKeySettingRefusedError("api_base", f"must reach a public address: {exc}") from None


def _blank_to_none(value: str | None) -> str | None:
    stripped = (value or "").strip()
    return stripped or None


def _validated_name(name: str) -> str:
    stripped = name.strip()
    if not stripped:
        raise WebSearchKeyNameRequiredError()
    return stripped


def _encrypted(api_key: str) -> tuple[str, str]:
    """Encrypt a plaintext key for storage, with the last four characters an operator tells keys apart by.

    Surrounding whitespace from a paste is dropped; whitespace or a control character inside is refused.
    """
    api_key = api_key.strip()
    if not is_sendable_api_key(api_key):
        raise WebSearchKeyMalformedError()
    try:
        return encrypt_secret(api_key), api_key[-4:]
    except SecretBoxUnavailableError:
        raise SecretBoxUnavailableTenancyError from None
