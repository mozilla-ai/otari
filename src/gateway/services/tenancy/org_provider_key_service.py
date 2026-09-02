"""Organization-scoped provider keys: CRUD, defaults, overrides, and the dispatch overlay.

Decided at otari-ai#1748: `org_provider_keys` and its two override
tables are the source of truth for organization-scoped BYO provider
credentials (see `gateway.models.provider_keys`). This module is kept
separate from `services/provider_store_service.py` on purpose, so that
module's config.yml-merge overlay for instance-keyed, deployment-global
credentials stays completely untouched: the two mechanisms are disjoint by
construction (see `services/provider_kwargs.py`), not layered on top of each
other.

**Authorization and CRUD** follow `organization_service.py`/
`workspace_service.py`'s conventions exactly: a `*_for_user` method resolves
the caller's organization or workspace and its own management-role check
before touching a row, and every route stays thin. Workspace resolution and
the workspace-management check both go through `services.tenancy.authorization`
directly (`resolve_visible_workspace`, `require_workspace_management_access`)
rather than through `WorkspaceService`, the same shared entry point
`WorkspaceBudgetDefaultService` uses and `WorkspaceService` itself delegates
to internally. Reading is open to any active member; every mutation on the
organization surface needs an organization owner/admin, and every mutation on
a workspace's override or model restrictions needs an organization
owner/admin *or* an owner/admin of that workspace.

**The dispatch overlay** mirrors `provider_store_service.py`'s shape one for
one: a module-global, process-wide cache refreshed on a TTL and immediately
on write, because the completion dispatch path is synchronous and holds no
database session (see `services/provider_kwargs.py`). The one real
difference is the cache's key: `provider_store_service` keys by instance name
alone, because a deployment has one credential per instance; this keys by
`(workspace_id, provider)`, because a workspace resolves to a *different*
key depending on that workspace's own pin, its organization's default, and
its organization's oldest key, in that order (`resolve_active_key`). One
global refresh reloads every organization in one pass rather than one query
per organization, per otari-ai#1748's second sub-question: the expected row
count (provider keys across a whole self-hosted deployment) stays small
enough that sharding the refresh by organization would add complexity for no
benefit.
"""

import asyncio
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from any_llm import LLMProvider
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.core.config import PROVIDER_TYPE_ALIASES
from gateway.core.database import create_session
from gateway.log_config import logger
from gateway.models.provider_keys import (
    OrgProviderKey,
    OrgProviderKeyCreateRequest,
    OrgProviderKeyPublic,
    OrgProviderKeysPublic,
    OrgProviderKeyUpdateRequest,
    WorkspaceProviderKeyOverride,
    WorkspaceProviderKeyOverridePublic,
    WorkspaceProviderKeyOverrideRequest,
    WorkspaceProviderKeyOverridesPublic,
    WorkspaceProviderModelRestriction,
    WorkspaceProviderModelRestrictionsPublic,
)
from gateway.models.secret_fields import restore_redacted_values
from gateway.models.tenancy import User, Workspace
from gateway.repositories.tenancy import (
    Candidate,
    OrgProviderKeyRepository,
    WorkspaceProviderKeyOverrideRepository,
    WorkspaceProviderModelRestrictionRepository,
    WorkspaceRepository,
    resolve_active_key,
)
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
)
from gateway.services.tenancy import authorization
from gateway.services.tenancy.errors import (
    OrgDefaultProviderKeyConflictError,
    OrgProviderKeyAlreadyExistsError,
    OrgProviderKeyArchivedError,
    OrgProviderKeyDisabledForWorkspaceError,
    OrgProviderKeyNameRequiredError,
    OrgProviderKeyNotArchivedError,
    OrgProviderKeyNotFoundError,
    OrgProviderKeyUnknownProviderError,
    OrgProviderKeyUnsafeApiBaseError,
    SecretBoxUnavailableTenancyError,
    WorkspaceProviderKeyOverrideConflictError,
)
from gateway.services.tenancy.organization_service import OrganizationService
from gateway.services.url_safety import UnsafeURLError, validate_provider_api_base

# Same value as provider_store_service.PROVIDER_CACHE_TTL_SECONDS, defined
# separately: the two overlays are independent mechanisms (see module
# docstring) and sharing a constant would imply a coupling that does not
# exist.
ORG_PROVIDER_CACHE_TTL_SECONDS = 30.0

# (workspace_id, provider) -> decrypted overlay entry, shaped like a
# config.providers value (api_key, api_base, client_args).
_org_cache: dict[tuple[uuid.UUID, str], dict[str, Any]] = {}
# (workspace_id, provider) -> the active key's model allow-list for that
# workspace. Absent = unrestricted (either no restriction rows exist, or no
# key resolved at all); present-and-empty is unreachable (an empty
# restriction set is stored as "no rows", not a row-less deny-all).
_org_model_restrictions: dict[tuple[uuid.UUID, str], list[str]] = {}
_org_cached_at: float | None = None


def _row_to_entry(key: OrgProviderKey) -> dict[str, Any]:
    """Build a config.providers-shaped overlay entry from a stored key.

    Raises ``SecretBoxUnavailableError`` / ``SecretDecryptionError`` when the
    key cannot be decrypted; the caller decides whether to skip it.
    """
    entry: dict[str, Any] = {}
    if key.api_base:
        entry["api_base"] = key.api_base
    if key.client_args:
        entry["client_args"] = dict(key.client_args)
    if key.encrypted_api_key:
        entry["api_key"] = decrypt_secret(key.encrypted_api_key)
    return entry


def key_is_usable(key: OrgProviderKey) -> bool:
    """Whether this key can actually supply a credential on this deployment.

    A row whose secret will not decrypt (no ``OTARI_SECRET_KEY``, or the wrong
    one) is skipped by the dispatch-path cache below, so anything that reports
    what a tenant may *reach* has to skip it too, or the catalog advertises
    models every request through that provider then fails on. Single-sourced
    here so the two answers cannot drift.
    """
    try:
        _row_to_entry(key)
    except (SecretBoxUnavailableError, SecretDecryptionError):
        return False
    return True


def cached_org_provider_kwargs(workspace_id: uuid.UUID, provider: str) -> dict[str, Any] | None:
    """The decrypted overlay entry this worker last loaded for this workspace+provider, if any."""
    entry = _org_cache.get((workspace_id, provider))
    return dict(entry) if entry is not None else None


def cached_org_model_restriction(workspace_id: uuid.UUID, provider: str) -> list[str] | None:
    """The active key's model allow-list for this workspace+provider, or ``None`` if unrestricted.

    Same convention `services.model_access.is_model_allowed` already uses for
    a caller's own allow-list: ``None`` means every model is permitted, a
    (non-empty) list narrows it. A workspace that has never restricted this
    key's models, or for which no organization-scoped key resolved at all,
    reads as ``None`` either way; the caller only reaches this when it
    already knows a key resolved (see `provider_kwargs.cached_org_provider_kwargs`).

    This is an access-control input, not only a credential cache: tightening
    an allow-list takes up to `ORG_PROVIDER_CACHE_TTL_SECONDS` to reach a
    sibling worker or replica, which keeps serving the wider list until its
    own refresh runs. The credential (`cached_org_provider_kwargs`) and the
    allow-list are also two separate reads on the request path with no shared
    snapshot between them, so a refresh landing in between can pair one key's
    credential with a different key's (now-current) list for one request.
    Neither is unbounded (the TTL caps the staleness window, and the pairing
    mismatch is one request, not a standing state), but a caller relying on a
    just-narrowed allow-list to take effect everywhere, immediately, is
    relying on a guarantee this cache does not make.
    """
    return _org_model_restrictions.get((workspace_id, provider))


def org_cache_is_stale(ttl: float = ORG_PROVIDER_CACHE_TTL_SECONDS) -> bool:
    """Whether the cache has never been loaded or has outlived ``ttl``."""
    return _org_cached_at is None or (time.monotonic() - _org_cached_at) >= ttl


def reset_org_provider_cache() -> None:
    """Drop the overlay cache so the next load starts clean (startup, tests)."""
    global _org_cached_at  # noqa: PLW0603

    _org_cache.clear()
    _org_model_restrictions.clear()
    _org_cached_at = None


async def refresh_org_provider_cache(db: AsyncSession) -> None:
    """Reload every organization's effective key per (workspace, provider) in one pass.

    Four queries total, independent of how many organizations or workspaces
    exist: every workspace's organization, every non-archived key, every
    override, and every model restriction. The precedence tiers
    (`resolve_active_key`) are then applied in Python once per (workspace,
    provider) pair that actually has a key, not once per workspace times
    every provider that exists anywhere.
    """
    global _org_cached_at  # noqa: PLW0603

    workspace_orgs = (await db.execute(select(col(Workspace.id), col(Workspace.organization_id)))).all()
    keys = (
        (
            await db.execute(
                select(OrgProviderKey)
                .where(col(OrgProviderKey.archived_at).is_(None))
                .order_by(col(OrgProviderKey.created_at), col(OrgProviderKey.id))
            )
        )
        .scalars()
        .all()
    )
    overrides = (await db.execute(select(WorkspaceProviderKeyOverride))).scalars().all()
    restrictions = (await db.execute(select(WorkspaceProviderModelRestriction))).scalars().all()

    keys_by_org_provider: dict[tuple[uuid.UUID, str], list[OrgProviderKey]] = defaultdict(list)
    providers_by_org: dict[uuid.UUID, set[str]] = defaultdict(set)
    for key in keys:
        keys_by_org_provider[(key.organization_id, key.provider)].append(key)
        providers_by_org[key.organization_id].add(key.provider)

    override_by_workspace_key = {(o.workspace_id, o.org_provider_key_id): o for o in overrides}

    models_by_workspace_key: dict[tuple[uuid.UUID, uuid.UUID], list[str]] = defaultdict(list)
    for restriction in restrictions:
        models_by_workspace_key[(restriction.workspace_id, restriction.org_provider_key_id)].append(restriction.model)

    new_cache: dict[tuple[uuid.UUID, str], dict[str, Any]] = {}
    new_restrictions: dict[tuple[uuid.UUID, str], list[str]] = {}
    for workspace_id, organization_id in workspace_orgs:
        for provider in providers_by_org.get(organization_id, ()):
            candidates: list[Candidate] = [
                (key, override_by_workspace_key.get((workspace_id, key.id)))
                for key in keys_by_org_provider[(organization_id, provider)]
            ]
            active = resolve_active_key(candidates)
            if active is None:
                continue
            try:
                new_cache[(workspace_id, provider)] = _row_to_entry(active)
            except (SecretBoxUnavailableError, SecretDecryptionError):
                logger.warning(
                    "Skipping organization provider key '%s' (%s) for workspace %s: "
                    "its API key could not be decrypted (check OTARI_SECRET_KEY).",
                    active.name,
                    provider,
                    workspace_id,
                )
                continue
            allowed_models = models_by_workspace_key.get((workspace_id, active.id))
            if allowed_models:
                new_restrictions[(workspace_id, provider)] = allowed_models

    _org_cache.clear()
    _org_cache.update(new_cache)
    _org_model_restrictions.clear()
    _org_model_restrictions.update(new_restrictions)
    _org_cached_at = time.monotonic()


async def load_org_provider_keys_at_startup(db: AsyncSession) -> None:
    """Prime the overlay so the first request does not race the first refresh.

    A failure here is logged rather than raised, the same posture
    `provider_store_service.load_providers_at_startup` takes: organization
    provider keys are additive to config.yml providers, and a gateway that
    boots with none cached is better than one that refuses to start because a
    credential load failed.
    """
    reset_org_provider_cache()
    try:
        await refresh_org_provider_cache(db)
    except Exception:
        logger.exception("Failed to load organization provider keys; continuing with none cached")
        return
    if _org_cache:
        logger.info("Loaded %d organization-scoped provider key overlay entr(y/ies)", len(_org_cache))


async def run_org_provider_refresher(interval: float = ORG_PROVIDER_CACHE_TTL_SECONDS) -> None:
    """Reload the organization provider key overlay forever so other writers' changes arrive.

    A write refreshes the worker that served it; this covers sibling workers
    and other replicas, which converge within ``interval``. Every error is
    swallowed and retried on the next tick so a database blip cannot kill the
    refresher and freeze the overlay. Cancelled at shutdown.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            async with create_session() as db:
                await refresh_org_provider_cache(db)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Organization provider key refresh failed; retrying in %ss", interval, exc_info=True)


def _encrypt_api_key(api_key: str | None) -> tuple[str | None, str | None]:
    """Encrypt a plaintext key for storage, or clear it. Never logs the plaintext."""
    if not api_key:
        return None, None
    try:
        return encrypt_secret(api_key), api_key[-4:]
    except SecretBoxUnavailableError:
        raise SecretBoxUnavailableTenancyError from None


def _validated_provider(provider: str) -> str:
    """Normalize and validate a provider name against any-llm's known implementations.

    ``OrgProviderKey.provider`` is stored verbatim and is exactly the string
    ``refresh_org_provider_cache`` keys its cache on, matched at dispatch
    against a resolved selector's ``LLMProvider.value``. Left unguarded, a
    typo, unexpected casing, or an unaliased name (``"OpenAI"``,
    ``"azure-openai"``, trailing whitespace) is accepted with a 201 and then
    never resolves at dispatch, with no error at either point. Mirrors
    ``/v1/provider-credentials``'s ``_validate_instance`` provider_type guard,
    including ``PROVIDER_TYPE_ALIASES`` so an aliased name still resolves; the
    canonical value is what gets stored, not the alias, so the cache key
    always matches what dispatch resolves to.
    """
    trimmed = provider.strip()
    canonical = PROVIDER_TYPE_ALIASES.get(trimmed, trimmed)
    try:
        return LLMProvider(canonical).value
    except ValueError:
        raise OrgProviderKeyUnknownProviderError(trimmed) from None


def _validated_name(name: str | None) -> str:
    """Reject a null or blank key name, the same guard `WorkspaceService` applies to a workspace name.

    ``OrgProviderKeyUpdateRequest.name`` is nullable so a client can send an
    explicit ``null``; ``OrgProviderKey.name`` is NOT NULL. Left unguarded, an
    explicit ``null`` (or a whitespace-only string) reaches the database as an
    integrity error that the surrounding duplicate-name handling then reports
    as a 409 naming a key called "None", rather than the 400 this is.
    """
    trimmed = (name or "").strip()
    if not trimmed:
        raise OrgProviderKeyNameRequiredError
    return trimmed


async def _gate_api_base(api_base: str | None) -> None:
    """Reject storing an internal ``api_base`` when the SSRF gate is on.

    Mirrors `api/routes/providers.py`'s ``_gate_api_base`` for
    `/v1/provider-credentials`: a no-op in the default allow-all state, but
    when the operator sets ``OTARI_PROVIDER_ALLOW_PRIVATE_HOSTS=false`` a
    private/link-local/reserved ``api_base`` is refused here rather than
    persisted. Truthy check so an empty/absent api_base (the "use the SDK
    default endpoint" case) is not treated as a URL.
    """
    if not api_base:
        return
    try:
        await validate_provider_api_base(api_base)
    except UnsafeURLError as exc:
        raise OrgProviderKeyUnsafeApiBaseError(str(exc)) from None


class OrgProviderKeyService:
    """Business logic for the organization provider key surface."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.keys = OrgProviderKeyRepository(db)
        self.overrides = WorkspaceProviderKeyOverrideRepository(db)
        self.restrictions = WorkspaceProviderModelRestrictionRepository(db)
        self.organizations = OrganizationService(db)

    # ------------------------------------------------------------------
    # Organization-scoped keys
    # ------------------------------------------------------------------

    async def list_keys_for_user(
        self,
        *,
        user: User,
        include_archived: bool = False,
        skip: int = 0,
        limit: int = 100,
    ) -> OrgProviderKeysPublic:
        """List the caller's organization's keys. Organization owners and admins only.

        Gated like every write on this surface, not open to the wider
        membership: a row names the provider, the endpoint and the credential's
        last four, which the roles matrix keeps out of a plain member's sight
        (otari-ai#1944). One audience for the whole surface is also what lets
        `OrgProviderKey.to_public` serialize ``client_args`` for every caller.
        ``count`` is the total matching rows, not the page size, so a caller
        can page correctly (mirrors ``WorkspaceService.list_workspaces``).
        """
        organization = await self.organizations.get_active_organization_for_user(user)
        await self.organizations.require_active_organization_management_access(user=user, organization=organization)
        rows, count = await self.keys.list_for_organization(
            organization.id, include_archived=include_archived, skip=skip, limit=limit
        )
        return OrgProviderKeysPublic(data=[row.to_public() for row in rows], count=count)

    async def create_key_for_user(
        self,
        *,
        user: User,
        request: OrgProviderKeyCreateRequest,
    ) -> OrgProviderKeyPublic:
        """Create a key in the caller's organization. Organization owners and admins only."""
        organization = await self.organizations.get_active_organization_for_user(user)
        await self.organizations.require_active_organization_management_access(user=user, organization=organization)

        provider = _validated_provider(request.provider)
        name = _validated_name(request.name)
        await _gate_api_base(request.api_base)

        if (
            await self.keys.get_by_org_provider_name(organization_id=organization.id, provider=provider, name=name)
            is not None
        ):
            raise OrgProviderKeyAlreadyExistsError(provider, name)

        encrypted_api_key, last4 = _encrypt_api_key(request.api_key)
        try:
            key = await self.keys.create_key(
                organization_id=organization.id,
                provider=provider,
                name=name,
                encrypted_api_key=encrypted_api_key,
                last4=last4,
                api_base=request.api_base,
                client_args=request.client_args,
            )
            await self.db.commit()
        except IntegrityError:
            # The pre-check above races the insert; the unique constraint is
            # what actually decides, as elsewhere in this slice.
            await self.db.rollback()
            raise OrgProviderKeyAlreadyExistsError(provider, name) from None

        await refresh_org_provider_cache(self.db)
        return key.to_public()

    async def update_key_for_user(
        self,
        *,
        user: User,
        key_id: uuid.UUID,
        request: OrgProviderKeyUpdateRequest,
    ) -> OrgProviderKeyPublic:
        """Change a key's name, credential, base URL, or client args. Organization owners and admins only."""
        organization = await self.organizations.get_active_organization_for_user(user)
        await self.organizations.require_active_organization_management_access(user=user, organization=organization)

        key = await self.keys.get_in_organization(key_id, organization.id)
        if key is None:
            raise OrgProviderKeyNotFoundError(key_id)
        if key.archived_at is not None:
            raise OrgProviderKeyArchivedError(key_id)

        update_data = request.model_dump(exclude_unset=True)
        if "name" in update_data:
            new_name = _validated_name(update_data["name"])
            update_data["name"] = new_name
            if new_name != key.name:
                if (
                    await self.keys.get_by_org_provider_name(
                        organization_id=organization.id, provider=key.provider, name=new_name
                    )
                    is not None
                ):
                    raise OrgProviderKeyAlreadyExistsError(key.provider, new_name)
        if "api_base" in update_data:
            await _gate_api_base(update_data["api_base"])
        if "client_args" in update_data:
            # ``to_public`` masks a credential-shaped entry, so an editor
            # resubmitting the whole object sends the mask for the entries it was
            # never shown; those keep their stored value.
            update_data["client_args"] = restore_redacted_values(update_data["client_args"], key.client_args)
        if "api_key" in update_data:
            encrypted_api_key, last4 = _encrypt_api_key(update_data.pop("api_key"))
            update_data["encrypted_api_key"] = encrypted_api_key
            update_data["last4"] = last4

        try:
            updated = await self.keys.update_key(key, update_data)
            await self.db.commit()
        except IntegrityError:
            await self.db.rollback()
            raise OrgProviderKeyAlreadyExistsError(key.provider, str(update_data.get("name", key.name))) from None

        await refresh_org_provider_cache(self.db)
        return updated.to_public()

    async def archive_key_for_user(self, *, user: User, key_id: uuid.UUID) -> OrgProviderKeyPublic:
        """Archive a key. Organization owners and admins only.

        Clears ``is_org_default`` at the same time: an archived key that was
        the default would otherwise silently become the default again on
        restore, with no re-election having happened.
        """
        organization = await self.organizations.get_active_organization_for_user(user)
        await self.organizations.require_active_organization_management_access(user=user, organization=organization)

        key = await self.keys.get_in_organization(key_id, organization.id)
        if key is None:
            raise OrgProviderKeyNotFoundError(key_id)

        updated = await self.keys.update_key(key, {"archived_at": datetime.now(UTC), "is_org_default": False})
        await self.db.commit()
        await refresh_org_provider_cache(self.db)
        return updated.to_public()

    async def restore_key_for_user(self, *, user: User, key_id: uuid.UUID) -> OrgProviderKeyPublic:
        """Restore an archived key. Organization owners and admins only."""
        organization = await self.organizations.get_active_organization_for_user(user)
        await self.organizations.require_active_organization_management_access(user=user, organization=organization)

        key = await self.keys.get_in_organization(key_id, organization.id)
        if key is None:
            raise OrgProviderKeyNotFoundError(key_id)

        updated = await self.keys.update_key(key, {"archived_at": None})
        await self.db.commit()
        await refresh_org_provider_cache(self.db)
        return updated.to_public()

    async def delete_key_for_user(self, *, user: User, key_id: uuid.UUID) -> None:
        """Permanently delete an archived key. Organization owners and admins only.

        Overrides and model restrictions naming it ride the database cascade.
        """
        organization = await self.organizations.get_active_organization_for_user(user)
        await self.organizations.require_active_organization_management_access(user=user, organization=organization)

        key = await self.keys.get_in_organization(key_id, organization.id)
        if key is None:
            raise OrgProviderKeyNotFoundError(key_id)
        if key.archived_at is None:
            raise OrgProviderKeyNotArchivedError(key_id)

        await self.keys.delete_key(key)
        await self.db.commit()
        await refresh_org_provider_cache(self.db)

    async def set_org_default_for_user(self, *, user: User, key_id: uuid.UUID) -> OrgProviderKeyPublic:
        """Make a key the organization's default for its provider. Organization owners and admins only."""
        organization = await self.organizations.get_active_organization_for_user(user)
        await self.organizations.require_active_organization_management_access(user=user, organization=organization)

        key = await self.keys.get_in_organization(key_id, organization.id)
        if key is None:
            raise OrgProviderKeyNotFoundError(key_id)
        if key.archived_at is not None:
            raise OrgProviderKeyArchivedError(key_id)

        try:
            updated = await self.keys.set_org_default(key)
            await self.db.commit()
        except IntegrityError:
            await self.db.rollback()
            raise OrgDefaultProviderKeyConflictError(key.provider) from None

        await refresh_org_provider_cache(self.db)
        return updated.to_public()

    # ------------------------------------------------------------------
    # Workspace overrides
    # ------------------------------------------------------------------

    async def list_effective_keys_for_workspace(
        self,
        *,
        user: User,
        workspace_id: uuid.UUID,
    ) -> WorkspaceProviderKeyOverridesPublic:
        """The effective view of every key visible to a workspace. Any member of the workspace may read it."""
        workspace = await authorization.resolve_visible_workspace(
            self.db, user=user, workspace_id=workspace_id, organizations=self.organizations
        )
        candidates = await self.overrides.all_candidates(
            organization_id=workspace.organization_id, workspace_id=workspace.id
        )

        by_provider: dict[str, list[Candidate]] = defaultdict(list)
        for key, override in candidates:
            by_provider[key.provider].append((key, override))
        effective_ids = {
            active.id for group in by_provider.values() if (active := resolve_active_key(group)) is not None
        }

        return WorkspaceProviderKeyOverridesPublic(
            data=[
                WorkspaceProviderKeyOverridePublic(
                    workspace_id=workspace.id,
                    org_provider_key_id=key.id,
                    is_default=override.is_default if override else False,
                    disabled=override.disabled if override else False,
                    is_effective_default=key.id in effective_ids,
                    is_effective_enabled=not (override.disabled if override else False),
                )
                for key, override in candidates
            ]
        )

    async def set_workspace_override_for_user(
        self,
        *,
        user: User,
        workspace_id: uuid.UUID,
        key_id: uuid.UUID,
        request: WorkspaceProviderKeyOverrideRequest,
    ) -> WorkspaceProviderKeyOverridePublic:
        """Pin or disable a key for one workspace.

        Tri-state: an omitted field leaves that flag unchanged. Sending one
        flag lets the other auto-resolve when they would otherwise conflict
        (pinning re-enables a disabled key, disabling un-pins a pinned one);
        sending both explicitly true is refused. Both flags false, whether
        merged or explicit, deletes the override row rather than storing a
        no-op: absence of a row already means full inheritance.
        """
        workspace = await authorization.resolve_visible_workspace(
            self.db, user=user, workspace_id=workspace_id, organizations=self.organizations
        )
        await authorization.require_workspace_management_access(
            self.db, user=user, workspace=workspace, organizations=self.organizations
        )

        key = await self.keys.get_in_organization(key_id, workspace.organization_id)
        if key is None:
            raise OrgProviderKeyNotFoundError(key_id)

        # Serializes the whole read-decide-write sequence below: two admins
        # concurrently pinning two different keys for the same workspace and
        # provider both have to see each other's clear, since no unique index
        # spans the variable set of override rows a "pin" can land in (unlike
        # `set_org_default`, which is a single row the partial unique index
        # already arbitrates).
        await WorkspaceRepository(self.db).lock(workspace.id)

        existing = await self.overrides.get(workspace_id=workspace.id, org_provider_key_id=key.id)
        current_default = existing.is_default if existing else False
        current_disabled = existing.disabled if existing else False
        new_default = request.is_default if request.is_default is not None else current_default
        new_disabled = request.disabled if request.disabled is not None else current_disabled

        if new_default and new_disabled:
            if request.is_default and request.disabled is None:
                new_disabled = False
            elif request.disabled and request.is_default is None:
                new_default = False
            else:
                raise WorkspaceProviderKeyOverrideConflictError

        if new_default:
            await self.overrides.clear_workspace_pinned_default(
                workspace_id=workspace.id,
                organization_id=workspace.organization_id,
                provider=key.provider,
                except_key_id=key.id,
            )

        if not new_default and not new_disabled:
            if existing is not None:
                await self.overrides.delete(existing)
            result_default, result_disabled = False, False
        elif existing is not None:
            updated = await self.overrides.update(existing, is_default=new_default, disabled=new_disabled)
            result_default, result_disabled = updated.is_default, updated.disabled
        else:
            created = await self.overrides.create(
                workspace_id=workspace.id,
                organization_id=workspace.organization_id,
                org_provider_key_id=key.id,
                is_default=new_default,
                disabled=new_disabled,
            )
            result_default, result_disabled = created.is_default, created.disabled

        if new_disabled and not current_disabled:
            await self.restrictions.delete_for_workspace_key(workspace_id=workspace.id, org_provider_key_id=key.id)

        await self.db.commit()
        await refresh_org_provider_cache(self.db)

        candidates = await self.overrides.candidates_for_provider(
            organization_id=workspace.organization_id,
            provider=key.provider,
            workspace_id=workspace.id,
        )
        active = resolve_active_key(candidates)
        return WorkspaceProviderKeyOverridePublic(
            workspace_id=workspace.id,
            org_provider_key_id=key.id,
            is_default=result_default,
            disabled=result_disabled,
            is_effective_default=active is not None and active.id == key.id,
            is_effective_enabled=not result_disabled,
        )

    async def reset_workspace_override_for_user(
        self, *, user: User, workspace_id: uuid.UUID, key_id: uuid.UUID
    ) -> None:
        """Remove a workspace's override, reverting to full inheritance. Idempotent."""
        workspace = await authorization.resolve_visible_workspace(
            self.db, user=user, workspace_id=workspace_id, organizations=self.organizations
        )
        await authorization.require_workspace_management_access(
            self.db, user=user, workspace=workspace, organizations=self.organizations
        )

        key = await self.keys.get_in_organization(key_id, workspace.organization_id)
        if key is None:
            raise OrgProviderKeyNotFoundError(key_id)

        existing = await self.overrides.get(workspace_id=workspace.id, org_provider_key_id=key.id)
        if existing is None:
            return
        await self.overrides.delete(existing)
        await self.db.commit()
        await refresh_org_provider_cache(self.db)

    # ------------------------------------------------------------------
    # Model restrictions
    # ------------------------------------------------------------------

    async def list_model_restrictions_for_user(
        self,
        *,
        user: User,
        workspace_id: uuid.UUID,
        key_id: uuid.UUID,
    ) -> WorkspaceProviderModelRestrictionsPublic:
        """List a workspace's model allow-list for a key. Empty means every model is allowed."""
        workspace = await authorization.resolve_visible_workspace(
            self.db, user=user, workspace_id=workspace_id, organizations=self.organizations
        )
        key = await self.keys.get_in_organization(key_id, workspace.organization_id)
        if key is None:
            raise OrgProviderKeyNotFoundError(key_id)
        models = await self.restrictions.list_for_workspace_key(workspace_id=workspace.id, org_provider_key_id=key.id)
        return WorkspaceProviderModelRestrictionsPublic(models=models)

    async def add_model_restriction_for_user(
        self,
        *,
        user: User,
        workspace_id: uuid.UUID,
        key_id: uuid.UUID,
        model: str,
    ) -> None:
        """Narrow a workspace's allow-list for a key to include one more model. Idempotent."""
        workspace = await authorization.resolve_visible_workspace(
            self.db, user=user, workspace_id=workspace_id, organizations=self.organizations
        )
        await authorization.require_workspace_management_access(
            self.db, user=user, workspace=workspace, organizations=self.organizations
        )

        key = await self.keys.get_in_organization(key_id, workspace.organization_id)
        if key is None:
            raise OrgProviderKeyNotFoundError(key_id)

        override = await self.overrides.get(workspace_id=workspace.id, org_provider_key_id=key.id)
        if override is not None and override.disabled:
            raise OrgProviderKeyDisabledForWorkspaceError

        if await self.restrictions.get(workspace_id=workspace.id, org_provider_key_id=key.id, model=model) is None:
            await self.restrictions.add(
                workspace_id=workspace.id,
                organization_id=workspace.organization_id,
                org_provider_key_id=key.id,
                model=model,
            )
            await self.db.commit()
            await refresh_org_provider_cache(self.db)

    async def remove_model_restriction_for_user(
        self,
        *,
        user: User,
        workspace_id: uuid.UUID,
        key_id: uuid.UUID,
        model: str,
    ) -> None:
        """Remove one model from a workspace's allow-list for a key. Idempotent."""
        workspace = await authorization.resolve_visible_workspace(
            self.db, user=user, workspace_id=workspace_id, organizations=self.organizations
        )
        await authorization.require_workspace_management_access(
            self.db, user=user, workspace=workspace, organizations=self.organizations
        )

        key = await self.keys.get_in_organization(key_id, workspace.organization_id)
        if key is None:
            raise OrgProviderKeyNotFoundError(key_id)

        restriction = await self.restrictions.get(workspace_id=workspace.id, org_provider_key_id=key.id, model=model)
        if restriction is not None:
            await self.restrictions.remove(restriction)
            await self.db.commit()
            await refresh_org_provider_cache(self.db)


__all__ = [
    "ORG_PROVIDER_CACHE_TTL_SECONDS",
    "OrgProviderKeyService",
    "cached_org_model_restriction",
    "cached_org_provider_kwargs",
    "load_org_provider_keys_at_startup",
    "org_cache_is_stale",
    "refresh_org_provider_cache",
    "reset_org_provider_cache",
    "run_org_provider_refresher",
]
