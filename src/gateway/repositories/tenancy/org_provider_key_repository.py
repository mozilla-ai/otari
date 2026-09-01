"""Data access for organization-scoped provider keys.

Three repositories, one per table in `gateway.models.provider_keys`. Every
column reference goes through `sqlmodel.col()` (see the package docstring).
Precedence resolution (`resolve_active_key`) is a module-level pure function
rather than a repository method: it takes plain in-memory data, so it is
unit-testable with no database and reusable from
`services/tenancy/org_provider_key_service.py`'s cache refresh, which needs
the same three-tier logic applied across every workspace at once rather than
one workspace at a time.
"""

import uuid
from collections.abc import Sequence

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.models.provider_keys import (
    OrgProviderKey,
    OrgProviderKeyCreateRequest,
    OrgProviderKeyUpdateRequest,
    WorkspaceProviderKeyOverride,
    WorkspaceProviderModelRestriction,
)
from gateway.repositories.base_repository import BaseRepository

# One (key, override) pair per candidate; the override is None when the
# workspace has never departed from inheriting this key.
Candidate = tuple[OrgProviderKey, WorkspaceProviderKeyOverride | None]


def resolve_active_key(candidates: Sequence[Candidate]) -> OrgProviderKey | None:
    """Pick the key a workspace should use for one provider.

    Ported from otari-ai's ``WorkspaceProviderKeyAccessRepository``, with every
    managed/phantom-bucket branch dropped: every otari-side key is BYO, so
    there is one tier ladder per provider rather than one per bucket.

    1. A workspace-pinned default (``is_default``, not ``disabled``) wins
       outright. At most one should exist per (workspace, provider): the
       service clears any sibling pin before setting a new one.
    2. Otherwise, the organization's default for this provider, unless this
       workspace has disabled it.
    3. Otherwise, the earliest-created key this workspace has not disabled.
       ``candidates`` must already be ordered oldest-first for this tier to be
       correct; callers query in that order rather than sorting here, so a
       cache-refresh pass building many workspaces' answers from one query
       does not re-sort per workspace.
    """
    enabled = [(key, override) for key, override in candidates if override is None or not override.disabled]

    for key, override in enabled:
        if override is not None and override.is_default:
            return key

    for key, _ in enabled:
        if key.is_org_default:
            return key

    return enabled[0][0] if enabled else None


class OrgProviderKeyRepository(
    BaseRepository[OrgProviderKey, OrgProviderKeyCreateRequest, OrgProviderKeyUpdateRequest]
):
    """Repository for `org_provider_keys` rows.

    Creation and update go through bespoke methods rather than the inherited
    generic ones: the plaintext key never lands in a column, so the service
    hands this repository an already-encrypted payload instead of the create
    request as-is.
    """

    def __init__(self, db: AsyncSession):
        super().__init__(db, OrgProviderKey)

    async def get_in_organization(self, key_id: uuid.UUID, organization_id: uuid.UUID) -> OrgProviderKey | None:
        """Return a key by id, scoped to the organization it must belong to.

        Scoping the read rather than checking afterward is what keeps a
        cross-organization id from being distinguishable from one that does
        not exist at all.
        """
        result = await self.db.execute(
            select(OrgProviderKey).where(
                col(OrgProviderKey.id) == key_id,
                col(OrgProviderKey.organization_id) == organization_id,
            )
        )
        return result.scalars().first()

    async def get_by_org_provider_name(
        self,
        *,
        organization_id: uuid.UUID,
        provider: str,
        name: str,
    ) -> OrgProviderKey | None:
        result = await self.db.execute(
            select(OrgProviderKey).where(
                col(OrgProviderKey.organization_id) == organization_id,
                col(OrgProviderKey.provider) == provider,
                col(OrgProviderKey.name) == name,
            )
        )
        return result.scalars().first()

    async def list_provider_names(self, organization_id: uuid.UUID) -> list[str]:
        """The distinct providers this organization holds a live key for.

        Unpaginated, unlike :meth:`list_for_organization`, because the result is
        one row per provider rather than per key: the catalog filter needs the
        whole set to decide what a member may be shown, and a page of it would
        silently hide providers.
        """
        result = await self.db.execute(
            select(col(OrgProviderKey.provider))
            .where(
                col(OrgProviderKey.organization_id) == organization_id,
                col(OrgProviderKey.archived_at).is_(None),
            )
            .distinct()
            .order_by(col(OrgProviderKey.provider))
        )
        return list(result.scalars().all())

    async def list_for_organization(
        self,
        organization_id: uuid.UUID,
        *,
        include_archived: bool = False,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[Sequence[OrgProviderKey], int]:
        """Return a page of an organization's keys, plus the total matching count.

        Mirrors ``WorkspaceRepository.get_by_organization``'s shape: a
        deployment with unbounded keys per organization would otherwise page
        every row on every list call, and the total lets a caller page
        correctly without a second unbounded query.
        """
        filters = [col(OrgProviderKey.organization_id) == organization_id]
        if not include_archived:
            filters.append(col(OrgProviderKey.archived_at).is_(None))

        count_result = await self.db.execute(select(func.count()).select_from(OrgProviderKey).where(*filters))
        count = count_result.scalar_one()

        stmt = (
            select(OrgProviderKey)
            .where(*filters)
            .order_by(col(OrgProviderKey.provider), col(OrgProviderKey.created_at), col(OrgProviderKey.id))
            .offset(skip)
            .limit(limit)
        )
        rows = (await self.db.execute(stmt)).scalars().all()
        return rows, count

    async def create_key(
        self,
        *,
        organization_id: uuid.UUID,
        provider: str,
        name: str,
        encrypted_api_key: str | None,
        last4: str | None,
        api_base: str | None,
        client_args: dict[str, object] | None,
    ) -> OrgProviderKey:
        key = OrgProviderKey(
            organization_id=organization_id,
            provider=provider,
            name=name,
            encrypted_api_key=encrypted_api_key,
            last4=last4,
            api_base=api_base,
            client_args=client_args,
        )
        self.db.add(key)
        await self.db.flush()
        await self.db.refresh(key)
        return key

    async def update_key(self, key: OrgProviderKey, update_data: dict[str, object]) -> OrgProviderKey:
        key.sqlmodel_update(update_data)
        self.db.add(key)
        await self.db.flush()
        await self.db.refresh(key)
        return key

    async def delete_key(self, key: OrgProviderKey) -> None:
        """Stage a deletion. Overrides and model restrictions ride the database cascade."""
        await self.db.delete(key)
        await self.db.flush()

    async def set_org_default(self, key: OrgProviderKey) -> OrgProviderKey:
        """Make ``key`` the organization's default for its provider.

        Clears any sibling default in the same ``(organization_id, provider)``
        first, in the same flush, so the two writes commit atomically rather
        than leaving a window with two (or zero) defaults visible to a
        concurrent reader. The partial unique index
        (``uq_org_provider_keys_org_default``) is the actual race arbiter: two
        concurrent calls for different keys both pass this method and one
        loses at flush/commit, which the service catches as ``IntegrityError``
        and maps to ``OrgDefaultProviderKeyConflictError``.
        """
        await self.db.execute(
            update(OrgProviderKey)
            .where(
                col(OrgProviderKey.organization_id) == key.organization_id,
                col(OrgProviderKey.provider) == key.provider,
                col(OrgProviderKey.id) != key.id,
                col(OrgProviderKey.is_org_default).is_(True),
            )
            .values(is_org_default=False)
            .execution_options(synchronize_session=False)
        )
        key.is_org_default = True
        self.db.add(key)
        await self.db.flush()
        await self.db.refresh(key)
        return key


class WorkspaceProviderKeyOverrideRepository:
    """Repository for `workspace_provider_key_overrides` rows.

    Not a ``BaseRepository``: every access is keyed by the (workspace, key)
    pair or resolves candidates across a whole provider, not by the override
    row's own id. Pure data access, no tri-state merging or conflict
    resolution: that business logic lives in the service, which is what
    `BaseRepository`'s own docstring asks of every repository here.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get(
        self, *, workspace_id: uuid.UUID, org_provider_key_id: uuid.UUID
    ) -> WorkspaceProviderKeyOverride | None:
        result = await self.db.execute(
            select(WorkspaceProviderKeyOverride).where(
                col(WorkspaceProviderKeyOverride.workspace_id) == workspace_id,
                col(WorkspaceProviderKeyOverride.org_provider_key_id) == org_provider_key_id,
            )
        )
        return result.scalars().first()

    async def create(
        self,
        *,
        workspace_id: uuid.UUID,
        organization_id: uuid.UUID,
        org_provider_key_id: uuid.UUID,
        is_default: bool,
        disabled: bool,
    ) -> WorkspaceProviderKeyOverride:
        override = WorkspaceProviderKeyOverride(
            workspace_id=workspace_id,
            organization_id=organization_id,
            org_provider_key_id=org_provider_key_id,
            is_default=is_default,
            disabled=disabled,
        )
        self.db.add(override)
        await self.db.flush()
        await self.db.refresh(override)
        return override

    async def update(
        self,
        override: WorkspaceProviderKeyOverride,
        *,
        is_default: bool,
        disabled: bool,
    ) -> WorkspaceProviderKeyOverride:
        override.is_default = is_default
        override.disabled = disabled
        self.db.add(override)
        await self.db.flush()
        await self.db.refresh(override)
        return override

    async def delete(self, override: WorkspaceProviderKeyOverride) -> None:
        await self.db.delete(override)
        await self.db.flush()

    async def clear_workspace_pinned_default(
        self,
        *,
        workspace_id: uuid.UUID,
        organization_id: uuid.UUID,
        provider: str,
        except_key_id: uuid.UUID,
    ) -> None:
        """Unset this workspace's pin on every *other* key of ``provider``, if any.

        Called before setting a new pin, so a workspace can never hold two
        pinned defaults for the same provider (the unique index is per
        (workspace, key), not per (workspace, provider), so nothing at the
        database enforces that on its own; the caller is expected to hold
        ``WorkspaceRepository.lock(workspace_id)`` for the duration of the
        read-clear-set sequence this is one step of, since a bare
        conditional UPDATE cannot serialize two different keys racing to
        become the same workspace's pin).

        ``except_key_id`` excludes the key about to be (re-)pinned. Without
        it, re-pinning a key that is already this workspace's default clears
        its own override row via this raw UPDATE, bypassing the ORM's change
        tracking; the caller then sets ``is_default = True`` on the very same
        Python value it loaded, which SQLAlchemy sees as no change and skips
        re-emitting the UPDATE, leaving the row cleared in the database while
        the caller's in-memory object (and its API response) still say pinned.
        """
        await self.db.execute(
            update(WorkspaceProviderKeyOverride)
            .where(
                col(WorkspaceProviderKeyOverride.workspace_id) == workspace_id,
                col(WorkspaceProviderKeyOverride.org_provider_key_id) != except_key_id,
                col(WorkspaceProviderKeyOverride.org_provider_key_id).in_(
                    select(col(OrgProviderKey.id)).where(
                        col(OrgProviderKey.organization_id) == organization_id,
                        col(OrgProviderKey.provider) == provider,
                    )
                ),
                col(WorkspaceProviderKeyOverride.is_default).is_(True),
            )
            .values(is_default=False)
            .execution_options(synchronize_session=False)
        )

    async def candidates_for_provider(
        self,
        *,
        organization_id: uuid.UUID,
        provider: str,
        workspace_id: uuid.UUID,
    ) -> list[Candidate]:
        """Every non-archived key of one provider in the organization, with this workspace's override.

        Ordered oldest-first, which is the order `resolve_active_key`'s
        fallback tier requires.
        """
        result = await self.db.execute(
            select(OrgProviderKey, WorkspaceProviderKeyOverride)
            .outerjoin(
                WorkspaceProviderKeyOverride,
                (col(WorkspaceProviderKeyOverride.org_provider_key_id) == col(OrgProviderKey.id))
                & (col(WorkspaceProviderKeyOverride.workspace_id) == workspace_id),
            )
            .where(
                col(OrgProviderKey.organization_id) == organization_id,
                col(OrgProviderKey.provider) == provider,
                col(OrgProviderKey.archived_at).is_(None),
            )
            .order_by(col(OrgProviderKey.created_at), col(OrgProviderKey.id))
        )
        return [(key, override) for key, override in result.all()]

    async def all_candidates(
        self,
        *,
        organization_id: uuid.UUID,
        workspace_id: uuid.UUID,
    ) -> list[Candidate]:
        """Every non-archived key in the organization, with this workspace's override.

        Used to build the effective view across every provider at once
        (`GET /v1/workspaces/{id}/provider-keys`) and by the dispatch-path
        cache refresh, which needs every workspace's answer for every
        provider in one pass rather than one query per (workspace, provider).
        """
        result = await self.db.execute(
            select(OrgProviderKey, WorkspaceProviderKeyOverride)
            .outerjoin(
                WorkspaceProviderKeyOverride,
                (col(WorkspaceProviderKeyOverride.org_provider_key_id) == col(OrgProviderKey.id))
                & (col(WorkspaceProviderKeyOverride.workspace_id) == workspace_id),
            )
            .where(
                col(OrgProviderKey.organization_id) == organization_id,
                col(OrgProviderKey.archived_at).is_(None),
            )
            .order_by(col(OrgProviderKey.provider), col(OrgProviderKey.created_at), col(OrgProviderKey.id))
        )
        return [(key, override) for key, override in result.all()]

    async def get_active_key_for_workspace_provider(
        self,
        *,
        organization_id: uuid.UUID,
        workspace_id: uuid.UUID,
        provider: str,
    ) -> OrgProviderKey | None:
        candidates = await self.candidates_for_provider(
            organization_id=organization_id,
            provider=provider,
            workspace_id=workspace_id,
        )
        return resolve_active_key(candidates)


class WorkspaceProviderModelRestrictionRepository:
    """Repository for `workspace_provider_model_restrictions` rows."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def list_for_workspace_key(self, *, workspace_id: uuid.UUID, org_provider_key_id: uuid.UUID) -> list[str]:
        result = await self.db.execute(
            select(col(WorkspaceProviderModelRestriction.model))
            .where(
                col(WorkspaceProviderModelRestriction.workspace_id) == workspace_id,
                col(WorkspaceProviderModelRestriction.org_provider_key_id) == org_provider_key_id,
            )
            .order_by(col(WorkspaceProviderModelRestriction.model))
        )
        return list(result.scalars().all())

    async def get(
        self, *, workspace_id: uuid.UUID, org_provider_key_id: uuid.UUID, model: str
    ) -> WorkspaceProviderModelRestriction | None:
        result = await self.db.execute(
            select(WorkspaceProviderModelRestriction).where(
                col(WorkspaceProviderModelRestriction.workspace_id) == workspace_id,
                col(WorkspaceProviderModelRestriction.org_provider_key_id) == org_provider_key_id,
                col(WorkspaceProviderModelRestriction.model) == model,
            )
        )
        return result.scalars().first()

    async def add(
        self,
        *,
        workspace_id: uuid.UUID,
        organization_id: uuid.UUID,
        org_provider_key_id: uuid.UUID,
        model: str,
    ) -> WorkspaceProviderModelRestriction:
        restriction = WorkspaceProviderModelRestriction(
            workspace_id=workspace_id,
            organization_id=organization_id,
            org_provider_key_id=org_provider_key_id,
            model=model,
        )
        self.db.add(restriction)
        await self.db.flush()
        await self.db.refresh(restriction)
        return restriction

    async def remove(self, restriction: WorkspaceProviderModelRestriction) -> None:
        await self.db.delete(restriction)
        await self.db.flush()

    async def delete_for_workspace_key(self, *, workspace_id: uuid.UUID, org_provider_key_id: uuid.UUID) -> None:
        """Drop every restriction a workspace holds for one key.

        Called when a workspace disables that key: a restriction on a key the
        workspace can no longer use is meaningless, and leaving it behind
        would resurface as soon as the key was re-enabled with a stale list
        nobody chose at that point.
        """
        await self.db.execute(
            delete(WorkspaceProviderModelRestriction)
            .where(
                col(WorkspaceProviderModelRestriction.workspace_id) == workspace_id,
                col(WorkspaceProviderModelRestriction.org_provider_key_id) == org_provider_key_id,
            )
            .execution_options(synchronize_session=False)
        )
        await self.db.flush()


__all__ = [
    "Candidate",
    "OrgProviderKeyRepository",
    "WorkspaceProviderKeyOverrideRepository",
    "WorkspaceProviderModelRestrictionRepository",
    "resolve_active_key",
]
