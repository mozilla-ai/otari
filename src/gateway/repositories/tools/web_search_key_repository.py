"""Data access for an organization's web search and fetch keys and its workspaces' overrides.

:func:`resolve_web_search_key` decides which key of a kind a workspace uses. It is a pure
function over rows the repository already loaded, so it is tested without a database.
"""

import uuid
from collections.abc import Callable, Sequence
from typing import Any

from sqlalchemy import Select, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.settings.tools import ToolKind
from gateway.core.unit_of_work import UnitOfWork, session_for
from gateway.models.tools import OrgWebSearchKey, WorkspaceWebSearchKeyOverride
from gateway.repositories.base_repository import BaseRepository
from gateway.schemas.tools import OrgWebSearchKeyCreateRequest, OrgWebSearchKeyUpdateRequest


class WebSearchKeyConflict(Exception):
    """A write collided with a key of the same kind, provider and name, or with a concurrent default.

    Raised in place of ``IntegrityError``, so the service maps it without importing the database layer.
    """


# One live key and this workspace's override of it, ``None`` where the workspace inherits.
SearchKeyCandidate = tuple[OrgWebSearchKey, WorkspaceWebSearchKeyOverride | None]


def resolve_web_search_key(
    candidates: Sequence[SearchKeyCandidate], usable: Callable[[OrgWebSearchKey], bool]
) -> OrgWebSearchKey | None:
    """The key of one kind a workspace uses, or ``None`` where it uses the deployment's instance.

    A request names no provider, so one key is chosen across all of them, among
    ``candidates`` of a single kind:

    1. The key the workspace pinned.
    2. Otherwise the organization's default.
    3. Otherwise the organization's oldest key.

    A key the workspace turned off, or one this deployment cannot decrypt, is passed over.
    ``usable`` is asked of keys in that order and only until one passes.
    ``candidates`` must be oldest-first, which is the order the repository returns.
    """
    live = [(key, override) for key, override in candidates if override is None or not override.disabled]
    pinned = [key for key, override in live if override is not None and override.is_default]
    defaults = [key for key, _ in live if key.is_org_default]
    return next((key for key in (*pinned, *defaults, *(key for key, _ in live)) if usable(key)), None)


class OrgWebSearchKeyRepository(
    BaseRepository[OrgWebSearchKey, OrgWebSearchKeyCreateRequest, OrgWebSearchKeyUpdateRequest]
):
    """``org_web_search_keys`` rows. Writes take ciphertext: the plaintext key never reaches a column."""

    def __init__(self, db: AsyncSession | UnitOfWork):
        super().__init__(db, OrgWebSearchKey)

    async def _flush_or_conflict(self) -> None:
        try:
            await self.db.flush()
        except IntegrityError:
            raise WebSearchKeyConflict from None

    async def get_in_organization(self, key_id: uuid.UUID, organization_id: uuid.UUID) -> OrgWebSearchKey | None:
        """A key by id, scoped to its organization, so another organization's key reads as absent."""
        result = await self.db.execute(
            select(OrgWebSearchKey).where(
                OrgWebSearchKey.id == key_id, OrgWebSearchKey.organization_id == organization_id
            )
        )
        return result.scalars().first()

    async def get_by_name(
        self, *, organization_id: uuid.UUID, kind: str, provider: str, name: str
    ) -> OrgWebSearchKey | None:
        result = await self.db.execute(
            select(OrgWebSearchKey).where(
                OrgWebSearchKey.organization_id == organization_id,
                OrgWebSearchKey.kind == kind,
                OrgWebSearchKey.provider == provider,
                OrgWebSearchKey.name == name,
            )
        )
        return result.scalars().first()

    async def list_for_organization(
        self, organization_id: uuid.UUID, *, include_archived: bool = False, skip: int = 0, limit: int = 100
    ) -> tuple[Sequence[OrgWebSearchKey], int]:
        """A page of the organization's keys and the total count."""
        filters = [OrgWebSearchKey.organization_id == organization_id]
        if not include_archived:
            filters.append(OrgWebSearchKey.archived_at.is_(None))
        count = (await self.db.execute(select(func.count()).select_from(OrgWebSearchKey).where(*filters))).scalar_one()
        rows = (
            await self.db.execute(
                select(OrgWebSearchKey)
                .where(*filters)
                .order_by(
                    OrgWebSearchKey.kind, OrgWebSearchKey.provider, OrgWebSearchKey.created_at, OrgWebSearchKey.id
                )
                .offset(skip)
                .limit(limit)
            )
        ).scalars()
        return list(rows), count

    async def create_key(
        self,
        *,
        organization_id: uuid.UUID,
        kind: ToolKind,
        provider: str,
        name: str,
        encrypted_api_key: str | None,
        last4: str | None,
        api_base: str | None,
        options: dict[str, Any] | None,
    ) -> OrgWebSearchKey:
        """Stage a new key.

        Raises:
            WebSearchKeyConflict: the organization has a key of that kind, provider and name.
        """
        key = OrgWebSearchKey(
            organization_id=organization_id,
            kind=kind,
            provider=provider,
            name=name,
            encrypted_api_key=encrypted_api_key,
            last4=last4,
            api_base=api_base,
            options=options,
        )
        self.db.add(key)
        await self._flush_or_conflict()
        await self.db.refresh(key)
        return key

    async def update_key(self, key: OrgWebSearchKey, values: dict[str, object]) -> OrgWebSearchKey:
        """Stage changes to a key.

        Raises:
            WebSearchKeyConflict: a rename collided with another key of the kind and provider.
        """
        for field, value in values.items():
            setattr(key, field, value)
        await self._flush_or_conflict()
        await self.db.refresh(key)
        return key

    async def delete_key(self, key: OrgWebSearchKey) -> None:
        """Stage a deletion. Overrides go with it by the database cascade."""
        await self.db.delete(key)
        await self.db.flush()

    async def set_org_default(self, key: OrgWebSearchKey) -> OrgWebSearchKey:
        """Make ``key`` its kind's default, clearing the other default of its kind in the same flush.

        Raises:
            WebSearchKeyConflict: a concurrent call made another key the default first.
        """
        await self.db.execute(
            update(OrgWebSearchKey)
            .where(
                OrgWebSearchKey.organization_id == key.organization_id,
                OrgWebSearchKey.kind == key.kind,
                OrgWebSearchKey.id != key.id,
                OrgWebSearchKey.is_org_default.is_(True),
            )
            .values(is_org_default=False)
            .execution_options(synchronize_session=False)
        )
        key.is_org_default = True
        await self._flush_or_conflict()
        await self.db.refresh(key)
        return key


class WorkspaceWebSearchKeyOverrideRepository:
    """``workspace_web_search_key_overrides`` rows, always addressed by (workspace, key)."""

    def __init__(self, db: AsyncSession | UnitOfWork):
        self._db = db

    @property
    def db(self) -> AsyncSession:
        return session_for(self._db) if isinstance(self._db, UnitOfWork) else self._db

    async def get(self, *, workspace_id: uuid.UUID, key_id: uuid.UUID) -> WorkspaceWebSearchKeyOverride | None:
        result = await self.db.execute(
            select(WorkspaceWebSearchKeyOverride).where(
                WorkspaceWebSearchKeyOverride.workspace_id == workspace_id,
                WorkspaceWebSearchKeyOverride.org_web_search_key_id == key_id,
            )
        )
        return result.scalars().first()

    async def save(
        self,
        override: WorkspaceWebSearchKeyOverride | None,
        *,
        workspace_id: uuid.UUID,
        organization_id: uuid.UUID,
        key_id: uuid.UUID,
        is_default: bool,
        disabled: bool,
    ) -> WorkspaceWebSearchKeyOverride:
        """Create the override, or update the one passed in."""
        if override is None:
            override = WorkspaceWebSearchKeyOverride(
                workspace_id=workspace_id, organization_id=organization_id, org_web_search_key_id=key_id
            )
            self.db.add(override)
        override.is_default = is_default
        override.disabled = disabled
        await self.db.flush()
        await self.db.refresh(override)
        return override

    async def delete(self, override: WorkspaceWebSearchKeyOverride) -> None:
        await self.db.delete(override)
        await self.db.flush()

    async def clear_other_pins(self, *, workspace_id: uuid.UUID, except_key_id: uuid.UUID, kind: str) -> None:
        """Unpin the workspace's other keys of ``kind``, so it holds at most one pin per kind.

        One pin per kind rather than per provider, because a request names no provider.
        The caller holds the workspace lock across the clear and the new pin.
        """
        await self.db.execute(
            update(WorkspaceWebSearchKeyOverride)
            .where(
                WorkspaceWebSearchKeyOverride.workspace_id == workspace_id,
                WorkspaceWebSearchKeyOverride.org_web_search_key_id != except_key_id,
                WorkspaceWebSearchKeyOverride.org_web_search_key_id.in_(
                    select(OrgWebSearchKey.id).where(OrgWebSearchKey.kind == kind)
                ),
                WorkspaceWebSearchKeyOverride.is_default.is_(True),
            )
            .values(is_default=False)
            .execution_options(synchronize_session=False)
        )

    def _candidates(
        self, organization_id: uuid.UUID, workspace_id: uuid.UUID, kind: str | None = None
    ) -> Select[tuple[OrgWebSearchKey, WorkspaceWebSearchKeyOverride]]:
        """The organization's live keys, of ``kind`` or every kind, each with this workspace's override or ``None``."""
        query = (
            select(OrgWebSearchKey, WorkspaceWebSearchKeyOverride)
            .outerjoin(
                WorkspaceWebSearchKeyOverride,
                (WorkspaceWebSearchKeyOverride.org_web_search_key_id == OrgWebSearchKey.id)
                & (WorkspaceWebSearchKeyOverride.workspace_id == workspace_id),
            )
            .where(OrgWebSearchKey.organization_id == organization_id, OrgWebSearchKey.archived_at.is_(None))
        )
        return query if kind is None else query.where(OrgWebSearchKey.kind == kind)

    async def preferred_candidates(
        self, *, organization_id: uuid.UUID, workspace_id: uuid.UUID, kind: str | None
    ) -> list[SearchKeyCandidate]:
        """The workspace's pins and the organization's defaults, of ``kind`` or every kind: at most two keys a kind."""
        result = await self.db.execute(
            self._candidates(organization_id, workspace_id, kind).where(
                WorkspaceWebSearchKeyOverride.is_default.is_(True) | OrgWebSearchKey.is_org_default.is_(True)
            )
        )
        return [(key, override) for key, override in result.tuples()]

    async def candidate_page(
        self,
        *,
        organization_id: uuid.UUID,
        workspace_id: uuid.UUID,
        skip: int,
        limit: int,
        kind: str | None = None,
        enabled_only: bool = False,
    ) -> list[SearchKeyCandidate]:
        """A page of the live keys of ``kind`` or every kind, oldest-first.

        ``enabled_only`` leaves out those the workspace turned off.
        """
        query = self._candidates(organization_id, workspace_id, kind)
        if enabled_only:
            query = query.where(
                WorkspaceWebSearchKeyOverride.disabled.is_(None) | WorkspaceWebSearchKeyOverride.disabled.is_(False)
            )
        result = await self.db.execute(
            query.order_by(OrgWebSearchKey.created_at, OrgWebSearchKey.id).offset(skip).limit(limit)
        )
        return [(key, override) for key, override in result.tuples()]

    async def count_candidates(self, *, organization_id: uuid.UUID) -> int:
        """How many live keys the organization holds."""
        result = await self.db.execute(
            select(func.count())
            .select_from(OrgWebSearchKey)
            .where(OrgWebSearchKey.organization_id == organization_id, OrgWebSearchKey.archived_at.is_(None))
        )
        return int(result.scalar_one())
