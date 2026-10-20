"""Data access for an organization's web search keys and its workspaces' overrides.

:func:`resolve_web_search_key` decides which key a workspace searches with. It is a pure
function over rows the repository already loaded, so it is tested without a database.
"""

import uuid
from collections.abc import Callable, Sequence

from sqlalchemy import Select, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.settings.tools import WEB_SEARCH_PROVIDERS
from gateway.core.unit_of_work import UnitOfWork, session_for
from gateway.models.tools import OrgWebSearchKey, WorkspaceWebSearchKeyOverride
from gateway.repositories.base_repository import BaseRepository
from gateway.schemas.tools import OrgWebSearchKeyCreateRequest, OrgWebSearchKeyUpdateRequest


class WebSearchKeyConflict(Exception):
    """A write collided with a key of the same provider and name, or with a concurrent default.

    Raised in place of ``IntegrityError``, so the service maps it without importing the database layer.
    """


# One live key and this workspace's override of it, ``None`` where the workspace inherits.
SearchKeyCandidate = tuple[OrgWebSearchKey, WorkspaceWebSearchKeyOverride | None]


def resolve_web_search_key(
    candidates: Sequence[SearchKeyCandidate], usable: Callable[[OrgWebSearchKey], bool]
) -> OrgWebSearchKey | None:
    """The key a workspace searches with, or ``None`` where it uses the deployment's search.

    A request names no search provider, so one key is chosen across all of them:

    1. The key the workspace pinned.
    2. Otherwise an organization default, in ``WEB_SEARCH_PROVIDERS`` order.
    3. Otherwise the organization's oldest key.

    A key the workspace turned off, or one this deployment cannot decrypt, is passed over.
    ``usable`` is asked of keys in that order and only until one passes.
    ``candidates`` must be oldest-first, which is the order the repository returns.
    """
    live = [(key, override) for key, override in candidates if override is None or not override.disabled]
    rank = {provider: index for index, provider in enumerate(WEB_SEARCH_PROVIDERS)}
    pinned = [key for key, override in live if override is not None and override.is_default]
    defaults = sorted((key for key, _ in live if key.is_org_default), key=lambda key: rank.get(key.provider, len(rank)))
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

    async def get_by_name(self, *, organization_id: uuid.UUID, provider: str, name: str) -> OrgWebSearchKey | None:
        result = await self.db.execute(
            select(OrgWebSearchKey).where(
                OrgWebSearchKey.organization_id == organization_id,
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
                .order_by(OrgWebSearchKey.provider, OrgWebSearchKey.created_at, OrgWebSearchKey.id)
                .offset(skip)
                .limit(limit)
            )
        ).scalars()
        return list(rows), count

    async def create_key(
        self, *, organization_id: uuid.UUID, provider: str, name: str, encrypted_api_key: str, last4: str
    ) -> OrgWebSearchKey:
        """Stage a new key.

        Raises:
            WebSearchKeyConflict: the organization has a key of that provider and name.
        """
        key = OrgWebSearchKey(
            organization_id=organization_id,
            provider=provider,
            name=name,
            encrypted_api_key=encrypted_api_key,
            last4=last4,
        )
        self.db.add(key)
        await self._flush_or_conflict()
        await self.db.refresh(key)
        return key

    async def update_key(self, key: OrgWebSearchKey, values: dict[str, object]) -> OrgWebSearchKey:
        """Stage changes to a key.

        Raises:
            WebSearchKeyConflict: a rename collided with another key of the provider.
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
        """Make ``key`` its provider's default, clearing the sibling default in the same flush.

        Raises:
            WebSearchKeyConflict: a concurrent call made another key the default first.
        """
        await self.db.execute(
            update(OrgWebSearchKey)
            .where(
                OrgWebSearchKey.organization_id == key.organization_id,
                OrgWebSearchKey.provider == key.provider,
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

    async def clear_other_pins(self, *, workspace_id: uuid.UUID, except_key_id: uuid.UUID) -> None:
        """Unpin every other key of the workspace, so it holds at most one pin.

        One pin per workspace rather than per provider, because a search request names no
        provider. The caller holds the workspace lock across the clear and the new pin.
        """
        await self.db.execute(
            update(WorkspaceWebSearchKeyOverride)
            .where(
                WorkspaceWebSearchKeyOverride.workspace_id == workspace_id,
                WorkspaceWebSearchKeyOverride.org_web_search_key_id != except_key_id,
                WorkspaceWebSearchKeyOverride.is_default.is_(True),
            )
            .values(is_default=False)
            .execution_options(synchronize_session=False)
        )

    def _candidates(
        self, organization_id: uuid.UUID, workspace_id: uuid.UUID
    ) -> Select[tuple[OrgWebSearchKey, WorkspaceWebSearchKeyOverride]]:
        """The organization's live keys, each with this workspace's override or ``None``."""
        return (
            select(OrgWebSearchKey, WorkspaceWebSearchKeyOverride)
            .outerjoin(
                WorkspaceWebSearchKeyOverride,
                (WorkspaceWebSearchKeyOverride.org_web_search_key_id == OrgWebSearchKey.id)
                & (WorkspaceWebSearchKeyOverride.workspace_id == workspace_id),
            )
            .where(OrgWebSearchKey.organization_id == organization_id, OrgWebSearchKey.archived_at.is_(None))
        )

    async def preferred_candidates(
        self, *, organization_id: uuid.UUID, workspace_id: uuid.UUID
    ) -> list[SearchKeyCandidate]:
        """The workspace's pin and the organization's defaults: at most one per provider and one more."""
        result = await self.db.execute(
            self._candidates(organization_id, workspace_id).where(
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
        enabled_only: bool = False,
    ) -> list[SearchKeyCandidate]:
        """A page of the live keys, oldest-first; ``enabled_only`` leaves out those the workspace turned off."""
        query = self._candidates(organization_id, workspace_id)
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
