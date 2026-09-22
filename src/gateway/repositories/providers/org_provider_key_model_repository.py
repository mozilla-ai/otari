"""Data access for the models an organization offers on its provider keys.

Built on the Unit of Work rather than a session: this table is new, so it
starts in the shape `backend-standards` prescribes rather than the one the
repository beside it is still in. Flushes, never commits.

Every column reference goes through `sqlmodel.col()`, as the package docstring
requires.
"""

import uuid
from collections.abc import Collection, Sequence

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.provider_keys import (
    OrgProviderKeyModel,
)
from gateway.repositories.base_repository import BaseRepository
from gateway.schemas.providers import (
    OrgProviderKeyModelCreateRequest,
    OrgProviderKeyModelUpdateRequest,
)


class OfferedModelConflict(Exception):
    """The unique index refused a model already offered on this key.

    Raised here rather than letting ``IntegrityError`` travel, because the
    service that maps this to its domain error may not import a database
    library (``scripts/check_architecture.py``, rule 11). It carries the model
    so the caller can name it without re-reading the row.
    """

    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.model = model


class OrgProviderKeyModelRepository(
    BaseRepository[OrgProviderKeyModel, OrgProviderKeyModelCreateRequest, OrgProviderKeyModelUpdateRequest]
):
    """Repository for `org_provider_key_models` rows."""

    def __init__(self, db: AsyncSession | UnitOfWork):
        """Bind to a unit of work, or to a session for a read outside one.

        New write code passes the unit of work. The catalog and the dispatch
        cache pass a session, because both are reads on code still in the older
        shape and neither opens a block.
        """
        super().__init__(db, OrgProviderKeyModel)

    async def get_in_key(self, model_id: uuid.UUID, org_provider_key_id: uuid.UUID) -> OrgProviderKeyModel | None:
        """Return one offered model by id, scoped to the key it must belong to.

        Scoping the read rather than checking afterward is what keeps a row id
        from another key indistinguishable from one that does not exist, the
        same rule `OrgProviderKeyRepository.get_in_organization` follows.
        """
        result = await self.db.execute(
            select(OrgProviderKeyModel).where(
                col(OrgProviderKeyModel.id) == model_id,
                col(OrgProviderKeyModel.org_provider_key_id) == org_provider_key_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_by_model(self, org_provider_key_id: uuid.UUID, model: str) -> OrgProviderKeyModel | None:
        """Return the row offering ``model`` on this key, if there is one."""
        result = await self.db.execute(
            select(OrgProviderKeyModel).where(
                col(OrgProviderKeyModel.org_provider_key_id) == org_provider_key_id,
                col(OrgProviderKeyModel.model) == model,
            )
        )
        return result.scalar_one_or_none()

    async def list_for_key(
        self, org_provider_key_id: uuid.UUID, *, skip: int = 0, limit: int = 500
    ) -> tuple[Sequence[OrgProviderKeyModel], int]:
        """One page of a key's offered models, and how many there are in total.

        ``count`` is the total rather than the page length, so a client knows
        whether another page is owed; counted in the database rather than by
        loading every row, because a provider can offer several hundred models.
        """
        rows = (
            (
                await self.db.execute(
                    select(OrgProviderKeyModel)
                    .where(col(OrgProviderKeyModel.org_provider_key_id) == org_provider_key_id)
                    .order_by(col(OrgProviderKeyModel.model))
                    .offset(skip)
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        total = (
            await self.db.execute(
                select(func.count())
                .select_from(OrgProviderKeyModel)
                .where(col(OrgProviderKeyModel.org_provider_key_id) == org_provider_key_id)
            )
        ).scalar_one()
        return rows, total

    async def list_all_for_key(self, org_provider_key_id: uuid.UUID) -> Sequence[OrgProviderKeyModel]:
        """Every offered row on one key, for the passes that reprice or reseed them."""
        result = await self.db.execute(
            select(OrgProviderKeyModel)
            .where(col(OrgProviderKeyModel.org_provider_key_id) == org_provider_key_id)
            .order_by(col(OrgProviderKeyModel.model))
        )
        return result.scalars().all()

    async def names_for_key(self, org_provider_key_id: uuid.UUID) -> set[str]:
        """The model names already offered on one key, for the additive refresh diff."""
        result = await self.db.execute(
            select(col(OrgProviderKeyModel.model)).where(
                col(OrgProviderKeyModel.org_provider_key_id) == org_provider_key_id
            )
        )
        return set(result.scalars().all())

    async def enabled_models_for_keys(
        self, org_provider_key_ids: Collection[uuid.UUID]
    ) -> dict[uuid.UUID, list[str]]:
        """The served models of each key that offers any, keyed by key id.

        **A key absent from the result offers nothing and is therefore
        unnarrowed**, which is the convention
        `WorkspaceProviderModelRestrictionRepository.list_for_workspace_keys`
        already sets: a caller must not read a missing key as "serves nothing".
        A key present with an empty list is the opposite answer, and is the one
        a key whose every model is switched off produces.
        """
        wanted = set(org_provider_key_ids)
        if not wanted:
            return {}
        result = await self.db.execute(
            select(
                col(OrgProviderKeyModel.org_provider_key_id),
                col(OrgProviderKeyModel.model),
                col(OrgProviderKeyModel.enabled),
            )
            .where(col(OrgProviderKeyModel.org_provider_key_id).in_(wanted))
            .order_by(col(OrgProviderKeyModel.model))
        )
        offered: dict[uuid.UUID, list[str]] = {}
        for key_id, model, enabled in result.all():
            served = offered.setdefault(key_id, [])
            if enabled:
                served.append(model)
        return offered

    async def create_many(self, rows: Sequence[OrgProviderKeyModel]) -> Sequence[OrgProviderKeyModel]:
        """Stage several offered rows at once. The caller owns the transaction.

        Flushed here rather than left to the commit, so the unique index answers
        while the caller can still say which model it refused. A pre-check races
        the insert, and the constraint is what actually decides.

        Raises:
            OfferedModelConflict: one of these models is already offered here.
        """
        if not rows:
            return []
        self.db.add_all(rows)
        try:
            await self.db.flush()
        except IntegrityError as exc:
            raise OfferedModelConflict(rows[0].model) from exc
        return rows

    async def save(self, row: OrgProviderKeyModel) -> OrgProviderKeyModel:
        """Flush a changed row and read back what the database filled in.

        Refreshed rather than only flushed because the UPDATE fires
        ``updated_at``'s ``onupdate``, which expires the attribute, and reading
        an expired attribute back on an async session is a synchronous lazy load
        that cannot run.
        """
        self.db.add(row)
        await self.db.flush()
        await self.db.refresh(row)
        return row

    async def delete_row(self, row: OrgProviderKeyModel) -> None:
        """Stop offering one model. The caller owns the transaction."""
        await self.db.delete(row)
        await self.db.flush()
