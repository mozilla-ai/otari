"""Data access for the deployment's hosted providers and the models offered on them.

Built on the Unit of Work. Flushes, never commits. Every column reference
goes through ``sqlmodel.col()``, as the SQLModel tables require.
"""

import uuid
from collections.abc import Collection, Sequence
from typing import Any, cast

from sqlalchemy import CursorResult, delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlmodel import col

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.providers import HostedProvider, HostedProviderModel
from gateway.repositories.base_repository import BaseRepository
from gateway.schemas.providers import (
    HostedModelCreateRequest,
    HostedModelUpdateRequest,
    HostedProviderCreateRequest,
    HostedProviderUpdateRequest,
)

# One row per provider, and any-llm knows a bounded set of them, so a full
# listing never pages. Named so it reads as a deliberate ceiling.
_ALL_PROVIDERS = 1000


class HostedProviderConflict(Exception):
    """The unique constraint refused a provider that already has a row.

    Raised in place of ``IntegrityError`` because the service that maps it to
    a domain error may not import a database library.
    """

    def __init__(self, provider: str) -> None:
        super().__init__(provider)
        self.provider = provider


class HostedModelConflict(Exception):
    """The unique index refused a model already offered on this provider."""

    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.model = model


class HostedProviderRepository(
    BaseRepository[HostedProvider, HostedProviderCreateRequest, HostedProviderUpdateRequest]
):
    """Repository for `hosted_providers` rows."""

    def __init__(self, uow: UnitOfWork):
        super().__init__(uow, HostedProvider)

    async def get_by_provider(self, provider: str) -> HostedProvider | None:
        """The row for ``provider``, if the deployment holds one."""
        result = await self.db.execute(select(HostedProvider).where(col(HostedProvider.provider) == provider))
        return result.scalar_one_or_none()

    async def list_page(self, *, skip: int = 0, limit: int = 100) -> tuple[Sequence[HostedProvider], int]:
        """One page of hosted providers by name, and how many there are in total."""
        rows = (
            (
                await self.db.execute(
                    select(HostedProvider).order_by(col(HostedProvider.provider)).offset(skip).limit(limit)
                )
            )
            .scalars()
            .all()
        )
        total = (await self.db.execute(select(func.count()).select_from(HostedProvider))).scalar_one()
        return rows, total

    async def list_all(self) -> Sequence[HostedProvider]:
        """Every hosted provider, for the runtime's listing and a catalog sweep."""
        result = await self.db.execute(
            select(HostedProvider).order_by(col(HostedProvider.provider)).limit(_ALL_PROVIDERS)
        )
        return result.scalars().all()

    async def insert(self, **fields: Any) -> HostedProvider:
        """Stage a new hosted provider.

        Flushed here rather than left to the commit, so the unique constraint
        answers while the caller can still say which provider it refused. A
        pre-check races the insert, and the constraint is what decides.

        Raises:
            HostedProviderConflict: a row for that provider already exists.
        """
        row = HostedProvider(**fields)
        self.db.add(row)
        try:
            await self.db.flush()
        except IntegrityError as exc:
            raise HostedProviderConflict(row.provider) from exc
        await self.db.refresh(row)
        return row

    async def save(self, row: HostedProvider) -> HostedProvider:
        """Flush a changed row and read back what the database filled in.

        Refreshed rather than only flushed because the UPDATE fires
        ``updated_at``'s ``onupdate``, which expires the attribute, and reading
        an expired attribute back on an async session is a synchronous lazy
        load that cannot run.
        """
        self.db.add(row)
        await self.db.flush()
        await self.db.refresh(row)
        return row

    async def delete_row(self, row: HostedProvider) -> None:
        """Remove a hosted provider. The caller owns the transaction."""
        await self.db.delete(row)
        await self.db.flush()


class HostedProviderModelRepository(
    BaseRepository[HostedProviderModel, HostedModelCreateRequest, HostedModelUpdateRequest]
):
    """Repository for `hosted_provider_models` rows."""

    def __init__(self, uow: UnitOfWork):
        super().__init__(uow, HostedProviderModel)

    async def get_in_provider(self, model_id: uuid.UUID, provider: str) -> HostedProviderModel | None:
        """One offered model by id, scoped to the provider it must belong to.

        Scoped rather than checked afterward, so a row id under another
        provider is indistinguishable from one that does not exist.
        """
        result = await self.db.execute(
            select(HostedProviderModel).where(
                col(HostedProviderModel.id) == model_id,
                col(HostedProviderModel.provider) == provider,
            )
        )
        return result.scalar_one_or_none()

    async def get_by_model(self, provider: str, model: str) -> HostedProviderModel | None:
        """The row offering ``model`` on ``provider``, if there is one."""
        result = await self.db.execute(
            select(HostedProviderModel).where(
                col(HostedProviderModel.provider) == provider,
                col(HostedProviderModel.model) == model,
            )
        )
        return result.scalar_one_or_none()

    async def list_for_provider(
        self, provider: str, *, skip: int = 0, limit: int = 500
    ) -> tuple[Sequence[HostedProviderModel], int]:
        """One page of a provider's offered models, and how many there are in total.

        Counted in the database rather than by loading every row, because a
        provider can offer several hundred models.
        """
        rows = (
            (
                await self.db.execute(
                    select(HostedProviderModel)
                    .where(col(HostedProviderModel.provider) == provider)
                    .order_by(col(HostedProviderModel.model))
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
                .select_from(HostedProviderModel)
                .where(col(HostedProviderModel.provider) == provider)
            )
        ).scalar_one()
        return rows, total

    async def list_seeded(self, provider: str) -> Sequence[HostedProviderModel]:
        """One provider's offered models whose price this surface seeded."""
        result = await self.db.execute(
            select(HostedProviderModel)
            .where(
                col(HostedProviderModel.provider) == provider,
                col(HostedProviderModel.seeded_price_at).is_not(None),
            )
            .order_by(col(HostedProviderModel.model))
        )
        return result.scalars().all()

    async def names_for_provider(self, provider: str) -> set[str]:
        """The model names already offered on one provider, for the additive refresh diff."""
        result = await self.db.execute(
            select(col(HostedProviderModel.model)).where(col(HostedProviderModel.provider) == provider)
        )
        return set(result.scalars().all())

    async def offered_keys(self) -> set[str]:
        """Every model the deployment offers, as the ``provider:model`` key pricing uses.

        Across every provider and both switch states, because the question this
        answers is which models the hosted-providers surface is responsible for,
        not which ones it is serving right now.
        """
        result = await self.db.execute(select(col(HostedProviderModel.provider), col(HostedProviderModel.model)))
        return {f"{provider}:{model}" for provider, model in result.all()}

    async def enabled_models_for_providers(self, providers: Collection[str]) -> dict[str, set[str]]:
        """The served models of each named provider, in one statement.

        A provider absent from the result serves nothing under it, which the
        caller reads as "advertises no model" rather than "unnarrowed": a hosted
        provider with no rows is served but not listed.
        """
        wanted = set(providers)
        if not wanted:
            return {}
        result = await self.db.execute(
            select(col(HostedProviderModel.provider), col(HostedProviderModel.model)).where(
                col(HostedProviderModel.provider).in_(wanted),
                col(HostedProviderModel.enabled).is_(True),
            )
        )
        served: dict[str, set[str]] = {}
        for provider, model in result.all():
            served.setdefault(provider, set()).add(model)
        return served

    async def create_many(self, rows: Sequence[HostedProviderModel]) -> Sequence[HostedProviderModel]:
        """Stage several offered rows at once. The caller owns the transaction.

        Flushed here rather than left to the commit, so the unique index answers
        while the caller can still say which model it refused.

        Raises:
            HostedModelConflict: one of these models is already offered here.
        """
        if not rows:
            return []
        self.db.add_all(rows)
        try:
            await self.db.flush()
        except IntegrityError as exc:
            raise HostedModelConflict(rows[0].model) from exc
        return rows

    async def save(self, row: HostedProviderModel) -> HostedProviderModel:
        """Flush a changed row and read back what the database filled in; see `HostedProviderRepository.save`."""
        self.db.add(row)
        await self.db.flush()
        await self.db.refresh(row)
        return row

    async def flush(self) -> None:
        """Flush rows edited in place, in one round trip rather than one per row."""
        await self.db.flush()

    async def delete_row(self, row: HostedProviderModel) -> None:
        """Stop offering one model. The caller owns the transaction."""
        await self.db.delete(row)
        await self.db.flush()

    async def delete_for_provider(self, provider: str) -> int:
        """Remove every offered row of one provider, returning how many went.

        What the database would do with a foreign key, done here instead,
        because the roster is keyed on the provider's name so it can outlive a
        row (a provider declared in configuration has none). One statement
        rather than a delete per row: a provider can offer several hundred
        models.
        """
        result = cast(
            CursorResult[Any],
            await self.db.execute(delete(HostedProviderModel).where(col(HostedProviderModel.provider) == provider)),
        )
        await self.db.flush()
        return result.rowcount
