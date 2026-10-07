"""Data access for the deployment's hosted providers and the models offered on them.

Built on the Unit of Work. Flushes, never commits. Creation goes through
bespoke methods rather than the inherited generic one: the plaintext key never
lands in a column, so the service hands these repositories encrypted payloads.
"""

import uuid
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Never, cast

from sqlalchemy import CursorResult, delete, func, select
from sqlalchemy.exc import IntegrityError

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.providers import HostedProvider, HostedProviderModel
from gateway.repositories.base_repository import BaseRepository

# One row per provider, and any-llm knows a bounded set of them, so a full
# listing never pages.
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
    """The unique constraint refused a model already offered on this provider."""

    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.model = model


class HostedProviderRepository(BaseRepository[HostedProvider, Never, Never]):
    """Repository for `hosted_providers` rows."""

    def __init__(self, uow: UnitOfWork):
        super().__init__(uow, HostedProvider)

    async def get_by_provider(self, provider: str) -> HostedProvider | None:
        result = await self.db.execute(select(HostedProvider).where(HostedProvider.provider == provider))
        return result.scalar_one_or_none()

    async def list_page(self, *, skip: int = 0, limit: int = 100) -> tuple[Sequence[HostedProvider], int]:
        """One page of hosted providers by name, and how many there are in total."""
        rows = (
            (await self.db.execute(select(HostedProvider).order_by(HostedProvider.provider).offset(skip).limit(limit)))
            .scalars()
            .all()
        )
        total = (await self.db.execute(select(func.count()).select_from(HostedProvider))).scalar_one()
        return rows, total

    async def list_all(self) -> Sequence[HostedProvider]:
        """Every hosted provider, bounded by the number of providers any-llm knows."""
        result = await self.db.execute(select(HostedProvider).order_by(HostedProvider.provider).limit(_ALL_PROVIDERS))
        return result.scalars().all()

    async def insert(self, **fields: Any) -> HostedProvider:
        """Stage a new hosted provider.

        Flushed here so the unique constraint answers while the caller can
        still say which provider it refused.

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
        """Flush a changed row and read back what the database filled in."""
        self.db.add(row)
        await self.db.flush()
        await self.db.refresh(row)
        return row

    async def delete_row(self, row: HostedProvider) -> None:
        await self.db.delete(row)
        await self.db.flush()


class HostedProviderModelRepository(BaseRepository[HostedProviderModel, Never, Never]):
    """Repository for `hosted_provider_models` rows."""

    def __init__(self, uow: UnitOfWork):
        super().__init__(uow, HostedProviderModel)

    async def get_in_provider(self, model_id: uuid.UUID, provider: str) -> HostedProviderModel | None:
        """One offered model by id, scoped to the provider it must belong to."""
        result = await self.db.execute(
            select(HostedProviderModel).where(
                HostedProviderModel.id == model_id, HostedProviderModel.provider == provider
            )
        )
        return result.scalar_one_or_none()

    async def get_by_model(self, provider: str, model: str) -> HostedProviderModel | None:
        result = await self.db.execute(
            select(HostedProviderModel).where(
                HostedProviderModel.provider == provider, HostedProviderModel.model == model
            )
        )
        return result.scalar_one_or_none()

    async def list_for_provider(
        self, provider: str, *, skip: int = 0, limit: int = 500
    ) -> tuple[Sequence[HostedProviderModel], int]:
        """One page of a provider's offered models, and how many there are in total."""
        rows = (
            (
                await self.db.execute(
                    select(HostedProviderModel)
                    .where(HostedProviderModel.provider == provider)
                    .order_by(HostedProviderModel.model)
                    .offset(skip)
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        total = (
            await self.db.execute(
                select(func.count()).select_from(HostedProviderModel).where(HostedProviderModel.provider == provider)
            )
        ).scalar_one()
        return rows, total

    async def list_all_for_provider(self, provider: str) -> Sequence[HostedProviderModel]:
        """Every offered row on one provider, for the pass that reprices them."""
        result = await self.db.execute(
            select(HostedProviderModel)
            .where(HostedProviderModel.provider == provider)
            .order_by(HostedProviderModel.model)
        )
        return result.scalars().all()

    async def names_for_provider(self, provider: str) -> set[str]:
        result = await self.db.execute(
            select(HostedProviderModel.model).where(HostedProviderModel.provider == provider)
        )
        return set(result.scalars().all())

    async def offered_keys(self) -> set[str]:
        """Every offered model as a ``provider:model`` key, across every provider and both switch states."""
        result = await self.db.execute(select(HostedProviderModel.provider, HostedProviderModel.model))
        return {f"{provider}:{model}" for provider, model in result.all()}

    async def enabled_models_for_providers(self, providers: Collection[str]) -> dict[str, set[str]]:
        """The served models of each named provider. A provider absent from the result serves none."""
        wanted = set(providers)
        if not wanted:
            return {}
        result = await self.db.execute(
            select(HostedProviderModel.provider, HostedProviderModel.model).where(
                HostedProviderModel.provider.in_(wanted), HostedProviderModel.enabled.is_(True)
            )
        )
        served: dict[str, set[str]] = {}
        for provider, model in result.all():
            served.setdefault(provider, set()).add(model)
        return served

    async def create_many(
        self, provider: str, names: Sequence[str], *, enabled: Mapping[str, bool]
    ) -> Sequence[HostedProviderModel]:
        """Stage one offered row per name, switched on where ``enabled`` says so.

        Raises:
            HostedModelConflict: one of these models is already offered here.
        """
        if not names:
            return []
        rows = [HostedProviderModel(provider=provider, model=name, enabled=enabled.get(name, True)) for name in names]
        self.db.add_all(rows)
        try:
            await self.db.flush()
        except IntegrityError as exc:
            raise HostedModelConflict(names[0]) from exc
        return rows

    async def save(self, row: HostedProviderModel) -> HostedProviderModel:
        """Flush a changed row and read back what the database filled in."""
        self.db.add(row)
        await self.db.flush()
        await self.db.refresh(row)
        return row

    async def flush(self) -> None:
        """Flush rows edited in place, in one round trip rather than one per row."""
        await self.db.flush()

    async def delete_row(self, row: HostedProviderModel) -> None:
        await self.db.delete(row)
        await self.db.flush()

    async def delete_for_provider(self, provider: str) -> int:
        """Remove every offered row of one provider, returning how many went.

        What the database would do with a foreign key, done here instead,
        because the roster is keyed on the provider's name so it can outlive a
        row (a provider declared in configuration has none).
        """
        result = cast(
            CursorResult[Any],
            await self.db.execute(delete(HostedProviderModel).where(HostedProviderModel.provider == provider)),
        )
        await self.db.flush()
        return result.rowcount
