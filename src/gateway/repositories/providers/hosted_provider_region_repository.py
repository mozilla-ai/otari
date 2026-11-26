"""Data access for the regions of the deployment's hosted providers.

Built on the Unit of Work. Flushes, never commits. As for the providers
themselves, the service hands in ciphertexts, so no plaintext key reaches a column.
"""

import uuid
from collections import defaultdict
from collections.abc import Collection, Sequence
from typing import Any, Never, cast

from sqlalchemy import CursorResult, inspect, select, update
from sqlalchemy.exc import IntegrityError

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.providers import HostedProviderRegion
from gateway.repositories.base_repository import BaseRepository

_SLUG_CONSTRAINT = "uq_hosted_provider_regions_provider_slug"


class HostedRegionConflict(Exception):
    """The unique constraint refused a slug the provider already has a region under.

    Raised in place of ``IntegrityError`` because the service that maps it to
    a domain error may not import a database library.
    """

    def __init__(self, slug: str) -> None:
        super().__init__(slug)
        self.slug = slug


class HostedProviderRegionRepository(BaseRepository[HostedProviderRegion, Never, Never]):
    """Repository for `hosted_provider_regions` rows."""

    def __init__(self, uow: UnitOfWork):
        super().__init__(uow, HostedProviderRegion)

    async def list_for_provider(self, provider_id: uuid.UUID, *, fresh: bool = False) -> list[HostedProviderRegion]:
        """One provider's regions, in the order they are tried.

        ``fresh`` overwrites what this session already holds of them, for a
        caller that just locked the provider and must see what committed since.
        """
        statement = (
            select(HostedProviderRegion)
            .where(HostedProviderRegion.hosted_provider_id == provider_id)
            .order_by(HostedProviderRegion.position, HostedProviderRegion.slug)
        )
        if fresh:
            statement = statement.execution_options(populate_existing=True)
        result = await self.db.execute(statement)
        return list(result.scalars().all())

    async def list_for_providers(
        self, provider_ids: Collection[uuid.UUID]
    ) -> dict[uuid.UUID, list[HostedProviderRegion]]:
        """The regions of each of ``provider_ids`` that has any, each list in the order they are tried."""
        if not provider_ids:
            return {}
        result = await self.db.execute(
            select(HostedProviderRegion)
            .where(HostedProviderRegion.hosted_provider_id.in_(provider_ids))
            .order_by(HostedProviderRegion.position, HostedProviderRegion.slug)
        )
        regions: dict[uuid.UUID, list[HostedProviderRegion]] = defaultdict(list)
        for row in result.scalars().all():
            regions[row.hosted_provider_id].append(row)
        return dict(regions)

    async def list_for_providers_detached(
        self, provider_ids: Collection[uuid.UUID]
    ) -> dict[uuid.UUID, list[HostedProviderRegion]]:
        """:meth:`list_for_providers` as objects no session holds, for a caller that keeps them past its transaction."""
        columns = [attribute.key for attribute in inspect(HostedProviderRegion).column_attrs]
        return {
            provider_id: [HostedProviderRegion(**{column: getattr(row, column) for column in columns}) for row in rows]
            for provider_id, rows in (await self.list_for_providers(provider_ids)).items()
        }

    async def list_encrypted(self) -> Sequence[HostedProviderRegion]:
        """Every region holding a ciphertext of its own, for a key rotation."""
        result = await self.db.execute(
            select(HostedProviderRegion).where(
                (HostedProviderRegion.encrypted_api_key.is_not(None))
                | (HostedProviderRegion.encrypted_client_args.is_not(None))
            )
        )
        return result.scalars().all()

    async def swap_ciphertexts(
        self,
        row_id: uuid.UUID,
        *,
        expected: tuple[str | None, str | None],
        replacement: tuple[str | None, str | None],
    ) -> bool:
        """Replace a region's key and extras ciphertexts if they still hold ``expected``.

        A conditional UPDATE, so an edit committed while a rotation ran is not
        overwritten. Returns whether the row was written.
        """
        key, client_args = expected
        result = cast(
            CursorResult[Any],
            await self.db.execute(
                update(HostedProviderRegion)
                .where(
                    HostedProviderRegion.id == row_id,
                    HostedProviderRegion.encrypted_api_key.is_(None)
                    if key is None
                    else HostedProviderRegion.encrypted_api_key == key,
                    HostedProviderRegion.encrypted_client_args.is_(None)
                    if client_args is None
                    else HostedProviderRegion.encrypted_client_args == client_args,
                )
                .values(encrypted_api_key=replacement[0], encrypted_client_args=replacement[1])
                .execution_options(synchronize_session=False)
            ),
        )
        return result.rowcount == 1

    async def insert(self, **fields: Any) -> HostedProviderRegion:
        """Stage a new region, flushed so the unique constraint answers here.

        Raises:
            HostedRegionConflict: the provider already has a region of that slug.
        """
        row = HostedProviderRegion(**fields)
        self.db.add(row)
        try:
            await self.db.flush()
        except IntegrityError as exc:
            if _SLUG_CONSTRAINT not in str(exc.orig):
                raise
            raise HostedRegionConflict(row.slug) from exc
        await self.db.refresh(row)
        return row

    async def save(self, row: HostedProviderRegion) -> HostedProviderRegion:
        """Flush a changed row and read back what the database filled in."""
        self.db.add(row)
        await self.db.flush()
        await self.db.refresh(row)
        return row

    async def flush(self) -> None:
        """Write what the caller changed on rows this session holds, in one flush."""
        await self.db.flush()

    async def delete_row(self, row: HostedProviderRegion) -> None:
        await self.db.delete(row)
        await self.db.flush()
