"""The country file an operator uploads, stored in the database so every replica reads the same one."""

import hashlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.inference_exceptions import (
    CountryDatabaseInvalidError,
    CountryDatabaseNotFoundError,
    CountryDatabaseTooLargeError,
)
from gateway.repositories.inference import CountryDatabaseRepository
from gateway.schemas.inference import CountryDatabaseStatus, StoredCountryDatabasePublic
from gateway.services.inference._client_country import CountryDatabaseError, inspect_country_database

# GeoLite2 Country and DB-IP Lite Country are each under 10 MiB.
COUNTRY_DATABASE_MAX_BYTES = 32 * 1024 * 1024


async def _read_upload(chunks: AsyncIterator[bytes]) -> bytes:
    content = bytearray()
    async for chunk in chunks:
        content += chunk
        if len(content) > COUNTRY_DATABASE_MAX_BYTES:
            raise CountryDatabaseTooLargeError(COUNTRY_DATABASE_MAX_BYTES)
    return bytes(content)


class CountryDatabaseStore:
    """Reads, replaces and removes the stored country file."""

    def __init__(self, uow: UnitOfWork, repository: CountryDatabaseRepository, config: GatewayConfig) -> None:
        self._uow = uow
        self._repository = repository
        self._config = config

    async def status(self) -> CountryDatabaseStatus:
        async with self._uow:
            row = await self._repository.find_summary()
            stored = (
                None
                if row is None
                else StoredCountryDatabasePublic(
                    database_type=row.database_type,
                    built_at=row.built_at,
                    size_bytes=row.size_bytes,
                    sha256=row.sha256,
                    uploaded_at=row.uploaded_at,
                )
            )
        return CountryDatabaseStatus(
            stored=stored, client_country_database_set=self._config.client_country_database is not None
        )

    async def store(self, chunks: AsyncIterator[bytes]) -> CountryDatabaseStatus:
        content = await _read_upload(chunks)
        try:
            inspected = inspect_country_database(content)
        except CountryDatabaseError as exc:
            raise CountryDatabaseInvalidError(str(exc)) from exc
        async with self._uow:
            await self._repository.replace(
                content=content,
                database_type=inspected.database_type,
                built_at=inspected.built_at,
                sha256=hashlib.sha256(content).hexdigest(),
                uploaded_at=datetime.now(UTC),
            )
        return await self.status()

    async def remove(self) -> None:
        async with self._uow:
            if not await self._repository.remove():
                raise CountryDatabaseNotFoundError
