"""The country file an operator uploads, stored in the database so every replica reads the same one.

Each process checks the stored SHA-256 on a timer and, when it changes, opens
the new file and swaps it in. A write is applied at once on the worker that
served it. ``client_country_database``, when set, wins, and the stored file is
not read at all.
"""

import asyncio
import hashlib
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime

from gateway.core.config import GatewayConfig
from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.exceptions.inference_exceptions import (
    CountryDatabaseInUseError,
    CountryDatabaseInvalidError,
    CountryDatabaseNotFoundError,
    CountryDatabaseTooLargeError,
)
from gateway.log_config import logger
from gateway.repositories.inference import CountryDatabaseRepository
from gateway.schemas.inference import CountryDatabaseStatus, StoredCountryDatabasePublic
from gateway.services.inference._client_country import (
    CountryDatabaseError,
    inspect_country_database,
    open_stored_country_database,
    reset_stored_country_database,
    stored_country_database_failed,
    stored_country_database_seen,
    use_stored_country_database,
)

# GeoLite2 Country and DB-IP Lite Country are each under 10 MiB.
COUNTRY_DATABASE_MAX_BYTES = 32 * 1024 * 1024
# How long another replica may look addresses up in the file stored before, the TTL the other dashboard overlays use.
COUNTRY_DATABASE_TTL_SECONDS = 30.0

# The enabled hosted providers that serve only some countries, by name.
CountryLimitedProviders = Callable[[], Awaitable[list[str]]]


async def _read_upload(chunks: AsyncIterator[bytes]) -> bytes:
    content = bytearray()
    async for chunk in chunks:
        content += chunk
        if len(content) > COUNTRY_DATABASE_MAX_BYTES:
            raise CountryDatabaseTooLargeError(COUNTRY_DATABASE_MAX_BYTES)
    return bytes(content)


async def refresh_stored_country_database(uow: UnitOfWork) -> bool:
    """Open the stored file when it is not the one open, returning whether a file is stored.

    A file that fails to open is logged once, and the file open before stays:
    with none, every address has no country, which a listed provider refuses.
    A temporary file that cannot be written is tried again on the next refresh.
    """
    async with uow:
        repository = CountryDatabaseRepository(uow)
        sha256 = await repository.find_sha256()
        if sha256 is not None and stored_country_database_seen(sha256):
            return True
        row = await repository.find() if sha256 is not None else None
        stored = (row.sha256, row.content) if row is not None else None
    if stored is None:
        use_stored_country_database(None)
        return False
    try:
        opened = await asyncio.to_thread(open_stored_country_database, *stored)
    except CountryDatabaseError as exc:
        stored_country_database_failed(stored[0])
        logger.warning("%s; keeping the country file open before, if any", exc)
        return True
    except OSError as exc:
        logger.warning(
            "The stored country file cannot be written to a temporary file (%s); keeping the country file open "
            "before, if any",
            type(exc).__name__,
        )
        return True
    use_stored_country_database(opened)
    return True


async def load_stored_country_database_at_startup(config: GatewayConfig) -> None:
    """Open the stored file before the first request.

    Raises:
        CountryDatabaseError: ``provider_supported_countries`` needs the stored file and none is stored.
    """
    reset_stored_country_database()
    if config.client_country_database is not None:
        return
    try:
        async with create_unit_of_work() as uow:
            stored = await refresh_stored_country_database(uow)
    except DATABASE_ERRORS as exc:
        logger.error(
            "Failed to load the stored country file (%s); every address has no country until it loads",
            describe_database_error(exc),
        )
        return
    except Exception:
        logger.exception("Failed to load the stored country file; every address has no country until it loads")
        return
    if not stored and config.country_gate_needs_stored_database():
        msg = (
            "provider_supported_countries needs client_country_header, client_country_database or a stored "
            "country file: without one no request has a country, and every listed provider would be refused"
        )
        raise CountryDatabaseError(msg)


async def run_stored_country_database_refresher(interval: float = COUNTRY_DATABASE_TTL_SECONDS) -> None:
    """Check the stored file forever, so another replica's upload arrives here within ``interval``."""
    while True:
        await asyncio.sleep(interval)
        try:
            async with create_unit_of_work() as uow:
                await refresh_stored_country_database(uow)
        except asyncio.CancelledError:
            raise
        except DATABASE_ERRORS as exc:
            logger.warning(
                "Stored country file refresh failed (%s); retrying in %ss", describe_database_error(exc), interval
            )
        except Exception:
            logger.warning("Stored country file refresh failed; retrying in %ss", interval, exc_info=True)


class CountryDatabaseStore:
    """Reads, replaces and removes the stored country file."""

    def __init__(
        self,
        uow: UnitOfWork,
        repository: CountryDatabaseRepository,
        config: GatewayConfig,
        *,
        country_limited_providers: CountryLimitedProviders,
    ) -> None:
        self._uow = uow
        self._repository = repository
        self._config = config
        self._country_limited_providers = country_limited_providers

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
        await self._apply()
        return await self.status()

    async def remove(self) -> None:
        config = self._config
        if config.country_gate_needs_stored_database():
            raise CountryDatabaseInUseError("provider_supported_countries")
        async with self._uow:
            if config.client_country_header is None and config.client_country_database is None:
                limited = await self._country_limited_providers()
                if limited:
                    raise CountryDatabaseInUseError(f"the countries of the hosted providers {', '.join(limited)}")
            if not await self._repository.remove():
                raise CountryDatabaseNotFoundError
        await self._apply()

    async def _apply(self) -> None:
        if self._config.client_country_database is None:
            await refresh_stored_country_database(self._uow)
