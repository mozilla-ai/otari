"""Tenancy timestamps stay UTC-aware on SQLite, the OSS edition's default engine.

``DateTime(timezone=True)`` is honored by PostgreSQL and is a no-op on SQLite,
which has no timestamp type: SQLAlchemy stores an ISO string and hands it back
naive. Every integration test runs on PostgreSQL, so the engine where the flag
does nothing had no coverage, while ``models/tenancy.py``'s own docstring lists
fixing the platform's naive-timestamp bug as a deliberate departure.

It matters on the wire rather than in the database. A naive datetime serializes
with no offset, and ``new Date("2026-08-18T23:47:00")`` in a browser is **local**
time, so the tenancy pages would render every timestamp shifted by the
deployment's UTC offset.

Driven through the models rather than through the type in isolation, because
what is under test is the whole path: the mixin, the column type, SQLite's
storage, and the ``Public`` schema a route answers with.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the shared metadata)
from gateway.models.tenancy import Organization, OrganizationPublic

T = TypeVar("T")

# Five hours behind UTC, so a wall-clock time stored without its offset is wrong
# by a visible amount rather than by something a reader could miss.
_ELSEWHERE = timezone(timedelta(hours=-5))


def _run(scenario: Callable[[AsyncSession], Awaitable[T]]) -> T:
    """Create the schema on a fresh in-memory SQLite database and run one scenario."""

    async def main() -> T:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            return await scenario(session)

    return asyncio.run(main())


async def _round_trip(session: AsyncSession, **fields: object) -> Organization:
    """Store an organization and read it back off the database, not the identity map."""
    session.add(Organization(name="Acme", slug="acme", **fields))
    await session.commit()
    session.expunge_all()
    return (await session.execute(select(Organization))).scalar_one()


def test_a_stored_timestamp_reads_back_utc_aware() -> None:
    stored = _run(_round_trip)

    assert stored.created_at.tzinfo is not None
    assert stored.created_at.utcoffset() == timedelta(0)


def test_the_wire_form_carries_an_offset() -> None:
    """The half a browser sees. Without one, it parses the value as local time."""
    stored = _run(_round_trip)

    created = json.loads(OrganizationPublic.model_validate(stored).model_dump_json())["created_at"]

    assert created.endswith("Z") or created.endswith("+00:00"), created


def test_a_value_written_in_another_zone_is_normalized_to_utc() -> None:
    """The bind direction, which SQLite gets wrong in the opposite way.

    Handed an aware value it stores the *wall clock* and drops the offset, so
    08:00-05:00 would read back as 08:00 UTC: the same instant reported five
    hours early rather than converted.
    """
    written = datetime(2026, 8, 18, 8, 0, tzinfo=_ELSEWHERE)

    stored = _run(lambda session: _round_trip(session, created_at=written))

    assert stored.created_at == written
    assert stored.created_at.hour == 13, "13:00 UTC is what 08:00-05:00 means"
