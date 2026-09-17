"""Reads and writes over ``guardrail_credentials``.

Module-level functions rather than a :class:`BaseRepository` subclass: that
generic is built around the SQLModel tenancy tables and their
``Create``/``Update`` schemas, and this is a declarative ``Base`` entity keyed by
a string whose writes are assembled by the service after it has split and
encrypted them.

Every write here flushes and never commits, as the repository layer does
everywhere: staging makes the change visible to the rest of the transaction
while the commit boundary stays with the service, which is the layer that knows
when a unit of work is complete.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.guardrails import GuardrailCredential

MAX_GUARDRAIL_CREDENTIALS = 500
"""Ceiling on one listing. The table is operator-authored and nothing like this
size, so the bound is here to keep the read from growing unbounded rather than to
paginate anything a deployment has today."""


async def list_guardrail_credentials(
    db: AsyncSession, *, limit: int = MAX_GUARDRAIL_CREDENTIALS
) -> list[GuardrailCredential]:
    """Stored guardrails ordered by name, bounded by ``limit``."""
    stmt = select(GuardrailCredential).order_by(GuardrailCredential.name).limit(limit)
    return list((await db.execute(stmt)).scalars().all())


async def list_encrypted_guardrail_credentials(db: AsyncSession) -> list[GuardrailCredential]:
    """Only the rows that carry a secret map, for a re-encryption pass."""
    stmt = select(GuardrailCredential).where(GuardrailCredential.encrypted_create_secrets.is_not(None))
    return list((await db.execute(stmt)).scalars().all())


async def get_guardrail_credential(db: AsyncSession, name: str) -> GuardrailCredential | None:
    """The stored guardrail called ``name``, or ``None``."""
    return await db.get(GuardrailCredential, name)


async def get_guardrail_credential_for_update(db: AsyncSession, name: str) -> GuardrailCredential | None:
    """Like :func:`get_guardrail_credential`, but locks the row ``FOR UPDATE``.

    So an optimistic-concurrency check and the write it guards run under one row
    lock, as the provider and search-tool stores do.
    """
    stmt = select(GuardrailCredential).where(GuardrailCredential.name == name).with_for_update()
    return (await db.execute(stmt)).scalar_one_or_none()


async def add_guardrail_credential(db: AsyncSession, row: GuardrailCredential) -> GuardrailCredential:
    """Stage a new row and flush it, so a unique-name collision surfaces here."""
    db.add(row)
    await db.flush()
    return row


async def delete_guardrail_credential(db: AsyncSession, row: GuardrailCredential) -> None:
    """Stage the row's removal and flush it."""
    await db.delete(row)
    await db.flush()
