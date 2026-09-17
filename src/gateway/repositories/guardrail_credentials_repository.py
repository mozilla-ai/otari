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

import uuid
from collections.abc import Sequence

from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.models.guardrails import GuardrailCredential, GuardrailCredentialWorkspace
from gateway.models.tenancy import Workspace

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


MAX_ENFORCED_GUARDRAILS = 10
"""Ceiling on how many definitions may be enabled at once.

Not a storage bound like the one above: an enabled definition runs before every
request of the workspaces it covers, and the checks run one after another, so
this bounds added latency rather than table size. Ten is
``MAX_GUARDRAILS_PER_ORGANIZATION``, which bounds the same cost on the layer
above for the same reason."""


async def count_enabled_guardrail_credentials(db: AsyncSession, *, excluding: str | None = None) -> int:
    """How many definitions are enabled, optionally ignoring one by name.

    ``excluding`` is the row a write is about to change, so an update that leaves
    it enabled is not counted against itself.
    """
    stmt = select(func.count()).select_from(GuardrailCredential).where(GuardrailCredential.enabled.is_(True))
    if excluding is not None:
        stmt = stmt.where(GuardrailCredential.name != excluding)
    return int((await db.execute(stmt)).scalar_one())


async def missing_workspace_ids(db: AsyncSession, workspace_ids: Sequence[uuid.UUID]) -> set[uuid.UUID]:
    """The ids of ``workspace_ids`` that no workspace row carries."""
    if not workspace_ids:
        return set()
    wanted = set(workspace_ids)
    stmt = select(col(Workspace.id)).where(col(Workspace.id).in_(wanted))
    return wanted - set((await db.execute(stmt)).scalars().all())


async def workspace_ids_by_credential(db: AsyncSession, *, name: str | None = None) -> dict[str, list[uuid.UUID]]:
    """Definition scopes keyed by name, for a listing that must not fan out.

    ``name`` narrows it to one row, which is what the single-row reads use.
    """
    stmt = select(GuardrailCredentialWorkspace.credential_name, GuardrailCredentialWorkspace.workspace_id).order_by(
        GuardrailCredentialWorkspace.credential_name, GuardrailCredentialWorkspace.workspace_id
    )
    if name is not None:
        stmt = stmt.where(GuardrailCredentialWorkspace.credential_name == name)
    scoped: dict[str, list[uuid.UUID]] = {}
    for name, workspace_id in (await db.execute(stmt)).all():
        scoped.setdefault(name, []).append(workspace_id)
    return scoped


async def replace_guardrail_credential_workspaces(
    db: AsyncSession, *, name: str, workspace_ids: Sequence[uuid.UUID]
) -> None:
    """Set one definition's scope to exactly ``workspace_ids``."""
    await db.execute(delete(GuardrailCredentialWorkspace).where(GuardrailCredentialWorkspace.credential_name == name))
    for workspace_id in workspace_ids:
        db.add(GuardrailCredentialWorkspace(credential_name=name, workspace_id=workspace_id))
    await db.flush()


async def list_enforced_guardrail_credentials(
    db: AsyncSession, *, workspace_id: uuid.UUID
) -> list[GuardrailCredential]:
    """The enabled definitions that check one workspace's requests, ordered by name.

    One indexed read, on every request that reaches a completion endpoint in
    standalone mode. Deliberately not cached, for the reason
    ``resolve_organization_guardrails`` gives about the layer above: an operator
    who turns a guardrail on expects the next request to run it, not the next
    process.

    Ordered by name so a request's guardrails run in a stable order and a test
    can assert one.
    """
    stmt = (
        select(GuardrailCredential)
        .outerjoin(
            GuardrailCredentialWorkspace,
            and_(
                GuardrailCredentialWorkspace.credential_name == GuardrailCredential.name,
                GuardrailCredentialWorkspace.workspace_id == workspace_id,
            ),
        )
        .where(
            GuardrailCredential.enabled.is_(True),
            or_(
                GuardrailCredential.applies_to_all_workspaces.is_(True),
                GuardrailCredentialWorkspace.workspace_id.is_not(None),
            ),
        )
        .order_by(GuardrailCredential.name)
    )
    return list((await db.execute(stmt)).scalars().all())
