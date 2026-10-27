"""Data access for ``trace_guardrail_definitions``. Flushes, never commits; a stored version is never changed."""

import uuid
from collections.abc import Collection, Sequence
from typing import Any, Never

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from gateway.core.sql import dialect_name
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.traces import TraceGuardrailDefinition
from gateway.repositories.base_repository import BaseRepository

# The first key of the two-key advisory lock, so this lock's keyspace is its own.
_ADMISSION_LOCK_CLASS = 0x6F74_6764  # "otgd"


class GuardrailDefinitionRepository(BaseRepository[TraceGuardrailDefinition, Never, Never]):
    """Store each version of a workspace's agent guardrails once, and read versions back by fingerprint."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, TraceGuardrailDefinition)

    async def insert_absent(self, workspace_id: uuid.UUID, digest: str, gates: list[dict[str, Any]]) -> None:
        """Stage this version unless the workspace already holds it."""
        insert = postgresql_insert if dialect_name(self.db) == "postgresql" else sqlite_insert
        await self.db.execute(
            insert(TraceGuardrailDefinition)
            .values(workspace_id=workspace_id, digest=digest, gates=gates)
            .on_conflict_do_nothing(
                index_elements=[TraceGuardrailDefinition.workspace_id, TraceGuardrailDefinition.digest]
            )
        )
        await self.db.flush()

    async def lock_workspace(self, workspace_id: uuid.UUID) -> None:
        """Serialize the workspace's new versions until the transaction ends, so two cannot both pass the count.

        A transaction-scoped advisory lock rather than a row lock: the workspace row belongs to
        the tenancy domain. A no-op on SQLite, which admits one writer at a time anyway.
        """
        if dialect_name(self.db) != "postgresql":
            return
        await self.db.execute(
            select(func.pg_advisory_xact_lock(_ADMISSION_LOCK_CLASS, func.hashtext(str(workspace_id))))
        )

    async def count_for_workspace(self, workspace_id: uuid.UUID) -> int:
        result = await self.db.execute(
            select(func.count())
            .select_from(TraceGuardrailDefinition)
            .where(TraceGuardrailDefinition.workspace_id == workspace_id)
        )
        return result.scalar_one()

    async def for_digests(
        self, workspace_id: uuid.UUID, digests: Collection[str]
    ) -> Sequence[TraceGuardrailDefinition]:
        if not digests:
            return ()
        result = await self.db.execute(
            select(TraceGuardrailDefinition).where(
                TraceGuardrailDefinition.workspace_id == workspace_id,
                TraceGuardrailDefinition.digest.in_(list(digests)),
            )
        )
        return result.scalars().all()
