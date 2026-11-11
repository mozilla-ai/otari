"""The lock modes the organization row is read under, pinned as rendered SQL.

Each mode is chosen for what it must not block, which no behavioral test states in one place: the
test that catches a mode too strong does so by timing out after two minutes, with nothing in the
failure naming the lock. These compile the statements and read the mode off them.
"""

import uuid
from collections.abc import Awaitable, Callable
from typing import Any, cast

import pytest
from sqlalchemy import Select
from sqlalchemy.dialects.postgresql.base import PGDialect
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.repositories.tenancy.organization_repository import OrganizationRepository


class _CapturingSession:
    """Records the statement it is asked to execute, and answers nothing."""

    def __init__(self) -> None:
        self.statements: list[Select[Any]] = []

    async def execute(self, statement: Select[Any], *_args: Any, **_kwargs: Any) -> None:
        self.statements.append(statement)
        return None


async def _rendered(lock: Callable[[OrganizationRepository, uuid.UUID], Awaitable[None]]) -> str:
    session = _CapturingSession()
    repository = OrganizationRepository(cast("AsyncSession", session))
    await lock(repository, uuid.uuid4())
    # SQLAlchemy does not type its dialect constructors.
    dialect = PGDialect()  # type: ignore[no-untyped-call]
    return str(session.statements[0].compile(dialect=dialect)).replace("\n", " ")


@pytest.mark.asyncio
async def test_the_settings_lock_does_not_block_a_row_referencing_the_organization() -> None:
    """A foreign key takes ``FOR KEY SHARE`` on the parent, which ``FOR UPDATE`` would conflict with.

    Taken as ``FOR UPDATE``, this lock stops another transaction inserting a connection, a
    membership or a workspace for the organization until the settings write commits.
    """
    assert "FOR NO KEY UPDATE" in await _rendered(OrganizationRepository.lock_settings)


@pytest.mark.asyncio
async def test_the_invariant_lock_holds_off_a_row_referencing_the_organization() -> None:
    """The counts this one guards are over child rows, so it has to block their insertion."""
    rendered = await _rendered(OrganizationRepository.lock)
    assert rendered.endswith("FOR UPDATE")
