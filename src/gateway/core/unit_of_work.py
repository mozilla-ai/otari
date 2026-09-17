"""The Unit of Work: where a business transaction starts, commits and rolls back.

Each request, and each worker job, has one Unit of Work over its one session.
A business step is one ``async with uow:`` block, and its transaction ends when the block does.
See Martin Fowler, *Patterns of Enterprise Application Architecture* (2002),
https://martinfowler.com/eaaCatalog/unitOfWork.html.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Self

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.database import DATABASE_ERRORS, create_log_session, create_session
from gateway.log_config import logger


class OutsideUnitOfWorkError(RuntimeError):
    """The session was asked for while no block of its Unit of Work was open."""


class UnitOfWorkRolledBackError(RuntimeError):
    """A step was rolled back because a block inside it failed, although the error was caught."""


class UnitOfWork:
    """One business transaction at a time over one session.

    A block commits when it ends, and rolls back and re-raises when it ends on an error.
    Blocks nest: an inner block joins the outer one, and only the outermost block commits,
    so a step that writes to two domains is atomic.
    A step in which an inner block failed is rolled back as a whole,
    even when the code around that block caught the error.
    Its outermost block then raises ``UnitOfWorkRolledBackError`` from the first such failure.
    A Unit of Work belongs to one task, because blocks that concurrent tasks open on it would commit each other's steps.

    NOTE: only a service should open a block.
    Repositories flush and never commit, and routes never open a block.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._depth = 0
        self._first_failure: BaseException | None = None

    async def __aenter__(self) -> Self:
        self._depth += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._depth -= 1
        if exc is not None and self._first_failure is None:
            self._first_failure = exc
        if self._depth > 0:
            return

        failure, self._first_failure = self._first_failure, None
        if failure is not None:
            await self._roll_back()
            if exc is None:
                msg = "The step was rolled back because a step inside it failed"
                raise UnitOfWorkRolledBackError(msg) from failure
            return

        try:
            await self._session.commit()
        except BaseException:
            await self._roll_back()
            raise

    async def _roll_back(self) -> None:
        """Roll back, leaving the error that ended the step as the one the caller sees."""
        try:
            await self._session.rollback()
        except DATABASE_ERRORS:
            logger.warning("Could not roll back a unit of work", exc_info=True)


def session_for(uow: UnitOfWork) -> AsyncSession:
    """Give a repository the session of the Unit of Work's open block.

    NOTE: only a repository should call this.
    Other code must reach the database through a repository, so that every query stays in the repository layer.

    Raises:
        OutsideUnitOfWorkError: no block is open.
    """
    if uow._depth == 0:
        msg = "The database session was used outside a unit of work block"
        raise OutsideUnitOfWorkError(msg)
    return uow._session


@asynccontextmanager
async def create_unit_of_work() -> AsyncIterator[UnitOfWork]:
    """Yield a Unit of Work for a worker job, on the request pool."""
    async with create_session() as session:
        yield UnitOfWork(session)


@asynccontextmanager
async def create_log_unit_of_work() -> AsyncIterator[UnitOfWork]:
    """Yield a Unit of Work for a usage-log job, on the metering pool."""
    async with create_log_session() as session:
        yield UnitOfWork(session)


__all__ = [
    "OutsideUnitOfWorkError",
    "UnitOfWork",
    "UnitOfWorkRolledBackError",
    "create_log_unit_of_work",
    "create_unit_of_work",
    "session_for",
]
