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

from gateway.core.database import create_log_session, create_session


class OutsideUnitOfWorkError(RuntimeError):
    """The session was asked for while no block of its Unit of Work was open."""


class UnitOfWorkRolledBackError(RuntimeError):
    """A step was rolled back because a block inside it failed, although the error was caught."""


class UnitOfWork:
    """One business transaction at a time over one session.

    A block commits when it ends, and rolls back and re-raises when it ends on an error.
    Blocks nest: an inner block joins the outer one, and only the outermost block commits,
    so a step that writes to two domains is atomic.
    A step in which any inner block failed is rolled back as a whole,
    even when the code around that block caught the error.

    NOTE: only a service should open a block.
    Repositories flush and never commit, and routes never open a block.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._depth = 0
        self._failed = False

    @property
    def session(self) -> AsyncSession:
        """The session for repositories to run on inside a block.

        Raises:
            OutsideUnitOfWorkError: no block is open.
        """
        if self._depth == 0:
            msg = "The database session was used outside a unit of work block"
            raise OutsideUnitOfWorkError(msg)
        return self._session

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
        if exc is not None:
            self._failed = True
        if self._depth > 0:
            return

        failed, self._failed = self._failed, False
        if failed:
            await self._session.rollback()
            if exc is None:
                msg = "The step was rolled back because a step inside it failed"
                raise UnitOfWorkRolledBackError(msg)
            return

        try:
            await self._session.commit()
        except BaseException:
            await self._session.rollback()
            raise


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
]
