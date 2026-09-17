"""A Unit of Work commits a business step once, when its outermost block ends."""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Annotated, Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import event, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel

from gateway.api.deps import get_unit_of_work
from gateway.core.config import GatewayConfig
from gateway.core.database import create_session, get_db, init_db, reset_db
from gateway.core.unit_of_work import (
    OutsideUnitOfWorkError,
    UnitOfWork,
    UnitOfWorkRolledBackError,
    create_log_unit_of_work,
    create_unit_of_work,
    session_for,
)
from gateway.models.tenancy import OAuthPendingState
from gateway.repositories.base_repository import BaseRepository


@pytest_asyncio.fixture
async def notes_database(tmp_path: Path) -> AsyncIterator[None]:
    reset_db()
    init_db(GatewayConfig(database_url=f"sqlite+aiosqlite:///{tmp_path / 'uow.db'}", auto_migrate=False))
    async with create_session() as session:
        await session.execute(text("CREATE TABLE notes (body TEXT NOT NULL)"))
        await session.commit()
    yield
    reset_db()


async def _write(session: AsyncSession, body: str) -> None:
    await session.execute(text("INSERT INTO notes (body) VALUES (:body)"), {"body": body})


async def _committed_bodies() -> list[str]:
    async with create_session() as session:
        rows = await session.execute(text("SELECT body FROM notes ORDER BY body"))
        return list(rows.scalars().all())


def _count_commits(session: AsyncSession) -> Callable[[], int]:
    commits = 0

    def _on_commit(_session: Any) -> None:
        nonlocal commits
        commits += 1

    event.listen(session.sync_session, "after_commit", _on_commit)
    return lambda: commits


@pytest.mark.asyncio
async def test_a_block_commits_when_it_ends(notes_database: None) -> None:
    async with create_unit_of_work() as uow:
        async with uow:
            await _write(session_for(uow), "a")

        assert await _committed_bodies() == ["a"]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ValueError("step failed"), asyncio.CancelledError()], ids=["error", "cancel"])
async def test_a_block_rolls_back_and_reraises_on_an_error(notes_database: None, error: BaseException) -> None:
    async with create_unit_of_work() as uow:
        with pytest.raises(type(error)):
            async with uow:
                session = session_for(uow)
                await _write(session, "a")
                raise error

        assert not session.in_transaction(), "the block ended with its transaction still open"
        assert await _committed_bodies() == []


@pytest.mark.asyncio
async def test_a_failed_commit_rolls_back_and_reraises(notes_database: None, monkeypatch: pytest.MonkeyPatch) -> None:
    async with create_session() as session:
        uow = UnitOfWork(session)
        failure = OperationalError("COMMIT", None, Exception("connection lost"))
        monkeypatch.setattr(session, "commit", AsyncMock(side_effect=failure))
        rollback = AsyncMock(wraps=session.rollback)
        monkeypatch.setattr(session, "rollback", rollback)

        with pytest.raises(OperationalError):
            async with uow:
                await _write(session_for(uow), "a")

        rollback.assert_awaited_once()


def _fail_rollback(monkeypatch: pytest.MonkeyPatch, session: AsyncSession) -> None:
    failure = OperationalError("ROLLBACK", None, Exception("connection lost"))
    monkeypatch.setattr(session, "rollback", AsyncMock(side_effect=failure))


@pytest.mark.asyncio
async def test_a_failed_rollback_keeps_the_error_that_ended_the_block(
    notes_database: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with create_session() as session:
        uow = UnitOfWork(session)
        _fail_rollback(monkeypatch, session)

        with pytest.raises(ValueError, match="step failed"):
            async with uow:
                await _write(session_for(uow), "a")
                raise ValueError("step failed")


@pytest.mark.asyncio
async def test_a_failed_rollback_keeps_the_error_of_a_failed_commit(
    notes_database: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with create_session() as session:
        uow = UnitOfWork(session)
        commit_failure = OperationalError("COMMIT", None, Exception("connection lost"))
        monkeypatch.setattr(session, "commit", AsyncMock(side_effect=commit_failure))
        _fail_rollback(monkeypatch, session)

        with pytest.raises(OperationalError) as caught:
            async with uow:
                await _write(session_for(uow), "a")

        assert caught.value is commit_failure


@pytest.mark.asyncio
async def test_a_failed_rollback_still_reports_a_step_whose_inner_block_failed(
    notes_database: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with create_session() as session:
        uow = UnitOfWork(session)
        _fail_rollback(monkeypatch, session)

        with pytest.raises(UnitOfWorkRolledBackError):
            async with uow:
                with pytest.raises(ValueError, match="inner step failed"):
                    async with uow:
                        raise ValueError("inner step failed")


@pytest.mark.asyncio
async def test_a_nested_block_joins_the_outer_one_and_only_the_outermost_commits(notes_database: None) -> None:
    async with create_session() as session:
        uow = UnitOfWork(session)
        commits = _count_commits(session)
        async with uow:
            await _write(session_for(uow), "outer")
            async with uow:
                await _write(session_for(uow), "inner")
            assert commits() == 0
            assert await _committed_bodies() == []

        assert commits() == 1
        assert await _committed_bodies() == ["inner", "outer"]


@pytest.mark.asyncio
async def test_a_step_whose_inner_block_failed_is_rolled_back_even_when_the_error_is_caught(
    notes_database: None,
) -> None:
    async with create_unit_of_work() as uow:
        with pytest.raises(UnitOfWorkRolledBackError):
            async with uow:
                await _write(session_for(uow), "outer")
                with pytest.raises(ValueError, match="inner step failed"):
                    async with uow:
                        await _write(session_for(uow), "inner")
                        raise ValueError("inner step failed")

        assert await _committed_bodies() == []

        async with uow:
            await _write(session_for(uow), "next step")
        assert await _committed_bodies() == ["next step"]


@pytest.mark.asyncio
async def test_a_rolled_back_step_names_the_inner_failure_as_its_cause(notes_database: None) -> None:
    inner_failure = ValueError("inner step failed")
    async with create_unit_of_work() as uow:
        with pytest.raises(UnitOfWorkRolledBackError) as rolled_back:
            async with uow:
                with pytest.raises(ValueError, match="inner step failed"):
                    async with uow:
                        raise inner_failure

    assert rolled_back.value.__cause__ is inner_failure


@pytest.mark.asyncio
async def test_the_session_is_unavailable_outside_a_block(notes_database: None) -> None:
    async with create_unit_of_work() as uow:
        with pytest.raises(OutsideUnitOfWorkError):
            _ = session_for(uow)
        async with uow:
            pass
        with pytest.raises(OutsideUnitOfWorkError):
            _ = session_for(uow)


@pytest.mark.asyncio
async def test_a_repository_built_on_a_unit_of_work_raises_outside_a_block(notes_database: None) -> None:
    async with create_unit_of_work() as uow:
        async with uow:
            table = SQLModel.metadata.tables["oauth_pending_state"]
            await session_for(uow).run_sync(lambda sync: table.create(sync.connection()))
        repository: BaseRepository[OAuthPendingState, Any, Any] = BaseRepository(uow, OAuthPendingState)

        with pytest.raises(OutsideUnitOfWorkError):
            await repository.count()
        async with uow:
            assert await repository.count() == 0


@pytest.mark.asyncio
async def test_a_repository_built_on_a_session_needs_no_block(notes_database: None) -> None:
    async with create_session() as session:
        repository: BaseRepository[OAuthPendingState, Any, Any] = BaseRepository(session, OAuthPendingState)

        assert repository.db is session


@pytest.mark.asyncio
@pytest.mark.parametrize("helper", [create_unit_of_work, create_log_unit_of_work], ids=["request pool", "log pool"])
async def test_a_worker_helper_produces_a_unit_of_work(
    notes_database: None, helper: Callable[[], AbstractAsyncContextManager[UnitOfWork]]
) -> None:
    async with helper() as uow:
        assert isinstance(uow, UnitOfWork)
        async with uow:
            await _write(session_for(uow), "from a worker")

    assert await _committed_bodies() == ["from a worker"]


def test_the_request_dependency_produces_a_unit_of_work_over_the_request_session(tmp_path: Path) -> None:
    reset_db()
    init_db(GatewayConfig(database_url=f"sqlite+aiosqlite:///{tmp_path / 'uow.db'}", auto_migrate=False))
    app = FastAPI()

    @app.get("/probe")
    async def probe(
        db: Annotated[AsyncSession, Depends(get_db)],
        uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    ) -> dict[str, bool]:
        async with uow:
            return {"is_unit_of_work": isinstance(uow, UnitOfWork), "wraps_request_session": session_for(uow) is db}

    try:
        response = TestClient(app).get("/probe")
    finally:
        reset_db()

    assert response.json() == {"is_unit_of_work": True, "wraps_request_session": True}
