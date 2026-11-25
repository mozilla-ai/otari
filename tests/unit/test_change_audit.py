"""A committed change to an audited model is recorded on the request's span, and nothing else is."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import ClassVar

from opentelemetry import trace
from opentelemetry.trace import INVALID_SPAN_CONTEXT, NonRecordingSpan
from opentelemetry.util.types import Attributes
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from gateway.core.change_audit import SETTING_CHANGED_EVENT, UNLOADED, install_change_audit


class _Base(DeclarativeBase):
    pass


class _Setting(_Base):
    __tablename__ = "audited_setting"
    __audited__: ClassVar[tuple[str, ...]] = ("enabled",)

    id: Mapped[int] = mapped_column(primary_key=True)
    enabled: Mapped[bool] = mapped_column(default=False)
    secret: Mapped[str] = mapped_column(default="")


class _Unaudited(_Base):
    __tablename__ = "unaudited_setting"

    id: Mapped[int] = mapped_column(primary_key=True)
    enabled: Mapped[bool] = mapped_column(default=False)


class _RecordingSpan(NonRecordingSpan):
    def __init__(self) -> None:
        super().__init__(INVALID_SPAN_CONTEXT)
        self.events: list[tuple[str, Attributes]] = []

    def add_event(self, name: str, attributes: Attributes = None, timestamp: int | None = None) -> None:
        self.events.append((name, attributes))


def _run(scenario: Callable[[AsyncSession], Awaitable[None]]) -> list[tuple[str, Attributes]]:
    """Run one scenario on a fresh in-memory database inside a span, and return the span's events."""
    install_change_audit()
    span = _RecordingSpan()

    async def main() -> None:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        try:
            async with engine.begin() as conn:
                await conn.run_sync(_Base.metadata.create_all)
            async with async_sessionmaker(engine, expire_on_commit=False)() as session:
                session.add_all([_Setting(id=1), _Unaudited(id=1)])
                await session.commit()
                with trace.use_span(span):
                    await scenario(session)
        finally:
            await engine.dispose()

    asyncio.run(main())
    return span.events


def test_a_committed_change_records_its_old_and_new_value() -> None:
    async def scenario(session: AsyncSession) -> None:
        row = await session.get(_Setting, 1)
        assert row is not None
        row.enabled = True
        await session.commit()

    assert _run(scenario) == [
        (
            SETTING_CHANGED_EVENT,
            {
                "change.table": "audited_setting",
                "change.operation": "update",
                "change.row": "1",
                "change.enabled": "False -> True",
            },
        )
    ]


def test_a_rolled_back_change_records_nothing() -> None:
    async def scenario(session: AsyncSession) -> None:
        row = await session.get(_Setting, 1)
        assert row is not None
        row.enabled = True
        await session.flush()
        await session.rollback()

    assert _run(scenario) == []


def test_a_column_the_model_does_not_name_is_never_recorded() -> None:
    async def scenario(session: AsyncSession) -> None:
        row = await session.get(_Setting, 1)
        assert row is not None
        row.secret = "sk-live-1234"
        await session.commit()

    assert _run(scenario) == []


def test_a_model_that_does_not_opt_in_records_nothing() -> None:
    async def scenario(session: AsyncSession) -> None:
        row = await session.get(_Unaudited, 1)
        assert row is not None
        row.enabled = True
        await session.commit()

    assert _run(scenario) == []


def test_a_new_and_a_deleted_row_are_recorded() -> None:
    async def scenario(session: AsyncSession) -> None:
        session.add(_Setting(id=2, enabled=True))
        old = await session.get(_Setting, 1)
        assert old is not None
        await session.delete(old)
        await session.commit()

    events = _run(scenario)

    assert sorted(str((attributes or {})["change.operation"]) for _, attributes in events) == ["delete", "insert"]
    inserted = next(attributes for _, attributes in events if (attributes or {})["change.operation"] == "insert")
    assert inserted == {
        "change.table": "audited_setting",
        "change.operation": "insert",
        "change.row": "2",
        "change.enabled": "None -> True",
    }


async def _set_enabled(session: AsyncSession, value: bool) -> None:
    row = await session.get(_Setting, 1)
    assert row is not None
    row.enabled = value
    await session.flush()


def _enabled_changes(events: list[tuple[str, Attributes]]) -> list[object]:
    return [(attributes or {}).get("change.enabled") for _, attributes in events]


def test_a_failed_savepoint_keeps_the_changes_made_before_it() -> None:
    async def scenario(session: AsyncSession) -> None:
        await _set_enabled(session, True)
        try:
            async with session.begin_nested():
                raise RuntimeError
        except RuntimeError:
            pass
        await session.commit()

    assert _enabled_changes(_run(scenario)) == ["False -> True"]


def test_a_change_inside_a_rolled_back_savepoint_is_not_recorded() -> None:
    async def scenario(session: AsyncSession) -> None:
        try:
            async with session.begin_nested():
                await _set_enabled(session, True)
                raise RuntimeError
        except RuntimeError:
            pass
        await session.commit()

    assert _run(scenario) == []


def test_a_released_savepoint_records_nothing_until_the_outer_transaction_commits() -> None:
    async def scenario(session: AsyncSession) -> None:
        async with session.begin_nested():
            await _set_enabled(session, True)
        await session.rollback()

    assert _run(scenario) == []


def test_a_released_savepoint_is_recorded_when_the_outer_transaction_commits() -> None:
    async def scenario(session: AsyncSession) -> None:
        async with session.begin_nested():
            await _set_enabled(session, True)
        await session.commit()

    assert _enabled_changes(_run(scenario)) == ["False -> True"]


def test_a_session_closed_without_committing_does_not_leak_into_its_next_commit() -> None:
    async def scenario(session: AsyncSession) -> None:
        await _set_enabled(session, True)
        await session.close()
        row = await session.get(_Unaudited, 1)
        assert row is not None
        row.enabled = True
        await session.commit()

    assert _run(scenario) == []


def test_a_value_the_session_never_loaded_is_not_reported_as_none() -> None:
    async def scenario(session: AsyncSession) -> None:
        row = await session.get(_Setting, 1)
        assert row is not None
        session.expire(row, ["enabled"])
        row.enabled = True
        await session.commit()

    assert _enabled_changes(_run(scenario)) == [f"{UNLOADED} -> True"]
