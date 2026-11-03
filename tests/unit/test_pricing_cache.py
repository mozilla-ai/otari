"""The process cache in front of the stored rates: what it saves, and what clears it.

A cached rate is only safe if a write in this process is seen at once, so every
way the gateway writes a price (an added row, a changed one, a bulk delete) is
shown clearing it, and an uncommitted write is shown reaching its writer alone.
"""

import asyncio
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, TypeVar

from sqlalchemy import delete, event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the shared metadata)
from gateway.models.pricing import OPERATOR_ORIGIN, ModelPricing, OrganizationModelPricing
from gateway.models.tenancy import Organization
from gateway.services.pricing_service import resolve_model_pricing

T = TypeVar("T")

_PAST = datetime(2026, 1, 1, tzinfo=UTC)


def _run(scenario: Callable[[async_sessionmaker[AsyncSession], list[str]], Awaitable[T]]) -> T:
    """Run one scenario on a fresh SQLite database, recording each statement it issues.

    A file rather than ``:memory:``, whose sessions all share one connection and
    so would see each other's uncommitted rows whatever the cache did.
    """

    async def main(directory: str) -> T:
        engine = create_async_engine(f"sqlite+aiosqlite:///{directory}/prices.db")
        statements: list[str] = []

        @event.listens_for(engine.sync_engine, "before_cursor_execute")
        def _record(_conn: Any, _cursor: Any, statement: str, *_args: Any) -> None:
            statements.append(statement)

        try:
            async with engine.begin() as conn:
                await conn.run_sync(SQLModel.metadata.create_all)
            statements.clear()
            return await scenario(async_sessionmaker(engine, expire_on_commit=False), statements)
        finally:
            await engine.dispose()

    with tempfile.TemporaryDirectory() as directory:
        return asyncio.run(main(directory))


def _price(rate: float) -> ModelPricing:
    return ModelPricing(
        model_key="openai:gpt-4o",
        effective_at=_PAST,
        input_price_per_million=Decimal(str(rate)),
        output_price_per_million=Decimal("1"),
    )


async def _input_rate(sessions: async_sessionmaker[AsyncSession], **kwargs: Any) -> Decimal | None:
    async with sessions() as session:
        resolved = await resolve_model_pricing(session, "openai", "gpt-4o", **kwargs)
    return None if resolved is None else resolved.pricing.input_price_per_million


def test_a_rate_read_once_is_served_without_a_statement() -> None:
    async def scenario(sessions: async_sessionmaker[AsyncSession], statements: list[str]) -> int:
        async with sessions() as session:
            session.add(_price(10))
            await session.commit()
        assert await _input_rate(sessions) == Decimal("10")
        statements.clear()
        assert await _input_rate(sessions) == Decimal("10")
        return len(statements)

    assert _run(scenario) == 0


def test_a_lookup_at_a_given_time_reads_the_database() -> None:
    async def scenario(sessions: async_sessionmaker[AsyncSession], statements: list[str]) -> int:
        async with sessions() as session:
            session.add(_price(10))
            await session.commit()
        await _input_rate(sessions)
        statements.clear()
        assert await _input_rate(sessions, as_of=datetime.now(UTC)) == Decimal("10")
        return len(statements)

    assert _run(scenario) > 0


def test_a_new_rate_is_seen_at_once_after_a_cached_miss() -> None:
    async def scenario(sessions: async_sessionmaker[AsyncSession], _statements: list[str]) -> Decimal | None:
        assert await _input_rate(sessions) is None
        async with sessions() as session:
            session.add(_price(10))
            await session.commit()
        return await _input_rate(sessions)

    assert _run(scenario) == Decimal("10")


def test_a_changed_rate_is_seen_at_once() -> None:
    async def scenario(sessions: async_sessionmaker[AsyncSession], _statements: list[str]) -> Decimal | None:
        async with sessions() as session:
            session.add(_price(10))
            await session.commit()
        assert await _input_rate(sessions) == Decimal("10")
        async with sessions() as session:
            row = (await session.execute(select(ModelPricing))).scalar_one()
            row.input_price_per_million = Decimal("4")
            await session.commit()
        return await _input_rate(sessions)

    assert _run(scenario) == Decimal("4")


def test_a_bulk_delete_is_seen_at_once() -> None:
    async def scenario(sessions: async_sessionmaker[AsyncSession], _statements: list[str]) -> Decimal | None:
        async with sessions() as session:
            session.add(_price(10))
            await session.commit()
        assert await _input_rate(sessions) == Decimal("10")
        async with sessions() as session:
            await session.execute(delete(ModelPricing).where(ModelPricing.model_key == "openai:gpt-4o"))
            await session.commit()
        return await _input_rate(sessions)

    assert _run(scenario) is None


def test_a_new_organization_override_is_seen_at_once() -> None:
    async def scenario(sessions: async_sessionmaker[AsyncSession], _statements: list[str]) -> Decimal | None:
        async with sessions() as session:
            organization = Organization(name="Acme", slug="acme")
            session.add_all([organization, _price(10)])
            await session.commit()
            organization_id: uuid.UUID = organization.id
        assert await _input_rate(sessions, organization_id=organization_id) == Decimal("10")
        async with sessions() as session:
            session.add(
                OrganizationModelPricing(
                    organization_id=organization_id,
                    model_key="openai:gpt-4o",
                    input_price_per_million=Decimal("2.5"),
                    output_price_per_million=Decimal("1"),
                    effective_from=_PAST,
                    origin=OPERATOR_ORIGIN,
                )
            )
            await session.commit()
        return await _input_rate(sessions, organization_id=organization_id)

    assert _run(scenario) == Decimal("2.5")


def test_an_uncommitted_rate_is_seen_only_by_its_writer() -> None:
    async def scenario(sessions: async_sessionmaker[AsyncSession], _statements: list[str]) -> Decimal | None:
        async with sessions() as session:
            session.add(_price(10))
            await session.flush()
            resolved = await resolve_model_pricing(session, "openai", "gpt-4o")
            assert resolved is not None
            assert await _input_rate(sessions) is None
            await session.rollback()
        return await _input_rate(sessions)

    assert _run(scenario) is None


def test_a_writer_sees_its_own_unflushed_change_over_a_cached_rate() -> None:
    async def scenario(sessions: async_sessionmaker[AsyncSession], _statements: list[str]) -> Decimal | None:
        async with sessions() as session:
            session.add(_price(10))
            await session.commit()
        assert await _input_rate(sessions) == Decimal("10")
        async with sessions() as session:
            row = (await session.execute(select(ModelPricing))).scalar_one()
            row.input_price_per_million = Decimal("4")
            resolved = await resolve_model_pricing(session, "openai", "gpt-4o")
            assert resolved is not None
            rate = resolved.pricing.input_price_per_million
            await session.rollback()
            return rate

    assert _run(scenario) == Decimal("4")


def test_a_writer_sees_its_own_change_over_a_rate_another_session_cached() -> None:
    async def scenario(sessions: async_sessionmaker[AsyncSession], _statements: list[str]) -> Decimal | None:
        async with sessions() as session:
            session.add(_price(10))
            await session.commit()
        async with sessions() as session:
            row = (await session.execute(select(ModelPricing))).scalar_one()
            row.input_price_per_million = Decimal("4")
            await session.flush()
            # Another session caches the committed rate after the flush.
            assert await _input_rate(sessions) == Decimal("10")
            resolved = await resolve_model_pricing(session, "openai", "gpt-4o")
            assert resolved is not None
            assert resolved.pricing.input_price_per_million == Decimal("4")
            await session.commit()
        return await _input_rate(sessions)

    assert _run(scenario) == Decimal("4")
