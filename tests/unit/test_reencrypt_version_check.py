"""Rotation never overwrites a credential someone changed while it was working.

otari#1127. ``reencrypt_*`` reads every row holding a secret, decrypts it and
re-encrypts it, and nothing pinned the write to the ciphertext it had read. A
PATCH committing in that window was overwritten with a re-encryption of the
value it replaced: a lost update on a credential, and a silent one.

The window is small and rotation is rare and hand-run, which is exactly why it
is worth a test: nobody would ever see it happen.

The racing cells use a second, real connection to the same database rather than
a stand-in, because the property under test is what the WHERE clause sees when
someone else has committed. A mock of "the row changed" would pass just as well
against the bug.
"""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeVar

import pytest
from sqlalchemy import create_engine, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.sql import Executable
from sqlmodel import SQLModel

from gateway.models.tools import SearchToolCredential
from gateway.services import search_tool_store_service as search_tool_store
from gateway.services.search_tool_store_service import reencrypt_search_tools
from gateway.services.secret_box import decrypt_secret, encrypt_secret, generate_secret_key

T = TypeVar("T")


def _run(scenario: Callable[[AsyncSession, str], Awaitable[T]]) -> T:
    """Run one scenario against a file-backed SQLite database in WAL mode.

    A file rather than ``:memory:`` so a second connection can reach the same
    rows, and WAL so that second connection can commit while the rotation holds
    its read.
    """

    async def main() -> T:
        with TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "otari-rotation.db")
            engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
            try:
                async with engine.begin() as conn:
                    await conn.execute(text("PRAGMA journal_mode=WAL"))
                    await conn.run_sync(SQLModel.metadata.create_all)
                session_factory = async_sessionmaker(engine, expire_on_commit=False)
                async with session_factory() as session:
                    return await scenario(session, db_path)
            finally:
                # aiosqlite runs each connection on its own thread, so an
                # undisposed engine leaks one per call.
                await engine.dispose()

    return asyncio.run(main())


def _commit_from_another_connection(db_path: str, statement: Executable) -> None:
    """Commit one statement on a separate connection, the way a PATCH would."""
    other = create_engine(f"sqlite:///{db_path}")
    try:
        with other.begin() as conn:
            conn.execute(statement)
    finally:
        other.dispose()


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())


async def _add_search_tool(session: AsyncSession, name: str, api_key: str, *, kind: str = "search") -> None:
    session.add(
        SearchToolCredential(
            name=name,
            kind=kind,
            provider="searxng" if kind == "search" else "exa",
            encrypted_api_key=encrypt_secret(api_key),
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
    )
    await session.commit()


class TestSearchToolRotation:
    """Same shape, second store. Both or neither: three stores with two shapes
    is worse than three with one, because the next person copies whichever they
    find first."""

    def test_an_untouched_row_is_reencrypted(self) -> None:
        async def scenario(session: AsyncSession, _db: str) -> tuple[int, int, int]:
            await _add_search_tool(session, "searxng", "key-live")
            return await reencrypt_search_tools(session)

        assert _run(scenario) == (1, 0, 0)

    def test_a_fetch_rows_key_is_rotated_with_the_search_rows(self) -> None:
        """Fetch instances share the table, so the one rotation covers both kinds."""

        async def scenario(session: AsyncSession, _db: str) -> tuple[tuple[int, int, int], str]:
            await _add_search_tool(session, "searxng", "key-search")
            await _add_search_tool(session, "exa-fetch", "key-fetch", kind="fetch")
            counts = await reencrypt_search_tools(session)
            await session.commit()
            stmt = select(SearchToolCredential).where(SearchToolCredential.name == "exa-fetch")
            row = (await session.execute(stmt.execution_options(populate_existing=True))).scalar_one()
            assert row.encrypted_api_key is not None
            return counts, decrypt_secret(row.encrypted_api_key)

        assert _run(scenario) == ((2, 0, 0), "key-fetch")

    def test_a_row_changed_under_the_rotation_is_skipped_not_clobbered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def scenario(session: AsyncSession, db_path: str) -> tuple[tuple[int, int, int], str]:
            await _add_search_tool(session, "searxng", "key-old")
            real_encrypt = encrypt_secret
            raced = False

            def encrypt_and_let_someone_else_commit(plaintext: str) -> str:
                nonlocal raced
                if not raced:
                    raced = True
                    _commit_from_another_connection(
                        db_path,
                        update(SearchToolCredential)
                        .where(SearchToolCredential.name == "searxng")
                        .values(encrypted_api_key=encrypt_secret("key-new")),
                    )
                return real_encrypt(plaintext)

            monkeypatch.setattr(search_tool_store, "encrypt_secret", encrypt_and_let_someone_else_commit)

            counts = await reencrypt_search_tools(session)
            await session.commit()
            session.expire_all()
            stored = (await session.execute(select(SearchToolCredential))).scalars().one()
            assert stored.encrypted_api_key is not None
            return counts, decrypt_secret(stored.encrypted_api_key)

        counts, stored_value = _run(scenario)
        assert counts == (0, 0, 1)
        assert stored_value == "key-new"
