"""Rotation never overwrites a credential someone changed while it was working.

otari#1127. ``reencrypt_*`` reads every row holding a secret, decrypts it and
re-encrypts it, and nothing pinned the write to the ciphertext it had read. A
PATCH committing in that window was overwritten with a re-encryption of the
value it replaced — a lost update on a credential, and a silent one.

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
from sqlmodel import SQLModel

from gateway.models.entities import ProviderCredential, SearchToolCredential
from gateway.services import provider_store_service as provider_store
from gateway.services import search_tool_store_service as search_tool_store
from gateway.services.provider_store_service import reencrypt_credentials
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


def _commit_from_another_connection(db_path: str, statement: object) -> None:
    """Commit one statement on a separate connection, the way a PATCH would."""
    other = create_engine(f"sqlite:///{db_path}")
    try:
        with other.begin() as conn:
            conn.execute(statement)  # type: ignore[arg-type]
    finally:
        other.dispose()


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())


async def _add_provider(session: AsyncSession, instance: str, api_key: str) -> None:
    session.add(
        ProviderCredential(
            instance=instance,
            provider_type="openai",
            encrypted_api_key=encrypt_secret(api_key),
            last4=api_key[-4:],
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
    )
    await session.commit()


async def _add_search_tool(session: AsyncSession, name: str, api_key: str) -> None:
    session.add(
        SearchToolCredential(
            name=name,
            provider="searxng",
            encrypted_api_key=encrypt_secret(api_key),
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
    )
    await session.commit()


class TestProviderRotation:
    def test_an_untouched_row_is_reencrypted(self) -> None:
        async def scenario(session: AsyncSession, _db: str) -> tuple[int, int, int]:
            await _add_provider(session, "openai", "sk-live-value")
            return await reencrypt_credentials(session)

        assert _run(scenario) == (1, 0, 0)

    def test_the_plaintext_survives_the_rotation(self) -> None:
        """A re-encryption that lost the value would pass every count above."""

        async def scenario(session: AsyncSession, _db: str) -> str:
            await _add_provider(session, "openai", "sk-live-value")
            before = (await session.execute(select(ProviderCredential))).scalars().one().encrypted_api_key
            await reencrypt_credentials(session)
            await session.commit()
            session.expire_all()
            after = (await session.execute(select(ProviderCredential))).scalars().one().encrypted_api_key
            assert after is not None
            assert after != before, "re-encryption produced the same ciphertext"
            return decrypt_secret(after)

        assert _run(scenario) == "sk-live-value"

    def test_a_row_changed_under_the_rotation_is_skipped_not_clobbered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point: the stored value must still be the competing edit's.

        The window is between the read and the write, so the competing commit has
        to land there and nowhere else. Hooking the service's own
        ``encrypt_secret`` puts it exactly there — the row has been read and
        decrypted, and its UPDATE has not run yet.
        """

        async def scenario(session: AsyncSession, db_path: str) -> tuple[tuple[int, int, int], str]:
            await _add_provider(session, "openai", "sk-old-value")
            real_encrypt = provider_store.encrypt_secret
            raced = False

            def encrypt_and_let_someone_else_commit(plaintext: str) -> str:
                nonlocal raced
                if not raced:
                    raced = True
                    _commit_from_another_connection(
                        db_path,
                        update(ProviderCredential)
                        .where(ProviderCredential.instance == "openai")
                        .values(encrypted_api_key=encrypt_secret("sk-new-value")),
                    )
                return real_encrypt(plaintext)

            monkeypatch.setattr(provider_store, "encrypt_secret", encrypt_and_let_someone_else_commit)

            counts = await reencrypt_credentials(session)
            await session.commit()
            session.expire_all()
            stored = (await session.execute(select(ProviderCredential))).scalars().one()
            assert stored.encrypted_api_key is not None
            return counts, decrypt_secret(stored.encrypted_api_key)

        counts, stored_value = _run(scenario)
        assert counts == (0, 0, 1)
        assert stored_value == "sk-new-value"

    def test_an_undecryptable_row_is_left_alone(self) -> None:
        async def scenario(session: AsyncSession, _db: str) -> tuple[int, int, int]:
            session.add(
                ProviderCredential(
                    instance="broken",
                    provider_type="openai",
                    encrypted_api_key="not-a-ciphertext",
                    last4="text",
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            )
            await session.commit()
            return await reencrypt_credentials(session)

        assert _run(scenario) == (0, 1, 0)


class TestSearchToolRotation:
    """Same shape, second store. Both or neither — three stores with two shapes
    is worse than three with one, because the next person copies whichever they
    find first."""

    def test_an_untouched_row_is_reencrypted(self) -> None:
        async def scenario(session: AsyncSession, _db: str) -> tuple[int, int, int]:
            await _add_search_tool(session, "searxng", "key-live")
            return await reencrypt_search_tools(session)

        assert _run(scenario) == (1, 0, 0)

    def test_a_row_changed_under_the_rotation_is_skipped_not_clobbered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def scenario(session: AsyncSession, db_path: str) -> tuple[tuple[int, int, int], str]:
            await _add_search_tool(session, "searxng", "key-old")
            real_encrypt = search_tool_store.encrypt_secret
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
