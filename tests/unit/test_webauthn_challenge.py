"""The challenge is spent exactly once, on the engine the OSS edition ships.

`tests/integration/test_webauthn_api.py` covers the ceremonies end to end
against PostgreSQL. This covers the one property that has to hold on SQLite too,
because a standalone deployment defaults to it: consuming a challenge is a
single conditional DELETE reading back what it removed, which needs SQLite's
`RETURNING` (3.35, March 2021). A build older than that would fail here rather
than in a browser, with a passkey sign-in that silently accepted replays.
"""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, col

import gateway.models  # noqa: F401  imports every model module, so create_all sees the whole schema
from gateway.models.tenancy import Organization, User, WebAuthnChallenge
from gateway.services.tenancy import webauthn_service
from gateway.services.tenancy.errors import PasskeyCeremonyError

CHALLENGE = b"\x01" * 32
OTHER_CHALLENGE = b"\x02" * 32


async def _session_factory(tmp_path: Path) -> async_sessionmaker[AsyncSession]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'webauthn.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    return async_sessionmaker(engine, expire_on_commit=False)


def test_a_challenge_is_spent_exactly_once(tmp_path: Path) -> None:
    async def _run() -> None:
        factory = await _session_factory(tmp_path)
        async with factory() as db:
            await webauthn_service._issue_challenge(db, CHALLENGE, ceremony="authentication", user_id=None)
            await db.commit()

        async with factory() as db:
            assert await webauthn_service._spend_challenge(db, CHALLENGE, ceremony="authentication") is None
            await db.commit()

        # The row is gone, so a replayed assertion carrying it matches nothing.
        async with factory() as db:
            with pytest.raises(PasskeyCeremonyError):
                await webauthn_service._spend_challenge(db, CHALLENGE, ceremony="authentication")

    asyncio.run(_run())


def test_an_unknown_challenge_is_refused(tmp_path: Path) -> None:
    async def _run() -> None:
        factory = await _session_factory(tmp_path)
        async with factory() as db:
            await webauthn_service._issue_challenge(db, CHALLENGE, ceremony="authentication", user_id=None)
            await db.commit()
        async with factory() as db:
            with pytest.raises(PasskeyCeremonyError):
                await webauthn_service._spend_challenge(db, OTHER_CHALLENGE, ceremony="authentication")

    asyncio.run(_run())


def test_the_wrong_ceremony_is_refused_and_the_challenge_survives_it(tmp_path: Path) -> None:
    """Answering a registration challenge as a sign-in is refused, and does not burn it.

    This pins what the routes actually do rather than what "single use" sounds
    like it should mean. The delete is staged on the caller's transaction and no
    route commits on a refusal, so the challenge is still there afterwards and
    dies on its TTL instead.

    The previous version of this test asserted the opposite and passed only
    because it committed by hand between the two calls, which no caller does. If
    this ever needs to become "spent either way", the change belongs in the
    routes' transaction handling, not here: committing inside ``_spend_challenge``
    would commit whatever else the request had staged, which is the hazard
    ``dashboard_session_service._revoke_deactivated_identity_sessions`` documents
    for the same reason.
    """

    async def _run() -> None:
        factory = await _session_factory(tmp_path)
        async with factory() as db:
            await webauthn_service._issue_challenge(db, CHALLENGE, ceremony="registration", user_id=None)
            await db.commit()

        # A refusal, and then a rollback: exactly what a route does with one.
        async with factory() as db:
            with pytest.raises(PasskeyCeremonyError):
                await webauthn_service._spend_challenge(db, CHALLENGE, ceremony="authentication")
            await db.rollback()

        # Still live, and still answerable by the ceremony it was issued for.
        async with factory() as db:
            assert await webauthn_service._spend_challenge(db, CHALLENGE, ceremony="registration") is None
            await db.commit()

        # And now it is gone, because that one completed.
        async with factory() as db:
            with pytest.raises(PasskeyCeremonyError):
                await webauthn_service._spend_challenge(db, CHALLENGE, ceremony="registration")

    asyncio.run(_run())


def test_a_registration_challenge_names_the_identity_it_was_issued_to(tmp_path: Path) -> None:
    """What ``finish_registration`` checks to keep one ceremony out of another's session."""

    async def _run() -> None:
        factory = await _session_factory(tmp_path)
        organization_id = uuid.uuid4()
        identity_id = uuid.uuid4()
        async with factory() as db:
            db.add(Organization(id=organization_id, name="Default", slug="default"))
            db.add(User(id=identity_id, email="op@example.com", active_organization_id=organization_id))
            await db.commit()

        async with factory() as db:
            await webauthn_service._issue_challenge(db, CHALLENGE, ceremony="registration", user_id=identity_id)
            await db.commit()

        async with factory() as db:
            assert await webauthn_service._spend_challenge(db, CHALLENGE, ceremony="registration") == identity_id

    asyncio.run(_run())


def test_issuing_a_challenge_sweeps_expired_ones(tmp_path: Path) -> None:
    """The table stays about as large as the ceremonies in flight."""

    async def _run() -> None:
        factory = await _session_factory(tmp_path)
        async with factory() as db:
            await webauthn_service._issue_challenge(db, CHALLENGE, ceremony="authentication", user_id=None)
            await db.commit()

        # Age it past its TTL, then start any other ceremony.
        async with factory() as db:
            await db.execute(
                update(WebAuthnChallenge)
                .where(col(WebAuthnChallenge.ceremony) == "authentication")
                .values(expires_at=datetime.now(UTC) - timedelta(seconds=1))
            )
            await db.commit()

        async with factory() as db:
            await webauthn_service._issue_challenge(db, OTHER_CHALLENGE, ceremony="authentication", user_id=None)
            await db.commit()

        async with factory() as db:
            with pytest.raises(PasskeyCeremonyError):
                await webauthn_service._spend_challenge(db, CHALLENGE, ceremony="authentication")

    asyncio.run(_run())
