"""The roster term of the tenant scope must match on SQLite as well as PostgreSQL.

``organization_member.user_id`` is a UUID column while ``users.user_id`` is a string
holding the hyphenated rendering of the same UUID. SQLite stores the former as
CHAR(32) hex, so a raw text comparison matched only on PostgreSQL, which let one
organization's operator name another organization's member.
"""

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the metadata)
from gateway.models.tenancy import Organization, OrganizationMember
from gateway.models.users import User
from gateway.repositories.users_repository import in_organization, owned_by_organization

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False)() as open_session:
            yield open_session
    finally:
        await engine.dispose()


async def _scoped_ids(session: AsyncSession, user_id: str, organization_id: uuid.UUID) -> list[str]:
    rows = await session.execute(select(User.user_id).where(User.user_id == user_id, in_organization(organization_id)))
    return [row[0] for row in rows.all()]


async def test_a_keyless_roster_member_is_scoped_to_their_own_organization(session: AsyncSession) -> None:
    own_org = Organization(name="Own", slug="own")
    other_org = Organization(name="Other", slug="other")
    identity_id = uuid.uuid4()
    member_id = str(identity_id)
    session.add_all([own_org, other_org, User(user_id=member_id)])
    await session.flush()
    # A member who holds no key and no usage: only the roster row puts them in reach.
    session.add(OrganizationMember(organization_id=own_org.id, user_id=identity_id))
    await session.flush()

    assert await _scoped_ids(session, member_id, own_org.id) == [member_id]
    assert await _scoped_ids(session, member_id, other_org.id) == []
    assert await owned_by_organization(session, member_id, other_org.id) is False
    assert await owned_by_organization(session, member_id, own_org.id) is True
