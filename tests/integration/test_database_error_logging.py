"""On PostgreSQL, a database error is described by SQLSTATE and constraint, not by its values.

asyncpg is the driver the gateway runs on, and the one whose error SQLAlchemy
wraps twice, so the description is checked against a real refusal from it.
"""

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.database import describe_database_error
from gateway.models.users import User

_SECRET = "sk-live-do-not-log"


@pytest.mark.asyncio
async def test_a_unique_violation_is_described_without_the_row(async_db: AsyncSession) -> None:
    async_db.add(User(user_id="dup", alias="first"))
    await async_db.commit()
    async_db.expunge_all()

    async_db.add(User(user_id="dup", alias=_SECRET))
    with pytest.raises(IntegrityError) as caught:
        await async_db.flush()
    await async_db.rollback()
    assert _SECRET in str(caught.value), "the message carries the parameters, which is why it is not logged"

    described = describe_database_error(caught.value)

    assert _SECRET not in described
    assert described == "IntegrityError UniqueViolationError sqlstate=23505 constraint=users_pkey"
