"""This Null Object adapter satisfies the account country port for a deployment that knows no account's country."""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.ports.account_country_port import AccountCountryPort


class NullAccountCountryAdapter(AccountCountryPort):
    """``AccountCountryPort`` adapter for a deployment with no record of where its accounts are.

    Holds no state, so the request's database session is unused.
    """

    def __init__(self, session: AsyncSession | None) -> None:
        # Accepted to match the container's per-request factory; unused, because
        # there is no account record to read.
        del session

    async def country(self, *, organization_id: uuid.UUID) -> str | None:
        return None
