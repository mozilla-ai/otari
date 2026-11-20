"""The country an organization's account is in, for the provider country gate.

The gate closes a provider to the country a request comes from, which the
request's address or a proxy's header says. This port adds where the account
lives, which a build that bills its customers knows from the billing address
and the core does not. The core adapter is a Null Object that knows no
country, so a self-hosted deployment is gated on the request's country alone.

An adapter reads on the request's session it is built with and commits
nothing. ``None`` means the deployment holds no country for the account, which
the gate takes as no opinion. An adapter that cannot answer, because the store
it reads is down, raises :class:`AccountCountryUnavailableError` instead: the
gate then refuses wherever a country list applies, so an outage cannot open it.
"""

import uuid
from typing import Protocol


class AccountCountryUnavailableError(Exception):
    """An adapter could not find out where the account is, as opposed to knowing it has no record."""


class AccountCountryPort(Protocol):
    """Where an organization's account is, as the build that bills it knows."""

    async def country(self, *, organization_id: uuid.UUID) -> str | None:
        """Return the organization's ISO 3166-1 alpha-2 country code, or ``None`` when the deployment holds none.

        Raises:
            AccountCountryUnavailableError: the adapter could not ask.
        """
        ...
