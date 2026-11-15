"""How many invitation requests an organization and an inviting account may make in a day.

How many of them may name one address in a day.
"""

import hashlib
import uuid
from datetime import timedelta
from types import TracebackType
from typing import Self

from gateway.core.addresses import stored_address
from gateway.exceptions.organizations_exceptions import InvitationLimitCeiling
from gateway.ports.rate_limit_store_port import RateLimitStorePort

# NOTE: A fixed window.
# A deployment that needs a different period can turn it into a setting.
INVITATION_LIMIT_WINDOW = timedelta(hours=24)
_WINDOW_SEC = INVITATION_LIMIT_WINDOW.total_seconds()


async def _settle_all(store: RateLimitStorePort, entries: list[tuple[str, str]]) -> None:
    """Settle every one of ``entries`` back to a count of 0."""
    for key, handle in entries:
        await store.settle(key, handle, 0)


class Admission:
    """The keys one invitation request has counted so far, as an async context manager.

    Leaving the block settles every one of them back to a count of 0, unless ``keep()`` was called.
    """

    def __init__(self, store: RateLimitStorePort, entries: list[tuple[str, str]]) -> None:
        self._store = store
        self._entries = entries
        self._keep = False

    def keep(self) -> None:
        """Keep the counted keys instead of settling them back when the block ends."""
        self._keep = True

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._keep:
            return
        entries, self._entries = self._entries, []
        await _settle_all(self._store, entries)


class InvitationLimits:
    """Bounds how many invitation requests an organization and an inviting account may make in a day.

    Also bounds how many of either may name one address in a day.
    A refused request leaves every count as it was.
    """

    def __init__(self, store: RateLimitStorePort, *, per_day: int, per_address_per_day: int) -> None:
        self._store = store
        self._per_day = per_day
        self._per_address_per_day = per_address_per_day

    async def admit(
        self, *, account_id: uuid.UUID, email: str, organization_id: uuid.UUID
    ) -> Admission | InvitationLimitCeiling:
        """Count one invitation request, or return the ceiling that refused it once one of its four keys is full.

        Raise ``ValueError`` when ``email`` is not an address that can be stored.
        """
        address = stored_address(email)
        if address is None:
            msg = "cannot count an invitation request for an email that cannot be stored"
            raise ValueError(msg)
        organization_address_hash = hashlib.sha256(f"organization:{organization_id}:{address}".encode()).hexdigest()
        account_address_hash = hashlib.sha256(f"account:{account_id}:{address}".encode()).hexdigest()
        keyed_limits = (
            (f"invitation:organization:{organization_id}", self._per_day, InvitationLimitCeiling.DAILY),
            (f"invitation:account:{account_id}", self._per_day, InvitationLimitCeiling.DAILY),
            (
                f"invitation:organization:{organization_id}:address:{organization_address_hash}",
                self._per_address_per_day,
                InvitationLimitCeiling.ADDRESS,
            ),
            (
                f"invitation:account:{account_id}:address:{account_address_hash}",
                self._per_address_per_day,
                InvitationLimitCeiling.ADDRESS,
            ),
        )
        entries: list[tuple[str, str]] = []
        try:
            for key, limit, ceiling in keyed_limits:
                window = await self._store.hit(key, limit, _WINDOW_SEC)
                if window.handle is None:
                    await _settle_all(self._store, entries)
                    return ceiling
                entries.append((key, window.handle))
        except BaseException:
            await _settle_all(self._store, entries)
            raise
        return Admission(self._store, entries)
