"""Removing invitations whose keep-time after expiry has passed."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from gateway.core.unit_of_work import UnitOfWork
from gateway.log_config import logger
from gateway.repositories.organizations import OrganizationRepositories
from gateway.services.organizations._poll import poll as background_poll

# NOTE: A fixed keep-time.
# A deployment that needs a different one can turn it into a setting.
EXPIRED_KEPT_FOR = timedelta(days=30)
# NOTE: A fixed interval.
# A deployment that needs a different one can turn it into a setting.
CLEANUP_SECONDS = 3600.0
# Keeps one batch from holding many rows locked at once; the poll loop runs another when this fills.
CLEANUP_BATCH_SIZE = 500


class ExpiredInvitationCleanup:
    """Removes an invitation once its keep-time after expiry has passed."""

    def __init__(
        self,
        uow: UnitOfWork,
        repositories: OrganizationRepositories,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._uow = uow
        self._invitations = repositories.invitations
        self._clock = clock

    async def delete_expired(self) -> int:
        """Delete up to ``CLEANUP_BATCH_SIZE`` of the oldest invitations whose keep-time after expiry has passed.

        Returns how many were deleted, which may be fewer than everything overdue.

        NOTE: The count is what was deleted, not what the batch picked:
        a row renewed between the two counts as neither,
        and a row the sweep skips because another writer holds it does too.
        ``poll`` continues only while that count fills a batch,
        so either one inside a full batch leaves the rest until the next sweep.
        """
        cutoff = self._clock() - EXPIRED_KEPT_FOR
        async with self._uow:
            deleted = await self._invitations.delete_expired_before(cutoff, limit=CLEANUP_BATCH_SIZE)
        if deleted:
            logger.info("Invitation cleanup removed %d expired invitations", deleted)
        return deleted

    @staticmethod
    async def poll(
        build: Callable[[UnitOfWork], "ExpiredInvitationCleanup"], interval: float = CLEANUP_SECONDS
    ) -> None:
        """Remove expired invitations every ``interval`` seconds, until the task is canceled.

        ``build`` receives a new Unit of Work for each batch.
        """

        async def step(uow: UnitOfWork) -> bool:
            deleted = await build(uow).delete_expired()
            return deleted >= CLEANUP_BATCH_SIZE

        await background_poll(step, interval=interval, name="Invitation cleanup")
