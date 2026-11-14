"""Which addresses the deployment's email blocklist matches."""

from collections.abc import Collection

from gateway.core.email_domains import blocklist_keys
from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.identity import BlockedEmailRepository


class EmailBlocklist:
    """Answers which addresses the deployment's email blocklist matches."""

    def __init__(self, uow: UnitOfWork, blocked_emails: BlockedEmailRepository) -> None:
        self._uow = uow
        self._blocked_emails = blocked_emails

    async def blocked_among(self, emails: Collection[str]) -> set[str]:
        """Return the members of ``emails`` that the blocklist matches, spelled as they were given.

        When this is called inside a caller's Unit of Work block,
        no entry is added to the blocklist until that block commits.
        """
        keys = {email: blocklist_keys(email) for email in emails}
        wanted = {key for email_keys in keys.values() for key in email_keys}
        if not wanted:
            return set()
        async with self._uow:
            await self._blocked_emails.lock_shared()
            found = await self._blocked_emails.patterns_among(wanted)
        return {email for email, email_keys in keys.items() if found.intersection(email_keys)}
