"""Deleting an account at its owner's request.

What names the person goes: the sign-in identity with its passwords, passkeys,
links and sessions, its Playground conversations, its agent traces and
telemetry, its files and its keys. What the deployment still needs to count
stays, under a new id that nothing links back to them: usage rows move onto a
nameless deleted user, so spend and token totals still add up and still count
one person, and an organization nobody else is active in keeps its rows under a
placeholder name and none of the secrets it held.
"""

import uuid
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Protocol

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.identity_exceptions import (
    AccountDeletionPasswordRequiredError,
    AccountDeletionUnconfirmedError,
    AccountErasureFailedError,
    BootstrapOperatorProtectedError,
    CurrentPasswordIncorrectError,
)
from gateway.log_config import logger
from gateway.models.tenancy import User
from gateway.ports.telemetry_storage_port import TelemetryStoragePort
from gateway.repositories.identity import IdentityRepositories
from gateway.services.api_keys import ApiKeyService
from gateway.services.files import FileService
from gateway.services.organizations import AccountRelease, ReleasedOrganizations, UserAccess
from gateway.services.password_service import verify_password_async
from gateway.services.traces import TraceService, TraceWriter
from gateway.services.usage_and_telemetry import UsageService

# How many times an end user a service key names while the account's records are erased sends the erasure round
# again before the deletion gives up and can be retried.
_ERASURE_ATTEMPTS = 3


class _LateEndUsersError(Exception):
    """The account has end users its records were not erased for, so the deletion was rolled back."""

    def __init__(self, user_ids: list[str]) -> None:
        super().__init__()
        self.user_ids = user_ids


class OrganizationSecrets(Protocol):
    """A store of credentials an organization or its workspaces hold."""

    async def forget_secrets(
        self, organization_ids: Collection[uuid.UUID], workspace_ids: Collection[uuid.UUID]
    ) -> None:
        """Stage clearing the credentials these organizations and workspaces hold."""
        ...


@dataclass(frozen=True)
class DeletedAccount:
    """The account that was deleted, as it was, for whoever must be told."""

    id: uuid.UUID
    email: str | None


class AccountDeletionService:
    """Delete an account at its owner's request. Each write runs in a Unit of Work block."""

    def __init__(
        self,
        uow: UnitOfWork,
        repositories: IdentityRepositories,
        *,
        bootstrap_operator_id: uuid.UUID | None,
        organizations: AccountRelease,
        users: UserAccess,
        keys: ApiKeyService,
        usage: UsageService,
        files: FileService | None,
        traces: TraceService,
        trace_writer: TraceWriter | None,
        telemetry: TelemetryStoragePort,
        secrets: Sequence[OrganizationSecrets],
    ) -> None:
        """``files`` is ``None`` where the deployment stores no files, ``trace_writer`` where it writes no traces.

        ``secrets`` are cleared in every organization the account leaves nobody active in.
        """
        self._uow = uow
        self._identities = repositories.users
        self._erased = repositories.erased_accounts
        self._bootstrap_operator_id = bootstrap_operator_id
        self._organizations = organizations
        self._users = users
        self._keys = keys
        self._usage = usage
        self._files = files
        self._traces = traces
        self._trace_writer = trace_writer
        self._telemetry = telemetry
        self._secrets = secrets

    async def delete(self, account: User, *, confirm_email: str, current_password: str | None) -> DeletedAccount:
        """Delete ``account`` and everything that names its owner.

        Raises:
            BootstrapOperatorProtectedError: The account is the deployment's bootstrap operator.
            AccountDeletionUnconfirmedError: ``confirm_email`` is not the account's address.
            AccountDeletionPasswordRequiredError: The account has a password and none was given.
            CurrentPasswordIncorrectError: The given password is not the account's.
            AccountDeletionLastOwnerError: The account is the last active owner of an organization others use.
            AccountErasureFailedError: Telemetry or traces could not be erased. Some may already be, but the
                account is not deleted, and a retry finishes the erasure.

        """
        if self._bootstrap_operator_id == account.id:
            raise BootstrapOperatorProtectedError("The deployment's bootstrap operator cannot delete its account")
        if account.email is None or confirm_email.strip().lower() != account.email.strip().lower():
            raise AccountDeletionUnconfirmedError
        if account.hashed_password is not None:
            if current_password is None:
                raise AccountDeletionPasswordRequiredError
            if not await verify_password_async(current_password, account.hashed_password):
                raise CurrentPasswordIncorrectError

        # Read now: a retried transaction rolls back, which expires the account's loaded attributes.
        account_id, email = account.id, account.email
        async with self._uow:
            await self._refuse_bootstrap_operator(account_id)
            await self._organizations.ensure_releasable(account_id)
            pending = await self._users.lock_with_end_users(str(account_id))
        erased: set[str] = set()
        for _ in range(_ERASURE_ATTEMPTS):
            # Erased first and outside the transaction, as an operator's deletion of a user does: telemetry
            # may be stored out of process, so the two cannot be atomic. Failing this way, or being refused
            # by a change to an organization meanwhile, leaves the account to retry on, where the other order
            # would leave records nobody could reach to erase.
            await self._erase_records(sorted(set(pending) - erased))
            erased.update(pending)
            try:
                released = await self._delete_rows(account_id, email, erased)
            except _LateEndUsersError as late:
                pending = late.user_ids
                continue
            break
        else:
            raise AccountErasureFailedError
        logger.warning(
            "Account deleted at its owner's request: account=%s organizations_released=%d",
            account_id,
            len(released.organization_ids),
        )
        return DeletedAccount(id=account_id, email=email)

    async def _delete_rows(self, identity_id: uuid.UUID, email: str, erased: set[str]) -> ReleasedOrganizations:
        """Delete the account's rows, rolling back with ``_LateEndUsersError`` if it has users not yet ``erased``.

        Holding the account's billing user locked keeps its service keys from adding an end user meanwhile.
        """
        async with self._uow:
            released = await self._organizations.release(identity_id)
            await self._keys.revoke_in_workspaces(released.workspace_ids)
            for store in self._secrets:
                await store.forget_secrets(released.organization_ids, released.workspace_ids)
            await self._organizations.withdraw_invitations(email)
            locked_ids = await self._users.lock_with_end_users(str(identity_id))
            if not erased.issuperset(locked_ids):
                raise _LateEndUsersError(locked_ids)
            if locked_ids:
                tombstone_id = await self._users.add_tombstone()
                await self._usage.reassign_owner(locked_ids, tombstone_id)
                if self._files is not None:
                    await self._files.release_owned(locked_ids, tombstone_id)
                await self._users.remove(locked_ids)
            identity = await self._identities.get_locked(identity_id)
            # Again under the account's lock, which a handover to it also takes, so the role cannot land
            # on the account between the checks above and its deletion.
            await self._refuse_bootstrap_operator(identity_id)
            if identity is not None:
                await self._identities.delete(identity)
            await self._erased.add(identity_id)
        return released

    async def _refuse_bootstrap_operator(self, identity_id: uuid.UUID) -> None:
        if await self._organizations.is_bootstrap_operator(identity_id):
            raise BootstrapOperatorProtectedError("The deployment's bootstrap operator cannot delete its account")

    async def _erase_records(self, user_ids: list[str]) -> None:
        """Erase these users' telemetry and agent traces, raising ``AccountErasureFailedError`` if a store refuses."""
        for user_id in user_ids:
            # Broad on purpose: what this raises depends on the store this build bound.
            try:
                await self._telemetry.purge_user(user_id=user_id)
            except Exception as exc:
                logger.error("Telemetry erasure failed for user %s: %s", user_id, type(exc).__name__)
                raise AccountErasureFailedError from None
            # The writer stops writing the user's traces first, or it would write some after the purge.
            if self._trace_writer is not None:
                await self._trace_writer.erase_user(user_id)
            try:
                await self._traces.purge_user(user_id)
            except Exception as exc:
                logger.error("Trace erasure failed for user %s: %s", user_id, type(exc).__name__)
                raise AccountErasureFailedError from None
