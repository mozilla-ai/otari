"""Personal access tokens: a person's own read-only credential for the management API."""

import hashlib
import secrets
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

from gateway.auth.models import PERSONAL_ACCESS_TOKEN_PREFIX, key_suffix
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.identity_exceptions import (
    PersonalAccessTokenLimitReachedError,
    PersonalAccessTokenNotFoundError,
)
from gateway.models.identity import MAX_ACCESS_TOKEN_NAME_LENGTH, PersonalAccessToken
from gateway.models.tenancy import User
from gateway.repositories.identity import PersonalAccessTokenRepository
from gateway.repositories.tenancy import UserRepository

# A ceiling on a table a signed-in caller writes in a loop they control.
MAX_PERSONAL_ACCESS_TOKENS = 50
# How stale ``last_used_at`` may get before a request writes it again, so a busy
# agent does not turn every read into a write.
_TOUCH_INTERVAL = timedelta(minutes=1)


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class PersonalAccessTokenService:
    def __init__(
        self,
        uow: UnitOfWork,
        tokens: PersonalAccessTokenRepository,
        users: UserRepository,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.uow, self.tokens, self.users, self.clock = uow, tokens, users, clock

    async def create(self, user: User, *, name: str, lifetime: timedelta | None) -> tuple[PersonalAccessToken, str]:
        """Mint a token for this account. The token is returned once and never stored.

        Raises:
            PersonalAccessTokenLimitReachedError: the account already holds the most it may.
        """
        token = PERSONAL_ACCESS_TOKEN_PREFIX + secrets.token_urlsafe(32)
        now = self.clock()
        async with self.uow:
            # Locked so two concurrent creates cannot both count one below the limit.
            await self.users.get_locked(user.id)
            if len(await self.tokens.list_for_user(user.id)) >= MAX_PERSONAL_ACCESS_TOKENS:
                raise PersonalAccessTokenLimitReachedError(MAX_PERSONAL_ACCESS_TOKENS)
            row = await self.tokens.add(
                user_id=user.id,
                name=name,
                token_hash=_hash(token),
                token_suffix=key_suffix(token),
                created_at=now,
                expires_at=None if lifetime is None else now + lifetime,
            )
        return row, token

    async def create_for_account(self, user_id: uuid.UUID, *, name: str, lifetime: timedelta | None) -> str | None:
        """Mint a token for the account by its ID, or None where it is gone or deactivated.

        Raises:
            PersonalAccessTokenLimitReachedError: the account already holds the most it may.
        """
        async with self.uow:
            owner = await self.users.get_locked(user_id)
            if owner is None or not owner.is_active:
                return None
            _row, token = await self.create(owner, name=name[:MAX_ACCESS_TOKEN_NAME_LENGTH], lifetime=lifetime)
        return token

    async def list(self, user: User) -> Sequence[PersonalAccessToken]:
        """The account's own tokens, newest first."""
        async with self.uow:
            return await self.tokens.list_for_user(user.id)

    async def revoke(self, user: User, token_id: uuid.UUID) -> None:
        """Delete one of the account's tokens.

        Raises:
            PersonalAccessTokenNotFoundError: the account holds no token by that ID.
        """
        async with self.uow:
            if not await self.tokens.delete_for_user(user.id, token_id):
                raise PersonalAccessTokenNotFoundError(token_id)

    async def revoke_presented(self, token: str) -> None:
        """Delete the token presented, whoever holds it; an unknown token is a no-op."""
        async with self.uow:
            await self.tokens.delete_by_hash(_hash(token))

    async def authenticate(self, token: str) -> User | None:
        """The account a stored, unexpired token speaks for, or None.

        A deactivated account's tokens resolve to nothing, as its dashboard sessions do,
        so deactivating someone ends their agents' access at once.
        """
        now = self.clock()
        async with self.uow:
            row = await self.tokens.find_by_hash(_hash(token))
            if row is None or (row.expires_at is not None and row.expires_at <= now):
                return None
            owner = await self.users.get(row.user_id)
            if owner is None or not owner.is_active:
                return None
            if row.last_used_at is None or now - row.last_used_at >= _TOUCH_INTERVAL:
                await self.tokens.touch(row.id, now)
            return owner
