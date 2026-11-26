"""Personal access tokens: a person's own read-only credential for one organization's management API."""

import hashlib
import secrets
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from gateway.auth.models import (
    MCP_ACCESS_TOKEN_PREFIX,
    MCP_REFRESH_TOKEN_PREFIX,
    PERSONAL_ACCESS_TOKEN_PREFIX,
    key_suffix,
)
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.identity_exceptions import (
    PersonalAccessTokenBlockedError,
    PersonalAccessTokenLimitReachedError,
    PersonalAccessTokenNoOrganizationError,
    PersonalAccessTokenNotFoundError,
    PersonalAccessTokenSessionEndedError,
)
from gateway.models.identity import MAX_ACCESS_TOKEN_NAME_LENGTH, PersonalAccessToken
from gateway.models.tenancy import User
from gateway.repositories.identity import DashboardSessionRepository, PersonalAccessTokenRepository
from gateway.repositories.tenancy import OrganizationMemberRepository, UserRepository
from gateway.services.identity._blocklist import EmailBlocklist

# A ceiling on a table a signed-in caller writes in a loop they control. Ended tokens do not count.
MAX_PERSONAL_ACCESS_TOKENS = 50
# How long an MCP client's access token lasts before its refresh token renews it.
MCP_ACCESS_TOKEN_LIFETIME = timedelta(hours=1)
# How stale ``last_used_at`` may get before a request writes it again, so a busy
# agent does not turn every read into a write.
_TOUCH_INTERVAL = timedelta(minutes=1)


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


@dataclass(frozen=True)
class McpSignIn:
    """What an MCP client is handed when it signs in or renews: a short-lived token and its renewal."""

    access_token: str
    refresh_token: str
    expires_in: int


class PersonalAccessTokenService:
    def __init__(
        self,
        uow: UnitOfWork,
        tokens: PersonalAccessTokenRepository,
        users: UserRepository,
        sessions: DashboardSessionRepository,
        members: OrganizationMemberRepository,
        blocklist: EmailBlocklist,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.uow, self.tokens, self.users = uow, tokens, users
        self.sessions, self.members, self.blocklist = sessions, members, blocklist
        self.clock = clock

    async def create(
        self, user: User, *, name: str, lifetime: timedelta, session_token: str | None = None
    ) -> tuple[PersonalAccessToken, str]:
        """Mint a token that reads the organization the person is acting in now. Returned once, never stored.

        ``session_token`` is the dashboard session the request authenticated with, re-read here
        because every event that ends an account's tokens also ends its sessions: a session that is
        gone by the time this commits means the token is one the cleanup meant to prevent.

        Raises:
            PersonalAccessTokenBlockedError: the account's address is blocked.
            PersonalAccessTokenLimitReachedError: the account already holds the most it may.
            PersonalAccessTokenNoOrganizationError: the account is acting in no organization.
            PersonalAccessTokenSessionEndedError: the account or the session asking ended meanwhile.
        """
        token = PERSONAL_ACCESS_TOKEN_PREFIX + secrets.token_urlsafe(32)
        now = self.clock()
        async with self.uow:
            # The blocklist's lock before any row lock, as a sign-in takes it: a block commits either
            # before this (and is read here) or after it (and deletes the token this makes).
            if user.email and await self.blocklist.blocked_among({user.email}):
                raise PersonalAccessTokenBlockedError()
            owner = await self._make_room(user.id, now)
            if owner is None or not owner.is_active:
                raise PersonalAccessTokenSessionEndedError()
            if session_token is not None and not await self.sessions.is_live(_hash(session_token), owner.id, now):
                raise PersonalAccessTokenSessionEndedError()
            if owner.active_organization_id is None:
                raise PersonalAccessTokenNoOrganizationError()
            row = await self.tokens.add(
                user_id=owner.id,
                name=name,
                token_hash=_hash(token),
                token_suffix=key_suffix(token),
                created_at=now,
                expires_at=now + lifetime,
                organization_id=owner.active_organization_id,
            )
        return row, token

    async def sign_in_client(
        self,
        user_id: uuid.UUID,
        *,
        organization_id: uuid.UUID,
        client_id: str,
        name: str,
        lifetime: timedelta,
    ) -> McpSignIn | None:
        """Start an MCP client's sign-in for the account by its ID, or None where it is gone or deactivated.

        Raises:
            PersonalAccessTokenLimitReachedError: the account already holds the most it may.
        """
        access, refresh = MCP_ACCESS_TOKEN_PREFIX + secrets.token_urlsafe(32), _new_refresh_token()
        now = self.clock()
        async with self.uow:
            owner = await self.users.get_locked(user_id)
            if owner is None or not owner.is_active:
                return None
            await self._make_room(user_id, now)
            expires_at = now + lifetime
            access_expires_at = min(now + MCP_ACCESS_TOKEN_LIFETIME, expires_at)
            await self.tokens.add(
                user_id=user_id,
                name=name[:MAX_ACCESS_TOKEN_NAME_LENGTH],
                token_hash=_hash(access),
                token_suffix=key_suffix(access),
                created_at=now,
                expires_at=expires_at,
                organization_id=organization_id,
                client_id=client_id,
                access_expires_at=access_expires_at,
                refresh_token_hash=_hash(refresh),
            )
        return McpSignIn(access, refresh, int((access_expires_at - now).total_seconds()))

    async def renew(self, client_id: str, refresh_token: str) -> McpSignIn | None:
        """Rotate a sign-in's tokens, or None where the refresh token renews nothing.

        A refresh token already rotated out ends its sign-in: only a copy someone else kept would
        still be in use (OAuth 2.0 Security BCP, section 4.14).
        """
        now = self.clock()
        presented = _hash(refresh_token)
        access, refresh = MCP_ACCESS_TOKEN_PREFIX + secrets.token_urlsafe(32), _new_refresh_token()
        async with self.uow:
            row = await self.tokens.find_by_refresh_hash_locked(presented)
            if row is None:
                await self.tokens.delete_by_previous_refresh_hash(presented)
                return None
            if row.client_id != client_id or row.expires_at is None or row.expires_at <= now:
                return None
            owner = await self.users.get(row.user_id)
            if owner is None or not owner.is_active or not await self._membership_is_live(row):
                return None
            access_expires_at = min(now + MCP_ACCESS_TOKEN_LIFETIME, row.expires_at)
            await self.tokens.rotate(
                row.id,
                token_hash=_hash(access),
                token_suffix=key_suffix(access),
                access_expires_at=access_expires_at,
                refresh_token_hash=_hash(refresh),
                previous_refresh_token_hash=presented,
            )
        return McpSignIn(access, refresh, int((access_expires_at - now).total_seconds()))

    async def _make_room(self, user_id: uuid.UUID, now: datetime) -> User | None:
        """Lock the account, clear its ended tokens, and refuse where it holds the most it may.

        The lock is what makes the count mean anything: it is held until the block commits, so a
        second create counts what the first one inserted.
        """
        owner = await self.users.get_locked(user_id)
        await self.tokens.delete_expired_for_user(user_id, now)
        if len(await self.tokens.list_for_user(user_id)) >= MAX_PERSONAL_ACCESS_TOKENS:
            raise PersonalAccessTokenLimitReachedError(MAX_PERSONAL_ACCESS_TOKENS)
        return owner

    async def _membership_is_live(self, token: PersonalAccessToken) -> bool:
        """Whether the account is still a member of the organization the token reads.

        The binding is the token's organization, put on the account for the request only. Every
        read resolves its organization from that pointer and falls back to another organization
        when the membership behind it is gone, so the membership is checked here rather than left
        to the read: a token reads one organization, and stops when that membership does.
        """
        if token.organization_id is None:
            return False
        membership = await self.members.get_active_by_organization_and_user(token.organization_id, token.user_id)
        return membership is not None

    async def list(self, user: User) -> Sequence[PersonalAccessToken]:
        """The account's own tokens and MCP sign-ins, newest first."""
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
        """Delete the token presented, or the sign-in a refresh token renews; an unknown token is a no-op."""
        async with self.uow:
            await self.tokens.delete_by_hash(_hash(token))

    async def authenticate(self, token: str) -> User | None:
        """The account a live token speaks for, acting in the token's organization, or None.

        A deactivated account's tokens resolve to nothing, as its dashboard sessions do,
        so deactivating someone ends their agents' access at once.
        """
        now = self.clock()
        async with self.uow:
            row = await self.tokens.find_by_hash(_hash(token))
            if row is None or _ended(row.expires_at, now) or _ended(row.access_expires_at, now):
                return None
            owner = await self.users.get(row.user_id)
            if owner is None or not owner.is_active or not await self._membership_is_live(row):
                return None
            if row.last_used_at is None or now - row.last_used_at >= _TOUCH_INTERVAL:
                await self.tokens.touch(row.id, now)
            self.users.act_in(owner, row.organization_id)
            return owner


def _new_refresh_token() -> str:
    return MCP_REFRESH_TOKEN_PREFIX + secrets.token_urlsafe(32)


def _ended(moment: datetime | None, now: datetime) -> bool:
    return moment is not None and moment <= now
