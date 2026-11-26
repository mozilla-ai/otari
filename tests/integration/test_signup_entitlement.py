"""What an invitation entitles a signup to, and when that is decided."""

from collections.abc import Callable

import pytest
from fastapi import BackgroundTasks
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.adapters.signup_invite_adapter import NullSignupInviteAdapter
from gateway.core.config import GatewayConfig
from gateway.core.tokens import hash_token
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tenancy import Organization, User
from gateway.ports.signup_invite_port import SignupInvitePort
from gateway.repositories.organizations.organization_invitation_repository import (
    OrganizationInvitationRepository,
)
from gateway.repositories.tenancy import OrganizationMemberRepository, UserRepository
from gateway.services.mail import MailNotConfiguredError
from gateway.services.tenancy import OrganizationService
from gateway.services.tenancy.user_service import SignupSubmission, create_user_for_signup

from .tenancy_helpers import NO_BLOCKED_ADDRESSES, create_organization, membership_listener_on

pytestmark = pytest.mark.asyncio

_EMAIL = "ada@example.com"
_PASSWORD = "a-str0ng-Passphrase!"
_TOKEN = "an-invitation-token"


class _EntitlingInvite:
    """Answers that its one token entitles an account."""

    async def allows_an_account(self, token: str, *, claimed_address: str) -> bool:
        del claimed_address
        return token == _TOKEN


class _ReadsThroughTheUnitOfWork:
    """Answers by reading through the Unit of Work it was built on, opening no block of its own.

    ``session_for`` raises ``OutsideUnitOfWorkError`` where no block is open, so this answers
    only while the caller holds one.
    """

    def __init__(self, uow: UnitOfWork) -> None:
        self._uow = uow

    async def allows_an_account(self, token: str, *, claimed_address: str) -> bool:
        del claimed_address
        await OrganizationInvitationRepository(self._uow).get_by_token_hash(hash_token(token))
        return True


def _config(*, sends_links: bool = True, open_signup: bool = False) -> GatewayConfig:
    return GatewayConfig(
        open_signup=open_signup,
        mail_from_email="otari@example.com",
        mail_transport="smtp" if sends_links else "console",
        public_base_url="https://otari.example.com" if sends_links else None,
        smtp_host="smtp.example.com",
    )


class _OpensItsOwnBlock:
    """Answers no, having opened a block on the caller's Unit of Work, as the real reader does."""

    def __init__(self, uow: UnitOfWork) -> None:
        self._uow = uow

    async def allows_an_account(self, token: str, *, claimed_address: str) -> bool:
        del claimed_address
        async with self._uow:
            await OrganizationInvitationRepository(self._uow).get_by_token_hash(hash_token(token))
        return False


async def _seed_unclaimed_identity(session: AsyncSession, *, still_wanted: bool = True) -> None:
    """An identity an admin added by address: password-less and unclaimed.

    An unverified identity is claimable on a closed deployment only while somebody still wants
    it, which ``still_wanted`` is.
    """
    organization = await create_organization(session, slug="signup")
    identity = await UserRepository(session).create_local_identity(
        full_name="Ada", active_organization_id=organization.id, email=_EMAIL
    )
    if still_wanted:
        await OrganizationMemberRepository(session).create_membership(
            organization_id=organization.id, user_id=identity.id, role="member"
        )
    await session.flush()


async def _organization_count(session: AsyncSession) -> int:
    return (await session.execute(select(func.count()).select_from(Organization))).scalar_one()


async def _signup(
    session: AsyncSession,
    *,
    invites: Callable[[UnitOfWork], SignupInvitePort],
    config: GatewayConfig | None = None,
) -> User | None:
    """Sign up through the service, with ``invites`` built on the same Unit of Work the write uses."""
    uow = UnitOfWork(session)
    return await create_user_for_signup(
        session,
        config or _config(),
        background_tasks=BackgroundTasks(),
        blocked_addresses=NO_BLOCKED_ADDRESSES,
        invites=invites(uow),
        organizations=OrganizationService(session, membership_listener=membership_listener_on(uow), uow=uow),
        submission=SignupSubmission(email=_EMAIL, password=_PASSWORD, invite_token=_TOKEN),
        uow=uow,
    )


async def test_an_entitled_signup_proves_the_address(async_db: AsyncSession) -> None:
    """An invitation entitles an account only where its link reached the mailbox."""
    await _seed_unclaimed_identity(async_db)
    claimed = await _signup(async_db, invites=lambda _: _EntitlingInvite())

    assert claimed is not None
    assert claimed.email_verified_at is not None
    assert claimed.email_proven_at is not None


async def test_an_entitled_signup_proceeds_where_no_link_can_be_mailed(async_db: AsyncSession) -> None:
    """The link already reached the mailbox, so no deployment setting gates it."""
    await _seed_unclaimed_identity(async_db)
    claimed = await _signup(async_db, invites=lambda _: _EntitlingInvite(), config=_config(sends_links=False))

    assert claimed is not None
    assert claimed.email_verified_at is not None


async def test_an_unentitled_signup_is_refused_where_no_link_can_be_mailed(async_db: AsyncSession) -> None:
    """Without an invitation the verification link is the only way out of the unverified state."""
    with pytest.raises(MailNotConfiguredError):
        await _signup(async_db, invites=lambda _: NullSignupInviteAdapter(), config=_config(sends_links=False))


async def test_the_entitlement_is_decided_inside_the_transaction_that_writes_the_account(
    async_db: AsyncSession,
) -> None:
    """The answer is read where the account is written, so a cancel in between is not honored.

    This reader opens no block of its own, which is what makes the placement observable.
    """
    await _seed_unclaimed_identity(async_db)
    claimed = await _signup(async_db, invites=_ReadsThroughTheUnitOfWork)

    assert claimed is not None
    assert claimed.email_verified_at is not None


async def test_an_entitled_signup_creates_an_account_where_signup_is_closed(async_db: AsyncSession) -> None:
    """The invitation is the way in, so the rule refusing an unknown address does not reach it."""
    claimed = await _signup(async_db, invites=lambda _: _EntitlingInvite())

    assert claimed is not None
    assert claimed.email_verified_at is not None


@pytest.mark.parametrize(
    "still_wanted",
    [None, True, False],
    ids=["unknown address", "an address an admin added", "an address nobody still wants"],
)
async def test_an_entitled_signup_founds_no_organization(async_db: AsyncSession, still_wanted: bool | None) -> None:
    """The membership is the invitation's to grant when it is accepted, so this writes the account alone.

    Each case reaches a different provisioning call, and the last is the one that would found an
    organization for an identity no membership still wants.
    """
    if still_wanted is not None:
        await _seed_unclaimed_identity(async_db, still_wanted=still_wanted)
    before = await _organization_count(async_db)

    claimed = await _signup(async_db, invites=lambda _: _EntitlingInvite())

    assert claimed is not None
    assert await _organization_count(async_db) == before


async def test_an_entitled_signup_claims_an_identity_nobody_still_wants(async_db: AsyncSession) -> None:
    """A revoked invitation leaves the identity behind, and a new one entitles it again."""
    await _seed_unclaimed_identity(async_db, still_wanted=False)

    claimed = await _signup(async_db, invites=lambda _: _EntitlingInvite())

    assert claimed is not None
    assert claimed.email_verified_at is not None


async def test_an_entitled_signup_founds_no_organization_on_an_open_deployment(async_db: AsyncSession) -> None:
    """Founding one is forbidden for an invited person whether this deployment's signup is open or closed.

    An open deployment is the only one that would found an organization for an identity no
    membership still wants, so it is the case the rule has to be checked against.
    """
    await _seed_unclaimed_identity(async_db, still_wanted=False)
    before = await _organization_count(async_db)

    claimed = await _signup(async_db, invites=lambda _: _EntitlingInvite(), config=_config(open_signup=True))

    assert claimed is not None
    assert await _organization_count(async_db) == before


async def test_a_mail_refusal_after_the_reader_opened_a_block_reaches_the_caller(async_db: AsyncSession) -> None:
    """A nested block joins the outer one and commits nothing, so the refusal is what the caller sees.

    The readiness check moved inside the block, where a reader may already have opened and
    closed one of its own.
    """
    await _seed_unclaimed_identity(async_db)

    with pytest.raises(MailNotConfiguredError):
        await _signup(async_db, invites=_OpensItsOwnBlock, config=_config(sends_links=False))
