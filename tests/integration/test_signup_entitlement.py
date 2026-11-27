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


@pytest.mark.parametrize(
    "invites",
    [lambda _: _EntitlingInvite(), lambda _: NullSignupInviteAdapter()],
    ids=["an invitation that allows an account", "no invitation"],
)
async def test_a_signup_is_refused_where_no_link_can_be_mailed(
    async_db: AsyncSession, invites: Callable[[UnitOfWork], SignupInvitePort]
) -> None:
    """The verification link is the only way out of the unverified state, invitation or not.

    The refusal is asked before any address is read, so it cannot differ by address either.
    """
    with pytest.raises(MailNotConfiguredError):
        await _signup(async_db, invites=invites, config=_config(sends_links=False))


async def test_the_entitlement_is_read_while_the_writing_block_is_open(
    async_db: AsyncSession,
) -> None:
    """The reader is asked inside the block that writes the account, so the two commit together.

    This reader opens no block of its own, which is what makes the placement observable. It says
    nothing about a cancellation, which the read does not lock against.
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


async def test_an_invited_address_with_no_account_gets_an_organization_of_its_own(
    async_db: AsyncSession,
) -> None:
    """A deployment that refuses an open signup still registers an invited address, with a tenancy.

    No signup leaves an account holding no organization. An address an organization invited already
    holds one, set when the invitation was created, so the only invitation reaching this branch is
    one to the deployment itself.
    """
    before = await _organization_count(async_db)

    claimed = await _signup(async_db, invites=lambda _: _EntitlingInvite())

    assert claimed is not None
    assert claimed.active_organization_id is not None
    assert await _organization_count(async_db) == before + 1


async def test_an_entitled_signup_claims_an_identity_nobody_still_wants(async_db: AsyncSession) -> None:
    """A revoked invitation leaves the identity behind, and a new one entitles it again."""
    await _seed_unclaimed_identity(async_db, still_wanted=False)

    claimed = await _signup(async_db, invites=lambda _: _EntitlingInvite())

    assert claimed is not None
    assert claimed.email_verified_at is not None


async def test_an_invitation_leaves_the_organization_question_to_the_rules_that_own_it(
    async_db: AsyncSession,
) -> None:
    """An invitation registers an address. It does not decide whether an organization is founded.

    An identity no membership still wants gets what any claim of it gets, which on an open
    deployment is an organization of its own.
    """
    await _seed_unclaimed_identity(async_db, still_wanted=False)
    before = await _organization_count(async_db)

    claimed = await _signup(async_db, invites=lambda _: _EntitlingInvite(), config=_config(open_signup=True))

    assert claimed is not None
    assert await _organization_count(async_db) == before + 1
