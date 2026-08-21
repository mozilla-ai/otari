"""The races the tenancy services resolve, driven concurrently.

Every check-then-act path here is decided by a unique constraint rather than by
its own pre-check, and the pre-check only makes the common case a clean 409. The
fixes for that are one ``except IntegrityError`` each, which is the kind of line
a later refactor removes without a test failing, so these drive the real race
with separate sessions rather than asserting the branch in isolation.
"""

import asyncio
import uuid
from collections.abc import Callable

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.auth.models import hash_key
from gateway.core.config import GatewayConfig
from gateway.models.entities import APIKey, WorkspaceActivationState
from gateway.models.tenancy import (
    ActiveOrganizationMemberCreateRequest,
    ActiveOrganizationMemberUpdateRequest,  # noqa: E402
    InviteOrganizationMemberRequest,
    Organization,
    User,
    WorkspaceAssignmentRequest,
    WorkspaceCreate,
)
from gateway.repositories.tenancy import (
    InvitationRepository,
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.services.tenancy import OrganizationService, WorkspaceService, user_service
from gateway.services.tenancy.errors import (
    EmailAlreadyInUseError,
    ForeignTenancyError,
    InvitationAlreadyPendingError,
    InvitationAlreadyUsedError,
    LastWorkspaceError,
    MembershipUpdateError,
    OrganizationMemberAlreadyExistsError,
    WorkspaceAlreadyExistsError,
    WorkspaceMemberAlreadyExistsError,
)
from gateway.services.tenancy.provisioning_service import (
    BOOTSTRAP_IDENTITY_KEY,
    ensure_bootstrap_identity,
)
from gateway.services.tenancy.user_service import set_password
from gateway.services.tenancy.workspace_activation_service import (
    ACTIVATION_KEY_NAME,
    WorkspaceActivationService,
)

pytestmark = pytest.mark.asyncio

_RACERS = 4


async def _seed_owner(db: AsyncSession) -> tuple[Organization, User]:
    organization = await OrganizationRepository(db).create_organization(
        name="Acme",
        slug=f"acme-{uuid.uuid4().hex[:8]}",
        created_by_user_id=None,
    )
    owner = await UserRepository(db).create_local_identity(
        full_name="Owner",
        active_organization_id=organization.id,
    )
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id,
        user_id=owner.id,
        role="owner",
    )
    await db.commit()
    return organization, owner


async def _race(
    session_factory: async_sessionmaker[AsyncSession],
    attempt: Callable[[AsyncSession], object],
) -> list[object]:
    """Run one attempt per racer, each on its own session, and collect outcomes."""

    async def run_one() -> object:
        async with session_factory() as session:
            try:
                return await attempt(session)  # type: ignore[misc]
            except Exception as exc:  # noqa: BLE001 - the outcome is the assertion
                return exc

    return list(await asyncio.gather(*(run_one() for _ in range(_RACERS))))


@pytest.fixture
def sessions(postgres_url: str) -> async_sessionmaker[AsyncSession]:
    url = postgres_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://").replace(
        "postgresql://", "postgresql+asyncpg://"
    )
    return async_sessionmaker(create_async_engine(url), expire_on_commit=False)


async def test_concurrent_workspace_creates_conflict_rather_than_fail(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    organization, owner = await _seed_owner(async_db)

    async def attempt(session: AsyncSession) -> object:
        user = await UserRepository(session).get(owner.id)
        assert user is not None
        return await WorkspaceService(session).create_workspace(
            user=user,
            workspace_create=WorkspaceCreate(name="Research"),
        )

    outcomes = await _race(sessions, attempt)

    created = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
    conflicts = [outcome for outcome in outcomes if isinstance(outcome, WorkspaceAlreadyExistsError)]
    assert len(created) == 1
    assert len(conflicts) == _RACERS - 1
    _, count = await WorkspaceRepository(async_db).get_by_organization(organization.id, limit=1)
    assert count == 1


async def test_concurrent_claims_of_one_address_leave_exactly_one_holder(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two identities claiming the same sign-in address at once.

    ``_claimable_email`` is a preflight, not a lock, so both racers can pass it
    and the unique index on ``user.email`` is what actually decides. The loser
    has to report the conflict its preflight would have reported rather than
    surfacing the driver's integrity error as a 500. Postgres is the engine that
    matters here: the unit suite runs on SQLite, whose error text differs, and
    the detector has to recognize both.
    """
    organization, _ = await _seed_owner(async_db)
    # Count how many losers came through the integrity-error mapping. Without
    # this the test passes whether the racers overlapped or not: a serialized
    # run refuses them at the ``_claimable_email`` preflight, which raises the
    # very same ``EmailAlreadyInUseError``, so the outcome assertions below
    # cannot tell the two routes apart and would green-light a fix that never
    # runs.
    mapped: list[bool] = []
    real_detector = user_service._is_email_conflict

    def counting_detector(exc: IntegrityError) -> bool:
        verdict = real_detector(exc)
        mapped.append(verdict)
        return verdict

    monkeypatch.setattr(user_service, "_is_email_conflict", counting_detector)

    users = UserRepository(async_db)
    racer_ids = [
        (
            await users.create_local_identity(
                full_name=f"Claimer {index}", active_organization_id=organization.id
            )
        ).id
        for index in range(_RACERS)
    ]
    await async_db.commit()
    # One identity per racer, handed out in order. `next` on a plain iterator is
    # safe here because the racers only interleave at their awaits.
    hand_out = iter(racer_ids)

    async def attempt(session: AsyncSession) -> object:
        # Each racer claims *its own* identity, all of them naming one address.
        identity = await UserRepository(session).get(next(hand_out))
        assert identity is not None
        await set_password(
            session,
            identity,
            new_password="a-real-password",
            email="contested@example.com",
        )
        # `set_password` returns None, so the racer that got through reports a
        # marker rather than a value indistinguishable from "nothing happened".
        return "claimed"

    outcomes = await _race(sessions, attempt)

    claimed = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
    refused = [outcome for outcome in outcomes if isinstance(outcome, EmailAlreadyInUseError)]
    escaped = [o for o in outcomes if isinstance(o, Exception) and not isinstance(o, EmailAlreadyInUseError)]
    assert not escaped, f"a raw database error reached the caller: {escaped}"
    assert len(claimed) == 1, outcomes
    assert len(refused) == _RACERS - 1, outcomes
    # Every loser reached the unique index rather than the preflight, which is
    # what makes this a test of the mapping and not of the pre-check.
    assert mapped == [True] * (_RACERS - 1), mapped

    holders = (
        (await async_db.execute(select(User).where(col(User.email) == "contested@example.com"))).scalars().all()
    )
    assert len(holders) == 1


async def test_concurrent_member_adds_create_one_identity(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """The email unique index decides, and the losers report the conflict."""
    _, owner = await _seed_owner(async_db)

    async def attempt(session: AsyncSession) -> object:
        user = await UserRepository(session).get(owner.id)
        assert user is not None
        return await OrganizationService(session).create_active_organization_member_for_user(
            user=user,
            request=ActiveOrganizationMemberCreateRequest(email="ada@example.com"),
        )

    outcomes = await _race(sessions, attempt)

    added = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
    conflicts = [outcome for outcome in outcomes if isinstance(outcome, OrganizationMemberAlreadyExistsError)]
    assert len(added) == 1
    assert len(conflicts) == _RACERS - 1
    assert await UserRepository(async_db).get_by_email("ada@example.com") is not None


async def test_concurrent_workspace_member_adds_conflict(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    _, owner = await _seed_owner(async_db)
    service = OrganizationService(async_db)
    owner_row = await UserRepository(async_db).get(owner.id)
    assert owner_row is not None
    added = await service.create_active_organization_member_for_user(
        user=owner_row,
        request=ActiveOrganizationMemberCreateRequest(email="ada@example.com"),
    )
    workspace = await WorkspaceService(async_db).create_workspace(
        user=owner_row,
        workspace_create=WorkspaceCreate(name="Research"),
    )
    assert added.user_id is not None

    async def attempt(session: AsyncSession) -> object:
        user = await UserRepository(session).get(owner.id)
        assert user is not None
        return await WorkspaceService(session).add_member(
            user=user,
            workspace_id=workspace.id,
            user_id=added.user_id,  # type: ignore[arg-type]
        )

    outcomes = await _race(sessions, attempt)

    joined = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
    conflicts = [outcome for outcome in outcomes if isinstance(outcome, WorkspaceMemberAlreadyExistsError)]
    assert len(joined) == 1
    assert len(conflicts) == _RACERS - 1


async def test_provisioning_refuses_to_shadow_an_organization_it_did_not_create(
    test_db: Session,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """A restored or imported tenancy must not be silently made unreachable.

    Provisioning adopts an organization slugged ``default``, which is the one it
    would have made itself. Anything else it would ignore, create its own beside,
    and point the marker at that; every route is scoped to the marked identity's
    organization, so the restored rows become invisible with no list, no switch
    and no by-id route to find them. The platform slugs organizations
    ``{name}-{prefix}``, so a restored hosted organization always lands here.
    """
    async with sessions() as db:
        await _seed_owner(db)

        with pytest.raises(ForeignTenancyError) as raised:
            await ensure_bootstrap_identity(db)

        # The message has to name the organization and the way out, because the
        # marker is not a settable key and nothing else can repoint it.
        assert "Acme" in str(raised.value)
        assert BOOTSTRAP_IDENTITY_KEY in str(raised.value)


async def test_provisioning_still_runs_on_an_empty_database(
    test_db: Session,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """The guard must not break first boot, which is the ordinary path."""
    async with sessions() as db:
        operator = await ensure_bootstrap_identity(db)

        assert operator.full_name == "Operator"


async def test_concurrent_demotions_cannot_strip_the_last_owner(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Two owners, each demoted by a different concurrent request.

    Unlike the three races above there is no unique index to lose to, so the
    ``IntegrityError`` guards never fire and only the row lock in
    ``OrganizationRepository.lock`` keeps the count the guard read still true when
    it writes. Without it both requests count two owners and both commit.
    """
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme",
        slug=f"acme-{uuid.uuid4().hex[:8]}",
        created_by_user_id=None,
    )
    members = OrganizationMemberRepository(async_db)
    users = UserRepository(async_db)
    owners = []
    for index in range(2):
        owner = await users.create_local_identity(
            full_name=f"Owner {index}",
            active_organization_id=organization.id,
            email=f"owner{index}@example.com",
        )
        membership = await members.create_membership(
            organization_id=organization.id,
            user_id=owner.id,
            role="owner",
        )
        owners.append((owner.id, membership.id))
    await async_db.commit()

    def demote(actor_id: uuid.UUID, target_membership_id: uuid.UUID) -> Callable[[AsyncSession], object]:
        async def attempt(session: AsyncSession) -> object:
            actor = await UserRepository(session).get(actor_id)
            assert actor is not None
            return await OrganizationService(session).update_active_organization_member_for_user(
                user=actor,
                organization_member_id=target_membership_id,
                request=ActiveOrganizationMemberUpdateRequest(role="member"),
            )

        return attempt

    attempts = [
        demote(owners[0][0], owners[1][1]),
        demote(owners[1][0], owners[0][1]),
    ]

    async def run_one(attempt: Callable[[AsyncSession], object]) -> object:
        async with sessions() as session:
            try:
                return await attempt(session)  # type: ignore[misc]
            except Exception as exc:  # noqa: BLE001 - the outcome is the assertion
                return exc

    outcomes = list(await asyncio.gather(*(run_one(attempt) for attempt in attempts)))

    refused = [outcome for outcome in outcomes if isinstance(outcome, MembershipUpdateError)]
    assert len(refused) == 1
    assert await OrganizationMemberRepository(async_db).count_active_owners(organization.id) == 1


async def test_concurrent_deletes_cannot_remove_the_last_workspace(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Two workspaces, deleted concurrently. One has to survive."""
    organization, owner = await _seed_owner(async_db)
    service = WorkspaceService(async_db)
    workspace_ids = [
        (await service.create_workspace(user=owner, workspace_create=WorkspaceCreate(name=name))).id
        for name in ("One", "Two")
    ]

    async def run_one(workspace_id: uuid.UUID) -> object:
        async with sessions() as session:
            user = await UserRepository(session).get(owner.id)
            assert user is not None
            try:
                await WorkspaceService(session).delete_workspace(user=user, workspace_id=workspace_id)
            except Exception as exc:  # noqa: BLE001 - the outcome is the assertion
                return exc
            return None

    outcomes = list(await asyncio.gather(*(run_one(workspace_id) for workspace_id in workspace_ids)))

    refused = [outcome for outcome in outcomes if isinstance(outcome, LastWorkspaceError)]
    assert len(refused) == 1
    _, remaining = await WorkspaceRepository(async_db).get_by_organization(organization.id, limit=1)
    assert remaining == 1


async def test_concurrent_invites_to_a_suspended_membership_produce_one_pending_invitation(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """A suspended membership has no unique index to lose to either.

    `organization_member_id` on `Invitation` carries no uniqueness (a
    membership can be invited, revoked, and re-invited more than once over its
    life), so nothing catches two concurrent invites to the same suspended
    membership as an `IntegrityError`. Without locking the organization before
    the status check that decides create/revive/refuse, both racers can read
    "suspended", both revive it, and both mint their own live pending
    invitation for the one membership.
    """
    organization, owner = await _seed_owner(async_db)
    owner_row = await UserRepository(async_db).get(owner.id)
    assert owner_row is not None
    added = await OrganizationService(async_db).create_active_organization_member_for_user(
        user=owner_row,
        request=ActiveOrganizationMemberCreateRequest(email="grace@example.com"),
    )
    assert added.organization_member_id is not None
    assert added.user_id is not None
    await OrganizationService(async_db).remove_active_organization_member_for_user(
        user=owner_row,
        organization_member_id=added.organization_member_id,
    )
    config = GatewayConfig()

    async def attempt(session: AsyncSession) -> object:
        user = await UserRepository(session).get(owner.id)
        assert user is not None
        return await OrganizationService(session).invite_active_organization_member_for_user(
            user=user,
            request=InviteOrganizationMemberRequest(email="grace@example.com"),
            config=config,
        )

    outcomes = await _race(sessions, attempt)

    invited = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
    # Whichever racer wins the lock leaves the membership `invited` with a
    # fresh, unexpired invitation, so every loser re-reads that and raises
    # InvitationAlreadyPendingError, not OrganizationMemberAlreadyExistsError
    # (that one is for an *active* membership, which none of the racers here
    # ever produce: the starting status is `suspended`).
    conflicts = [outcome for outcome in outcomes if isinstance(outcome, InvitationAlreadyPendingError)]
    assert len(invited) == 1
    assert len(conflicts) == _RACERS - 1

    membership = await OrganizationMemberRepository(async_db).get_by_organization_and_user(
        organization.id, added.user_id
    )
    assert membership is not None
    pending = await InvitationRepository(async_db).get_pending_by_organization_members([membership.id])
    assert len(pending) == 1


async def test_concurrent_accepts_of_one_invitation_produce_one_active_membership(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """accept_invitation's pending check is check-then-act too, with a worse failure mode.

    Without locking before the re-check, two concurrent accepts of the same
    token both see `pending`, both flip the membership, and both reach
    `_apply_workspace_assignments`, whose existing-then-create shape lets the
    second racer's insert violate `uq_workspace_member_workspace_user` as an
    uncaught `IntegrityError` on a public, unauthenticated endpoint, rather
    than the mapped `InvitationAlreadyUsedError` every other double-use path
    already answers with.
    """
    organization, owner = await _seed_owner(async_db)
    owner_row = await UserRepository(async_db).get(owner.id)
    assert owner_row is not None
    workspace = await WorkspaceService(async_db).create_workspace(
        user=owner_row,
        workspace_create=WorkspaceCreate(name="Research"),
    )
    config = GatewayConfig()
    invited = await OrganizationService(async_db).invite_active_organization_member_for_user(
        user=owner_row,
        request=InviteOrganizationMemberRequest(
            email="hank@example.com",
            workspace_assignments=[WorkspaceAssignmentRequest(workspace_id=workspace.id, role="viewer")],
        ),
        config=config,
    )
    token = invited.accept_link.split("token=")[1]

    async def attempt(session: AsyncSession) -> object:
        return await OrganizationService(session).accept_invitation(token)

    outcomes = await _race(sessions, attempt)

    accepted = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
    already_used = [outcome for outcome in outcomes if isinstance(outcome, InvitationAlreadyUsedError)]
    assert len(accepted) == 1
    assert len(already_used) == _RACERS - 1

    # async_db's own invite call above left the membership row resident in this
    # session's identity map with status "invited"; with expire_on_commit=False,
    # a plain get() would return that unexpired cached instance rather than
    # querying the row the race committed through separate sessions.
    async_db.expire_all()
    membership = await OrganizationMemberRepository(async_db).get(invited.organization_member_id)
    assert membership is not None
    assert membership.status == "active"
    # Exactly one workspace_member row, not one per racer that reached
    # _apply_workspace_assignments before the lock closed this off.
    workspace_members = await WorkspaceMemberRepository(async_db).get_by_workspaces_and_user(
        {workspace.id: "viewer"}, membership.user_id
    )
    assert len(workspace_members) == 1


async def test_concurrent_accept_and_revoke_of_one_invitation_produce_one_consistent_outcome(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """revoke_organization_member_invitation_for_user races accept_invitation too.

    Both are check-then-act on the same invitation's `pending` status, and
    until revoke's organization lock moved ahead of its own reads, this had a
    worse failure mode than the accept-vs-accept race above: revoke's read of
    the invitation and membership happened before it took any lock, so a
    revoke that started just ahead of a winning accept could sit on the lock
    call inside `_validate_membership_update`, wake up once that accept had
    fully committed, and then unconditionally overwrite the accept's
    `active`/`accepted` state with its own stale, pre-lock `suspended`/
    `cancelled` write. Both operations would report success, and the invitee
    who just accepted would silently lose the membership they were told they
    had.
    """
    organization, owner = await _seed_owner(async_db)
    owner_row = await UserRepository(async_db).get(owner.id)
    assert owner_row is not None
    invited = await OrganizationService(async_db).invite_active_organization_member_for_user(
        user=owner_row,
        request=InviteOrganizationMemberRequest(email="ivy@example.com"),
        config=GatewayConfig(),
    )
    token = invited.accept_link.split("token=")[1]
    assert invited.invitation_id is not None

    async def accept(session: AsyncSession) -> object:
        return await OrganizationService(session).accept_invitation(token)

    async def revoke(session: AsyncSession) -> object:
        user = await UserRepository(session).get(owner.id)
        assert user is not None
        assert invited.invitation_id is not None
        await OrganizationService(session).revoke_organization_member_invitation_for_user(
            user=user,
            invitation_id=invited.invitation_id,
        )
        return None

    async def run_one(attempt: Callable[[AsyncSession], object]) -> object:
        async with sessions() as session:
            try:
                return await attempt(session)  # type: ignore[misc]
            except Exception as exc:  # noqa: BLE001 - the outcome is the assertion
                return exc

    outcomes = list(await asyncio.gather(*(run_one(attempt) for attempt in (accept, revoke))))

    # Exactly one side has to lose, and lose with the mapped error: accept and
    # revoke are different operations, but they are still racing the same
    # pending-to-something-else transition, so only one may win it.
    losses = [outcome for outcome in outcomes if isinstance(outcome, InvitationAlreadyUsedError)]
    assert len(losses) == 1

    async_db.expire_all()
    invitation = await InvitationRepository(async_db).get(invited.invitation_id)
    membership = await OrganizationMemberRepository(async_db).get(invited.organization_member_id)
    assert invitation is not None
    assert membership is not None
    # The invitation and its paired membership have to agree on which
    # operation won, never a mix showing one operation's write and the other's
    # leftover state.
    assert (invitation.status, membership.status) in {
        ("accepted", "active"),
        ("cancelled", "suspended"),
    }


async def test_concurrent_first_issuance_leaves_one_setup_key(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Every racer asking a fresh workspace for its setup key gets the same one.

    The guide's promise is one "Setup guide" key per workspace, rotated in place,
    and on a first call there is no state row to lock, so the serialization point
    is the insert of that row rather than a pre-check. Minting the key before
    creating it (which is what this used to do) let each racer commit a key of
    its own, leaving a workspace with several live credentials nobody asked for
    and a state row naming whichever committed last.
    """
    organization, owner = await _seed_owner(async_db)
    workspace = await WorkspaceRepository(async_db).create_workspace(
        name="Engineering",
        organization_id=organization.id,
        created_by_user_id=owner.id,
    )
    await WorkspaceMemberRepository(async_db).create(
        workspace_id=workspace.id,
        user_id=owner.id,
        role="owner",
    )
    await async_db.commit()

    async def attempt(session: AsyncSession) -> object:
        user = await UserRepository(session).get(owner.id)
        assert user is not None
        return await WorkspaceActivationService(session, GatewayConfig()).issue_api_key(
            user=user,
            workspace_id=workspace.id,
        )

    workspace_id = workspace.id
    outcomes = await _race(sessions, attempt)

    # Every racer, not merely one: a losing call has to come back with a key
    # rather than with the integrity error it hit on the way, which is the half
    # of this a "one row survived" assertion cannot see.
    issued = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
    assert len(issued) == _RACERS, f"some racers failed: {outcomes}"

    # Expired first, and every value read out into a local before the next
    # await: this session predates the racers' commits, and a lazily refreshed
    # attribute on an ``AsyncSession`` raises ``MissingGreenlet`` rather than
    # reloading.
    async_db.expire_all()
    keys = (await async_db.execute(select(APIKey).where(col(APIKey.workspace_id) == workspace_id))).scalars().all()
    key_ids = [key.id for key in keys]
    key_names = [key.key_name for key in keys]
    key_hash = keys[0].key_hash
    state = await async_db.get(WorkspaceActivationState, workspace_id)
    assert state is not None
    named_key = state.api_key_id

    # One row, and every racer that got a key got that row's, so no plaintext was
    # handed out for a credential the workspace does not carry.
    assert key_names == [ACTIVATION_KEY_NAME]
    assert {issue.key_id for issue in issued} == set(key_ids)  # type: ignore[attr-defined]
    assert named_key == key_ids[0]

    # Exactly one of the plaintexts authenticates, and that is the design rather
    # than a gap in it: each call rotates the row, so the last writer's key is
    # the live one and the earlier ones are invalidated exactly as a page reload
    # invalidates the key before it. Asserting it here keeps that a stated
    # property, so a future change that quietly handed out several live keys for
    # one row would fail rather than look like an improvement.
    live = [issue for issue in issued if hash_key(issue.key) == key_hash]  # type: ignore[attr-defined]
    assert len(live) == 1
