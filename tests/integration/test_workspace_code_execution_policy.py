"""Per-workspace code-execution policy: CRUD, authorization, and what the request path reads.

Exercised at the service layer, matching `test_workspace_member_budget_policies.py`:
the API can only ever act as the one superuser operator identity a standalone
deployment has, so the rules that matter most (a plain member refused, a
foreign workspace refused) are only reachable by calling the service with
identities built at whatever role a case needs.
"""

import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from gateway.models.entities import WorkspaceCodeExecutionPolicy
from gateway.models.tenancy import Organization, User, Workspace
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.services.tenancy.errors import NotAuthorizedError, WorkspaceNotFoundError
from gateway.services.tenancy.workspace_code_execution_policy_service import (
    WorkspaceCodeExecutionPolicyService,
    WorkspaceCodeExecutionPolicyUpdate,
    resolve_workspace_code_execution_policy,
)

pytestmark = pytest.mark.asyncio


async def _organization(db: AsyncSession, *, slug: str) -> Organization:
    return await OrganizationRepository(db).create_organization(name=slug.title(), slug=slug, created_by_user_id=None)


async def _member(db: AsyncSession, organization: Organization, *, role: str, full_name: str) -> User:
    user = await UserRepository(db).create_local_identity(
        full_name=full_name,
        active_organization_id=organization.id,
    )
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id,
        user_id=user.id,
        role=role,
    )
    return user


async def _workspace(db: AsyncSession, organization: Organization, *, name: str, owner: User) -> Workspace:
    workspace = await WorkspaceRepository(db).create_workspace(
        name=name,
        organization_id=organization.id,
        created_by_user_id=owner.id,
    )
    await WorkspaceMemberRepository(db).create(workspace_id=workspace.id, user_id=owner.id, role="owner")
    return workspace


def _service(db: AsyncSession, *, sandbox_configured: bool = True) -> WorkspaceCodeExecutionPolicyService:
    return WorkspaceCodeExecutionPolicyService(db, sandbox_configured=sandbox_configured)


async def test_a_workspace_with_no_policy_reads_as_unconfigured_and_narrows_nothing(
    async_db: AsyncSession,
) -> None:
    """The zero-rows case #655 requires: nothing configured, nothing narrowed."""
    org = await _organization(async_db, slug="acme-unset")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)

    policy = await _service(async_db).get_policy(user=owner, workspace_id=workspace.id)

    assert policy.configured is False
    assert policy.enabled is True
    assert policy.max_iterations is None
    assert policy.exec_timeout_s is None
    # And the request path sees the same thing: no row at all, so no narrowing.
    assert await resolve_workspace_code_execution_policy(async_db, workspace.id) is None


async def test_set_then_read_round_trips_and_the_request_path_sees_it(async_db: AsyncSession) -> None:
    org = await _organization(async_db, slug="acme-set")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)

    stored = await _service(async_db).set_policy(
        user=owner,
        workspace_id=workspace.id,
        request=WorkspaceCodeExecutionPolicyUpdate(
            enabled=True,
            default_purpose_hint="Prefer running code",
            max_iterations=3,
            exec_timeout_s=5,
        ),
    )

    assert stored.configured is True
    assert stored.default_purpose_hint == "Prefer running code"
    assert stored.max_iterations == 3
    assert stored.exec_timeout_s == 5

    read_back = await _service(async_db).get_policy(user=owner, workspace_id=workspace.id)
    assert read_back.max_iterations == 3

    resolved = await resolve_workspace_code_execution_policy(async_db, workspace.id)
    assert resolved is not None
    assert resolved.enabled is True
    assert resolved.default_purpose_hint == "Prefer running code"
    assert resolved.max_iterations == 3
    assert resolved.exec_timeout_s == 5


async def test_setting_twice_replaces_rather_than_accumulates(async_db: AsyncSession) -> None:
    """PUT semantics: an omitted limit is cleared, not carried over."""
    org = await _organization(async_db, slug="acme-replace")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)

    service = _service(async_db)
    await service.set_policy(
        user=owner,
        workspace_id=workspace.id,
        request=WorkspaceCodeExecutionPolicyUpdate(enabled=True, max_iterations=4, exec_timeout_s=9),
    )
    replaced = await service.set_policy(
        user=owner,
        workspace_id=workspace.id,
        request=WorkspaceCodeExecutionPolicyUpdate(enabled=False),
    )

    assert replaced.enabled is False
    assert replaced.max_iterations is None
    assert replaced.exec_timeout_s is None


async def test_a_blank_hint_is_stored_as_absent(async_db: AsyncSession) -> None:
    """A cleared text field must not become a configured empty hint."""
    org = await _organization(async_db, slug="acme-blank")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)

    stored = await _service(async_db).set_policy(
        user=owner,
        workspace_id=workspace.id,
        request=WorkspaceCodeExecutionPolicyUpdate(enabled=True, default_purpose_hint="   "),
    )

    assert stored.default_purpose_hint is None


async def test_clearing_returns_the_workspace_to_the_deployment_default(async_db: AsyncSession) -> None:
    org = await _organization(async_db, slug="acme-clear")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)

    service = _service(async_db)
    await service.set_policy(
        user=owner,
        workspace_id=workspace.id,
        request=WorkspaceCodeExecutionPolicyUpdate(enabled=False),
    )
    cleared = await service.clear_policy(user=owner, workspace_id=workspace.id)

    assert cleared.configured is False
    assert cleared.enabled is True
    assert await resolve_workspace_code_execution_policy(async_db, workspace.id) is None

    # Idempotent: clearing again is the state already asked for, not a 404.
    assert (await service.clear_policy(user=owner, workspace_id=workspace.id)).configured is False


async def test_sandbox_configured_reports_whether_the_deployment_can_run_any(async_db: AsyncSession) -> None:
    """The capability ceiling the page sits under, independent of the workspace's own answer."""
    org = await _organization(async_db, slug="acme-capability")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)

    enabled_on_a_deployment_with_no_sandbox = await _service(async_db, sandbox_configured=False).set_policy(
        user=owner,
        workspace_id=workspace.id,
        request=WorkspaceCodeExecutionPolicyUpdate(enabled=True),
    )

    assert enabled_on_a_deployment_with_no_sandbox.enabled is True
    assert enabled_on_a_deployment_with_no_sandbox.sandbox_configured is False


async def test_a_plain_member_may_neither_read_nor_write(async_db: AsyncSession) -> None:
    """Unlike the budget defaults next door, reads take the management role too.

    The policy is the workspace's security and billing posture rather than one
    member's allowance, which is the rule the hosted service enforces.
    """
    org = await _organization(async_db, slug="acme-member")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    plain = await _member(async_db, org, role="member", full_name="Plain")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)
    await WorkspaceMemberRepository(async_db).create(workspace_id=workspace.id, user_id=plain.id, role="member")

    service = _service(async_db)
    with pytest.raises(NotAuthorizedError):
        await service.get_policy(user=plain, workspace_id=workspace.id)
    with pytest.raises(NotAuthorizedError):
        await service.set_policy(
            user=plain,
            workspace_id=workspace.id,
            request=WorkspaceCodeExecutionPolicyUpdate(enabled=False),
        )


async def test_a_workspace_admin_may_write_it(async_db: AsyncSession) -> None:
    """Looser than the hosted rule on purpose: this repo's management gate admits a workspace admin."""
    org = await _organization(async_db, slug="acme-wsadmin")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    admin = await _member(async_db, org, role="member", full_name="Workspace Admin")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)
    await WorkspaceMemberRepository(async_db).create(workspace_id=workspace.id, user_id=admin.id, role="admin")

    stored = await _service(async_db).set_policy(
        user=admin,
        workspace_id=workspace.id,
        request=WorkspaceCodeExecutionPolicyUpdate(enabled=False),
    )

    assert stored.enabled is False


async def test_a_workspace_the_caller_cannot_see_is_not_found(async_db: AsyncSession) -> None:
    """404, not 403: another organization's workspace must not be distinguishable from none."""
    org = await _organization(async_db, slug="acme-foreign")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    other_org = await _organization(async_db, slug="rival-foreign")
    other_owner = await _member(async_db, other_org, role="owner", full_name="Rival Owner")
    foreign = await _workspace(async_db, other_org, name="Theirs", owner=other_owner)

    with pytest.raises(WorkspaceNotFoundError):
        await _service(async_db).get_policy(user=owner, workspace_id=foreign.id)


async def test_an_unknown_workspace_is_not_found(async_db: AsyncSession) -> None:
    org = await _organization(async_db, slug="acme-missing")
    owner = await _member(async_db, org, role="owner", full_name="Owner")

    with pytest.raises(WorkspaceNotFoundError):
        await _service(async_db).get_policy(user=owner, workspace_id=uuid.uuid4())


async def test_a_limit_above_the_deployment_ceiling_is_refused(async_db: AsyncSession) -> None:
    """A policy may only narrow, so a value that could never take effect is refused at the write."""
    with pytest.raises(ValueError):
        WorkspaceCodeExecutionPolicyUpdate(enabled=True, max_iterations=10_000)
    with pytest.raises(ValueError):
        WorkspaceCodeExecutionPolicyUpdate(enabled=True, exec_timeout_s=10_000)
    with pytest.raises(ValueError):
        WorkspaceCodeExecutionPolicyUpdate(enabled=True, max_iterations=0)


@pytest_asyncio.fixture
async def sessions(postgres_url: str) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A session factory on its own engine, disposed after the test.

    Copied from `test_workspace_member_budget_policies.py`, and disposed for
    the same reason: an undisposed engine leaves an asyncpg pool alive until
    garbage collection, which surfaces later as connection-limit or
    closed-loop noise in an unrelated test.
    """
    url = postgres_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://").replace(
        "postgresql://", "postgresql+asyncpg://"
    )
    engine = create_async_engine(url)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def test_a_write_that_lost_the_insert_race_applies_over_the_winner(
    async_db: AsyncSession,
    sessions: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two writers can both read no row and both insert; the loser must not 500.

    Driven rather than waited for: the racing session's first read is forced to
    answer "no row" while one exists, which is what a writer that read before a
    concurrent insert committed would have seen. The primary key then refuses
    its insert, and the recovery re-reads and applies the caller's own values,
    which is the outcome it would have reached had it arrived a moment later.
    """
    org = await _organization(async_db, slug="acme-race")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)
    await _service(async_db).set_policy(
        user=owner,
        workspace_id=workspace.id,
        request=WorkspaceCodeExecutionPolicyUpdate(enabled=True, max_iterations=2),
    )
    # The racing session is a separate connection and must see the row above
    # committed, not merely flushed.
    await async_db.commit()

    async with sessions() as session:
        real_get = session.get
        stale_reads = {"remaining": 1}

        async def get_with_one_stale_read(entity: Any, ident: Any) -> Any:
            if entity is WorkspaceCodeExecutionPolicy and stale_reads["remaining"]:
                stale_reads["remaining"] -= 1
                return None
            return await real_get(entity, ident)

        monkeypatch.setattr(session, "get", get_with_one_stale_read)
        racing_owner = await UserRepository(session).get(owner.id)
        assert racing_owner is not None
        stored = await _service(session).set_policy(
            user=racing_owner,
            workspace_id=workspace.id,
            request=WorkspaceCodeExecutionPolicyUpdate(enabled=False, max_iterations=5),
        )

    assert stored.enabled is False
    assert stored.max_iterations == 5


async def test_deleting_the_workspace_takes_its_policy_with_it(async_db: AsyncSession) -> None:
    """The row rides the workspace's own delete rather than needing separate cleanup."""
    org = await _organization(async_db, slug="acme-cascade")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)
    await _service(async_db).set_policy(
        user=owner,
        workspace_id=workspace.id,
        request=WorkspaceCodeExecutionPolicyUpdate(enabled=False),
    )

    await WorkspaceRepository(async_db).delete_workspace(workspace)
    await async_db.commit()

    assert await resolve_workspace_code_execution_policy(async_db, workspace.id) is None
