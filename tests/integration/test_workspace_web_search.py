"""Per-workspace web-search configuration: CRUD, authorization, and what the request path reads.

Exercised at the service layer, matching `test_workspace_code_execution_policy.py`:
the API can only ever act as the one superuser operator identity a standalone
deployment has, so the rules that matter most (a plain member refused, a foreign
workspace refused) are only reachable by calling the service with identities
built at whatever role a case needs.
"""

import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from gateway.models.entities import WorkspaceWebSearchConfig
from gateway.models.tenancy import Organization, User, Workspace
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.services.tenancy.errors import NotAuthorizedError, WorkspaceNotFoundError
from gateway.services.tenancy.workspace_web_search_service import (
    WorkspaceWebSearchConfigUpdate,
    WorkspaceWebSearchService,
    resolve_workspace_web_search_config,
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


def _service(db: AsyncSession, *, web_search_configured: bool = True) -> WorkspaceWebSearchService:
    return WorkspaceWebSearchService(db, web_search_configured=web_search_configured)


async def test_a_workspace_with_no_row_reads_as_unconfigured_and_narrows_nothing(
    async_db: AsyncSession,
) -> None:
    """The zero-rows case #655 requires: nothing configured, nothing narrowed."""
    org = await _organization(async_db, slug="ws-unset")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)

    config = await _service(async_db).get_config(user=owner, workspace_id=workspace.id)

    assert config.configured is False
    assert config.enabled is True
    assert config.max_results is None
    assert config.allowed_domains is None
    assert config.blocked_domains is None
    # And the request path sees the same thing: no row at all, so no narrowing.
    assert await resolve_workspace_web_search_config(async_db, workspace.id) is None


async def test_set_then_read_round_trips_and_the_request_path_sees_it(async_db: AsyncSession) -> None:
    org = await _organization(async_db, slug="ws-set")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)

    stored = await _service(async_db).set_config(
        user=owner,
        workspace_id=workspace.id,
        request=WorkspaceWebSearchConfigUpdate(
            enabled=True,
            max_results=4,
            purpose_hint="Cite your sources",
            allowed_domains=["Wikipedia.org", " arxiv.org "],
            blocked_domains=["example.invalid"],
            provider_options={"search_depth": "advanced"},
        ),
    )

    assert stored.configured is True
    assert stored.max_results == 4
    assert stored.purpose_hint == "Cite your sources"
    # Normalized on the way in, which is what makes the request-path comparison
    # a plain string match.
    assert stored.allowed_domains == ["wikipedia.org", "arxiv.org"]
    assert stored.provider_options == {"search_depth": "advanced"}

    read_back = await _service(async_db).get_config(user=owner, workspace_id=workspace.id)
    assert read_back.max_results == 4

    resolved = await resolve_workspace_web_search_config(async_db, workspace.id)
    assert resolved is not None
    assert resolved.enabled is True
    assert resolved.max_results == 4
    assert resolved.purpose_hint == "Cite your sources"
    assert resolved.allowed_domains == ("wikipedia.org", "arxiv.org")
    assert resolved.blocked_domains == ("example.invalid",)
    assert resolved.provider_options == {"search_depth": "advanced"}


async def test_setting_twice_replaces_rather_than_accumulates(async_db: AsyncSession) -> None:
    """PUT semantics: an omitted field is cleared, not carried over."""
    org = await _organization(async_db, slug="ws-replace")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)

    service = _service(async_db)
    await service.set_config(
        user=owner,
        workspace_id=workspace.id,
        request=WorkspaceWebSearchConfigUpdate(
            enabled=True,
            max_results=4,
            blocked_domains=["example.invalid"],
            provider_options={"topic": "news"},
        ),
    )
    replaced = await service.set_config(
        user=owner,
        workspace_id=workspace.id,
        request=WorkspaceWebSearchConfigUpdate(enabled=False),
    )

    assert replaced.enabled is False
    assert replaced.max_results is None
    assert replaced.blocked_domains is None
    assert replaced.provider_options is None


async def test_a_blank_hint_and_an_all_blank_domain_list_are_stored_as_absent(async_db: AsyncSession) -> None:
    """A cleared form field must not become a configured empty value.

    An empty allow-list is the case that matters: stored as ``[]`` it would read
    as "permit nothing" to a person and as "no allow-list" to the narrowing,
    which is a policy that means two different things at once.
    """
    org = await _organization(async_db, slug="ws-blank")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)

    stored = await _service(async_db).set_config(
        user=owner,
        workspace_id=workspace.id,
        request=WorkspaceWebSearchConfigUpdate(
            enabled=True,
            purpose_hint="   ",
            allowed_domains=["", "  "],
            provider_options={},
        ),
    )

    assert stored.purpose_hint is None
    assert stored.allowed_domains is None
    assert stored.provider_options is None


async def test_clearing_returns_the_workspace_to_the_deployment_default(async_db: AsyncSession) -> None:
    org = await _organization(async_db, slug="ws-clear")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)

    service = _service(async_db)
    await service.set_config(
        user=owner,
        workspace_id=workspace.id,
        request=WorkspaceWebSearchConfigUpdate(enabled=False),
    )
    cleared = await service.clear_config(user=owner, workspace_id=workspace.id)

    assert cleared.configured is False
    assert cleared.enabled is True
    assert await resolve_workspace_web_search_config(async_db, workspace.id) is None

    # Idempotent: clearing again is the state already asked for, not a 404.
    assert (await service.clear_config(user=owner, workspace_id=workspace.id)).configured is False


async def test_web_search_configured_reports_whether_the_deployment_can_search_at_all(
    async_db: AsyncSession,
) -> None:
    """The capability ceiling the page sits under, independent of the workspace's own answer."""
    org = await _organization(async_db, slug="ws-capability")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)

    enabled_with_no_backend = await _service(async_db, web_search_configured=False).set_config(
        user=owner,
        workspace_id=workspace.id,
        request=WorkspaceWebSearchConfigUpdate(enabled=True),
    )

    assert enabled_with_no_backend.enabled is True
    assert enabled_with_no_backend.web_search_configured is False


async def test_a_plain_member_may_neither_read_nor_write(async_db: AsyncSession) -> None:
    """Reads take the management role too: the row is the workspace's posture, not a member's allowance."""
    org = await _organization(async_db, slug="ws-member")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    plain = await _member(async_db, org, role="member", full_name="Plain")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)
    await WorkspaceMemberRepository(async_db).create(workspace_id=workspace.id, user_id=plain.id, role="member")

    service = _service(async_db)
    with pytest.raises(NotAuthorizedError):
        await service.get_config(user=plain, workspace_id=workspace.id)
    with pytest.raises(NotAuthorizedError):
        await service.set_config(
            user=plain,
            workspace_id=workspace.id,
            request=WorkspaceWebSearchConfigUpdate(enabled=False),
        )


async def test_a_workspace_admin_may_write_it(async_db: AsyncSession) -> None:
    """Looser than the hosted rule on purpose: this repo's management gate admits a workspace admin."""
    org = await _organization(async_db, slug="ws-wsadmin")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    admin = await _member(async_db, org, role="member", full_name="Workspace Admin")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)
    await WorkspaceMemberRepository(async_db).create(workspace_id=workspace.id, user_id=admin.id, role="admin")

    stored = await _service(async_db).set_config(
        user=admin,
        workspace_id=workspace.id,
        request=WorkspaceWebSearchConfigUpdate(enabled=False),
    )

    assert stored.enabled is False


async def test_a_workspace_the_caller_cannot_see_is_not_found(async_db: AsyncSession) -> None:
    """404, not 403: another organization's workspace must not be distinguishable from none."""
    org = await _organization(async_db, slug="ws-foreign")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    other_org = await _organization(async_db, slug="ws-rival")
    other_owner = await _member(async_db, other_org, role="owner", full_name="Rival Owner")
    foreign = await _workspace(async_db, other_org, name="Theirs", owner=other_owner)

    with pytest.raises(WorkspaceNotFoundError):
        await _service(async_db).get_config(user=owner, workspace_id=foreign.id)


async def test_an_unknown_workspace_is_not_found(async_db: AsyncSession) -> None:
    org = await _organization(async_db, slug="ws-missing")
    owner = await _member(async_db, org, role="owner", full_name="Owner")

    with pytest.raises(WorkspaceNotFoundError):
        await _service(async_db).get_config(user=owner, workspace_id=uuid.uuid4())


async def test_a_ceiling_above_what_the_backend_honors_is_refused(async_db: AsyncSession) -> None:
    """A row may only narrow, so a value that could never take effect is refused at the write."""
    with pytest.raises(ValueError):
        WorkspaceWebSearchConfigUpdate(enabled=True, max_results=10_000)
    with pytest.raises(ValueError):
        WorkspaceWebSearchConfigUpdate(enabled=True, max_results=0)


async def test_oversized_lists_and_option_bags_are_refused(async_db: AsyncSession) -> None:
    """One workspace's row cannot grow without limit."""
    with pytest.raises(ValueError):
        WorkspaceWebSearchConfigUpdate(enabled=True, allowed_domains=[f"host{i}.example" for i in range(101)])
    with pytest.raises(ValueError):
        WorkspaceWebSearchConfigUpdate(enabled=True, blocked_domains=["x" * 254 + ".example"])
    with pytest.raises(ValueError):
        WorkspaceWebSearchConfigUpdate(enabled=True, provider_options={f"k{i}": i for i in range(31)})
    with pytest.raises(ValueError):
        WorkspaceWebSearchConfigUpdate(enabled=True, provider_options={"blob": "x" * 5000})


@pytest_asyncio.fixture
async def sessions(postgres_url: str) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A session factory on its own engine, disposed after the test.

    Copied from `test_workspace_code_execution_policy.py`, and disposed for the
    same reason: an undisposed engine leaves an asyncpg pool alive until garbage
    collection, which surfaces later as connection-limit or closed-loop noise in
    an unrelated test.
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
    its insert, and the recovery re-reads and applies the caller's own values.
    """
    org = await _organization(async_db, slug="ws-race")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)
    await _service(async_db).set_config(
        user=owner,
        workspace_id=workspace.id,
        request=WorkspaceWebSearchConfigUpdate(enabled=True, max_results=2),
    )
    # The racing session is a separate connection and must see the row above
    # committed, not merely flushed.
    await async_db.commit()

    async with sessions() as session:
        real_get = session.get
        stale_reads = {"remaining": 1}

        async def get_with_one_stale_read(entity: Any, ident: Any) -> Any:
            if entity is WorkspaceWebSearchConfig and stale_reads["remaining"]:
                stale_reads["remaining"] -= 1
                return None
            return await real_get(entity, ident)

        monkeypatch.setattr(session, "get", get_with_one_stale_read)
        racing_owner = await UserRepository(session).get(owner.id)
        assert racing_owner is not None
        stored = await _service(session).set_config(
            user=racing_owner,
            workspace_id=workspace.id,
            request=WorkspaceWebSearchConfigUpdate(enabled=False, max_results=5),
        )

    assert stored.enabled is False
    assert stored.max_results == 5


async def test_deleting_the_workspace_takes_its_row_with_it(async_db: AsyncSession) -> None:
    """The row rides the workspace's own delete rather than needing separate cleanup."""
    org = await _organization(async_db, slug="ws-cascade")
    owner = await _member(async_db, org, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, org, name="Engineering", owner=owner)
    await _service(async_db).set_config(
        user=owner,
        workspace_id=workspace.id,
        request=WorkspaceWebSearchConfigUpdate(enabled=False),
    )

    await WorkspaceRepository(async_db).delete_workspace(workspace)
    await async_db.commit()

    assert await resolve_workspace_web_search_config(async_db, workspace.id) is None
