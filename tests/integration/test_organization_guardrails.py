"""Organization guardrails: CRUD, authorization, scope, and what the request path reads.

Exercised at the service layer, matching `test_workspace_mcp_servers.py` and
`test_org_provider_keys.py`: the API can only ever act as the one bootstrap
operator identity a standalone deployment has, who is always an owner, so the
rules that matter most (a plain member refused, another organization's workspace
invisible) are only reachable by calling the service with identities built at
whatever role a case needs.

URLs here are IP literals in public ranges, or are rejected before any lookup
happens. ``validate_mcp_url`` resolves a hostname through DNS, so a test naming
one would pass or fail on whether the runner has egress.
"""

from collections.abc import Iterator

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import OrganizationGuardrail, OrganizationGuardrailWorkspace
from gateway.models.tenancy import Organization, User, Workspace
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.services.secret_box import decrypt_secret, generate_secret_key
from gateway.services.tenancy.errors import (
    NotAuthorizedError,
    OrganizationGuardrailAlreadyExistsError,
    OrganizationGuardrailLimitReachedError,
    OrganizationGuardrailNotFoundError,
    OrganizationGuardrailScopeConflictError,
    OrganizationGuardrailUnsafeUrlError,
    WorkspaceNotFoundError,
)
from gateway.services.tenancy.organization_guardrail_service import (
    MAX_GUARDRAILS_PER_ORGANIZATION,
    OrganizationGuardrailCreate,
    OrganizationGuardrailService,
    OrganizationGuardrailUpdate,
    resolve_organization_guardrails,
)

pytestmark = pytest.mark.asyncio

# A public IP literal, so the safety check never reaches a DNS resolver.
PUBLIC_URL = "https://93.184.216.34/guardrails"


async def _organization(db: AsyncSession, *, slug: str = "acme") -> Organization:
    return await OrganizationRepository(db).create_organization(name=slug.title(), slug=slug, created_by_user_id=None)


async def _member(db: AsyncSession, organization: Organization, *, role: str, full_name: str) -> User:
    user = await UserRepository(db).create_local_identity(
        full_name=full_name,
        active_organization_id=organization.id,
    )
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id, user_id=user.id, role=role
    )
    return user


async def _workspace(
    db: AsyncSession, organization: Organization, *, name: str = "Default", owner: User | None = None
) -> Workspace:
    workspace = await WorkspaceRepository(db).create_workspace(
        name=name, organization_id=organization.id, created_by_user_id=owner.id if owner else None
    )
    if owner is not None:
        await WorkspaceMemberRepository(db).create(workspace_id=workspace.id, user_id=owner.id, role="owner")
    return workspace


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


def _create(**overrides: object) -> OrganizationGuardrailCreate:
    fields: dict[str, object] = {"profile": "prompt-injection"}
    fields.update(overrides)
    return OrganizationGuardrailCreate(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #


async def test_crud_round_trip(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = OrganizationGuardrailService(async_db)

    created = await service.create_guardrail(
        user=owner,
        request=_create(mode="block", validate_kwargs={"threshold": 0.8}, workspace_ids=[workspace.id]),
    )
    assert created.organization_id == organization.id
    assert created.mode == "block"
    assert created.on_unavailable == "block", "the enforcing default a request-body entry also has"
    assert created.enabled is True
    assert created.applies_to_all_workspaces is False
    assert created.has_credential is False
    assert created.workspace_ids == [workspace.id]

    listed = await service.list_guardrails(user=owner)
    assert listed.count == 1
    assert [entry.id for entry in listed.data] == [created.id]

    updated = await service.update_guardrail(
        user=owner,
        guardrail_id=created.id,
        request=OrganizationGuardrailUpdate(mode="monitor", enabled=False),
    )
    assert updated.mode == "monitor"
    assert updated.enabled is False
    assert updated.validate_kwargs == {"threshold": 0.8}, "an omitted field is left in place"
    assert updated.workspace_ids == [workspace.id], "and so is the scope"

    await service.delete_guardrail(user=owner, guardrail_id=created.id)
    assert (await service.list_guardrails(user=owner)).count == 0


async def test_one_profile_per_organization(async_db: AsyncSession) -> None:
    """Two rows of one profile could never both run, so the second is refused at the write."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationGuardrailService(async_db)

    await service.create_guardrail(user=owner, request=_create())
    with pytest.raises(OrganizationGuardrailAlreadyExistsError):
        await service.create_guardrail(user=owner, request=_create(mode="block"))


async def test_renaming_onto_an_existing_profile_is_refused(async_db: AsyncSession) -> None:
    """The collision the unique index catches when a rename is flushed."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationGuardrailService(async_db)
    await service.create_guardrail(user=owner, request=_create(profile="pii"))
    second = await service.create_guardrail(user=owner, request=_create(profile="prompt-injection"))

    with pytest.raises(OrganizationGuardrailAlreadyExistsError):
        await service.update_guardrail(
            user=owner, guardrail_id=second.id, request=OrganizationGuardrailUpdate(profile="pii")
        )


async def test_renaming_onto_an_existing_profile_is_refused_alongside_a_scope_change(
    async_db: AsyncSession,
) -> None:
    """The same collision, on the update that also rewrites the scope.

    ``_replace_scope`` issues a DELETE, and the autoflush that triggers is what
    emits the renamed row's UPDATE. Without an explicit flush before it, the
    unique violation escapes from inside that helper as a raw ``IntegrityError``
    on an unrolled-back session, so the caller sees a 500 rather than the
    conflict.
    """
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = OrganizationGuardrailService(async_db)
    await service.create_guardrail(user=owner, request=_create(profile="pii"))
    second = await service.create_guardrail(user=owner, request=_create(profile="prompt-injection"))

    with pytest.raises(OrganizationGuardrailAlreadyExistsError):
        await service.update_guardrail(
            user=owner,
            guardrail_id=second.id,
            request=OrganizationGuardrailUpdate(profile="pii", workspace_ids=[workspace.id]),
        )


async def test_the_credential_is_encrypted_at_rest_and_never_returned(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationGuardrailService(async_db)

    created = await service.create_guardrail(
        user=owner, request=_create(url=PUBLIC_URL, credential="s3cret", applies_to_all_workspaces=True)
    )
    assert created.has_credential is True
    assert "s3cret" not in created.model_dump_json()

    row = (
        await async_db.execute(select(OrganizationGuardrail).where(OrganizationGuardrail.id == created.id))
    ).scalar_one()
    assert row.encrypted_credential is not None
    assert row.encrypted_credential != "s3cret"
    assert decrypt_secret(row.encrypted_credential) == "s3cret"


async def test_a_credential_is_left_alone_unless_the_update_names_it(async_db: AsyncSession) -> None:
    """The three-state rule: omitted leaves it, empty clears it, a value replaces it."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationGuardrailService(async_db)
    created = await service.create_guardrail(
        user=owner, request=_create(url=PUBLIC_URL, credential="first", applies_to_all_workspaces=True)
    )

    untouched = await service.update_guardrail(
        user=owner, guardrail_id=created.id, request=OrganizationGuardrailUpdate(mode="block")
    )
    assert untouched.has_credential is True

    rotated = await service.update_guardrail(
        user=owner, guardrail_id=created.id, request=OrganizationGuardrailUpdate(credential="second")
    )
    assert rotated.has_credential is True
    row = (
        await async_db.execute(select(OrganizationGuardrail).where(OrganizationGuardrail.id == created.id))
    ).scalar_one()
    assert row.encrypted_credential is not None
    assert decrypt_secret(row.encrypted_credential) == "second"

    cleared = await service.update_guardrail(
        user=owner, guardrail_id=created.id, request=OrganizationGuardrailUpdate(credential="")
    )
    assert cleared.has_credential is False


async def test_an_unsafe_endpoint_is_refused_at_the_write(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationGuardrailService(async_db)

    with pytest.raises(OrganizationGuardrailUnsafeUrlError):
        await service.create_guardrail(user=owner, request=_create(url="ftp://93.184.216.34/guardrails"))
    with pytest.raises(OrganizationGuardrailUnsafeUrlError):
        await service.create_guardrail(
            user=owner, request=_create(url="http://93.184.216.34/guardrails", credential="s3cret")
        )


async def test_adding_a_credential_rechecks_a_url_the_update_never_mentioned(async_db: AsyncSession) -> None:
    """An http endpoint is admissible until a credential would ride on it."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationGuardrailService(async_db)
    created = await service.create_guardrail(user=owner, request=_create(url="http://93.184.216.34/guardrails"))

    with pytest.raises(OrganizationGuardrailUnsafeUrlError):
        await service.update_guardrail(
            user=owner, guardrail_id=created.id, request=OrganizationGuardrailUpdate(credential="s3cret")
        )


async def test_the_entry_count_is_bounded(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationGuardrailService(async_db)
    for index in range(MAX_GUARDRAILS_PER_ORGANIZATION):
        await service.create_guardrail(user=owner, request=_create(profile=f"profile-{index}"))

    with pytest.raises(OrganizationGuardrailLimitReachedError):
        await service.create_guardrail(user=owner, request=_create(profile="one-too-many"))


# --------------------------------------------------------------------------- #
# Authorization
# --------------------------------------------------------------------------- #


async def test_a_plain_member_may_neither_read_nor_write(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    member = await _member(async_db, organization, role="member", full_name="Member")
    service = OrganizationGuardrailService(async_db)
    created = await service.create_guardrail(user=owner, request=_create())

    with pytest.raises(NotAuthorizedError):
        await service.list_guardrails(user=member)
    with pytest.raises(NotAuthorizedError):
        await service.create_guardrail(user=member, request=_create(profile="pii"))
    with pytest.raises(NotAuthorizedError):
        await service.delete_guardrail(user=member, guardrail_id=created.id)


async def test_another_organizations_entry_is_not_found(async_db: AsyncSession) -> None:
    theirs = await _organization(async_db, slug="theirs")
    their_owner = await _member(async_db, theirs, role="owner", full_name="Their Owner")
    ours = await _organization(async_db, slug="ours")
    our_owner = await _member(async_db, ours, role="owner", full_name="Our Owner")
    service = OrganizationGuardrailService(async_db)
    created = await service.create_guardrail(user=their_owner, request=_create())

    with pytest.raises(OrganizationGuardrailNotFoundError):
        await service.update_guardrail(
            user=our_owner, guardrail_id=created.id, request=OrganizationGuardrailUpdate(enabled=False)
        )


async def test_a_workspace_in_another_organization_cannot_be_scoped(async_db: AsyncSession) -> None:
    """Reported as not found, so the scope is not an existence oracle across tenants."""
    theirs = await _organization(async_db, slug="theirs")
    their_owner = await _member(async_db, theirs, role="owner", full_name="Their Owner")
    their_workspace = await _workspace(async_db, theirs, owner=their_owner)
    ours = await _organization(async_db, slug="ours")
    our_owner = await _member(async_db, ours, role="owner", full_name="Our Owner")
    service = OrganizationGuardrailService(async_db)

    with pytest.raises(WorkspaceNotFoundError):
        await service.create_guardrail(user=our_owner, request=_create(workspace_ids=[their_workspace.id]))


# --------------------------------------------------------------------------- #
# Scope
# --------------------------------------------------------------------------- #


async def test_a_scope_alongside_applies_to_all_workspaces_is_refused(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = OrganizationGuardrailService(async_db)

    with pytest.raises(ValidationError):
        _create(applies_to_all_workspaces=True, workspace_ids=[workspace.id])

    created = await service.create_guardrail(user=owner, request=_create(applies_to_all_workspaces=True))
    with pytest.raises(OrganizationGuardrailScopeConflictError):
        await service.update_guardrail(
            user=owner,
            guardrail_id=created.id,
            request=OrganizationGuardrailUpdate(workspace_ids=[workspace.id]),
        )


async def test_the_scope_is_replaced_whole_and_cleared_by_an_empty_list(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    first = await _workspace(async_db, organization, name="First", owner=owner)
    second = await _workspace(async_db, organization, name="Second", owner=owner)
    service = OrganizationGuardrailService(async_db)
    created = await service.create_guardrail(user=owner, request=_create(workspace_ids=[first.id]))

    replaced = await service.update_guardrail(
        user=owner, guardrail_id=created.id, request=OrganizationGuardrailUpdate(workspace_ids=[second.id])
    )
    assert replaced.workspace_ids == [second.id]

    cleared = await service.update_guardrail(
        user=owner, guardrail_id=created.id, request=OrganizationGuardrailUpdate(workspace_ids=[])
    )
    assert cleared.workspace_ids == []


async def test_switching_to_every_workspace_clears_the_scope_it_had(async_db: AsyncSession) -> None:
    """So switching back does not silently reinstate a set nobody looked at."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = OrganizationGuardrailService(async_db)
    created = await service.create_guardrail(user=owner, request=_create(workspace_ids=[workspace.id]))

    await service.update_guardrail(
        user=owner,
        guardrail_id=created.id,
        request=OrganizationGuardrailUpdate(applies_to_all_workspaces=True),
    )
    back = await service.update_guardrail(
        user=owner,
        guardrail_id=created.id,
        request=OrganizationGuardrailUpdate(applies_to_all_workspaces=False),
    )

    assert back.workspace_ids == []
    assert (
        await resolve_organization_guardrails(async_db, organization_id=organization.id, workspace_id=workspace.id)
    ) == []


async def test_deleting_an_entry_takes_its_scope_rows_with_it(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = OrganizationGuardrailService(async_db)
    created = await service.create_guardrail(user=owner, request=_create(workspace_ids=[workspace.id]))

    await service.delete_guardrail(user=owner, guardrail_id=created.id)

    remaining = (
        (
            await async_db.execute(
                select(OrganizationGuardrailWorkspace).where(
                    OrganizationGuardrailWorkspace.organization_guardrail_id == created.id
                )
            )
        )
        .scalars()
        .all()
    )
    assert remaining == []


# --------------------------------------------------------------------------- #
# What the request path reads
# --------------------------------------------------------------------------- #


async def test_an_organization_with_no_entries_resolves_to_nothing(async_db: AsyncSession) -> None:
    """The zero-rows requirement #655 puts on every one of the four planes."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)

    resolved = await resolve_organization_guardrails(
        async_db, organization_id=organization.id, workspace_id=workspace.id
    )
    assert resolved == []


async def test_only_the_scoped_workspace_resolves_the_entry(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    scoped = await _workspace(async_db, organization, name="Scoped", owner=owner)
    other = await _workspace(async_db, organization, name="Other", owner=owner)
    service = OrganizationGuardrailService(async_db)
    await service.create_guardrail(
        user=owner, request=_create(url=PUBLIC_URL, credential="s3cret", workspace_ids=[scoped.id])
    )

    in_scope = await resolve_organization_guardrails(async_db, organization_id=organization.id, workspace_id=scoped.id)
    assert [entry.config.profile for entry in in_scope] == ["prompt-injection"]
    assert in_scope[0].config.url == PUBLIC_URL
    assert in_scope[0].credential == "s3cret", "decrypted for the request path and nowhere else"

    assert (
        await resolve_organization_guardrails(async_db, organization_id=organization.id, workspace_id=other.id)
    ) == []


async def test_a_new_workspace_inherits_the_organization_default_and_nothing_else(
    async_db: AsyncSession,
) -> None:
    """The inheritance rule otari#654 asks for, asserted on a workspace created afterwards."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    existing = await _workspace(async_db, organization, name="Existing", owner=owner)
    service = OrganizationGuardrailService(async_db)
    await service.create_guardrail(user=owner, request=_create(profile="everywhere", applies_to_all_workspaces=True))
    await service.create_guardrail(user=owner, request=_create(profile="scoped", workspace_ids=[existing.id]))

    fresh = await _workspace(async_db, organization, name="Fresh", owner=owner)

    resolved = await resolve_organization_guardrails(async_db, organization_id=organization.id, workspace_id=fresh.id)
    assert [entry.config.profile for entry in resolved] == ["everywhere"]


async def test_a_disabled_entry_resolves_nowhere(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = OrganizationGuardrailService(async_db)
    created = await service.create_guardrail(user=owner, request=_create(applies_to_all_workspaces=True))
    await service.update_guardrail(
        user=owner, guardrail_id=created.id, request=OrganizationGuardrailUpdate(enabled=False)
    )

    resolved = await resolve_organization_guardrails(
        async_db, organization_id=organization.id, workspace_id=workspace.id
    )
    assert resolved == []


async def test_a_workspace_of_another_organization_resolves_nothing(async_db: AsyncSession) -> None:
    theirs = await _organization(async_db, slug="theirs")
    their_owner = await _member(async_db, theirs, role="owner", full_name="Their Owner")
    ours = await _organization(async_db, slug="ours")
    our_owner = await _member(async_db, ours, role="owner", full_name="Our Owner")
    our_workspace = await _workspace(async_db, ours, owner=our_owner)
    service = OrganizationGuardrailService(async_db)
    await service.create_guardrail(user=their_owner, request=_create(applies_to_all_workspaces=True))

    resolved = await resolve_organization_guardrails(async_db, organization_id=ours.id, workspace_id=our_workspace.id)
    assert resolved == []
