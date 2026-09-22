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

import uuid
from collections.abc import Iterator

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.guardrails import (
    OrganizationGuardrail,
    OrganizationGuardrailDefinition,
    OrganizationGuardrailWorkspace,
)
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
    OrganizationGuardrailCredentialNeedsUrlError,
    OrganizationGuardrailDefinitionNotFoundError,
    OrganizationGuardrailLimitReachedError,
    OrganizationGuardrailNotFoundError,
    OrganizationGuardrailScopeConflictError,
    OrganizationGuardrailSingleBackendError,
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


async def _definition(
    db: AsyncSession, organization: Organization, *, name: str = "lakera"
) -> OrganizationGuardrailDefinition:
    """A definition row, added with the session directly.

    The mandate is what is under test, and going through
    `organization_guardrail_definition_service` would put its catalog rules in
    front of every case here.
    """
    definition = OrganizationGuardrailDefinition(
        organization_id=organization.id,
        name=name,
        guardrail_name="lakera_guard",
        create_kwargs={"endpoint": PUBLIC_URL},
    )
    db.add(definition)
    await db.flush()
    return definition


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


async def test_a_credential_needs_an_endpoint_of_its_own(async_db: AsyncSession) -> None:
    """Without one the bearer would ride to the deployment url, which may be plain http."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationGuardrailService(async_db)

    with pytest.raises(OrganizationGuardrailCredentialNeedsUrlError):
        await service.create_guardrail(user=owner, request=_create(credential="s3cret"))


async def test_a_credential_cannot_be_added_to_an_entry_that_has_no_endpoint(async_db: AsyncSession) -> None:
    """The likelier way in: the entry is stored first and the credential arrives later."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationGuardrailService(async_db)
    created = await service.create_guardrail(user=owner, request=_create())

    with pytest.raises(OrganizationGuardrailCredentialNeedsUrlError):
        await service.update_guardrail(
            user=owner, guardrail_id=created.id, request=OrganizationGuardrailUpdate(credential="s3cret")
        )


async def test_clearing_the_endpoint_of_a_credentialed_entry_is_refused(async_db: AsyncSession) -> None:
    """The other half of the same rule, reached by editing the url rather than the credential."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationGuardrailService(async_db)
    created = await service.create_guardrail(user=owner, request=_create(url=PUBLIC_URL, credential="s3cret"))

    with pytest.raises(OrganizationGuardrailCredentialNeedsUrlError):
        await service.update_guardrail(user=owner, guardrail_id=created.id, request=OrganizationGuardrailUpdate(url=""))


async def test_a_profile_of_only_whitespace_is_refused_rather_than_stored_empty(
    async_db: AsyncSession,
) -> None:
    """Stored blank it would fail `GuardrailConfig` on the request path, after the budget hold.

    That failure is neither an `HTTPException` nor a `SQLAlchemyError`, so it
    escapes the release arm in `prepare_gateway_tools` and strands the
    reservation. Refused at the schema instead.
    """
    with pytest.raises(ValidationError):
        _create(profile="   ")
    with pytest.raises(ValidationError):
        OrganizationGuardrailUpdate(profile="  ")

    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    stored = await OrganizationGuardrailService(async_db).create_guardrail(
        user=owner, request=_create(profile="  pii  ")
    )
    assert stored.profile == "pii", "and a padded one is stored trimmed"


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
# The definition a mandate runs
# --------------------------------------------------------------------------- #


async def test_a_mandate_can_name_a_definition(async_db: AsyncSession) -> None:
    """The link, and the read that reports it."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    definition = await _definition(async_db, organization)
    service = OrganizationGuardrailService(async_db)

    created = await service.create_guardrail(user=owner, request=_create(definition_id=definition.id))
    assert created.definition_id == definition.id
    assert created.url is None

    listed = (await service.list_guardrails(user=owner)).data
    assert [entry.definition_id for entry in listed] == [definition.id]


async def test_a_mandate_names_one_backend_or_the_other(async_db: AsyncSession) -> None:
    """Both at once has no resolution rule, so it is refused rather than decided."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    definition = await _definition(async_db, organization)
    service = OrganizationGuardrailService(async_db)

    with pytest.raises(OrganizationGuardrailSingleBackendError):
        await service.create_guardrail(
            user=owner, request=_create(url=PUBLIC_URL, definition_id=definition.id)
        )


async def test_a_credential_beside_a_definition_is_refused_as_the_contradiction(
    async_db: AsyncSession,
) -> None:
    """Not as "a credential needs a url", which would send the caller the wrong way.

    A credential is only ever sent to a guardrails service, so one beside a
    definition is the same two-backend contradiction a step earlier. The check
    constraint does not cover this pair, which makes the service the only place
    it is caught.
    """
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    definition = await _definition(async_db, organization)
    service = OrganizationGuardrailService(async_db)

    with pytest.raises(OrganizationGuardrailSingleBackendError):
        await service.create_guardrail(
            user=owner, request=_create(credential="sk-guardrails", definition_id=definition.id)
        )


async def test_a_definition_cannot_be_added_to_a_mandate_that_has_an_endpoint(async_db: AsyncSession) -> None:
    """The stored half the request never mentions is what it is checked against."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    definition = await _definition(async_db, organization)
    service = OrganizationGuardrailService(async_db)
    created = await service.create_guardrail(user=owner, request=_create(url=PUBLIC_URL))

    with pytest.raises(OrganizationGuardrailSingleBackendError):
        await service.update_guardrail(
            user=owner,
            guardrail_id=created.id,
            request=OrganizationGuardrailUpdate(definition_id=definition.id),
        )


async def test_an_endpoint_cannot_be_added_to_a_mandate_that_has_a_definition(async_db: AsyncSession) -> None:
    """The same crossing from the other side."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    definition = await _definition(async_db, organization)
    service = OrganizationGuardrailService(async_db)
    created = await service.create_guardrail(user=owner, request=_create(definition_id=definition.id))

    with pytest.raises(OrganizationGuardrailSingleBackendError):
        await service.update_guardrail(
            user=owner, guardrail_id=created.id, request=OrganizationGuardrailUpdate(url=PUBLIC_URL)
        )


async def test_an_explicit_null_clears_the_link(async_db: AsyncSession) -> None:
    """The divergence from ``url`` and ``credential``, which a null leaves alone.

    A definition id is returned on every read, so a client sending null is
    sending back a field it was shown.
    """
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    definition = await _definition(async_db, organization)
    service = OrganizationGuardrailService(async_db)
    created = await service.create_guardrail(user=owner, request=_create(definition_id=definition.id))

    cleared = await service.update_guardrail(
        user=owner, guardrail_id=created.id, request=OrganizationGuardrailUpdate(definition_id=None)
    )
    assert cleared.definition_id is None

    # And the endpoint it was exclusive with is now available.
    remote = await service.update_guardrail(
        user=owner, guardrail_id=created.id, request=OrganizationGuardrailUpdate(url=PUBLIC_URL)
    )
    assert remote.url == PUBLIC_URL


async def test_an_edit_that_never_mentions_the_link_keeps_it(async_db: AsyncSession) -> None:
    """The case that makes the null above a deliberate clear rather than a side effect."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    definition = await _definition(async_db, organization)
    service = OrganizationGuardrailService(async_db)
    created = await service.create_guardrail(user=owner, request=_create(definition_id=definition.id))

    updated = await service.update_guardrail(
        user=owner, guardrail_id=created.id, request=OrganizationGuardrailUpdate(mode="block")
    )
    assert updated.mode == "block"
    assert updated.definition_id == definition.id


async def test_another_organizations_definition_is_not_found(async_db: AsyncSession) -> None:
    """A 404 about the definition, not the 409 about a profile the database would produce.

    The composite foreign key refuses the write either way. What is under test
    is the answer: every `IntegrityError` on this path is reported as a profile
    collision, so an unchecked foreign id would deny a field the caller got
    right.
    """
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    other = await _organization(async_db, slug="other")
    theirs = await _definition(async_db, other, name="theirs")
    service = OrganizationGuardrailService(async_db)

    with pytest.raises(OrganizationGuardrailDefinitionNotFoundError):
        await service.create_guardrail(user=owner, request=_create(definition_id=theirs.id))


async def test_a_definition_that_does_not_exist_is_not_found(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationGuardrailService(async_db)

    with pytest.raises(OrganizationGuardrailDefinitionNotFoundError):
        await service.create_guardrail(user=owner, request=_create(definition_id=uuid.uuid4()))


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
    assert in_scope[0].definition_id is None, "a mandate naming an endpoint runs no definition of ours"

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


async def test_a_linked_mandate_resolves_with_the_definition_that_serves_it(async_db: AsyncSession) -> None:
    """The request path learns which definition to run, and nothing else changes.

    The link is what tells the two shapes apart after the merge, where an
    in-process entry's empty `url` is otherwise indistinguishable from a remote
    entry falling back to the deployment's guardrails service.
    """
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    definition = await _definition(async_db, organization)
    service = OrganizationGuardrailService(async_db)
    await service.create_guardrail(
        user=owner, request=_create(definition_id=definition.id, applies_to_all_workspaces=True)
    )

    resolved = await resolve_organization_guardrails(
        async_db, organization_id=organization.id, workspace_id=workspace.id
    )
    assert [entry.config.profile for entry in resolved] == ["prompt-injection"]
    assert resolved[0].definition_id == definition.id
    assert resolved[0].config.url is None
    assert resolved[0].credential is None, "the build secrets are the runner's, and no bearer is sent anywhere"
