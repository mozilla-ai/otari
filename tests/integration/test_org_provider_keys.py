"""Integration tests for organization-scoped provider keys against a real DB.

Exercises the `org_provider_keys` / `workspace_provider_key_overrides` /
`workspace_provider_model_restrictions` migration, encryption at rest,
`set_org_default`, workspace override inheritance, model restrictions, the
dispatch-path overlay, and the tenancy authorization rules, at the service
layer (see `test_tenancy_authorization.py`'s module docstring for why: the
routes can only ever act as the one bootstrap operator, who is always an
owner, so the rules that matter most are only reachable by building
identities at whatever role a case needs).
"""

import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.models.provider_keys import (
    OrgProviderKey,
    OrgProviderKeyCreateRequest,
    OrgProviderKeyUpdateRequest,
    WorkspaceProviderKeyOverrideRequest,
)
from gateway.models.tenancy import Organization, User, Workspace
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.services.provider_kwargs import resolve_provider_selector
from gateway.services.secret_box import generate_secret_key
from gateway.services.tenancy import OrgProviderKeyService
from gateway.services.tenancy.errors import (
    NotAuthorizedError,
    OrgProviderKeyAlreadyExistsError,
    OrgProviderKeyArchivedError,
    OrgProviderKeyDisabledForWorkspaceError,
    OrgProviderKeyNotArchivedError,
    OrgProviderKeyNotFoundError,
    WorkspaceNotFoundError,
    WorkspaceProviderKeyOverrideConflictError,
)
from gateway.services.tenancy.org_provider_key_service import (
    cached_org_model_restriction,
    refresh_org_provider_cache,
    reset_org_provider_cache,
)

pytestmark = pytest.mark.asyncio


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
def _secret_key_and_clean_cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    reset_org_provider_cache()
    yield
    reset_org_provider_cache()


def _create_request(
    *, provider: str = "openai", name: str = "primary", api_key: str | None = "sk-live-1234"
) -> OrgProviderKeyCreateRequest:
    return OrgProviderKeyCreateRequest(provider=provider, name=name, api_key=api_key)


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #


async def test_crud_round_trip(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrgProviderKeyService(async_db)

    created = await service.create_key_for_user(user=owner, request=_create_request())
    assert created.last4 == "1234"
    assert created.is_org_default is False

    listed = await service.list_keys_for_user(user=owner)
    assert [k.id for k in listed.data] == [created.id]

    updated = await service.update_key_for_user(
        user=owner, key_id=created.id, request=OrgProviderKeyUpdateRequest(api_base="https://proxy/v1")
    )
    assert updated.api_base == "https://proxy/v1"
    assert updated.last4 == "1234", "omitted api_key is left in place"

    archived = await service.archive_key_for_user(user=owner, key_id=created.id)
    assert archived.archived_at is not None

    restored = await service.restore_key_for_user(user=owner, key_id=created.id)
    assert restored.archived_at is None

    await service.archive_key_for_user(user=owner, key_id=created.id)
    await service.delete_key_for_user(user=owner, key_id=created.id)
    assert (await service.list_keys_for_user(user=owner, include_archived=True)).count == 0


async def test_delete_requires_archived_first(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrgProviderKeyService(async_db)
    created = await service.create_key_for_user(user=owner, request=_create_request())

    with pytest.raises(OrgProviderKeyNotArchivedError):
        await service.delete_key_for_user(user=owner, key_id=created.id)


async def test_updating_a_nonexistent_key_is_not_found(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrgProviderKeyService(async_db)

    with pytest.raises(OrgProviderKeyNotFoundError):
        await service.update_key_for_user(
            user=owner, key_id=uuid.uuid4(), request=OrgProviderKeyUpdateRequest(name="new-name")
        )


async def test_a_key_from_another_organization_is_not_found(async_db: AsyncSession) -> None:
    organization_a = await _organization(async_db, slug="a")
    organization_b = await _organization(async_db, slug="b")
    owner_a = await _member(async_db, organization_a, role="owner", full_name="Owner A")
    owner_b = await _member(async_db, organization_b, role="owner", full_name="Owner B")
    service = OrgProviderKeyService(async_db)
    key = await service.create_key_for_user(user=owner_a, request=_create_request())

    with pytest.raises(OrgProviderKeyNotFoundError):
        await service.archive_key_for_user(user=owner_b, key_id=key.id)


async def test_create_duplicate_provider_name_conflicts(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrgProviderKeyService(async_db)
    await service.create_key_for_user(user=owner, request=_create_request())

    with pytest.raises(OrgProviderKeyAlreadyExistsError):
        await service.create_key_for_user(user=owner, request=_create_request())


async def test_plain_member_cannot_create_a_key(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    member = await _member(async_db, organization, role="member", full_name="Member")
    service = OrgProviderKeyService(async_db)

    with pytest.raises(NotAuthorizedError):
        await service.create_key_for_user(user=member, request=_create_request())


async def test_plain_member_does_not_see_client_args_but_admin_does(async_db: AsyncSession) -> None:
    """`client_args` is arbitrary JSON an admin can set, so the list read
    (open to every active member) redacts it wholesale for a non-admin reader
    while the admin who set it still sees the non-credential fields back."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    member = await _member(async_db, organization, role="member", full_name="Member")
    service = OrgProviderKeyService(async_db)
    await service.create_key_for_user(
        user=owner,
        request=OrgProviderKeyCreateRequest(
            provider="openai", name="primary", api_key="sk-live-1234", client_args={"region_name": "us-east-1"}
        ),
    )

    as_owner = await service.list_keys_for_user(user=owner)
    assert as_owner.data[0].client_args == {"region_name": "us-east-1"}

    as_member = await service.list_keys_for_user(user=member)
    assert as_member.data[0].client_args is None


async def test_credential_shaped_client_args_are_redacted_even_for_the_admin_who_set_them(
    async_db: AsyncSession,
) -> None:
    """A credential-shaped field placed in `client_args` (this gateway's own
    Bedrock support genuinely needs `aws_access_key_id`/`aws_secret_access_key`
    there, so the field cannot simply be rejected outright) never round-trips,
    the same treatment `encrypted_api_key` itself already gets: only `last4`
    comes back, never the plaintext. This holds even for the admin who set it,
    not only for a lower-privileged reader -- that is a separate, coarser gate
    `include_client_args` already covers."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrgProviderKeyService(async_db)
    await service.create_key_for_user(
        user=owner,
        request=OrgProviderKeyCreateRequest(
            provider="bedrock",
            name="primary",
            api_key="bearer-token",
            client_args={
                "region_name": "us-east-1",
                "aws_access_key_id": "AKIAABCDEFGHIJKLMNOP",
                "aws_secret_access_key": "supersecretvalue",
            },
        ),
    )

    as_owner = await service.list_keys_for_user(user=owner)
    assert as_owner.data[0].client_args == {
        "region_name": "us-east-1",
        "aws_access_key_id": "***",
        "aws_secret_access_key": "***",
    }


# --------------------------------------------------------------------------- #
# set_org_default
# --------------------------------------------------------------------------- #


async def test_set_org_default_clears_the_previous_default(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrgProviderKeyService(async_db)

    first = await service.create_key_for_user(user=owner, request=_create_request(name="first"))
    second = await service.create_key_for_user(user=owner, request=_create_request(name="second"))

    first = await service.set_org_default_for_user(user=owner, key_id=first.id)
    assert first.is_org_default is True

    second = await service.set_org_default_for_user(user=owner, key_id=second.id)
    assert second.is_org_default is True
    refreshed_first = (await service.list_keys_for_user(user=owner)).data
    assert next(k for k in refreshed_first if k.id == first.id).is_org_default is False


async def test_set_org_default_refuses_an_archived_key(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrgProviderKeyService(async_db)
    key = await service.create_key_for_user(user=owner, request=_create_request())
    await service.archive_key_for_user(user=owner, key_id=key.id)

    with pytest.raises(OrgProviderKeyArchivedError):
        await service.set_org_default_for_user(user=owner, key_id=key.id)


async def test_the_partial_unique_index_admits_only_one_default_per_org_and_provider(
    async_db: AsyncSession,
) -> None:
    """The database-level guarantee `set_org_default`'s race handling depends on:
    a second row with ``is_org_default=True`` for the same (organization,
    provider) violates ``uq_org_provider_keys_org_default`` even though the two
    rows are otherwise unrelated. Exercised directly against the constraint in
    one session rather than with two genuinely concurrent transactions: the
    repository's ``set_org_default`` clears every sibling before setting its
    own row, so the only way two rows both end up ``is_org_default=True`` is
    a race the index alone decides, which is what this pins."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrgProviderKeyService(async_db)
    first = await service.create_key_for_user(user=owner, request=_create_request(name="first"))
    await service.set_org_default_for_user(user=owner, key_id=first.id)

    second_row = OrgProviderKey(organization_id=organization.id, provider="openai", name="second", is_org_default=True)
    async_db.add(second_row)
    with pytest.raises(IntegrityError):
        await async_db.flush()


# --------------------------------------------------------------------------- #
# Workspace inheritance and overrides
# --------------------------------------------------------------------------- #


async def test_workspace_with_no_override_inherits_org_default(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = OrgProviderKeyService(async_db)

    key = await service.create_key_for_user(user=owner, request=_create_request())
    await service.set_org_default_for_user(user=owner, key_id=key.id)

    effective = await service.list_effective_keys_for_workspace(user=owner, workspace_id=workspace.id)
    (view,) = effective.data
    assert view.is_effective_default is True
    assert view.is_effective_enabled is True
    assert view.is_default is False, "no override row exists; the effective flag comes from the org default"


async def test_sibling_workspace_can_pin_a_different_key(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace_a = await _workspace(async_db, organization, name="A", owner=owner)
    workspace_b = await _workspace(async_db, organization, name="B", owner=owner)
    service = OrgProviderKeyService(async_db)

    default_key = await service.create_key_for_user(user=owner, request=_create_request(name="default"))
    await service.set_org_default_for_user(user=owner, key_id=default_key.id)
    other_key = await service.create_key_for_user(user=owner, request=_create_request(name="other"))

    await service.set_workspace_override_for_user(
        user=owner,
        workspace_id=workspace_b.id,
        key_id=other_key.id,
        request=WorkspaceProviderKeyOverrideRequest(is_default=True),
    )

    a_effective = await service.list_effective_keys_for_workspace(user=owner, workspace_id=workspace_a.id)
    assert next(k for k in a_effective.data if k.is_effective_default).org_provider_key_id == default_key.id

    b_effective = await service.list_effective_keys_for_workspace(user=owner, workspace_id=workspace_b.id)
    assert next(k for k in b_effective.data if k.is_effective_default).org_provider_key_id == other_key.id


async def test_archiving_the_default_falls_through_to_earliest_fallback(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = OrgProviderKeyService(async_db)

    fallback = await service.create_key_for_user(user=owner, request=_create_request(name="fallback"))
    default_key = await service.create_key_for_user(user=owner, request=_create_request(name="default"))
    await service.set_org_default_for_user(user=owner, key_id=default_key.id)

    await service.archive_key_for_user(user=owner, key_id=default_key.id)

    effective = await service.list_effective_keys_for_workspace(user=owner, workspace_id=workspace.id)
    assert [k.org_provider_key_id for k in effective.data] == [fallback.id], "the archived key is not a candidate"
    assert effective.data[0].is_effective_default is True


async def test_pinning_reenables_a_disabled_key_and_disabling_unpins(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = OrgProviderKeyService(async_db)
    key = await service.create_key_for_user(user=owner, request=_create_request())

    disabled = await service.set_workspace_override_for_user(
        user=owner, workspace_id=workspace.id, key_id=key.id, request=WorkspaceProviderKeyOverrideRequest(disabled=True)
    )
    assert disabled.disabled is True

    pinned = await service.set_workspace_override_for_user(
        user=owner,
        workspace_id=workspace.id,
        key_id=key.id,
        request=WorkspaceProviderKeyOverrideRequest(is_default=True),
    )
    assert pinned.is_default is True
    assert pinned.disabled is False, "pinning re-enables"

    reset = await service.set_workspace_override_for_user(
        user=owner, workspace_id=workspace.id, key_id=key.id, request=WorkspaceProviderKeyOverrideRequest(disabled=True)
    )
    assert reset.disabled is True
    assert reset.is_default is False, "disabling un-pins"


async def test_repinning_an_already_pinned_key_stays_pinned(async_db: AsyncSession) -> None:
    """Regression: `clear_workspace_pinned_default` must exclude the key being
    (re-)pinned. Clearing it too raced the ORM's own change tracking: a bulk
    UPDATE with ``synchronize_session=False`` cleared the row in the database
    without the loaded object noticing, so the following ``is_default = True``
    assignment looked like a no-op to SQLAlchemy and was never re-flushed,
    silently un-pinning a key an admin asked to keep pinned."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = OrgProviderKeyService(async_db)
    key = await service.create_key_for_user(user=owner, request=_create_request())

    first = await service.set_workspace_override_for_user(
        user=owner,
        workspace_id=workspace.id,
        key_id=key.id,
        request=WorkspaceProviderKeyOverrideRequest(is_default=True),
    )
    assert first.is_default is True

    again = await service.set_workspace_override_for_user(
        user=owner,
        workspace_id=workspace.id,
        key_id=key.id,
        request=WorkspaceProviderKeyOverrideRequest(is_default=True),
    )
    assert again.is_default is True


async def test_conflicting_explicit_pin_and_disable_is_refused(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = OrgProviderKeyService(async_db)
    key = await service.create_key_for_user(user=owner, request=_create_request())

    with pytest.raises(WorkspaceProviderKeyOverrideConflictError):
        await service.set_workspace_override_for_user(
            user=owner,
            workspace_id=workspace.id,
            key_id=key.id,
            request=WorkspaceProviderKeyOverrideRequest(is_default=True, disabled=True),
        )


async def test_disabling_a_key_cascades_deleting_its_model_restrictions(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = OrgProviderKeyService(async_db)
    key = await service.create_key_for_user(user=owner, request=_create_request())

    await service.add_model_restriction_for_user(user=owner, workspace_id=workspace.id, key_id=key.id, model="gpt-4o")
    assert (
        await service.list_model_restrictions_for_user(user=owner, workspace_id=workspace.id, key_id=key.id)
    ).models == ["gpt-4o"]

    await service.set_workspace_override_for_user(
        user=owner, workspace_id=workspace.id, key_id=key.id, request=WorkspaceProviderKeyOverrideRequest(disabled=True)
    )

    assert (
        await service.list_model_restrictions_for_user(user=owner, workspace_id=workspace.id, key_id=key.id)
    ).models == []


async def test_restricting_models_on_a_disabled_key_is_refused(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = OrgProviderKeyService(async_db)
    key = await service.create_key_for_user(user=owner, request=_create_request())
    await service.set_workspace_override_for_user(
        user=owner, workspace_id=workspace.id, key_id=key.id, request=WorkspaceProviderKeyOverrideRequest(disabled=True)
    )

    with pytest.raises(OrgProviderKeyDisabledForWorkspaceError):
        await service.add_model_restriction_for_user(
            user=owner, workspace_id=workspace.id, key_id=key.id, model="gpt-4o"
        )


async def test_model_restriction_is_cached_for_the_active_key(async_db: AsyncSession) -> None:
    """The dispatch-path cache carries the active key's model allow-list too
    (otari#643 review), not only its credentials: a workspace restricted to
    one model must not be able to dispatch every model with the same key."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = OrgProviderKeyService(async_db)
    key = await service.create_key_for_user(user=owner, request=_create_request())
    await service.set_org_default_for_user(user=owner, key_id=key.id)
    await service.add_model_restriction_for_user(user=owner, workspace_id=workspace.id, key_id=key.id, model="gpt-4o")

    await refresh_org_provider_cache(async_db)

    assert cached_org_model_restriction(workspace.id, "openai") == ["gpt-4o"]


async def test_adding_a_model_restriction_refreshes_the_cache_immediately(async_db: AsyncSession) -> None:
    """Regression: add_model_restriction_for_user/remove_model_restriction_for_user
    must refresh the overlay themselves, like every other mutation on this
    surface, rather than leaving dispatch to serve the stale allow-list for up
    to ORG_PROVIDER_CACHE_TTL_SECONDS on the very worker that made the change."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = OrgProviderKeyService(async_db)
    key = await service.create_key_for_user(user=owner, request=_create_request())
    await service.set_org_default_for_user(user=owner, key_id=key.id)
    await refresh_org_provider_cache(async_db)
    assert cached_org_model_restriction(workspace.id, "openai") is None

    await service.add_model_restriction_for_user(user=owner, workspace_id=workspace.id, key_id=key.id, model="gpt-4o")
    assert cached_org_model_restriction(workspace.id, "openai") == ["gpt-4o"], "no manual refresh call here"

    await service.remove_model_restriction_for_user(
        user=owner, workspace_id=workspace.id, key_id=key.id, model="gpt-4o"
    )
    assert cached_org_model_restriction(workspace.id, "openai") is None, "no manual refresh call here either"


async def test_model_restriction_is_absent_with_no_restrictions_configured(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = OrgProviderKeyService(async_db)
    key = await service.create_key_for_user(user=owner, request=_create_request())
    await service.set_org_default_for_user(user=owner, key_id=key.id)

    await refresh_org_provider_cache(async_db)

    assert cached_org_model_restriction(workspace.id, "openai") is None


async def test_model_restriction_follows_the_active_key_not_a_stale_one(async_db: AsyncSession) -> None:
    """Restricting key A's models must not leak onto key B once B becomes the
    workspace's active key: the cache keys restrictions by the resolved
    key's id, not by (workspace, provider) alone."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = OrgProviderKeyService(async_db)
    key_a = await service.create_key_for_user(user=owner, request=_create_request(name="a"))
    await service.set_org_default_for_user(user=owner, key_id=key_a.id)
    await service.add_model_restriction_for_user(user=owner, workspace_id=workspace.id, key_id=key_a.id, model="gpt-4o")

    key_b = await service.create_key_for_user(user=owner, request=_create_request(name="b"))
    await service.set_org_default_for_user(user=owner, key_id=key_b.id)

    await refresh_org_provider_cache(async_db)

    assert cached_org_model_restriction(workspace.id, "openai") is None


async def test_workspace_owner_who_is_not_an_org_admin_can_still_set_overrides(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    admin = await _member(async_db, organization, role="owner", full_name="Admin")
    plain_member = await _member(async_db, organization, role="member", full_name="Member")
    workspace = await _workspace(async_db, organization, owner=plain_member)
    service = OrgProviderKeyService(async_db)
    key = await service.create_key_for_user(user=admin, request=_create_request())

    # The workspace's own owner, an organization plain member otherwise,
    # may still pin/disable a key for that one workspace.
    result = await service.set_workspace_override_for_user(
        user=plain_member,
        workspace_id=workspace.id,
        key_id=key.id,
        request=WorkspaceProviderKeyOverrideRequest(is_default=True),
    )
    assert result.is_default is True


async def test_org_member_with_no_workspace_membership_gets_workspace_not_found(async_db: AsyncSession) -> None:
    """Not a member of the workspace at all: 404, matching `WorkspaceService`'s
    own rule that a workspace the caller cannot see must be indistinguishable
    from one that does not exist."""
    organization = await _organization(async_db)
    admin = await _member(async_db, organization, role="owner", full_name="Admin")
    outsider = await _member(async_db, organization, role="member", full_name="Outsider")
    workspace = await _workspace(async_db, organization, owner=admin)
    service = OrgProviderKeyService(async_db)
    key = await service.create_key_for_user(user=admin, request=_create_request())

    with pytest.raises(WorkspaceNotFoundError):
        await service.set_workspace_override_for_user(
            user=outsider,
            workspace_id=workspace.id,
            key_id=key.id,
            request=WorkspaceProviderKeyOverrideRequest(is_default=True),
        )


async def test_workspace_member_without_management_role_cannot_set_overrides(async_db: AsyncSession) -> None:
    """A plain member of the workspace can see it, but not manage its keys: 403."""
    organization = await _organization(async_db)
    admin = await _member(async_db, organization, role="owner", full_name="Admin")
    plain_workspace_member = await _member(async_db, organization, role="member", full_name="Plain")
    workspace = await _workspace(async_db, organization, owner=admin)
    await WorkspaceMemberRepository(async_db).create(
        workspace_id=workspace.id, user_id=plain_workspace_member.id, role="member"
    )
    service = OrgProviderKeyService(async_db)
    key = await service.create_key_for_user(user=admin, request=_create_request())

    with pytest.raises(NotAuthorizedError):
        await service.set_workspace_override_for_user(
            user=plain_workspace_member,
            workspace_id=workspace.id,
            key_id=key.id,
            request=WorkspaceProviderKeyOverrideRequest(is_default=True),
        )


# --------------------------------------------------------------------------- #
# Dispatch-path integration
# --------------------------------------------------------------------------- #


async def test_dispatch_resolves_an_organization_scoped_key_with_no_config_entry(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = OrgProviderKeyService(async_db)
    key = await service.create_key_for_user(user=owner, request=_create_request(api_key="sk-org-scoped-5678"))
    await service.set_org_default_for_user(user=owner, key_id=key.id)

    await refresh_org_provider_cache(async_db)

    config = GatewayConfig(providers={})
    resolved = resolve_provider_selector(config, "openai:gpt-4o", workspace_id=workspace.id)
    assert resolved.kwargs["api_key"] == "sk-org-scoped-5678"


async def test_an_instance_addressed_selector_never_consults_organization_scoped_keys(
    async_db: AsyncSession,
) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(async_db, organization, owner=owner)
    service = OrgProviderKeyService(async_db)
    key = await service.create_key_for_user(user=owner, request=_create_request(api_key="sk-org-scoped-5678"))
    await service.set_org_default_for_user(user=owner, key_id=key.id)
    await refresh_org_provider_cache(async_db)

    # "openai" here is a config.yml *instance* name, disjoint from the bare
    # "openai:model" selector the previous test used; the config-file key wins
    # outright and the organization-scoped one is never consulted.
    config = GatewayConfig(providers={"openai": {"api_key": "sk-config-file"}})
    resolved = resolve_provider_selector(config, "openai:gpt-4o", workspace_id=workspace.id)
    assert resolved.kwargs["api_key"] == "sk-config-file"


async def test_dispatch_without_workspace_id_ignores_organization_scoped_keys(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrgProviderKeyService(async_db)
    key = await service.create_key_for_user(user=owner, request=_create_request())
    await service.set_org_default_for_user(user=owner, key_id=key.id)
    await refresh_org_provider_cache(async_db)

    # No config.yml provider and no workspace_id: this must behave exactly as
    # it did before this feature existed, i.e. no credentials found (any-llm's
    # own env-var fallback would apply next, at the actual dispatch call, not
    # here), rather than a silent org-scoped resolution finding the org default
    # anyway because it happens to be the only key configured anywhere.
    config = GatewayConfig(providers={})
    resolved = resolve_provider_selector(config, "openai:gpt-4o")
    assert resolved.kwargs == {}
