"""Writing an organization's guardrail definitions: CRUD, authorization, and the columns.

Exercised at the service layer, matching `test_organization_guardrails.py`: the
API can only ever act as the one bootstrap operator identity a standalone
deployment has, who is always an owner, so the rules that matter most (a plain
member refused, another organization's row invisible) are only reachable by
calling the service with identities built at whatever role a case needs.

Two neighbours cover the rest. `test_organization_guardrail_definitions.py` is
what the *database* refuses about these rows, which holds whatever a write path
checks; `tests/unit/test_organization_guardrail_definition_rules.py` is each
catalog rule on its own. What is here is that the write path runs those rules,
and what a column ends up holding.
"""

import json
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Any, cast

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.guardrails import OrganizationGuardrail, OrganizationGuardrailDefinition
from gateway.models.tenancy import Organization, User
from gateway.repositories.tenancy import (
    OrganizationGuardrailDefinitionRepository,
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
)
from gateway.services.secret_box import decrypt_secret, generate_secret_key
from gateway.services.tenancy import organization_guardrail_runner as runner
from gateway.services.tenancy.errors import (
    NotAuthorizedError,
    OrganizationGuardrailDefinitionAlreadyExistsError,
    OrganizationGuardrailDefinitionArgumentsError,
    OrganizationGuardrailDefinitionInUseError,
    OrganizationGuardrailDefinitionLimitReachedError,
    OrganizationGuardrailDefinitionNotFoundError,
    OrganizationGuardrailDefinitionUnsafeUrlError,
    OrganizationGuardrailNotBuildableError,
    OrganizationGuardrailNotDefinableError,
)
from gateway.services.tenancy.organization_guardrail_definition_service import (
    MAX_DEFINITIONS_PER_ORGANIZATION,
    OrganizationGuardrailDefinitionCreate,
    OrganizationGuardrailDefinitionPublic,
    OrganizationGuardrailDefinitionService,
    OrganizationGuardrailDefinitionUpdate,
)
from gateway.services.tenancy.organization_service import OrganizationService

pytestmark = pytest.mark.asyncio

# The write path checks every stored address, so a fixture's endpoint has to be
# one that passes without a resolver: an IP literal in the public range
# example.com has used for years, the convention
# `test_organization_guardrail_definitions.py` already follows and
# `test_organization_guardrails.py` explains.
VENDOR_ENDPOINT = "https://93.184.216.34/v2"
OTHER_ENDPOINT = "https://93.184.216.35/v2"


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


def _service(db: AsyncSession) -> OrganizationGuardrailDefinitionService:
    """The service as `api/deps.py` builds it: a Unit of Work over the request's session."""
    uow = UnitOfWork(db)
    return OrganizationGuardrailDefinitionService(
        definitions=OrganizationGuardrailDefinitionRepository(uow),
        organizations=OrganizationService(db, membership_listener=None),
        uow=uow,
        build_state=runner.build_state,
        rebuild=runner.rebuild_definition,
        handle=runner.handle,
    )


async def _after_a_refusal(db: AsyncSession, user: User) -> None:
    """Reload an identity a rolled-back block expired.

    A refusal inside a Unit of Work block rolls the block back, and a rollback
    expires every instance in the session, the caller's own identity included. A
    request is over by then and the next one arrives on a session of its own, so
    only a test keeps using the same objects; without this the next role gate
    reads an expired column and lazily loads it where no lazy load can be
    awaited.
    """
    await db.refresh(user)


def _create(**overrides: object) -> OrganizationGuardrailDefinitionCreate:
    fields: dict[str, object] = {
        "name": "prod-lakera",
        "guardrail_name": "lakera_guard",
        "create_kwargs": {"api_key": "lakera-key", "endpoint": VENDOR_ENDPOINT},
    }
    fields.update(overrides)
    return OrganizationGuardrailDefinitionCreate(**fields)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


@pytest.fixture(autouse=True)
def _empty_runner() -> Iterator[None]:
    """The runner is process-global, so a held entry would outlive its test."""
    runner.reset_guardrail_runner()
    yield
    runner.reset_guardrail_runner()


@pytest.fixture(autouse=True)
def _stub_vendor(monkeypatch: pytest.MonkeyPatch) -> None:
    """A write rebuilds, so every write here would otherwise construct a real vendor client.

    Nothing in this module is about any-guardrail. The build is stubbed so the
    suite neither depends on an upstream constructor nor pays for one, and the
    cases that are about the build say so by replacing this.
    """

    class _Stub:
        @staticmethod
        def create(_guardrail_name: Any, **_kwargs: Any) -> Any:
            return object()

    monkeypatch.setattr(runner, 'AnyGuardrail', _Stub)


def _hold(organization_id: uuid.UUID, definition: OrganizationGuardrailDefinitionPublic, *, guardrail: object) -> None:
    """Say what this worker holds, without building anything.

    ``guardrail=None`` is a build that failed, which is the state the read has
    to report and the one no test can produce by dialing a vendor.
    """
    runner._held[(organization_id, definition.id)] = runner._Held(
        fingerprint=datetime.fromisoformat(definition.updated_at),
        guardrail_name=definition.guardrail_name,
        guardrail=cast(Any, guardrail),
    )


async def _row(db: AsyncSession, definition_id: uuid.UUID) -> OrganizationGuardrailDefinition:
    """The stored row, read back outside the service, to see what a column actually holds."""
    stored = (
        await db.execute(
            select(OrganizationGuardrailDefinition).where(OrganizationGuardrailDefinition.id == definition_id)
        )
    ).scalar_one()
    await db.refresh(stored)
    return stored


def _mandate(organization: Organization, *, profile: str, definition_id: uuid.UUID) -> OrganizationGuardrail:
    """A mandate pointing at a definition, added with the session directly.

    No write path sets ``definition_id`` yet, so the row is built here rather
    than through `organization_guardrail_service`, the way
    `test_organization_guardrail_definitions.py` builds one.
    """
    return OrganizationGuardrail(
        organization_id=organization.id,
        profile=profile,
        definition_id=definition_id,
        applies_to_all_workspaces=True,
    )


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #


async def test_crud_round_trip(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)

    created = await service.create_definition(user=owner, request=_create())
    assert created.organization_id == organization.id
    assert created.guardrail_name == "lakera_guard"
    assert created.enabled is True
    assert created.create_kwargs == {"endpoint": VENDOR_ENDPOINT}
    assert created.create_secrets == {"api_key": "***"}
    assert created.secrets_decryptable is True

    listed = await service.list_definitions(user=owner)
    assert listed.count == 1
    assert [entry.id for entry in listed.data] == [created.id]

    updated = await service.update_definition(
        user=owner,
        definition_id=created.id,
        request=OrganizationGuardrailDefinitionUpdate(name="eu-lakera", enabled=False),
    )
    assert updated.name == "eu-lakera"
    assert updated.enabled is False

    await service.delete_definition(user=owner, definition_id=created.id)
    assert (await service.list_definitions(user=owner)).count == 0


async def test_an_organization_admin_manages_definitions_too(async_db: AsyncSession) -> None:
    """The audience this whole surface exists for: an owner or an admin, not an operator."""
    organization = await _organization(async_db)
    admin = await _member(async_db, organization, role="admin", full_name="Admin")
    service = _service(async_db)

    created = await service.create_definition(user=admin, request=_create())

    assert (await service.list_definitions(user=admin)).count == 1
    await service.delete_definition(user=admin, definition_id=created.id)


async def test_a_plain_member_neither_reads_nor_writes(async_db: AsyncSession) -> None:
    """One gate over reads and writes alike, as the mandates next door have.

    These rows name the vendors an organization holds accounts with and say which
    of those accounts have credentials stored here, which is not something every
    member needs.
    """
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    member = await _member(async_db, organization, role="member", full_name="Member")
    service = _service(async_db)
    created = await service.create_definition(user=owner, request=_create())

    with pytest.raises(NotAuthorizedError):
        await service.list_definitions(user=member)
    with pytest.raises(NotAuthorizedError):
        await service.create_definition(user=member, request=_create(name="member-lakera"))
    with pytest.raises(NotAuthorizedError):
        await service.update_definition(
            user=member,
            definition_id=created.id,
            request=OrganizationGuardrailDefinitionUpdate(enabled=False),
        )
    with pytest.raises(NotAuthorizedError):
        await service.delete_definition(user=member, definition_id=created.id)


async def test_another_organizations_definition_is_not_found(async_db: AsyncSession) -> None:
    """Not found rather than forbidden, so this is no existence oracle across tenants.

    The composite foreign key on the mandate makes a cross-tenant *link*
    impossible in the database. This is the read half of the same rule.
    """
    theirs = await _organization(async_db, slug="theirs")
    their_owner = await _member(async_db, theirs, role="owner", full_name="Their Owner")
    ours = await _organization(async_db, slug="ours")
    our_owner = await _member(async_db, ours, role="owner", full_name="Our Owner")
    service = _service(async_db)

    created = await service.create_definition(user=their_owner, request=_create())

    with pytest.raises(OrganizationGuardrailDefinitionNotFoundError):
        await service.update_definition(
            user=our_owner,
            definition_id=created.id,
            request=OrganizationGuardrailDefinitionUpdate(enabled=False),
        )
    await _after_a_refusal(async_db, our_owner)

    with pytest.raises(OrganizationGuardrailDefinitionNotFoundError):
        await service.delete_definition(user=our_owner, definition_id=created.id)
    await _after_a_refusal(async_db, our_owner)

    assert (await service.list_definitions(user=our_owner)).count == 0


async def test_one_definition_per_name(async_db: AsyncSession) -> None:
    """The unique index is the guard, and the conflict is what a caller is owed.

    Also the rename case, which no prior read can rule out: the second write
    loses at the flush rather than at a check.
    """
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    await service.create_definition(user=owner, request=_create(name="first"))
    second = await service.create_definition(user=owner, request=_create(name="second"))

    with pytest.raises(OrganizationGuardrailDefinitionAlreadyExistsError):
        await service.create_definition(user=owner, request=_create(name="first"))
    await _after_a_refusal(async_db, owner)

    with pytest.raises(OrganizationGuardrailDefinitionAlreadyExistsError):
        await service.update_definition(
            user=owner,
            definition_id=second.id,
            request=OrganizationGuardrailDefinitionUpdate(name="first"),
        )
    await _after_a_refusal(async_db, owner)

    assert {entry.name for entry in (await service.list_definitions(user=owner)).data} == {"first", "second"}


async def test_the_same_name_in_two_organizations(async_db: AsyncSession) -> None:
    """Unique per organization, not globally: the key is the pair."""
    theirs = await _organization(async_db, slug="theirs")
    their_owner = await _member(async_db, theirs, role="owner", full_name="Their Owner")
    ours = await _organization(async_db, slug="ours")
    our_owner = await _member(async_db, ours, role="owner", full_name="Our Owner")
    service = _service(async_db)

    await service.create_definition(user=their_owner, request=_create(name="prod"))
    await service.create_definition(user=our_owner, request=_create(name="prod"))

    assert (await service.list_definitions(user=our_owner)).count == 1


async def test_the_ceiling_bounds_what_one_organization_defines(async_db: AsyncSession) -> None:
    """Its own ceiling, bounding built vendor clients held per worker."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)

    for index in range(MAX_DEFINITIONS_PER_ORGANIZATION):
        await service.create_definition(user=owner, request=_create(name=f"lakera-{index}"))

    with pytest.raises(OrganizationGuardrailDefinitionLimitReachedError):
        await service.create_definition(user=owner, request=_create(name="one-too-many"))
    await _after_a_refusal(async_db, owner)

    assert (await service.list_definitions(user=owner)).count == MAX_DEFINITIONS_PER_ORGANIZATION


# --------------------------------------------------------------------------- #
# The rules, reaching the write path
# --------------------------------------------------------------------------- #


async def test_the_catalog_decides_what_may_be_defined(async_db: AsyncSession) -> None:
    """Three refusals, each a different answer, and none of them stores a row.

    The rules are unit-tested; what this asserts is that the write path runs
    them, and that a refused write leaves nothing behind.
    """
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)

    # Each of these three is refused before a block is opened, the guardrail
    # being resolved from the catalog before anything is read or written.
    with pytest.raises(OrganizationGuardrailNotBuildableError):
        await service.create_definition(user=owner, request=_create(guardrail_name="susfactor"))
    with pytest.raises(OrganizationGuardrailNotDefinableError):
        await service.create_definition(user=owner, request=_create(guardrail_name="any_llm", create_kwargs={}))
    with pytest.raises(OrganizationGuardrailDefinitionArgumentsError):
        await service.create_definition(
            user=owner, request=_create(create_kwargs={"api_key": "k", "not_a_parameter": 1})
        )

    assert (await service.list_definitions(user=owner)).count == 0


async def test_a_refused_edit_leaves_the_row_as_it_was(async_db: AsyncSession) -> None:
    """The block rolls back, so a half-applied definition is not a state this can reach."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    created = await service.create_definition(user=owner, request=_create())

    with pytest.raises(OrganizationGuardrailDefinitionArgumentsError):
        await service.update_definition(
            user=owner,
            definition_id=created.id,
            request=OrganizationGuardrailDefinitionUpdate(name="renamed", create_kwargs={"bad_argument": 1}),
        )
    await _after_a_refusal(async_db, owner)

    listed = (await service.list_definitions(user=owner)).data
    assert [entry.name for entry in listed] == ["prod-lakera"]
    assert listed[0].create_kwargs == {"endpoint": VENDOR_ENDPOINT}


# --------------------------------------------------------------------------- #
# What a column holds
# --------------------------------------------------------------------------- #


async def test_a_credential_reaches_only_the_encrypted_column(async_db: AsyncSession) -> None:
    """Split by the catalog's flag, so the plain column never sees the vendor key."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)

    created = await service.create_definition(user=owner, request=_create())

    stored = await _row(async_db, created.id)
    assert stored.create_kwargs == {"endpoint": VENDOR_ENDPOINT}
    assert stored.encrypted_create_secrets is not None
    assert "lakera-key" not in stored.encrypted_create_secrets
    assert json.loads(decrypt_secret(stored.encrypted_create_secrets)) == {"api_key": "lakera-key"}


async def test_a_definition_with_no_secret_stores_no_ciphertext(async_db: AsyncSession) -> None:
    """Nothing to keep, so nothing is written: an empty map is not a credential."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)

    created = await service.create_definition(
        user=owner,
        request=_create(create_kwargs={"endpoint": VENDOR_ENDPOINT}),
    )

    assert created.create_secrets == {}
    assert (await _row(async_db, created.id)).encrypted_create_secrets is None


async def test_editing_another_field_keeps_the_stored_credential(async_db: AsyncSession) -> None:
    """An edit that never mentions the arguments must not destroy what it was never shown."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    created = await service.create_definition(user=owner, request=_create())

    await service.update_definition(
        user=owner,
        definition_id=created.id,
        request=OrganizationGuardrailDefinitionUpdate(enabled=False),
    )

    stored = await _row(async_db, created.id)
    assert stored.enabled is False
    assert stored.encrypted_create_secrets is not None
    assert json.loads(decrypt_secret(stored.encrypted_create_secrets)) == {"api_key": "lakera-key"}


async def test_a_mask_keeps_the_credential_and_a_value_rotates_it(async_db: AsyncSession) -> None:
    """How a form that only ever saw ``***`` saves a change to the endpoint beside it."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    created = await service.create_definition(user=owner, request=_create())

    kept = await service.update_definition(
        user=owner,
        definition_id=created.id,
        request=OrganizationGuardrailDefinitionUpdate(create_kwargs={"api_key": "***", "endpoint": OTHER_ENDPOINT}),
    )
    assert kept.create_kwargs == {"endpoint": OTHER_ENDPOINT}
    stored = await _row(async_db, created.id)
    assert stored.encrypted_create_secrets is not None
    assert json.loads(decrypt_secret(stored.encrypted_create_secrets)) == {"api_key": "lakera-key"}

    await service.update_definition(
        user=owner,
        definition_id=created.id,
        request=OrganizationGuardrailDefinitionUpdate(create_kwargs={"api_key": "rotated"}),
    )
    stored = await _row(async_db, created.id)
    assert stored.create_kwargs == {}, "the endpoint was left out, so it is gone with the rest"
    assert stored.encrypted_create_secrets is not None
    assert json.loads(decrypt_secret(stored.encrypted_create_secrets)) == {"api_key": "rotated"}


async def test_changing_the_guardrail_re_splits_the_stored_arguments(async_db: AsyncSession) -> None:
    """The split is the old class's answer, so the row cannot keep it and stay honest.

    Both of Lakera's stored halves are merged and split again under OpenAI
    moderation, which takes ``api_key`` as a secret and ``base_url`` rather than
    ``endpoint``, so the stale argument is refused rather than carried.
    """
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    created = await service.create_definition(user=owner, request=_create())

    with pytest.raises(OrganizationGuardrailDefinitionArgumentsError) as refused:
        await service.update_definition(
            user=owner,
            definition_id=created.id,
            request=OrganizationGuardrailDefinitionUpdate(guardrail_name="openai_moderation"),
        )
    assert "endpoint" in str(refused.value)
    await _after_a_refusal(async_db, owner)

    switched = await service.update_definition(
        user=owner,
        definition_id=created.id,
        request=OrganizationGuardrailDefinitionUpdate(
            guardrail_name="openai_moderation",
            create_kwargs={"api_key": "***", "base_url": OTHER_ENDPOINT},
        ),
    )
    assert switched.guardrail_name == "openai_moderation"
    assert switched.create_kwargs == {"base_url": OTHER_ENDPOINT}
    stored = await _row(async_db, created.id)
    assert stored.encrypted_create_secrets is not None
    assert json.loads(decrypt_secret(stored.encrypted_create_secrets)) == {"api_key": "lakera-key"}


async def test_a_row_whose_credentials_cannot_be_read_still_lists(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One unreadable row must not cost an organization the page that would let them fix it.

    Reported rather than raised, the way `routes/providers.py` reports an
    undecryptable provider credential. The row stays editable, so the way out is
    to send the credential again.
    """
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    created = await service.create_definition(user=owner, request=_create())

    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())

    listed = (await service.list_definitions(user=owner)).data
    assert [entry.secrets_decryptable for entry in listed] == [False]
    assert listed[0].create_secrets == {}
    assert listed[0].create_kwargs == {"endpoint": VENDOR_ENDPOINT}

    repaired = await service.update_definition(
        user=owner,
        definition_id=created.id,
        request=OrganizationGuardrailDefinitionUpdate(
            create_kwargs={"api_key": "retyped", "endpoint": VENDOR_ENDPOINT}
        ),
    )
    assert repaired.secrets_decryptable is True
    assert repaired.create_secrets == {"api_key": "***"}


# --------------------------------------------------------------------------- #
# A definition a mandate still names
# --------------------------------------------------------------------------- #


async def test_deleting_a_definition_a_mandate_names_is_refused(async_db: AsyncSession) -> None:
    """The database's RESTRICT reaches the caller as a conflict naming the mandate.

    Unhandled it would be a 500, because the refusal arrives from the DELETE as
    the same `IntegrityError` a duplicate name raises.
    """
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    created = await service.create_definition(user=owner, request=_create())
    async_db.add(_mandate(organization, profile="prompt-injection", definition_id=created.id))
    await async_db.commit()

    with pytest.raises(OrganizationGuardrailDefinitionInUseError) as refused:
        await service.delete_definition(user=owner, definition_id=created.id)
    assert "prompt-injection" in str(refused.value)

    # The refused DELETE was its own SAVEPOINT, so the service can still read
    # the definition it just declined to drop.
    await _after_a_refusal(async_db, owner)
    assert [entry.id for entry in (await service.list_definitions(user=owner)).data] == [created.id]


async def test_dropping_the_mandate_frees_the_definition(async_db: AsyncSession) -> None:
    """The guard holds a reference and not the row, so removing it releases the definition."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    created = await service.create_definition(user=owner, request=_create())
    mandate = _mandate(organization, profile="prompt-injection", definition_id=created.id)
    async_db.add(mandate)
    await async_db.commit()

    await async_db.delete(mandate)
    await async_db.commit()

    await service.delete_definition(user=owner, definition_id=created.id)
    assert (await service.list_definitions(user=owner)).data == []


# --------------------------------------------------------------------------- #
# The endpoints a definition may name
# --------------------------------------------------------------------------- #


async def test_refuses_an_endpoint_this_gateway_must_not_dial(async_db: AsyncSession) -> None:
    """The rule is unit-tested; what this asserts is that the write path runs it."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)

    with pytest.raises(OrganizationGuardrailDefinitionUnsafeUrlError) as refused:
        await service.create_definition(
            user=owner,
            request=_create(create_kwargs={"api_key": "lakera-key", "endpoint": "https://169.254.169.254/latest/"}),
        )

    assert "endpoint" in str(refused.value)
    assert (await service.list_definitions(user=owner)).count == 0


async def test_refuses_an_edit_that_turns_the_endpoint_inward(async_db: AsyncSession) -> None:
    """The stored endpoint and the stored credential both survive the refusal."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    created = await service.create_definition(user=owner, request=_create())

    with pytest.raises(OrganizationGuardrailDefinitionUnsafeUrlError):
        await service.update_definition(
            user=owner,
            definition_id=created.id,
            request=OrganizationGuardrailDefinitionUpdate(
                create_kwargs={"api_key": "***", "endpoint": "https://10.0.0.5/v2"}
            ),
        )
    await _after_a_refusal(async_db, owner)

    stored = await _row(async_db, created.id)
    assert stored.create_kwargs == {"endpoint": VENDOR_ENDPOINT}
    assert stored.encrypted_create_secrets is not None
    assert json.loads(decrypt_secret(stored.encrypted_create_secrets)) == {"api_key": "lakera-key"}


async def test_an_address_nested_in_an_argument_is_checked_too(async_db: AsyncSession) -> None:
    """Alinia's ``detection_config`` is a real argument that holds a whole object."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)

    with pytest.raises(OrganizationGuardrailDefinitionUnsafeUrlError) as refused:
        await service.create_definition(
            user=owner,
            request=_create(
                name="prod-alinia",
                guardrail_name="alinia",
                create_kwargs={"detection_config": {"webhook": "http://10.0.0.5/collect"}},
            ),
        )

    assert "detection_config.webhook" in str(refused.value)
    assert (await service.list_definitions(user=owner)).count == 0


# --------------------------------------------------------------------------- #
# What a read says about the build
# --------------------------------------------------------------------------- #


async def test_a_worker_that_holds_nothing_reads_as_pending(async_db: AsyncSession) -> None:
    """What every worker but the one that served the write says, until the tick reaches it.

    It must not read as `failed`, which is what an admin would act on, and it
    must not read as `built`, which would hide a mandate nothing is evaluating.
    """
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    await service.create_definition(user=owner, request=_create())

    # Which is what a sibling worker is: it holds nothing for this row yet.
    runner.reset_guardrail_runner()

    assert [entry.build_state for entry in (await service.list_definitions(user=owner)).data] == ["pending"]


async def test_a_definition_this_worker_built_reads_as_built(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    created = await service.create_definition(user=owner, request=_create())
    _hold(organization.id, created, guardrail=object())

    page = await service.list_definitions(user=owner)

    assert [entry.build_state for entry in page.data] == ["built"]


async def test_a_definition_this_worker_could_not_build_reads_as_failed(async_db: AsyncSession) -> None:
    """The reading this field exists for: saved, and not running."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    created = await service.create_definition(user=owner, request=_create())
    _hold(organization.id, created, guardrail=None)

    page = await service.list_definitions(user=owner)

    assert [entry.build_state for entry in page.data] == ["failed"]


async def test_a_build_of_an_older_version_does_not_read_as_built(async_db: AsyncSession) -> None:
    """A guardrail built from arguments the row no longer has is not this row's health.

    Reporting it as `built` would answer for a definition nobody is looking at,
    which is the reading an admin acts on after repairing one.
    """
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    created = await service.create_definition(user=owner, request=_create())

    # What a worker looks like mid-write: still holding the previous build.
    held = runner._held[(organization.id, created.id)]
    runner._held[(organization.id, created.id)] = runner._Held(
        fingerprint=held.fingerprint - timedelta(seconds=1),
        guardrail_name=held.guardrail_name,
        guardrail=held.guardrail,
    )

    assert [entry.build_state for entry in (await service.list_definitions(user=owner)).data] == ["pending"]


async def test_a_disabled_definition_reads_as_disabled(async_db: AsyncSession) -> None:
    """Nothing is built for it on purpose, so `pending` would promise a build that is not coming."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    created = await service.create_definition(user=owner, request=_create())

    updated = await service.update_definition(
        user=owner, definition_id=created.id, request=OrganizationGuardrailDefinitionUpdate(enabled=False)
    )

    assert updated.build_state == "disabled"


async def test_another_organizations_build_is_not_reported_as_this_ones(async_db: AsyncSession) -> None:
    """The state is keyed on the pair, so two organizations cannot read each other's health."""
    mine = await _organization(async_db, slug="mine")
    theirs = await _organization(async_db, slug="theirs")
    owner = await _member(async_db, mine, role="owner", full_name="Owner")
    service = _service(async_db)
    created = await service.create_definition(user=owner, request=_create())
    runner.reset_guardrail_runner()
    _hold(theirs.id, created, guardrail=object())

    page = await service.list_definitions(user=owner)

    assert [entry.build_state for entry in page.data] == ["pending"]


async def test_a_row_whose_secrets_will_not_decrypt_still_reports_its_build(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two independent failures, and a listing reports both rather than dropping the row."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    created = await service.create_definition(user=owner, request=_create())
    _hold(organization.id, created, guardrail=None)

    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    listed = (await service.list_definitions(user=owner)).data

    assert listed[0].secrets_decryptable is False
    assert listed[0].build_state == "failed"


# --------------------------------------------------------------------------- #
# What a write does about the build
# --------------------------------------------------------------------------- #


@pytest.fixture
def vendor(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub any-guardrail with one that builds, returning the log of what it built."""
    built: list[str] = []

    class _Stub:
        @staticmethod
        def create(guardrail_name: Any, **_kwargs: Any) -> Any:
            built.append(str(guardrail_name))
            return object()

    monkeypatch.setattr(runner, "AnyGuardrail", _Stub)
    return built


async def test_a_create_builds_the_definition_and_says_so(async_db: AsyncSession, vendor: list[str]) -> None:
    """The moment the admin pressed Save is the moment to tell them it is running."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")

    created = await _service(async_db).create_definition(user=owner, request=_create())

    assert created.build_state == "built"
    assert len(vendor) == 1
    assert runner.handle(organization.id, created.id) is not None


async def test_a_create_whose_build_fails_still_saves_and_reports_it(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The row is the truth and the runner is a copy of it.

    "Saved successfully" and "saved successfully, and every mandate pointing
    here now refuses requests" must not be the same response, and a vendor that
    will not answer must not roll back a definition the form accepted.
    """
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")

    class _Stub:
        @staticmethod
        def create(_guardrail_name: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("vendor refused the key")

    monkeypatch.setattr(runner, "AnyGuardrail", _Stub)
    service = _service(async_db)

    created = await service.create_definition(user=owner, request=_create())

    assert created.build_state == "failed"
    stored = (await service.list_definitions(user=owner)).data
    assert [entry.id for entry in stored] == [created.id]


async def test_a_patch_that_repairs_a_definition_reports_it_running_again(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loop an admin is actually in: see `failed`, fix the key, see `built`."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")

    working = False

    class _Stub:
        @staticmethod
        def create(_guardrail_name: Any, **_kwargs: Any) -> Any:
            if not working:
                raise RuntimeError("vendor refused the key")
            return object()

    monkeypatch.setattr(runner, "AnyGuardrail", _Stub)
    service = _service(async_db)
    created = await service.create_definition(user=owner, request=_create())
    assert created.build_state == "failed"

    working = True
    repaired = await service.update_definition(
        user=owner,
        definition_id=created.id,
        request=OrganizationGuardrailDefinitionUpdate(
            create_kwargs={"api_key": "the-right-key", "endpoint": VENDOR_ENDPOINT}
        ),
    )

    assert repaired.build_state == "built"


async def test_disabling_a_definition_stops_it_on_the_worker_that_served_the_write(
    async_db: AsyncSession, vendor: list[str]
) -> None:
    """Thirty seconds late is not good enough for a kill switch on the worker you reached."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    created = await service.create_definition(user=owner, request=_create())
    assert runner.handle(organization.id, created.id) is not None

    updated = await service.update_definition(
        user=owner, definition_id=created.id, request=OrganizationGuardrailDefinitionUpdate(enabled=False)
    )

    assert updated.build_state == "disabled"
    assert runner.handle(organization.id, created.id) is None


async def test_deleting_a_definition_drops_what_the_worker_held(
    async_db: AsyncSession, vendor: list[str]
) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    created = await service.create_definition(user=owner, request=_create())
    assert runner.handle(organization.id, created.id) is not None

    await service.delete_definition(user=owner, definition_id=created.id)

    assert runner.handle(organization.id, created.id) is None


async def test_a_patch_that_changed_nothing_does_not_dial_the_vendor_again(
    async_db: AsyncSession, vendor: list[str]
) -> None:
    """The stamp only moves when a column does, so renaming nothing costs no handshake."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = _service(async_db)
    created = await service.create_definition(user=owner, request=_create())
    assert len(vendor) == 1

    unchanged = await service.update_definition(
        user=owner, definition_id=created.id, request=OrganizationGuardrailDefinitionUpdate()
    )

    assert unchanged.build_state == "built"
    assert len(vendor) == 1


async def test_a_rebuild_that_cannot_run_does_not_fail_the_write(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The write is committed by then, so reporting it as a 500 would be a lie."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")

    async def _unavailable(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("the database went away")

    service = _service(async_db)
    monkeypatch.setattr(OrganizationGuardrailDefinitionRepository, "get_in_organization", _unavailable, raising=True)

    created = await service.create_definition(user=owner, request=_create())

    assert created.build_state == "pending"
