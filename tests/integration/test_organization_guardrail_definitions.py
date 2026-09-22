"""What the database refuses about an organization's guardrail definitions.

No service writes these rows yet, so the constraints *are* the enforcement, and
that is the point: the composite foreign key holds when a write path forgets to
check which organization a definition belongs to, and the check constraint holds
when one forgets that the two backends are exclusive.

On PostgreSQL rather than SQLite, because SQLite enforces a foreign key only
with ``PRAGMA foreign_keys`` on and the schema-chain unit test already covers
the check constraint there.

Rows are inserted with the session directly. There is no service to go through,
and a helper that hid the columns would hide exactly what is under test.
"""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.models.guardrails import OrganizationGuardrail, OrganizationGuardrailDefinition
from gateway.models.tenancy import Organization
from gateway.repositories.tenancy import OrganizationRepository

pytestmark = pytest.mark.asyncio

# A public IP literal, so nothing here depends on a resolver.
PUBLIC_URL = "https://93.184.216.34/guardrails"


async def _organization(db: AsyncSession, *, slug: str) -> Organization:
    return await OrganizationRepository(db).create_organization(name=slug.title(), slug=slug, created_by_user_id=None)


def _definition(organization: Organization, *, name: str = "lakera") -> OrganizationGuardrailDefinition:
    return OrganizationGuardrailDefinition(
        organization_id=organization.id,
        name=name,
        guardrail_name="lakera_guard",
        create_kwargs={"endpoint": PUBLIC_URL},
        encrypted_create_secrets=None,
        enabled=True,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _mandate(
    organization: Organization,
    *,
    profile: str = "prompt-injection",
    url: str | None = None,
    definition_id: uuid.UUID | None = None,
) -> OrganizationGuardrail:
    return OrganizationGuardrail(
        organization_id=organization.id,
        profile=profile,
        url=url,
        definition_id=definition_id,
        mode="monitor",
        on_unavailable="block",
        enabled=True,
        applies_to_all_workspaces=True,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


async def test_a_mandate_with_neither_backend_still_stores(async_db: AsyncSession) -> None:
    """The shape every existing row is in, and the reason the check is not an XOR."""
    organization = await _organization(async_db, slug="acme")
    async_db.add(_mandate(organization))
    await async_db.commit()

    stored = (await async_db.execute(select(OrganizationGuardrail))).scalar_one()
    assert stored.url is None
    assert stored.definition_id is None


async def test_a_mandate_cannot_name_both_a_url_and_a_definition(async_db: AsyncSession) -> None:
    """One check runs on one backend. Both set is a contradiction, not a preference."""
    organization = await _organization(async_db, slug="acme")
    definition = _definition(organization)
    async_db.add(definition)
    await async_db.flush()

    async_db.add(_mandate(organization, url=PUBLIC_URL, definition_id=definition.id))
    with pytest.raises(IntegrityError):
        await async_db.commit()
    await async_db.rollback()


async def test_a_mandate_cannot_reach_another_organizations_definition(async_db: AsyncSession) -> None:
    """The composite foreign key, and the whole tenant-isolation story for the link.

    A definition holds a vendor credential. Refused by the database, so it holds
    against a write path that trusted its input.
    """
    acme = await _organization(async_db, slug="acme")
    other = await _organization(async_db, slug="globex")
    theirs = _definition(other)
    async_db.add(theirs)
    await async_db.flush()

    async_db.add(_mandate(acme, definition_id=theirs.id))
    with pytest.raises(IntegrityError):
        await async_db.commit()
    await async_db.rollback()


async def test_two_mandates_can_share_one_definition(async_db: AsyncSession) -> None:
    """What the split buys: one vendor key, one rotation, however many policies."""
    organization = await _organization(async_db, slug="acme")
    definition = _definition(organization)
    async_db.add(definition)
    await async_db.flush()

    async_db.add(_mandate(organization, profile="strict", definition_id=definition.id))
    async_db.add(_mandate(organization, profile="lenient", definition_id=definition.id))
    await async_db.commit()

    linked = (await async_db.execute(select(OrganizationGuardrail.definition_id))).scalars().all()
    assert linked == [definition.id, definition.id]


async def test_one_definition_name_per_organization(async_db: AsyncSession) -> None:
    """A mandate points at a definition by name, so two of one name is ambiguous."""
    organization = await _organization(async_db, slug="acme")
    async_db.add(_definition(organization, name="lakera"))
    await async_db.flush()

    async_db.add(_definition(organization, name="lakera"))
    with pytest.raises(IntegrityError):
        await async_db.commit()
    await async_db.rollback()


async def test_two_organizations_may_use_the_same_definition_name(async_db: AsyncSession) -> None:
    """The name is the organization's own label, not a deployment-wide key."""
    acme = await _organization(async_db, slug="acme")
    globex = await _organization(async_db, slug="globex")
    async_db.add(_definition(acme, name="lakera"))
    async_db.add(_definition(globex, name="lakera"))
    await async_db.commit()

    assert len((await async_db.execute(select(OrganizationGuardrailDefinition))).scalars().all()) == 2


async def test_deleting_a_mandated_definition_is_refused(async_db: AsyncSession) -> None:
    """RESTRICT, because CASCADE would silently stop a guardrail running."""
    organization = await _organization(async_db, slug="acme")
    definition = _definition(organization)
    async_db.add(definition)
    await async_db.flush()
    async_db.add(_mandate(organization, definition_id=definition.id))
    await async_db.commit()

    with pytest.raises(IntegrityError):
        await async_db.execute(
            delete(OrganizationGuardrailDefinition).where(OrganizationGuardrailDefinition.id == definition.id)
        )
    await async_db.rollback()


async def test_an_unmandated_definition_can_be_deleted(async_db: AsyncSession) -> None:
    """RESTRICT guards a reference, not the row: nothing names this one."""
    organization = await _organization(async_db, slug="acme")
    definition = _definition(organization)
    async_db.add(definition)
    await async_db.commit()

    await async_db.execute(
        delete(OrganizationGuardrailDefinition).where(OrganizationGuardrailDefinition.id == definition.id)
    )
    await async_db.commit()

    assert (await async_db.execute(select(OrganizationGuardrailDefinition))).scalars().all() == []


async def test_deleting_the_organization_removes_both(async_db: AsyncSession) -> None:
    """The one case RESTRICT must not block.

    Both tables cascade from ``organization``, and both cascades run inside the
    one statement, so by the time the RESTRICT check fires at the end of it the
    mandate is already gone.
    """
    organization = await _organization(async_db, slug="acme")
    definition = _definition(organization)
    async_db.add(definition)
    await async_db.flush()
    async_db.add(_mandate(organization, definition_id=definition.id))
    await async_db.commit()

    await async_db.execute(delete(Organization).where(col(Organization.id) == organization.id))
    await async_db.commit()

    assert (await async_db.execute(select(OrganizationGuardrailDefinition))).scalars().all() == []
    assert (await async_db.execute(select(OrganizationGuardrail))).scalars().all() == []
