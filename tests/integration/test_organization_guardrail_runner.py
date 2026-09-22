"""What the guardrail runner holds after a refresh, and what a second refresh redoes.

The build of one definition is covered next door in
`tests/unit/test_organization_guardrail_runner.py`. What needs rows, and so
needs to be here, is the diff: which entries survive a tick untouched, which are
rebuilt, and which are dropped. That is the part that decides whether a worker
does real vendor I/O twice a minute or almost none.

any-guardrail is stubbed. A build here is a counter, so "was this rebuilt" is a
question the test can actually ask.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from any_guardrail import GuardrailName
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.guardrails import OrganizationGuardrailDefinition
from gateway.models.tenancy import Organization
from gateway.repositories.tenancy import OrganizationRepository
from gateway.services.tenancy import organization_guardrail_runner as runner

pytestmark = pytest.mark.asyncio

# A public IP literal, so nothing here depends on a resolver.
VENDOR_ENDPOINT = "https://93.184.216.34/v2"


@pytest.fixture(autouse=True)
def _empty_runner() -> Any:
    runner.reset_guardrail_runner()
    yield
    runner.reset_guardrail_runner()


class _Built:
    """One constructed guardrail. Identity is what the diff test compares."""


@pytest.fixture
def builds(monkeypatch: pytest.MonkeyPatch) -> list[GuardrailName]:
    """Replace any-guardrail with a stub, returning the log of builds it did."""
    done: list[GuardrailName] = []

    class _Stub:
        @staticmethod
        def create(guardrail_name: GuardrailName, **_kwargs: Any) -> Any:
            done.append(guardrail_name)
            return _Built()

    monkeypatch.setattr(runner, "AnyGuardrail", _Stub)
    return done


async def _organization(db: AsyncSession, *, slug: str) -> Organization:
    return await OrganizationRepository(db).create_organization(name=slug.title(), slug=slug, created_by_user_id=None)


async def _definition(
    db: AsyncSession, organization: Organization, *, name: str, enabled: bool = True, endpoint: str = VENDOR_ENDPOINT
) -> OrganizationGuardrailDefinition:
    definition = OrganizationGuardrailDefinition(
        organization_id=organization.id,
        name=name,
        guardrail_name="lakera_guard",
        create_kwargs={"endpoint": endpoint},
        encrypted_create_secrets=None,
        enabled=enabled,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    db.add(definition)
    await db.flush()
    return definition


async def _refresh(db: AsyncSession) -> None:
    await runner.refresh_guardrail_runner(UnitOfWork(db))


async def test_a_refresh_holds_every_organizations_enabled_definitions(
    async_db: AsyncSession, builds: list[GuardrailName]
) -> None:
    """One pass over the deployment, and the disabled row is not built at all."""
    first = await _organization(async_db, slug="runner-holds-one")
    second = await _organization(async_db, slug="runner-holds-two")
    held = await _definition(async_db, first, name="held")
    off = await _definition(async_db, first, name="off", enabled=False)
    elsewhere = await _definition(async_db, second, name="held")

    await _refresh(async_db)

    assert runner.build_state(first.id, held.id) == "built"
    assert runner.build_state(second.id, elsewhere.id) == "built"
    assert runner.build_state(first.id, off.id) is None
    assert len(builds) == 2


async def test_a_second_refresh_rebuilds_nothing_that_did_not_move(
    async_db: AsyncSession, builds: list[GuardrailName]
) -> None:
    """The whole point of the fingerprint. Twice a minute, per worker, forever."""
    organization = await _organization(async_db, slug="runner-steady")
    definition = await _definition(async_db, organization, name="held")

    await _refresh(async_db)
    first_object = runner._held[(organization.id, definition.id)].guardrail
    await _refresh(async_db)

    assert len(builds) == 1
    assert runner._held[(organization.id, definition.id)].guardrail is first_object


async def test_a_definition_that_was_written_to_is_rebuilt(
    async_db: AsyncSession, builds: list[GuardrailName]
) -> None:
    """``updated_at`` carries every write, so the fingerprint needs no second column."""
    organization = await _organization(async_db, slug="runner-edited")
    definition = await _definition(async_db, organization, name="held")

    await _refresh(async_db)
    definition.updated_at = definition.updated_at + timedelta(seconds=1)
    await async_db.flush()
    await _refresh(async_db)

    assert len(builds) == 2


async def test_a_definition_that_was_disabled_is_dropped(
    async_db: AsyncSession, builds: list[GuardrailName]
) -> None:
    """A kill switch that left the built client in memory would not be one."""
    organization = await _organization(async_db, slug="runner-disabled")
    definition = await _definition(async_db, organization, name="held")

    await _refresh(async_db)
    definition.enabled = False
    await async_db.flush()
    await _refresh(async_db)

    assert runner.handle(organization.id, definition.id) is None
    assert runner.build_state(organization.id, definition.id) is None


async def test_a_definition_that_was_deleted_is_dropped(
    async_db: AsyncSession, builds: list[GuardrailName]
) -> None:
    """Nothing points at it any more, so nothing should hold its vendor client."""
    organization = await _organization(async_db, slug="runner-deleted")
    definition = await _definition(async_db, organization, name="held")
    definition_id = definition.id

    await _refresh(async_db)
    await async_db.delete(definition)
    await async_db.flush()
    await _refresh(async_db)

    assert runner.build_state(organization.id, definition_id) is None


async def test_one_row_that_will_not_build_does_not_cost_the_others_theirs(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The posture that matters at startup: a bad row must not take the boot down."""
    organization = await _organization(async_db, slug="runner-mixed")
    good = await _definition(async_db, organization, name="good")
    bad = await _definition(async_db, organization, name="bad", endpoint="https://93.184.216.35/broken")

    class _Stub:
        @staticmethod
        def create(guardrail_name: GuardrailName, **kwargs: Any) -> Any:
            del guardrail_name
            if kwargs["endpoint"].endswith("/broken"):
                raise RuntimeError("vendor is unhappy")
            return _Built()

    monkeypatch.setattr(runner, "AnyGuardrail", _Stub)

    await _refresh(async_db)

    assert runner.build_state(organization.id, good.id) == "built"
    assert runner.build_state(organization.id, bad.id) == "failed"
    assert runner.handle(organization.id, bad.id) is None


async def test_a_definition_this_worker_never_held_is_not_a_handle(async_db: AsyncSession) -> None:
    """A request must not be able to mistake "not loaded here" for "not mandated"."""
    organization = await _organization(async_db, slug="runner-empty")

    assert runner.handle(organization.id, uuid.uuid4()) is None
