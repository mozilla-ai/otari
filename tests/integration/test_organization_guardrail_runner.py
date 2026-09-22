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

import time
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

    assert runner.build_state(first.id, held.id, held.updated_at) == "built"
    assert runner.build_state(second.id, elsewhere.id, elsewhere.updated_at) == "built"
    assert runner.build_state(first.id, off.id, off.updated_at) == "pending"
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
    assert runner.build_state(organization.id, definition.id, definition.updated_at) == "pending"


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

    assert runner.build_state(organization.id, definition_id, datetime.now(UTC)) == "pending"


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

    assert runner.build_state(organization.id, good.id, good.updated_at) == "built"
    assert runner.build_state(organization.id, bad.id, bad.updated_at) == "failed"
    assert runner.handle(organization.id, bad.id) is None


async def test_a_build_that_ran_out_of_time_is_tried_again_on_the_next_tick(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The distinction the deadline exists to make: unfinished is not failed.

    A failed build is held until the row changes, so a timeout recorded as one
    would take a definition out of service until an admin edited it for no
    reason. Nothing is held, and the next tick tries again.
    """
    organization = await _organization(async_db, slug="slow-vendor")
    await _definition(async_db, organization, name="lakera")

    attempts = 0

    def _outlast_the_deadline(guardrail_name: GuardrailName, **_kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            time.sleep(0.5)
        return _Built()

    class _Stub:
        create = staticmethod(_outlast_the_deadline)

    monkeypatch.setattr(runner, "AnyGuardrail", _Stub)
    monkeypatch.setattr(runner, "_BUILD_TIMEOUT_SECONDS", 0.05)

    await _refresh(async_db)
    assert runner._held == {}

    monkeypatch.setattr(runner, "_BUILD_TIMEOUT_SECONDS", 20.0)
    await _refresh(async_db)

    assert len(runner._held) == 1
    assert next(iter(runner._held.values())).guardrail is not None


async def test_a_build_that_failed_is_not_tried_again(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the same rule, so the retry above cannot quietly widen."""
    organization = await _organization(async_db, slug="bad-credential")
    await _definition(async_db, organization, name="lakera")

    attempts = 0

    def _explode(_guardrail_name: GuardrailName, **_kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("vendor refused the key")

    class _Stub:
        create = staticmethod(_explode)

    monkeypatch.setattr(runner, "AnyGuardrail", _Stub)

    await _refresh(async_db)
    await _refresh(async_db)

    assert attempts == 1
    assert next(iter(runner._held.values())).guardrail is None


# --------------------------------------------------------------------------- #
# Rebuilding one definition, which is what a write asks for
# --------------------------------------------------------------------------- #


async def test_one_definition_is_rebuilt_without_touching_the_others(
    async_db: AsyncSession, builds: list[GuardrailName]
) -> None:
    """A write is about one row, and an unrelated organization's vendor is not dialed for it."""
    mine = await _organization(async_db, slug="mine")
    theirs = await _organization(async_db, slug="theirs")
    definition = await _definition(async_db, mine, name="lakera")
    await _definition(async_db, theirs, name="lakera")

    state = await runner.rebuild_definition(UnitOfWork(async_db), mine.id, definition.id)

    assert state == "built"
    assert len(builds) == 1
    assert list(runner._held) == [(mine.id, definition.id)]


async def test_a_rebuild_reports_a_failure_rather_than_raising(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The write is already committed, so the build's outcome is news, not an error."""
    organization = await _organization(async_db, slug="bad-key")
    definition = await _definition(async_db, organization, name="lakera")

    class _Stub:
        @staticmethod
        def create(_guardrail_name: GuardrailName, **_kwargs: Any) -> Any:
            raise RuntimeError("vendor refused the key")

    monkeypatch.setattr(runner, "AnyGuardrail", _Stub)

    assert await runner.rebuild_definition(UnitOfWork(async_db), organization.id, definition.id) == "failed"


async def test_a_rebuild_that_runs_out_of_time_is_pending_rather_than_failed(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An admin waits ten seconds at most, and a slow vendor is not a broken definition."""
    organization = await _organization(async_db, slug="slow")
    definition = await _definition(async_db, organization, name="lakera")

    class _Stub:
        @staticmethod
        def create(_guardrail_name: GuardrailName, **_kwargs: Any) -> Any:
            time.sleep(0.5)
            return _Built()

    monkeypatch.setattr(runner, "AnyGuardrail", _Stub)
    monkeypatch.setattr(runner, "_WRITE_BUILD_TIMEOUT_SECONDS", 0.05)

    assert await runner.rebuild_definition(UnitOfWork(async_db), organization.id, definition.id) == "pending"
    assert runner._held == {}


async def test_a_rebuild_of_a_disabled_definition_drops_what_was_held(
    async_db: AsyncSession, builds: list[GuardrailName]
) -> None:
    """Turning a definition off has to stop the guardrail on the worker that served the write."""
    organization = await _organization(async_db, slug="switched-off")
    definition = await _definition(async_db, organization, name="lakera")
    await _refresh(async_db)
    assert runner.handle(organization.id, definition.id) is not None

    definition.enabled = False
    await async_db.flush()

    assert await runner.rebuild_definition(UnitOfWork(async_db), organization.id, definition.id) == "pending"
    assert runner.handle(organization.id, definition.id) is None


async def test_a_rebuild_of_a_deleted_definition_drops_what_was_held(
    async_db: AsyncSession, builds: list[GuardrailName]
) -> None:
    organization = await _organization(async_db, slug="deleted")
    definition = await _definition(async_db, organization, name="lakera")
    await _refresh(async_db)
    definition_id = definition.id

    await async_db.delete(definition)
    await async_db.flush()

    assert await runner.rebuild_definition(UnitOfWork(async_db), organization.id, definition_id) == "pending"
    assert runner.handle(organization.id, definition_id) is None


async def test_a_rebuild_will_not_reach_another_organizations_definition(
    async_db: AsyncSession, builds: list[GuardrailName]
) -> None:
    """The tenant predicate is the repository's, so a leaked id alone builds nothing."""
    theirs = await _organization(async_db, slug="not-yours")
    definition = await _definition(async_db, theirs, name="lakera")

    state = await runner.rebuild_definition(UnitOfWork(async_db), uuid.uuid4(), definition.id)

    assert state == "pending"
    assert builds == []


async def test_a_rebuild_of_an_unchanged_row_builds_nothing_again(
    async_db: AsyncSession, builds: list[GuardrailName]
) -> None:
    """A write that changed no column leaves the stamp alone, so the vendor is not dialed twice."""
    organization = await _organization(async_db, slug="untouched")
    definition = await _definition(async_db, organization, name="lakera")
    await _refresh(async_db)
    assert len(builds) == 1

    assert await runner.rebuild_definition(UnitOfWork(async_db), organization.id, definition.id) == "built"
    assert len(builds) == 1


async def test_a_definition_this_worker_never_held_is_not_a_handle(async_db: AsyncSession) -> None:
    """A request must not be able to mistake "not loaded here" for "not mandated"."""
    organization = await _organization(async_db, slug="runner-empty")

    assert runner.handle(organization.id, uuid.uuid4()) is None
