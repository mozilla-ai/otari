"""The budget migration preflight against a real schema.

The rounding and partitioning logic is unit tested; what needs a database is the
join that pairs budgets with the users attached to them, and the CLI command
that opens the database read only.
"""

import json
from collections.abc import Generator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from click.testing import CliRunner
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

import gateway.cli as gateway_cli
from gateway.db import Base, Budget, BudgetResetLog, User, reset_db
from gateway.services.budget_migration_report import build_migration_report

_DAY = 86_400


def _budgets() -> list[Budget]:
    """One budget of each kind the report has to tell apart."""
    return [
        Budget(budget_id="b-daily", name="Daily", max_budget=100.0, budget_duration_sec=_DAY),
        Budget(budget_id="b-sprint", name="Sprint", max_budget=250.0, budget_duration_sec=3 * _DAY),
        Budget(budget_id="b-forever", name="No reset", max_budget=50.0, budget_duration_sec=None),
        Budget(budget_id="b-orphan", name="Orphan", max_budget=10.0, budget_duration_sec=7 * _DAY),
    ]


def _users(now: datetime) -> list[User]:
    return [
        User(user_id="alice", spend=12.5, reserved=1.0, budget_id="b-daily"),
        User(user_id="bob", spend=40.0, reserved=0.0, budget_id="b-sprint"),
        User(user_id="carol", spend=7.25, reserved=0.5, budget_id="b-sprint"),
        # Soft deleted: migrates as a deactivated identity, so no live cap.
        User(user_id="dave", spend=3.0, reserved=0.0, budget_id="b-forever", deleted_at=now),
        # No budget attached at all.
        User(user_id="erin", spend=0.0, reserved=0.0, budget_id=None),
    ]


def _reset_log(now: datetime) -> BudgetResetLog:
    return BudgetResetLog(user_id="alice", budget_id="b-daily", previous_spend=5.0, reset_at=now)


@pytest_asyncio.fixture
async def seeded(async_db: AsyncSession) -> AsyncSession:
    now = datetime.now(UTC)
    async_db.add_all(_budgets())
    await async_db.flush()
    async_db.add_all(_users(now))
    async_db.add(_reset_log(now))
    await async_db.commit()
    return async_db


@pytest.fixture
def released_engine() -> Generator[None]:
    """Dispose the process-wide engine a CLI command leaves behind.

    ``init_db`` installs a global session factory, which would otherwise point
    the next test at this test's database.
    """
    yield
    reset_db()


@pytest.fixture
def seeded_url(postgres_url: str) -> Generator[str]:
    """Seed the same rows for the CLI, which drives its own synchronous session.

    The command calls ``asyncio.run``, so its tests cannot be async, and an async
    fixture would leave them without a running loop to seed from. Alembic is not
    needed here: the report reads three tables the models already describe.
    """
    engine = create_engine(postgres_url, pool_pre_ping=True)
    Base.metadata.create_all(bind=engine)
    now = datetime.now(UTC)
    with sessionmaker(bind=engine)() as session:
        session.add_all(_budgets())
        session.flush()
        session.add_all(_users(now))
        session.add(_reset_log(now))
        session.commit()
    try:
        yield postgres_url
    finally:
        # The command leaves a process-wide engine behind, which would otherwise
        # point the next test at this database.
        reset_db()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.mark.asyncio
async def test_report_pairs_budgets_with_their_attached_users(seeded: AsyncSession) -> None:
    report = await build_migration_report(seeded)

    assert {plan.budget_id for plan in report.budgets} == {"b-daily", "b-sprint", "b-forever", "b-orphan"}
    assert [plan.budget_id for plan in report.shared_pools] == ["b-sprint"]
    (shared,) = report.shared_pools
    assert [(user.user_id, user.spend, user.reserved) for user in shared.live_attached] == [
        ("bob", 40.0, 0.0),
        ("carol", 7.25, 0.5),
    ]
    # alice, bob and carol; dave is soft deleted and erin has no budget.
    assert report.member_budgets_to_create == 3
    assert report.deleted_attachments == 1
    assert report.reset_log_count == 1


@pytest.mark.asyncio
async def test_report_partitions_durations_and_orphans(seeded: AsyncSession) -> None:
    report = await build_migration_report(seeded)

    assert [plan.budget_id for plan in report.rounded] == ["b-sprint"]
    assert [plan.budget_id for plan in report.periodless] == ["b-forever"]
    # b-forever's only user is soft deleted, so nothing live points at it either.
    assert {plan.budget_id for plan in report.unattached} == {"b-forever", "b-orphan"}


@pytest.mark.asyncio
async def test_report_is_empty_on_a_gateway_with_no_budgets(async_db: AsyncSession) -> None:
    report = await build_migration_report(async_db)

    assert report.budgets == []
    assert report.member_budgets_to_create == 0


def test_cli_prints_the_report(seeded_url: str) -> None:
    result = CliRunner().invoke(gateway_cli.budgets_migration_report, ["--database-url", seeded_url])

    assert result.exit_code == 0, result.output
    assert "4 budget(s) in this gateway." in result.output
    assert "3 member budget(s)" in result.output
    assert "3d -> daily" in result.output
    assert '"Sprint" (b-sprint)' in result.output


def test_cli_json_output_is_machine_readable(seeded_url: str) -> None:
    result = CliRunner().invoke(gateway_cli.budgets_migration_report, ["--database-url", seeded_url, "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["summary"]["shared_pools"] == 1
    assert payload["summary"]["member_budgets_to_create"] == 3


def test_cli_does_not_print_the_database_password(released_engine: None) -> None:
    # Port 1 refuses immediately, so this fails at connect without a live server.
    result = CliRunner().invoke(
        gateway_cli.budgets_migration_report,
        ["--database-url", "postgresql://otari:hunter2@127.0.0.1:1/otari"],
    )

    assert result.exit_code != 0
    assert "hunter2" not in result.output
    # The host is still named, because a failure that does not say which database
    # it tried is not actionable.
    assert "127.0.0.1:1/otari" in result.output


def test_cli_reports_a_database_it_cannot_read(tmp_path: object) -> None:
    try:
        result = CliRunner().invoke(
            gateway_cli.budgets_migration_report,
            ["--database-url", f"sqlite:///{tmp_path}/absent.db"],
        )
    finally:
        reset_db()

    assert result.exit_code != 0
    assert "Could not read budgets" in result.output
    assert "otari migrate" in result.output
