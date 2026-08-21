"""The two consolidation revisions' Alembic chain, exercised on SQLite with data.

The OSS base ships SQLite by default, and both revisions drop columns and add a
foreign key on existing tables, neither of which SQLite has an ``ALTER TABLE``
for: Alembic's batch mode rebuilds the table instead, and a rebuild silently
drops what reflection could not see. Both tables carry *partial* unique indexes,
which is exactly that, so ``copy_from`` in each revision is what has to carry
them across.

Every integration run migrates PostgreSQL and nothing migrates SQLite, so this is
the only coverage of that path. Modeled on ``test_scoped_budget_alignment_chain``,
with one addition: both revisions **mint rows**, so this drives them with data and
asserts what the backfill produced, not only that the schema arrived.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_DEFAULTS_REVISION = "e2f4a6c8b0d3"
_CEILINGS_REVISION = "f3a5c7e9d1b4"
_PREVIOUS_REVISION = "db8fbf901ee0"

_DEFAULT_INDEXES = {
    "uq_workspace_budget_defaults_with_key",
    "uq_workspace_budget_defaults_no_key",
    "ix_workspace_budget_defaults_workspace_id",
    "ix_workspace_budget_defaults_budget_id",
}
_CEILING_INDEXES = {
    "uq_scoped_budgets_scope_with_key",
    "uq_scoped_budgets_scope_no_key",
    "ix_scoped_budgets_scope",
    "ix_scoped_budgets_budget_id",
}


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


def _seed(engine: Engine) -> None:
    """A workspace with a default, two members materialized from it, and a stray ceiling.

    The two materialized ceilings hold the same figures as the default, which is
    what materialization produced before this pair of revisions. The stray one
    holds a shape no default names, so it has to mint a budget of its own.
    """
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO organization (id, name, slug, created_at)"
                " VALUES ('11111111-1111-1111-1111-111111111111', 'Acme', 'acme', '2026-08-21')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO workspace (id, organization_id, name, activation_classification, created_at)"
                " VALUES ('22222222-2222-2222-2222-222222222222',"
                " '11111111-1111-1111-1111-111111111111', 'Alpha', 'eligible', '2026-08-21')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO workspace_budget_defaults"
                " (id, workspace_id, provider_key_id, name, max_budget, budget_duration_sec,"
                "  created_at, updated_at)"
                " VALUES ('d1', '22222222-2222-2222-2222-222222222222', NULL, 'Alpha member default',"
                " 50.0, 2592000, '2026-08-21', '2026-08-21')"
            )
        )
        for member in ("m1", "m2"):
            connection.execute(
                text(
                    "INSERT INTO scoped_budgets"
                    " (id, scope_type, scope_id, provider_key_id, name, max_budget, current_spend,"
                    "  reserved_spend, budget_duration_sec, reset_alignment, created_at, updated_at)"
                    f" VALUES ('sb-{member}', 'workspace_member', '{member}', NULL, NULL, 50.0, 3.0, 0.0,"
                    " 2592000, NULL, '2026-08-21', '2026-08-21')"
                )
            )
        connection.execute(
            text(
                "INSERT INTO scoped_budgets"
                " (id, scope_type, scope_id, provider_key_id, name, max_budget, current_spend,"
                "  reserved_spend, budget_duration_sec, reset_alignment, created_at, updated_at)"
                " VALUES ('sb-org', 'organization', 'org-1', NULL, NULL, NULL, 1.0, 0.0,"
                " NULL, 'calendar_month', '2026-08-21', '2026-08-21')"
            )
        )


@pytest.fixture
def seeded_sqlite(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    """A SQLite database at the revision before the pair, carrying rows to migrate."""
    database_url = f"sqlite:///{tmp_path / 'consolidation.db'}"
    config = _alembic_config(database_url)
    command.upgrade(config, _PREVIOUS_REVISION)
    engine = create_engine(database_url)
    _seed(engine)
    try:
        yield config, engine
    finally:
        engine.dispose()


def test_the_pair_lands_every_index_through_two_sqlite_rebuilds(seeded_sqlite: tuple[Config, Engine]) -> None:
    """Both tables keep their partial unique indexes and gain the new ones."""
    config, engine = seeded_sqlite
    command.upgrade(config, _CEILINGS_REVISION)

    inspector = inspect(engine)
    defaults = {index["name"] for index in inspector.get_indexes("workspace_budget_defaults")}
    ceilings = {index["name"] for index in inspector.get_indexes("scoped_budgets")}
    assert _DEFAULT_INDEXES <= defaults, f"missing on workspace_budget_defaults: {_DEFAULT_INDEXES - defaults}"
    assert _CEILING_INDEXES <= ceilings, f"missing on scoped_budgets: {_CEILING_INDEXES - ceilings}"

    # The figures moved off both tables and onto the budget each row names.
    default_columns = {column["name"] for column in inspector.get_columns("workspace_budget_defaults")}
    ceiling_columns = {column["name"] for column in inspector.get_columns("scoped_budgets")}
    assert "budget_id" in default_columns
    assert "budget_id" in ceiling_columns
    assert not ({"max_budget", "budget_duration_sec", "reset_alignment"} & ceiling_columns)
    assert not ({"max_budget", "budget_duration_sec", "name"} & default_columns)


def test_a_default_and_the_ceilings_it_materialized_land_on_one_budget(
    seeded_sqlite: tuple[Config, Engine],
) -> None:
    """The whole point of the pair, and the way it is easiest to get wrong.

    The second revision mints a budget per distinct ceiling *shape*. If it does
    not first look at what the first revision already created, a workspace's
    default and the ceilings materialized from it end up naming two different
    budgets holding equal numbers: the budgets list would say a budget was a
    workspace's default while nobody in that workspace was enforced against it,
    and editing it would move no one. That is the state the pair exists to
    remove, so re-creating it here would make the migration a no-op in practice.
    """
    config, engine = seeded_sqlite
    command.upgrade(config, _CEILINGS_REVISION)

    with engine.connect() as connection:
        default_budget = connection.execute(
            text("SELECT budget_id FROM workspace_budget_defaults WHERE id = 'd1'")
        ).scalar_one()
        materialized = connection.execute(
            text("SELECT budget_id FROM scoped_budgets WHERE scope_type = 'workspace_member' ORDER BY id")
        ).scalars().all()
        stray = connection.execute(text("SELECT budget_id FROM scoped_budgets WHERE id = 'sb-org'")).scalar_one()
        budgets = connection.execute(
            text("SELECT budget_id, name, max_budget, budget_duration_sec, reset_alignment FROM budgets")
        ).all()

    assert materialized == [default_budget, default_budget], "both members must be held to the default's own budget"
    assert stray != default_budget, "a ceiling whose shape no default names needs a budget of its own"
    # Two budgets, not four: one for the default's shape, one for the stray's.
    assert len(budgets) == 2, budgets
    by_id = {row.budget_id: row for row in budgets}
    assert by_id[default_budget].name == "Alpha member default", "the default's own budget keeps its name"
    assert by_id[default_budget].max_budget == 50.0
    assert by_id[stray].reset_alignment == "calendar_month", "the calendar cadence survives onto the budget"


def test_the_pair_round_trips_with_its_figures_intact(seeded_sqlite: tuple[Config, Engine]) -> None:
    """Down and up again, which is what a rollback of a bad deploy actually runs.

    The minted budgets are deliberately left behind by the downgrade, since
    nothing records which of them existed only to back a ceiling. What has to
    come back is every figure, on both tables, and the counters the ceilings hold.
    """
    config, engine = seeded_sqlite
    command.upgrade(config, _CEILINGS_REVISION)
    command.downgrade(config, _PREVIOUS_REVISION)

    with engine.connect() as connection:
        ceilings = connection.execute(
            text(
                "SELECT id, max_budget, budget_duration_sec, reset_alignment, current_spend"
                " FROM scoped_budgets ORDER BY id"
            )
        ).all()
        default = connection.execute(
            text("SELECT name, max_budget, budget_duration_sec FROM workspace_budget_defaults WHERE id = 'd1'")
        ).one()

    assert [(row.max_budget, row.budget_duration_sec, row.reset_alignment) for row in ceilings] == [
        (50.0, 2592000, None),
        (50.0, 2592000, None),
        (None, None, "calendar_month"),
    ]
    assert [row.current_spend for row in ceilings] == [3.0, 3.0, 1.0], "counters are the ceiling's own and must survive"
    assert (default.name, default.max_budget, default.budget_duration_sec) == ("Alpha member default", 50.0, 2592000)

    # And forward again, which a redeploy after a rollback runs.
    command.upgrade(config, _CEILINGS_REVISION)
    with engine.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM scoped_budgets WHERE budget_id IS NULL")).scalar_one() == 0
