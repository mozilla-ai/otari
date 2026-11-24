"""The endpoint credential revision's Alembic chain, exercised on SQLite.

The OSS base ships SQLite by default, and this revision adds a CHECK
constraint to an existing table, which SQLite has no ``ALTER TABLE ... ADD
CONSTRAINT`` for. Alembic's batch mode rebuilds the table instead, and a
rebuild is what can lose an index or a constraint. ``org_provider_keys``
carries two *partial* unique indexes, so the revision passes ``copy_from`` and
the rebuild takes the table from that declaration rather than from reflection.

Every integration run migrates PostgreSQL and nothing migrates SQLite, so this
is the only coverage of that path. Driven against a real file database rather
than in-memory because the rebuild is what is under test. Modeled on
``test_scoped_budget_alignment_chain.py``.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import CheckConstraint, Engine, create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the shared metadata)

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_ENDPOINT_AUTH_REVISION = "c7e1b9d4a260"
_PREVIOUS_REVISION = "f6b8d0a2c4e7"
_CHECK = "ck_org_provider_keys_endpoint_auth_all_or_none"
_INDEXES = {
    "ix_org_provider_keys_organization_id",
    "uq_org_provider_keys_org_default",
    "uq_org_provider_keys_org_slug",
}
_COLUMNS = (
    "encrypted_endpoint_auth_secret",
    "endpoint_auth_expires_at",
    "endpoint_auth_header_name",
    "endpoint_auth_last4",
    "endpoint_auth_method",
)
_WHOLE = {
    "encrypted_endpoint_auth_secret": "ciphertext",
    "endpoint_auth_expires_at": "2027-11-20",
    "endpoint_auth_header_name": "Fastly-Key",
    "endpoint_auth_last4": "cret",
    "endpoint_auth_method": "header",
}

_INSERT = text(
    "INSERT INTO org_provider_keys"
    " (id, organization_id, provider, name, endpoint_auth_method, endpoint_auth_header_name,"
    "  encrypted_endpoint_auth_secret, endpoint_auth_last4, endpoint_auth_expires_at,"
    "  created_at, updated_at)"
    " VALUES (:id, :id, 'openai', :id, :endpoint_auth_method, :endpoint_auth_header_name,"
    "  :encrypted_endpoint_auth_secret, :endpoint_auth_last4, :endpoint_auth_expires_at,"
    "  '2026-11-20', '2026-11-20')"
)


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture
def sqlite_at_head(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    """A SQLite database migrated to the endpoint credential revision."""
    database_url = f"sqlite:///{tmp_path / 'endpoint_auth.db'}"
    config = _alembic_config(database_url)
    command.upgrade(config, _ENDPOINT_AUTH_REVISION)
    engine = create_engine(database_url)
    try:
        yield config, engine
    finally:
        engine.dispose()


def _normalized(sql: str) -> str:
    """``sql`` with every run of whitespace collapsed, so two renderings compare."""
    return " ".join(sql.split())


def _insert(engine: Engine, key_id: str, **columns: str | None) -> None:
    with engine.begin() as connection:
        connection.execute(_INSERT, {"id": key_id, **{name: columns.get(name) for name in _COLUMNS}})


def test_upgrade_adds_the_five_columns_and_the_check(sqlite_at_head: tuple[Config, Engine]) -> None:
    _, engine = sqlite_at_head
    inspector = inspect(engine)

    added = {column["name"]: column for column in inspector.get_columns("org_provider_keys")}
    assert set(_COLUMNS) <= set(added)
    assert all(added[name]["nullable"] is True for name in _COLUMNS)
    assert _CHECK in {check["name"] for check in inspector.get_check_constraints("org_provider_keys")}


def test_the_partial_unique_indexes_survive_the_batch_rebuild(sqlite_at_head: tuple[Config, Engine]) -> None:
    """A rebuild that dropped a WHERE clause would widen both uniqueness rules,
    which then refuse a second key the organization is allowed to hold."""
    _, engine = sqlite_at_head
    assert _INDEXES <= {index["name"] for index in inspect(engine).get_indexes("org_provider_keys")}

    with engine.begin() as connection:
        definitions = connection.execute(
            text("SELECT name, sql FROM sqlite_master WHERE type = 'index' AND tbl_name = 'org_provider_keys'")
        ).all()
    partial = {name: sql for name, sql in definitions if name.startswith("uq_")}
    assert "WHERE slug IS NOT NULL" in partial["uq_org_provider_keys_org_slug"]
    default_index = partial["uq_org_provider_keys_org_default"]
    assert "WHERE is_org_default AND archived_at IS NULL AND slug IS NULL" in default_index


def test_both_legal_states_are_storable(sqlite_at_head: tuple[Config, Engine]) -> None:
    """No credential, and a whole one. Existing rows are the first, so the
    CHECK must not have made them unwritable."""
    _, engine = sqlite_at_head

    _insert(engine, "none")
    _insert(engine, "whole", **_WHOLE)

    with engine.begin() as connection:
        stored = connection.execute(text("SELECT id FROM org_provider_keys ORDER BY id")).scalars().all()
    assert list(stored) == ["none", "whole"]


@pytest.mark.parametrize("column", _COLUMNS)
def test_the_check_rejects_a_row_holding_only_one_of_the_five(
    sqlite_at_head: tuple[Config, Engine], column: str
) -> None:
    """One column alone declares a credential with nothing behind it."""
    _, engine = sqlite_at_head
    with pytest.raises(IntegrityError):
        _insert(engine, column, **{column: _WHOLE[column]})


@pytest.mark.parametrize("column", _COLUMNS)
def test_the_check_rejects_a_row_missing_one_of_the_five(sqlite_at_head: tuple[Config, Engine], column: str) -> None:
    """Four columns set is the shape a partial write leaves behind."""
    _, engine = sqlite_at_head
    partial = {name: value for name, value in _WHOLE.items() if name != column}
    with pytest.raises(IntegrityError):
        _insert(engine, column, **partial)


def test_downgrade_drops_the_columns_and_keeps_the_rest(sqlite_at_head: tuple[Config, Engine]) -> None:
    config, engine = sqlite_at_head
    _insert(engine, "whole", **_WHOLE)

    command.downgrade(config, _PREVIOUS_REVISION)

    inspector = inspect(engine)
    assert not set(_COLUMNS) & {column["name"] for column in inspector.get_columns("org_provider_keys")}
    assert _CHECK not in {check["name"] for check in inspector.get_check_constraints("org_provider_keys")}
    assert _INDEXES <= {index["name"] for index in inspector.get_indexes("org_provider_keys")}
    with engine.begin() as connection:
        assert connection.execute(text("SELECT provider FROM org_provider_keys")).scalar() == "openai"


def test_upgrade_downgrade_upgrade_round_trips(sqlite_at_head: tuple[Config, Engine]) -> None:
    config, engine = sqlite_at_head

    command.downgrade(config, _PREVIOUS_REVISION)
    command.upgrade(config, _ENDPOINT_AUTH_REVISION)

    inspector = inspect(engine)
    assert set(_COLUMNS) <= {column["name"] for column in inspector.get_columns("org_provider_keys")}
    assert _CHECK in {check["name"] for check in inspector.get_check_constraints("org_provider_keys")}
    assert _INDEXES <= {index["name"] for index in inspector.get_indexes("org_provider_keys")}


def test_the_model_declares_the_same_check(sqlite_at_head: tuple[Config, Engine]) -> None:
    """A migrated schema and a ``create_all`` schema both have to carry the rule.

    Nothing else in the suite compares a check constraint between the two, so a
    model that forgot this one, or spelled it differently, would leave every
    ``create_all`` database enforcing something else.
    """
    _, engine = sqlite_at_head
    table = SQLModel.metadata.tables["org_provider_keys"]
    declared = {
        constraint.name: constraint for constraint in table.constraints if isinstance(constraint, CheckConstraint)
    }
    assert _CHECK in declared

    with engine.begin() as connection:
        migrated = connection.execute(
            text("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'org_provider_keys'")
        ).scalar_one()
    assert _normalized(str(declared[_CHECK].sqltext)) in _normalized(migrated)
