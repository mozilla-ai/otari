"""The tenancy revision's Alembic chain, exercised on SQLite.

The OSS base ships SQLite by default, so the reconciled schema has to migrate
there as well as on PostgreSQL, and this revision has a step that behaves
differently on each: ``user.active_organization_id`` and
``organization.created_by_user_id`` reference each other, so one of the two
foreign keys can only be added after both tables exist. PostgreSQL takes that as
an ``ALTER TABLE``; SQLite has no ``ADD CONSTRAINT``, so Alembic's batch mode
rebuilds the table, which is the step that can silently lose a constraint.

Every integration run migrates PostgreSQL and nothing migrates SQLite, so this
is the only coverage of that path. Driven against a real file database rather
than in-memory because batch mode's rebuild is what is under test.

The later revisions that reshape the same tables are exercised here too, for the
same reason: the workspace scoping that rebuilds four request-plane tables, the
credential columns added to ``user``, whose downgrade depends on SQLite's refusal
to drop an indexed column, the identity column added to ``dashboard_sessions``,
which rebuilds that table to tighten a column to NOT NULL and point it at
``user``, and the two revisions that finish scoping the gateway survivals: one
swaps the alias and policy uniqueness constraints (a batch rebuild with the
partial indexes taken out and put back around it), the other adds
``workspace_id`` to three more tables.
"""

import json
import subprocess
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Connection, Engine, create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the shared metadata)
from gateway.models.tenancy import UtcDateTime

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_TENANCY_REVISION = "c4b6d8e0f2a3"
_PREVIOUS_REVISION = "b2d4f6a8c0e1"
_TENANCY_TABLES = {"user", "organization", "organization_member", "workspace", "workspace_member"}

_SESSION_IDENTITY_REVISION = "b6d8f0a2c4e7"
_BEFORE_SESSION_IDENTITY = "7ff4e082eb0c"
_SESSION_USER_INDEX = "ix_dashboard_sessions_user_id"
_SESSION_EXPIRY_INDEX = "ix_dashboard_sessions_expires_at"
BOOTSTRAP_IDENTITY_KEY = "tenancy_bootstrap_user_id"

_CREDENTIAL_REVISION = "f2a4c6d8b0e3"
_BEFORE_CREDENTIALS = "a3c7e1b9d5f2"
_CREDENTIAL_COLUMNS = {
    "hashed_password",
    "terms_accepted_at",
    "oauth_provider",
    "email_verification_token",
    "email_verified_at",
}
_TOKEN_INDEX = "ix_user_email_verification_token"

_ALIAS_WIDEN_REVISION = "c1e4a7b9d3f6"
_SURVIVALS_REVISION = "d2f5b8c0e4a7"
_SURVIVAL_TABLES = ("routing_memory", "router_preferences", "file_objects")


def _parent_of(revision: str) -> str:
    """The revision immediately below ``revision``, read from the chain itself.

    Deliberately derived rather than written down. This pair of revisions sits at
    the end of the chain, so every migration that lands on ``main`` while the
    branch is open re-points the lower one's ``down_revision``, and a hardcoded
    constant here is a second place to remember. Reading it back means the
    downgrade targets below follow the re-point on their own, and the test keeps
    asserting what it means ("roll back to just before this revision") rather
    than a literal that goes stale.
    """
    parent = ScriptDirectory.from_config(_alembic_config("sqlite://")).get_revision(revision).down_revision
    assert isinstance(parent, str), f"{revision} should have exactly one parent, got {parent!r}"
    return parent

_TOKEN_REVISION = "db8fbf901ee0"
_BEFORE_TOKENS = "c8e2a4f6b0d3"
_TOKEN_COLUMNS = {
    "email_verification_token_hash",
    "email_verification_token_expires_at",
    "password_reset_token_hash",
    "password_reset_token_expires_at",
}
_VERIFICATION_TOKEN_HASH_INDEX = "ix_user_email_verification_token_hash"
_RESET_TOKEN_HASH_INDEX = "ix_user_password_reset_token_hash"


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture
def sqlite_at_head(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    """A SQLite database migrated to head, with its config for further steps."""
    database_url = f"sqlite:///{tmp_path / 'tenancy.db'}"
    config = _alembic_config(database_url)
    command.upgrade(config, "head")
    engine = create_engine(database_url)
    try:
        yield config, engine
    finally:
        engine.dispose()


def test_upgrade_creates_every_tenancy_table(sqlite_at_head: tuple[Config, Engine]) -> None:
    _, engine = sqlite_at_head
    assert _TENANCY_TABLES <= set(inspect(engine).get_table_names())


def test_circular_foreign_keys_survive_the_batch_rebuild(sqlite_at_head: tuple[Config, Engine]) -> None:
    """Both halves of the cycle are present, pointing at each other."""
    _, engine = sqlite_at_head
    inspector = inspect(engine)

    user_targets = {
        (fk["referred_table"], tuple(fk["constrained_columns"])) for fk in inspector.get_foreign_keys("user")
    }
    organization_targets = {
        (fk["referred_table"], tuple(fk["constrained_columns"])) for fk in inspector.get_foreign_keys("organization")
    }

    assert ("organization", ("active_organization_id",)) in user_targets
    assert ("user", ("created_by_user_id",)) in organization_targets


def test_rebuilt_table_keeps_its_named_unique_constraint(sqlite_at_head: tuple[Config, Engine]) -> None:
    """The rebuild must not degrade ``uq_organization_slug`` into an unnamed index.

    ``copy_from`` in the revision is what guarantees this; without it the
    constraint's identity depends on SQLite reflection.
    """
    _, engine = sqlite_at_head
    constraint_names = {constraint["name"] for constraint in inspect(engine).get_unique_constraints("organization")}
    assert "uq_organization_slug" in constraint_names


def test_email_is_uniquely_indexed_and_nullable(sqlite_at_head: tuple[Config, Engine]) -> None:
    """A local identity has no email, so the column is nullable but still unique."""
    _, engine = sqlite_at_head
    inspector = inspect(engine)

    email_indexes = [index for index in inspector.get_indexes("user") if index["column_names"] == ["email"]]
    assert [index["unique"] for index in email_indexes] == [True]

    email_column = next(column for column in inspector.get_columns("user") if column["name"] == "email")
    assert email_column["nullable"] is True


def test_workspace_activation_classification_is_constrained(sqlite_at_head: tuple[Config, Engine]) -> None:
    """The check constraint travels with the column it guards."""
    _, engine = sqlite_at_head
    checks = inspect(engine).get_check_constraints("workspace")
    assert "check_workspace_activation_classification" in {check["name"] for check in checks}


def test_downgrade_removes_the_tenancy_tables_and_leaves_the_rest(sqlite_at_head: tuple[Config, Engine]) -> None:
    """Rolling back this revision alone must not disturb the gateway's own tables."""
    config, engine = sqlite_at_head

    command.downgrade(config, _PREVIOUS_REVISION)

    remaining = set(inspect(engine).get_table_names())
    assert not (_TENANCY_TABLES & remaining)
    # The gateway's own tables predate this revision and must be untouched.
    assert {"users", "api_keys", "usage_logs"} <= remaining


def test_upgrade_downgrade_upgrade_round_trips(sqlite_at_head: tuple[Config, Engine]) -> None:
    """A downgrade leaves nothing behind for the next upgrade to collide with."""
    config, engine = sqlite_at_head

    command.downgrade(config, _PREVIOUS_REVISION)
    command.upgrade(config, _TENANCY_REVISION)

    assert _TENANCY_TABLES <= set(inspect(engine).get_table_names())


def test_naming_one_model_module_registers_them_all() -> None:
    """``Base.metadata`` is whole however few model modules the caller imported.

    ``alembic/env.py`` names only ``gateway.models.entities`` and relies on the
    package ``__init__`` to pull in the rest. If that import chain breaks, the
    metadata silently loses the tenancy tables and autogenerate proposes
    ``DROP TABLE`` for them, which is data-loss-class and invisible until
    someone runs it. Asserting it needs a fresh interpreter, because by the time
    a test runs in this one every model module is already imported.
    """
    source = (
        "from gateway.models.entities import Base;"
        "import json,sys;"
        "sys.stdout.write(json.dumps(sorted(Base.metadata.tables)))"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=True,
    )

    assert _TENANCY_TABLES <= set(json.loads(result.stdout))


def test_workspace_scope_seeds_a_default_and_backfills_existing_rows(tmp_path: Path) -> None:
    """The workspace column is NOT NULL, so the migration has to supply a value.

    Provisioning is lazy, so a gateway that has only ever served completions has
    no organization and no workspace to backfill onto. The migration seeds them
    under the slug and name provisioning looks up, so a later first boot adopts
    these rather than creating a second default.
    """
    url = f"sqlite:///{tmp_path / 'backfill.db'}"
    config = _alembic_config(url)

    command.upgrade(config, "c4b6d8e0f2a3")
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO users (user_id, spend, reserved, blocked, created_at, updated_at, metadata) "
                "VALUES ('alice', 0, 0, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, '{}')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO api_keys (id, key_hash, key_name, user_id, created_at, is_active, "
                "exclude_from_budget, metadata) "
                "VALUES ('k1', 'h1', 'ada', 'alice', CURRENT_TIMESTAMP, 1, 0, '{}')"
            )
        )
        assert connection.execute(text("SELECT COUNT(*) FROM workspace")).scalar_one() == 0

    command.upgrade(config, "head")

    with engine.begin() as connection:
        seeded = connection.execute(
            text(
                "SELECT w.id FROM workspace w "
                "JOIN organization o ON o.id = w.organization_id WHERE o.slug = 'default'"
            )
        ).scalar_one()
        assert connection.execute(text("SELECT workspace_id FROM api_keys")).scalar_one() == seeded
    engine.dispose()


def test_workspace_scope_keeps_the_partial_indexes_it_rebuilds_around(
    sqlite_at_head: tuple[Config, Engine],
) -> None:
    """Adding the column must not cost the alias and policy partial indexes.

    Both tables carry a ``user_id IS NULL`` partial unique index, and SQLite's
    batch mode reflects one poorly. The rebuild happens regardless, because the
    foreign key needs it, so this is what proves the indexes survive it rather
    than what proves it was avoided (the migration's own docstring retracts that
    earlier rationale).

    The names are the workspace-scoped ones because ``c1e4a7b9d3f6`` widened both
    indexes to lead with ``workspace_id``; what is under test here is unchanged,
    that a partial index survives a batch rebuild of its table.
    """
    _, engine = sqlite_at_head
    with engine.begin() as connection:
        partial = {
            name
            for (name,) in connection.execute(
                text("SELECT name FROM sqlite_master WHERE type='index' AND sql LIKE '%WHERE%'")
            )
        }
    assert {
        "uq_model_aliases_workspace_global_name",
        "uq_routing_policies_workspace_global_name",
    } <= partial


def _insert_identity(connection: Connection, *, email: str | None = None, token: str | None = None) -> str:
    """Store one organization-scoped identity, returning its id.

    Raw SQL rather than the models, because what is under test is the migrated
    table: going through SQLModel would assert against the metadata that
    produced the revision instead of against what the revision built. The
    literals are shaped for SQLite, which is the engine every test here drives:
    a UUID is CHAR(32) hex and a boolean is 1 or 0.
    """
    organization_id = uuid.uuid4().hex
    user_id = uuid.uuid4().hex
    connection.execute(
        text(
            "INSERT INTO organization (id, name, slug, created_at) "
            "VALUES (:id, :name, :slug, CURRENT_TIMESTAMP)"
        ),
        {"id": organization_id, "name": f"Org {organization_id[:6]}", "slug": organization_id[:6]},
    )
    connection.execute(
        text(
            'INSERT INTO "user" '
            "(id, email, is_active, is_superuser, full_name, active_organization_id, created_at) "
            "VALUES (:id, :email, 1, 0, 'Ada', :org, CURRENT_TIMESTAMP)"
        ),
        {"id": user_id, "email": email, "org": organization_id},
    )
    if token is not None:
        connection.execute(
            text('UPDATE "user" SET email_verification_token = :token WHERE id = :id'),
            {"token": token, "id": user_id},
        )
    return user_id


def test_the_credential_columns_arrive_nullable(sqlite_at_head: tuple[Config, Engine]) -> None:
    """All five, and every one of them optional.

    Nothing reads them yet, and the master key remains the API credential, so a
    standalone row with all five null is the normal state. A NOT NULL here would
    make the session flow's arrival a data migration rather than a code change.
    """
    _, engine = sqlite_at_head
    columns = {column["name"]: column for column in inspect(engine).get_columns("user")}

    assert _CREDENTIAL_COLUMNS <= set(columns)
    assert [columns[name]["nullable"] for name in sorted(_CREDENTIAL_COLUMNS)] == [True] * 5


def test_the_verification_token_is_uniquely_indexed(sqlite_at_head: tuple[Config, Engine]) -> None:
    """Unique like the platform's, because a shared token confirms the wrong address."""
    _, engine = sqlite_at_head
    indexes = [
        index
        for index in inspect(engine).get_indexes("user")
        if index["column_names"] == ["email_verification_token"]
    ]

    assert [index["name"] for index in indexes] == [_TOKEN_INDEX]
    assert [index["unique"] for index in indexes] == [True]


def test_two_identities_cannot_share_a_verification_token(sqlite_at_head: tuple[Config, Engine]) -> None:
    """The error path the unique index exists for."""
    _, engine = sqlite_at_head

    with engine.begin() as connection:
        _insert_identity(connection, email="ada@example.com", token="tok-1")

    with pytest.raises(IntegrityError), engine.begin() as connection:
        _insert_identity(connection, email="grace@example.com", token="tok-1")


def test_identities_without_a_token_coexist_under_that_index(sqlite_at_head: tuple[Config, Engine]) -> None:
    """Which is every existing row: both engines allow repeated NULLs in a unique index.

    Counted as a delta rather than as a total, so the assertion still describes
    the two rows it inserted if a later revision ever seeds an identity of its
    own (the workspace-scope revision deliberately seeds none today).
    """
    _, engine = sqlite_at_head
    without_a_token = text('SELECT COUNT(*) FROM "user" WHERE email_verification_token IS NULL')

    with engine.begin() as connection:
        before = connection.execute(without_a_token).scalar_one()
        _insert_identity(connection)
        _insert_identity(connection)
        after = connection.execute(without_a_token).scalar_one()

    assert after - before == 2


def test_an_existing_database_upgrades_with_its_rows_untouched(tmp_path: Path) -> None:
    """The upgrade an operator on v0.x runs: additive, and null everywhere it lands."""
    url = f"sqlite:///{tmp_path / 'credentials.db'}"
    config = _alembic_config(url)
    command.upgrade(config, _BEFORE_CREDENTIALS)
    engine = create_engine(url)

    with engine.begin() as connection:
        user_id = _insert_identity(connection, email="ada@example.com")

    command.upgrade(config, _CREDENTIAL_REVISION)

    with engine.begin() as connection:
        row = connection.execute(
            text(
                "SELECT email, full_name, is_active, hashed_password, terms_accepted_at, oauth_provider, "
                'email_verification_token, email_verified_at FROM "user" WHERE id = :id'
            ),
            {"id": user_id},
        ).mappings().one()

    assert (row["email"], row["full_name"], row["is_active"]) == ("ada@example.com", "Ada", 1)
    assert all(row[column] is None for column in _CREDENTIAL_COLUMNS)
    engine.dispose()


def test_the_credential_revision_round_trips(sqlite_at_head: tuple[Config, Engine]) -> None:
    """Down drops the five columns and the index; up puts them back.

    The downgrade drops the index first on purpose: SQLite refuses
    ``DROP COLUMN`` while an index covers the column, and this is the only
    coverage of that ordering, since nothing else migrates SQLite.
    """
    config, engine = sqlite_at_head

    command.downgrade(config, _BEFORE_CREDENTIALS)

    inspector = inspect(engine)
    assert not (_CREDENTIAL_COLUMNS & {column["name"] for column in inspector.get_columns("user")})
    assert _TOKEN_INDEX not in {index["name"] for index in inspector.get_indexes("user")}

    command.upgrade(config, _CREDENTIAL_REVISION)

    assert _CREDENTIAL_COLUMNS <= {column["name"] for column in inspect(engine).get_columns("user")}


def test_the_migrated_columns_match_the_model(sqlite_at_head: tuple[Config, Engine]) -> None:
    """Hand-written revision, so nothing else would notice the two drifting apart.

    The timestamps are the half worth pinning: the platform stores them naive,
    and this schema's departure is that every tenancy timestamp reads back
    UTC-aware on both engines, which ``UtcDateTime`` is what delivers.
    """
    _, engine = sqlite_at_head

    declared = SQLModel.metadata.tables["user"]
    migrated = {column["name"] for column in inspect(engine).get_columns("user")}

    assert migrated == set(declared.columns.keys())
    for name in ("terms_accepted_at", "email_verified_at"):
        assert isinstance(declared.columns[name].type, UtcDateTime)


def test_the_token_columns_arrive_nullable(sqlite_at_head: tuple[Config, Engine]) -> None:
    """All four, and every one of them optional: nothing has claimed or reset yet."""
    _, engine = sqlite_at_head
    columns = {column["name"]: column for column in inspect(engine).get_columns("user")}

    assert _TOKEN_COLUMNS <= set(columns)
    assert [columns[name]["nullable"] for name in sorted(_TOKEN_COLUMNS)] == [True] * 4


def test_the_token_hashes_are_uniquely_indexed(sqlite_at_head: tuple[Config, Engine]) -> None:
    """Unique like the invitation token's hash: two identities cannot share one."""
    _, engine = sqlite_at_head
    indexes_by_column = {
        index["column_names"][0]: index for index in inspect(engine).get_indexes("user") if index["column_names"]
    }

    verification_index = indexes_by_column["email_verification_token_hash"]
    reset_index = indexes_by_column["password_reset_token_hash"]
    assert verification_index["name"] == _VERIFICATION_TOKEN_HASH_INDEX
    assert verification_index["unique"]
    assert reset_index["name"] == _RESET_TOKEN_HASH_INDEX
    assert reset_index["unique"]


def test_an_existing_database_upgrades_with_its_token_columns_untouched(tmp_path: Path) -> None:
    """The upgrade an operator on an earlier v0.x runs: additive, and null everywhere it lands."""
    url = f"sqlite:///{tmp_path / 'tokens.db'}"
    config = _alembic_config(url)
    command.upgrade(config, _BEFORE_TOKENS)
    engine = create_engine(url)

    with engine.begin() as connection:
        user_id = _insert_identity(connection, email="ada@example.com")

    command.upgrade(config, _TOKEN_REVISION)

    with engine.begin() as connection:
        row = connection.execute(
            text(
                "SELECT email_verification_token_hash, email_verification_token_expires_at, "
                'password_reset_token_hash, password_reset_token_expires_at FROM "user" WHERE id = :id'
            ),
            {"id": user_id},
        ).mappings().one()

    assert all(row[column] is None for column in _TOKEN_COLUMNS)
    engine.dispose()


def test_the_token_revision_round_trips(sqlite_at_head: tuple[Config, Engine]) -> None:
    """Down drops the four columns and both indexes; up puts them back.

    The downgrade drops both indexes before their columns on purpose: SQLite
    refuses ``DROP COLUMN`` while an index covers the column, mirroring the
    credential revision's own ordering.
    """
    config, engine = sqlite_at_head

    command.downgrade(config, _BEFORE_TOKENS)

    inspector = inspect(engine)
    assert not (_TOKEN_COLUMNS & {column["name"] for column in inspector.get_columns("user")})
    index_names = {index["name"] for index in inspector.get_indexes("user")}
    assert _VERIFICATION_TOKEN_HASH_INDEX not in index_names
    assert _RESET_TOKEN_HASH_INDEX not in index_names

    command.upgrade(config, _TOKEN_REVISION)

    assert _TOKEN_COLUMNS <= {column["name"] for column in inspect(engine).get_columns("user")}


def test_the_revision_chain_has_one_head() -> None:
    """A second head is not a merge conflict, so nothing else fails when one appears.

    Several migrations land in parallel during the M5 rehome, and Alembic accepts
    two revisions naming the same ``down_revision`` without complaint until
    ``upgrade head`` refuses to choose between them.
    """
    heads = ScriptDirectory.from_config(_alembic_config("sqlite://")).get_heads()

    assert len(heads) == 1, heads


def _insert_session(connection: Connection, token_hash: str) -> None:
    """Store one pre-#647 dashboard session: a token hash and its two timestamps."""
    connection.execute(
        text(
            "INSERT INTO dashboard_sessions (token_hash, created_at, expires_at) "
            "VALUES (:hash, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        ),
        {"hash": token_hash},
    )


def _mark_bootstrap_identity(connection: Connection, value: str) -> None:
    """Point the provisioning marker at ``value``, in the dashed form it holds."""
    connection.execute(
        text("INSERT INTO runtime_settings (key, value, updated_at) VALUES (:key, :value, CURRENT_TIMESTAMP)"),
        {"key": BOOTSTRAP_IDENTITY_KEY, "value": value},
    )


def _at_revision(tmp_path: Path, name: str, revision: str) -> tuple[Config, Engine]:
    """A SQLite database migrated to one revision, for a step-by-step upgrade."""
    url = f"sqlite:///{tmp_path / name}"
    config = _alembic_config(url)
    command.upgrade(config, revision)
    return config, create_engine(url)


def test_a_live_session_is_bound_to_the_bootstrap_operator(tmp_path: Path) -> None:
    """The signed-in operator stays signed in across the upgrade.

    ``user_id`` is NOT NULL, so an existing session has to be attributed to
    someone, and the only right answer is the identity master-key auth already
    resolves to: the one the provisioning marker names.
    """
    config, engine = _at_revision(tmp_path, "session-backfill.db", _BEFORE_SESSION_IDENTITY)
    with engine.begin() as connection:
        identity = _insert_identity(connection)
        _mark_bootstrap_identity(connection, str(uuid.UUID(identity)))
        _insert_session(connection, "hash-of-a-live-session")

    command.upgrade(config, _SESSION_IDENTITY_REVISION)

    with engine.begin() as connection:
        bound = connection.execute(text("SELECT token_hash, user_id FROM dashboard_sessions")).one()
    assert bound == ("hash-of-a-live-session", identity)
    engine.dispose()


def test_a_session_with_nobody_to_name_is_revoked(tmp_path: Path) -> None:
    """A deployment that never served a tenancy request has no identity to bind to.

    Provisioning is lazy, so there is nothing to attribute the row to and the
    column cannot be null: the session is revoked and the operator signs in once
    more, which is what a master-key rotation already does to them.
    """
    config, engine = _at_revision(tmp_path, "session-unattributed.db", _BEFORE_SESSION_IDENTITY)
    with engine.begin() as connection:
        _insert_session(connection, "hash-of-an-unattributed-session")

    command.upgrade(config, _SESSION_IDENTITY_REVISION)

    with engine.begin() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM dashboard_sessions")).scalar_one() == 0
    engine.dispose()


def test_a_marker_naming_a_missing_identity_revokes_rather_than_failing(tmp_path: Path) -> None:
    """The marker can outlive the row it names, and the runtime tolerates that.

    Binding a session to it would fail the foreign key mid-upgrade, so the
    revision checks that the identity exists rather than trusting the marker.
    """
    config, engine = _at_revision(tmp_path, "session-stale-marker.db", _BEFORE_SESSION_IDENTITY)
    with engine.begin() as connection:
        _mark_bootstrap_identity(connection, str(uuid.uuid4()))
        _insert_session(connection, "hash-of-a-session-whose-operator-is-gone")

    command.upgrade(config, _SESSION_IDENTITY_REVISION)

    with engine.begin() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM dashboard_sessions")).scalar_one() == 0
    engine.dispose()


def test_the_session_identity_column_cascades_from_its_identity(sqlite_at_head: tuple[Config, Engine]) -> None:
    """Deleting an identity revokes its sessions instead of orphaning them."""
    _, engine = sqlite_at_head
    foreign_keys = [
        fk for fk in inspect(engine).get_foreign_keys("dashboard_sessions") if fk["referred_table"] == "user"
    ]

    assert [tuple(fk["constrained_columns"]) for fk in foreign_keys] == [("user_id",)]
    assert foreign_keys[0]["options"].get("ondelete") == "CASCADE"


def test_the_session_identity_revision_round_trips(sqlite_at_head: tuple[Config, Engine]) -> None:
    """Down drops the column and its index; up puts them back.

    The expiry index is the one that has to survive both directions: the table is
    rebuilt in each, and a rebuild driven by reflection alone is what loses it
    (hence ``copy_from`` in the revision).
    """
    config, engine = sqlite_at_head

    command.downgrade(config, _BEFORE_SESSION_IDENTITY)

    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns("dashboard_sessions")}
    indexes = {index["name"] for index in inspector.get_indexes("dashboard_sessions")}
    assert "user_id" not in columns
    assert _SESSION_USER_INDEX not in indexes
    assert _SESSION_EXPIRY_INDEX in indexes

    command.upgrade(config, _SESSION_IDENTITY_REVISION)

    inspector = inspect(engine)
    assert "user_id" in {column["name"] for column in inspector.get_columns("dashboard_sessions")}
    assert {_SESSION_USER_INDEX, _SESSION_EXPIRY_INDEX} <= {
        index["name"] for index in inspector.get_indexes("dashboard_sessions")
    }


def test_the_migrated_session_table_matches_the_model(sqlite_at_head: tuple[Config, Engine]) -> None:
    """Hand-written revision, so nothing else would notice the two drifting apart."""
    _, engine = sqlite_at_head

    declared = SQLModel.metadata.tables["dashboard_sessions"]
    migrated = {column["name"]: column for column in inspect(engine).get_columns("dashboard_sessions")}

    assert set(migrated) == set(declared.columns.keys())
    assert migrated["user_id"]["nullable"] is False


# ---------------------------------------------------------------------------
# Workspace-scoped survivals (otari-ai#1643): the two revisions that finish
# scoping what stays in the gateway. Both reshape existing tables, so SQLite's
# batch rebuild is the risk, and this module is its only coverage.
# ---------------------------------------------------------------------------


def _two_workspaces(connection: Connection) -> tuple[str, str]:
    """The migrated default workspace plus a second one beside it, as stored ids.

    A UUID is CHAR(32) hex on SQLite, which is the engine every test here drives,
    so the returned ids are what the ``workspace_id`` columns actually hold.
    """
    default = connection.execute(
        text(
            "SELECT w.id FROM workspace w JOIN organization o ON o.id = w.organization_id "
            "WHERE o.slug = 'default'"
        )
    ).scalar_one()
    organization_id = connection.execute(
        text("SELECT organization_id FROM workspace WHERE id = :id"), {"id": default}
    ).scalar_one()
    second = uuid.uuid4().hex
    connection.execute(
        text(
            "INSERT INTO workspace "
            "(id, organization_id, name, description, created_by_user_id, "
            " activation_classification, created_at) "
            "VALUES (:id, :org, 'Second workspace', NULL, NULL, 'eligible', CURRENT_TIMESTAMP)"
        ),
        {"id": second, "org": organization_id},
    )
    return str(default), second


def test_alias_and_policy_uniqueness_leads_with_the_workspace(
    sqlite_at_head: tuple[Config, Engine],
) -> None:
    """The composite constraint and the partial index both gain ``workspace_id``.

    The pair is what the widening is: the constraint keeps one row per
    (workspace, name, user), and the partial index keeps one workspace-wide row
    per (workspace, name), which the constraint cannot, both engines treating a
    NULL ``user_id`` as distinct.
    """
    _, engine = sqlite_at_head
    inspector = inspect(engine)

    for table, constraint, index in (
        ("model_aliases", "uq_model_aliases_workspace_name_user", "uq_model_aliases_workspace_global_name"),
        (
            "routing_policies",
            "uq_routing_policies_workspace_name_user",
            "uq_routing_policies_workspace_global_name",
        ),
    ):
        unique = {c["name"]: c["column_names"] for c in inspector.get_unique_constraints(table)}
        assert unique[constraint] == ["workspace_id", "name", "user_id"]
        partial = {i["name"]: i["column_names"] for i in inspector.get_indexes(table)}
        assert partial[index] == ["workspace_id", "name"]


def test_the_widening_keeps_the_plain_indexes_the_rebuild_would_drop(
    sqlite_at_head: tuple[Config, Engine],
) -> None:
    """``copy_from`` replaces reflection, so an index left out of it is lost.

    Both tables carry a ``user_id`` and a ``workspace_id`` index that predate this
    revision and that nothing else would notice going missing: a dropped index is
    a slow query, not a failure.
    """
    _, engine = sqlite_at_head
    inspector = inspect(engine)

    for table in ("model_aliases", "routing_policies"):
        names = {index["name"] for index in inspector.get_indexes(table)}
        assert {f"ix_{table}_user_id", f"ix_{table}_workspace_id"} <= names


def test_two_workspaces_can_hold_the_same_alias_name(tmp_path: Path) -> None:
    """The row the narrower constraint refused and the widened one admits.

    Written against the migrated schema rather than the models, so it is the
    revision under test rather than the metadata that produced it.
    """
    url = f"sqlite:///{tmp_path / 'widen.db'}"
    config = _alembic_config(url)
    command.upgrade(config, "head")
    engine = create_engine(url)

    with engine.begin() as connection:
        first, second = _two_workspaces(connection)
        for index, workspace in enumerate((first, second)):
            connection.execute(
                text(
                    "INSERT INTO model_aliases (id, name, target, user_id, workspace_id, created_at, updated_at) "
                    "VALUES (:id, 'fast', :target, NULL, :ws, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                {"id": f"alias-{index}", "target": f"anthropic:model-{index}", "ws": workspace},
            )
        stored = connection.execute(text("SELECT COUNT(*) FROM model_aliases WHERE name = 'fast'")).scalar_one()

    assert stored == 2
    engine.dispose()


def test_one_workspace_still_cannot_hold_two_of_a_name(tmp_path: Path) -> None:
    """Widening admits rows; it does not stop admitting the duplicate it existed for."""
    url = f"sqlite:///{tmp_path / 'widen-dup.db'}"
    config = _alembic_config(url)
    command.upgrade(config, "head")
    engine = create_engine(url)

    with engine.begin() as connection:
        workspace, _second = _two_workspaces(connection)
        connection.execute(
            text(
                "INSERT INTO model_aliases (id, name, target, user_id, workspace_id, created_at, updated_at) "
                "VALUES ('a1', 'fast', 'anthropic:one', NULL, :ws, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            ),
            {"ws": workspace},
        )

    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO model_aliases (id, name, target, user_id, workspace_id, created_at, updated_at) "
                "VALUES ('a2', 'fast', 'anthropic:two', NULL, :ws, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            ),
            {"ws": workspace},
        )
    engine.dispose()


def test_the_widening_round_trips(sqlite_at_head: tuple[Config, Engine]) -> None:
    """Down narrows both constraints back; up widens them again.

    The partial indexes are dropped before each batch rebuild and recreated
    after, so this is what proves neither direction leaves one behind under a
    stale name.
    """
    config, engine = sqlite_at_head

    command.downgrade(config, _parent_of(_ALIAS_WIDEN_REVISION))

    inspector = inspect(engine)
    for table in ("model_aliases", "routing_policies"):
        assert {c["name"] for c in inspector.get_unique_constraints(table)} == {f"uq_{table}_name_user"}
        names = {index["name"] for index in inspector.get_indexes(table)}
        assert f"uq_{table}_global_name" in names
        assert f"uq_{table}_workspace_global_name" not in names

    command.upgrade(config, "head")

    inspector = inspect(engine)
    for table in ("model_aliases", "routing_policies"):
        assert {c["name"] for c in inspector.get_unique_constraints(table)} == {
            f"uq_{table}_workspace_name_user"
        }


def test_the_survivals_carry_a_restricting_workspace_foreign_key(
    sqlite_at_head: tuple[Config, Engine],
) -> None:
    """NOT NULL and RESTRICT on all three, matching ``api_keys.workspace_id``.

    RESTRICT rather than cascade because deleting a workspace must not silently
    take a user's uploads or a router's training data with it.
    """
    _, engine = sqlite_at_head
    inspector = inspect(engine)

    for table in _SURVIVAL_TABLES:
        column = next(c for c in inspector.get_columns(table) if c["name"] == "workspace_id")
        assert column["nullable"] is False
        workspace_fk = [fk for fk in inspector.get_foreign_keys(table) if fk["referred_table"] == "workspace"]
        assert [tuple(fk["constrained_columns"]) for fk in workspace_fk] == [("workspace_id",)]
        assert workspace_fk[0]["options"].get("ondelete") == "RESTRICT"
        # The user foreign key is untouched: the workspace is a second axis, not
        # a re-parenting (otari-ai#1643).
        user_fk = [fk for fk in inspector.get_foreign_keys(table) if fk["referred_table"] == "users"]
        assert [tuple(fk["constrained_columns"]) for fk in user_fk] == [("user_id",)]
        assert user_fk[0]["options"].get("ondelete") == "CASCADE"


def test_the_router_indexes_are_rebuilt_leading_with_the_workspace(
    sqlite_at_head: tuple[Config, Engine],
) -> None:
    """Every read filters on the workspace, so it leads each composite index."""
    _, engine = sqlite_at_head
    inspector = inspect(engine)

    memory = {index["name"]: index["column_names"] for index in inspector.get_indexes("routing_memory")}
    assert memory["ix_routing_memory_workspace_user_model"] == ["workspace_id", "user_id", "embedding_model"]
    assert memory["ix_routing_memory_workspace_user_created"] == ["workspace_id", "user_id", "created_at"]
    assert memory["ix_routing_memory_workspace_user_model_task"] == [
        "workspace_id",
        "user_id",
        "embedding_model",
        "task_id",
    ]
    assert "ix_routing_memory_user_model" not in memory

    preferences = {i["name"]: i["column_names"] for i in inspector.get_indexes("router_preferences")}
    assert preferences["ix_router_preferences_workspace_user_created"] == [
        "workspace_id",
        "user_id",
        "created_at",
    ]


def test_existing_survival_rows_are_backfilled_onto_the_default_workspace(tmp_path: Path) -> None:
    """The upgrade an operator with uploaded files and a taught router runs.

    NOT NULL means the migration has to supply a value, and the only right one is
    the workspace every other request-plane row was already backfilled onto: no
    data moves and nothing is re-issued.
    """
    url = f"sqlite:///{tmp_path / 'survivals.db'}"
    config = _alembic_config(url)
    command.upgrade(config, _ALIAS_WIDEN_REVISION)
    engine = create_engine(url)

    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO users (user_id, spend, reserved, blocked, created_at, updated_at, metadata) "
                "VALUES ('alice', 0, 0, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, '{}')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO file_objects "
                "(id, user_id, filename, mime_type, bytes, purpose, storage_ref, created_at, metadata) "
                "VALUES ('file-1', 'alice', 'notes.txt', 'text/plain', 4, 'user_data', 'ref-1', "
                " CURRENT_TIMESTAMP, '{}')"
            )
        )
        connection.execute(
            text(
                "INSERT INTO routing_memory "
                "(id, user_id, embedding_model, embedding, qualities, task_id, label_source, created_at) "
                "VALUES ('m1', 'alice', 'openai:text-embedding-3-small', '[0.1]', '{}', NULL, 'human', "
                " CURRENT_TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO router_preferences "
                "(id, user_id, prompt, task_id, scores, label_source, created_at) "
                "VALUES ('p1', 'alice', 'hello', NULL, '{}', 'human', CURRENT_TIMESTAMP)"
            )
        )

    command.upgrade(config, _SURVIVALS_REVISION)

    with engine.begin() as connection:
        default = connection.execute(
            text(
                "SELECT w.id FROM workspace w "
                "JOIN organization o ON o.id = w.organization_id WHERE o.slug = 'default'"
            )
        ).scalar_one()
        for table in _SURVIVAL_TABLES:
            landed = connection.execute(text(f"SELECT workspace_id FROM {table}")).scalar_one()  # noqa: S608
            assert landed == default, table
    engine.dispose()


def test_the_survivals_revision_round_trips(sqlite_at_head: tuple[Config, Engine]) -> None:
    """Down drops the column, its index and the renamed ones; up puts them back.

    The downgrade drops the workspace index before its column on purpose: SQLite
    refuses ``DROP COLUMN`` while an index covers it, the same ordering the
    credential revision needed.
    """
    config, engine = sqlite_at_head

    command.downgrade(config, _ALIAS_WIDEN_REVISION)

    inspector = inspect(engine)
    for table in _SURVIVAL_TABLES:
        assert "workspace_id" not in {column["name"] for column in inspector.get_columns(table)}
        assert f"ix_{table}_workspace_id" not in {index["name"] for index in inspector.get_indexes(table)}
    memory = {index["name"] for index in inspector.get_indexes("routing_memory")}
    assert "ix_routing_memory_user_model" in memory
    assert "ix_routing_memory_workspace_user_model" not in memory

    command.upgrade(config, "head")

    inspector = inspect(engine)
    for table in _SURVIVAL_TABLES:
        assert "workspace_id" in {column["name"] for column in inspector.get_columns(table)}


def test_the_migrated_survival_tables_match_their_models(sqlite_at_head: tuple[Config, Engine]) -> None:
    """Hand-written revisions, so nothing else would notice the two drifting apart."""
    _, engine = sqlite_at_head

    for table in _SURVIVAL_TABLES:
        declared = SQLModel.metadata.tables[table]
        migrated = {column["name"] for column in inspect(engine).get_columns(table)}
        assert migrated == set(declared.columns.keys()), table
