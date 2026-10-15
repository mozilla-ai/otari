"""The ``api_key_federation_rules`` revision's Alembic chain, exercised on SQLite.

The integration suite migrates PostgreSQL only, so this is the SQLite half: the
table refuses a rule naming neither a subject nor an email, goes with its key,
and the downgrade removes it.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.exc import IntegrityError

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "557fc0f5bfda"
_BEFORE = "b7d9f1a3c5e8"


@pytest.fixture
def sqlite_at_revision(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    database_url = f"sqlite:///{tmp_path / 'federation_rules.db'}"
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    command.upgrade(config, _REVISION)
    engine = create_engine(database_url)
    event.listen(engine, "connect", lambda connection, _record: connection.execute("PRAGMA foreign_keys=ON"))
    try:
        yield config, engine
    finally:
        engine.dispose()


def _add_key(engine: Engine) -> None:
    with engine.begin() as connection:
        workspace = connection.execute(text("SELECT id FROM workspace LIMIT 1")).scalar_one()
        connection.execute(
            text(
                "INSERT INTO api_keys"
                " (id, key_hash, workspace_id, created_at, is_active, exclude_from_budget, metadata)"
                " VALUES ('key-1', 'hash-1', :workspace, CURRENT_TIMESTAMP, 1, 0, '{}')"
            ),
            {"workspace": workspace},
        )


def _add_rule(engine: Engine, rule_id: str, subject: str | None, email: str | None) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO api_key_federation_rules (id, api_key_id, issuer, audience, subject, email)"
                " VALUES (:id, 'key-1', 'https://accounts.google.com', 'aud', :subject, :email)"
            ),
            {"id": rule_id, "subject": subject, "email": email},
        )


def test_a_rule_names_a_principal_and_goes_with_its_key(sqlite_at_revision: tuple[Config, Engine]) -> None:
    _config, engine = sqlite_at_revision
    _add_key(engine)
    _add_rule(engine, "r1", None, "bot@example.com")

    with pytest.raises(IntegrityError):
        _add_rule(engine, "r2", None, None)

    with engine.begin() as connection:
        connection.execute(text("DELETE FROM api_keys WHERE id = 'key-1'"))
        assert connection.execute(text("SELECT count(*) FROM api_key_federation_rules")).scalar_one() == 0


def test_the_downgrade_drops_the_table(sqlite_at_revision: tuple[Config, Engine]) -> None:
    config, engine = sqlite_at_revision

    command.downgrade(config, _BEFORE)

    assert "api_key_federation_rules" not in inspect(engine).get_table_names()
    command.upgrade(config, _REVISION)
    assert "api_key_federation_rules" in inspect(engine).get_table_names()
