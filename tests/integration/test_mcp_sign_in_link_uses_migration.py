"""The ``mcp_sign_in_link_uses`` revision on PostgreSQL, down and up again.

The SQLite half is ``tests/unit/test_mcp_sign_in_link_uses_schema_chain.py``.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.orm import Session

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "e5b9d2f7a4c1"
_BEFORE = "a9d3f5b7c1e2"
_TABLE = "mcp_sign_in_link_uses"
_INDEX = "ix_mcp_sign_in_link_uses_expires_at"


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


def test_the_table_comes_and_goes_with_its_revision_on_postgresql(test_db: Session, postgres_url: str) -> None:
    engine = test_db.get_bind()
    # Released before the DDL: an idle transaction would block it.
    test_db.close()
    config = _alembic_config(postgres_url)
    try:
        command.downgrade(config, _BEFORE)
        assert _TABLE not in inspect(engine).get_table_names()

        command.upgrade(config, _REVISION)
        assert inspect(engine).get_pk_constraint(_TABLE)["constrained_columns"] == ["ticket_hash"]
        assert _INDEX in {index["name"] for index in inspect(engine).get_indexes(_TABLE)}
    finally:
        # Back at head whatever happened, or every later test on this worker gets a schema its models do not match.
        test_db.close()
        command.upgrade(config, "head")
