"""The revision recording a sign-in's workspace on PostgreSQL, down and up again.

The SQLite half is ``tests/unit/test_mcp_sign_in_workspace_schema_chain.py``.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.orm import Session

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "da9ab6d90202"
_BEFORE = "124a7328c1b3"
_TABLE = "mcp_oauth_flows"


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


def test_the_column_comes_and_goes_with_its_revision_on_postgresql(test_db: Session, postgres_url: str) -> None:
    engine = test_db.get_bind()
    # Released before the DDL: an idle transaction would block it.
    test_db.close()
    config = _alembic_config(postgres_url)
    try:
        command.downgrade(config, _BEFORE)
        assert "workspace_id" not in {column["name"] for column in inspect(engine).get_columns(_TABLE)}

        command.upgrade(config, _REVISION)
        assert "workspace_id" in {column["name"] for column in inspect(engine).get_columns(_TABLE)}
        assert "ix_mcp_oauth_flows_workspace_id" in {index["name"] for index in inspect(engine).get_indexes(_TABLE)}
        assert {key["referred_table"] for key in inspect(engine).get_foreign_keys(_TABLE)} >= {"workspace"}
    finally:
        # Back at head whatever happened, or every later test on this worker gets a schema its models do not match.
        test_db.close()
        command.upgrade(config, "head")
