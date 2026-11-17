"""The MCP server governance revision on PostgreSQL: which empty read labels it keeps, down and up again.

The SQLite half, and the cases, are in ``tests/unit/test_mcp_governance_schema_chain.py``.
"""

from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy.orm import Session

from conftest import seed_workspace_id

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_CASES_PATH = Path(__file__).resolve().parents[1] / "unit" / "test_mcp_governance_schema_chain.py"
_spec = importlib.util.spec_from_file_location("mcp_governance_schema_chain", _CASES_PATH)
assert _spec is not None and _spec.loader is not None
chain = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chain)


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


def test_empty_read_labels_stay_empty_where_a_rule_uses_read_on_postgresql(test_db: Session, postgres_url: str) -> None:
    workspace_id = seed_workspace_id(test_db)
    test_db.commit()
    # Released before the ALTER: an idle transaction would block it.
    test_db.close()
    config = _alembic_config(postgres_url)
    try:
        command.downgrade(config, chain._BEFORE)
        with test_db.bind.begin() as connection:  # type: ignore[union-attr]
            chain.seed_servers(connection, uuid.UUID(str(workspace_id)))

        command.upgrade(config, chain._REVISION)
        with test_db.bind.connect() as connection:  # type: ignore[union-attr]
            assert chain.read_labels(connection) == chain.EXPECTED_AFTER_UPGRADE

        command.downgrade(config, chain._BEFORE)
        with test_db.bind.connect() as connection:  # type: ignore[union-attr]
            assert all(labels is not None for labels in chain.read_labels(connection).values())

        command.upgrade(config, chain._REVISION)
        with test_db.bind.connect() as connection:  # type: ignore[union-attr]
            assert chain.read_labels(connection) == chain.EXPECTED_AFTER_UPGRADE
    finally:
        # Back at head whatever happened, or every later test on this worker gets a schema its models do not match.
        test_db.close()
        command.upgrade(config, "head")
