"""The provider-files revision round-trips on the standalone SQLite engine."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import CheckConstraint, create_engine, inspect
from sqlmodel import SQLModel

from gateway.models import provider_files  # noqa: F401

TABLES = {
    "provider_account_generations",
    "provider_file_bindings",
    "provider_file_output_operations",
    "provider_file_rate_windows",
}


def test_provider_file_migration_round_trip(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    script = ScriptDirectory.from_config(config)
    revision = script.get_revision("c3e5a7b9d1f4")
    assert revision is not None and revision.down_revision == "d5f8b2a4c6e9"
    assert len(script.get_heads()) == 1
    url = f"sqlite:///{tmp_path / 'files.db'}"
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "c3e5a7b9d1f4")
    engine = create_engine(url)
    try:
        assert TABLES <= set(inspect(engine).get_table_names())
        assert {"purpose", "provider_created_at"} <= {
            column["name"] for column in inspect(engine).get_columns("provider_file_bindings")
        }
        command.downgrade(config, "d5f8b2a4c6e9")
        assert "ix_api_keys_internal_dispatch" in {index["name"] for index in inspect(engine).get_indexes("api_keys")}
        assert not TABLES & set(inspect(engine).get_table_names())
        command.upgrade(config, "c3e5a7b9d1f4")
        assert TABLES <= set(inspect(engine).get_table_names())
        assert {"purpose", "provider_created_at"} <= {
            column["name"] for column in inspect(engine).get_columns("provider_file_bindings")
        }
        for table_name in TABLES:
            expected = {
                constraint.name: str(constraint.sqltext)
                for constraint in SQLModel.metadata.tables[table_name].constraints
                if isinstance(constraint, CheckConstraint)
            }
            actual = {
                constraint["name"]: constraint["sqltext"]
                for constraint in inspect(engine).get_check_constraints(table_name)
            }
            assert actual == expected
    finally:
        engine.dispose()
