"""How autogenerate decides which tables in a database belong to otari's migration chain."""

import shutil
import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.util import CommandError

_REPO_ROOT = Path(__file__).resolve().parents[2]

_UNREADABLE_REVISION = '''"""A revision that names its table with a call."""

import sqlalchemy as sa
from alembic import op

revision = "f0f0f0f0f0f0"
down_revision = "{head}"
branch_labels = None
depends_on = None


def _table_name() -> str:
    return "computed_widgets"


def upgrade() -> None:
    pass


def downgrade() -> None:
    op.create_table(_table_name(), sa.Column("id", sa.Integer(), primary_key=True))
'''


@pytest.mark.parametrize("foreign_table", [False, True])
def test_autogenerate_refuses_a_revision_whose_table_name_it_cannot_read(tmp_path: Path, foreign_table: bool) -> None:
    # Guessing would either drop a table another chain owns or keep one otari retired.
    script_location = tmp_path / "alembic"
    shutil.copytree(_REPO_ROOT / "alembic", script_location, ignore=shutil.ignore_patterns("__pycache__"))
    database = tmp_path / "otari.db"
    config = Config()
    config.set_main_option("script_location", str(script_location))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database}")
    head = ScriptDirectory.from_config(config).get_current_head()
    (script_location / "versions" / "f0f0f0f0f0f0_unreadable_table_name.py").write_text(
        _UNREADABLE_REVISION.format(head=head)
    )
    command.upgrade(config, "head")
    if foreign_table:
        connection = sqlite3.connect(database)
        try:
            connection.execute("CREATE TABLE edition_widgets (id INTEGER PRIMARY KEY)")
            connection.commit()
        finally:
            connection.close()

    with pytest.raises(CommandError, match=r"Revision f0f0f0f0f0f0 names a table with _table_name\(\)"):
        command.check(config)
