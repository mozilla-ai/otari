"""Reading which tables otari's migration chain owns."""

from pathlib import Path

import pytest
from alembic.script import ScriptDirectory
from alembic.util import CommandError

import gateway.models  # noqa: F401
from gateway.core.migration_chain import tables_the_chain_creates
from gateway.models.base import Base

_REPO_ROOT = Path(__file__).resolve().parents[2]

_REVISION = '''"""A revision under test."""

import sqlalchemy as sa
from alembic import op

revision = "a1a1a1a1a1a1"
down_revision = None
branch_labels = None
depends_on = None

_TABLE = "constant_widgets"


def _table_name() -> str:
    return "computed_widgets"


def upgrade() -> None:
{body}


def downgrade() -> None:
    pass
'''


def _chain_with(tmp_path: Path, body: str) -> ScriptDirectory:
    versions = tmp_path / "versions"
    versions.mkdir()
    (versions / "a1a1a1a1a1a1_under_test.py").write_text(_REVISION.format(body=body))
    return ScriptDirectory(str(tmp_path))


def test_the_real_chain_creates_every_table_the_models_declare() -> None:
    assert set(Base.metadata.tables) - tables_the_chain_creates(ScriptDirectory(str(_REPO_ROOT / "alembic"))) == set()


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('    op.create_table("literal_widgets", sa.Column("id", sa.Integer()))', {"literal_widgets"}),
        ('    op.create_table(_TABLE, sa.Column("id", sa.Integer()))', {"constant_widgets"}),
        ('    op.create_table(table_name="keyword_widgets")', {"keyword_widgets"}),
        ('    op.rename_table("old_widgets", "new_widgets")', {"new_widgets"}),
        ('    op.rename_table("old_widgets", new_table_name="new_widgets")', {"new_widgets"}),
        ('    op.add_column("widgets", sa.Column("name", sa.String()))', set()),
    ],
)
def test_a_readable_table_name_is_owned(tmp_path: Path, body: str, expected: set[str]) -> None:
    assert tables_the_chain_creates(_chain_with(tmp_path, body)) == expected


@pytest.mark.parametrize(
    ("body", "written"),
    [
        ('    op.create_table(_table_name(), sa.Column("id", sa.Integer()))', r"_table_name\(\)"),
        ("    op.create_table()", "no name"),
        ('    op.create_table(UNDEFINED, sa.Column("id", sa.Integer()))', "UNDEFINED"),
    ],
)
def test_an_unreadable_table_name_is_refused(tmp_path: Path, body: str, written: str) -> None:
    with pytest.raises(CommandError, match=rf"Revision a1a1a1a1a1a1 names a table with {written}"):
        tables_the_chain_creates(_chain_with(tmp_path, body))
