"""The provider-width downgrade refuses while a stored name is too long for the old column.

An organization's own connection is named by its issuer URL, which the old widths cannot hold, so
narrowing the column has to refuse rather than fail with the engine's own error or lose the name
that identifies an account.
"""

import importlib.util
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import Connection, create_engine, text

_MIGRATION = (
    Path(__file__).resolve().parents[2] / "alembic" / "versions" / "086a7895681c_widen_oidc_provider_columns.py"
)

# A real issuer, at the length one reaches in practice: 80 characters, which the old 50-character
# column cannot hold. A short stand-in would make this test pass with the check removed.
_LONG_PROVIDER = "oidc:https://login.microsoftonline.com/9188040d-6c67-4c5b-b112-36a304b66dad/v2.0"

_SCHEMA = (
    "CREATE TABLE oauth_identity_link (provider VARCHAR(255) NOT NULL)",
    "CREATE TABLE oauth_pending_state (provider VARCHAR(255) NOT NULL)",
    'CREATE TABLE "user" (oauth_provider VARCHAR(255))',
)


def _load(connection: Connection, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the migration with ``op.get_bind()`` bound to ``connection``."""
    spec = importlib.util.spec_from_file_location("migration_086a7895681c", _MIGRATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.op, "get_bind", lambda: connection)
    return module


@pytest.fixture
def conn() -> Iterator[Connection]:
    engine = create_engine("sqlite://")
    with engine.connect() as connection:
        for statement in _SCHEMA:
            connection.execute(text(statement))
        yield connection


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("oauth_identity_link", "provider"),
        ("oauth_pending_state", "provider"),
        ('"user"', "oauth_provider"),
    ],
)
def test_a_name_too_long_for_its_old_column_refuses_the_downgrade(
    conn: Connection, monkeypatch: pytest.MonkeyPatch, table: str, column: str
) -> None:
    migration = _load(conn, monkeypatch)
    conn.execute(text(f"INSERT INTO {table} ({column}) VALUES (:v)"), {"v": _LONG_PROVIDER})  # noqa: S608

    with pytest.raises(RuntimeError) as refusal:
        migration._refuse_names_that_no_longer_fit()

    reason = str(refusal.value)
    assert table.strip('"') in reason
    assert column in reason
    assert "migrate again" in reason


def test_names_that_still_fit_let_the_downgrade_proceed(conn: Connection, monkeypatch: pytest.MonkeyPatch) -> None:
    migration = _load(conn, monkeypatch)
    for table, column in (("oauth_identity_link", "provider"), ("oauth_pending_state", "provider")):
        conn.execute(text(f"INSERT INTO {table} ({column}) VALUES ('google')"))  # noqa: S608
    conn.execute(text("INSERT INTO \"user\" (oauth_provider) VALUES ('github')"))

    migration._refuse_names_that_no_longer_fit()
