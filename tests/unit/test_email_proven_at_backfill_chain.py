"""``e2b7c4d9f1a3``'s address-proof backfill, on SQLite, over the three populations.

The revision replaces a negative flag with a positive one, so the backfill decides
which existing rows carry a proved address. Getting it wrong in the permissive
direction would mark an address an organization merely recorded as one its owner
proved, which is what domain auto-join and a second organization's add both refuse
on. That decision is what is under test here rather than the ADD COLUMN.

SQLite because the OSS edition ships it by default.
"""

from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text

import gateway.models  # noqa: F401  (registers every table on the shared metadata)

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "e2b7c4d9f1a3"
_BEFORE = "b4d6f8a0c2e5"
_VERIFIED = "2026-08-31 00:00:00"
_RECORDED = "2026-09-01 00:00:00"


def _alembic_config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture
def sqlite_before(tmp_path: Path) -> Iterator[tuple[Config, Engine]]:
    database_url = f"sqlite:///{tmp_path / 'address-proof.db'}"
    config = _alembic_config(database_url)
    command.upgrade(config, _BEFORE)
    engine = create_engine(database_url)
    try:
        yield config, engine
    finally:
        engine.dispose()


def _account(engine: Engine, *, email: str, verified_at: str | None, recorded_at: str | None) -> str:
    # `uuid4().hex`, not `str(uuid4())`: SQLAlchemy's `Uuid` stores as CHAR(32) on
    # SQLite, so a dashed string would not compare equal to a row written by the app.
    account_id = uuid4().hex
    with engine.begin() as connection:
        connection.execute(
            text(
                'INSERT INTO "user" (id, email, is_active, is_superuser, email_verified_at, email_vouched_at) '
                "VALUES (:id, :email, 1, 0, :verified_at, :recorded_at)"
            ),
            {"id": account_id, "email": email, "verified_at": verified_at, "recorded_at": recorded_at},
        )
    return account_id


def _proven_at(engine: Engine, account_id: str) -> object:
    with engine.begin() as connection:
        return connection.execute(
            text('SELECT email_proven_at FROM "user" WHERE id = :id'), {"id": account_id}
        ).scalar_one()


def _recorded_at(engine: Engine, account_id: str) -> object:
    with engine.begin() as connection:
        return connection.execute(
            text('SELECT email_vouched_at FROM "user" WHERE id = :id'), {"id": account_id}
        ).scalar_one()


def test_an_address_nothing_verified_carries_no_proof(sqlite_before: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before
    account_id = _account(engine, email="none@example.com", verified_at=None, recorded_at=None)

    command.upgrade(config, _REVISION)

    assert _proven_at(engine, account_id) is None


def test_an_address_an_organization_recorded_carries_no_proof(sqlite_before: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before
    account_id = _account(engine, email="recorded@example.com", verified_at=_VERIFIED, recorded_at=_RECORDED)

    command.upgrade(config, _REVISION)

    assert _proven_at(engine, account_id) is None


def test_a_verification_nothing_contradicted_carries_the_proof(sqlite_before: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before
    account_id = _account(engine, email="proved@example.com", verified_at=_VERIFIED, recorded_at=None)

    command.upgrade(config, _REVISION)

    assert _proven_at(engine, account_id) == _VERIFIED


def test_the_downgrade_restores_the_flag_each_row_carried(sqlite_before: tuple[Config, Engine]) -> None:
    config, engine = sqlite_before
    unverified = _account(engine, email="none@example.com", verified_at=None, recorded_at=None)
    recorded = _account(engine, email="recorded@example.com", verified_at=_VERIFIED, recorded_at=_RECORDED)
    proved = _account(engine, email="proved@example.com", verified_at=_VERIFIED, recorded_at=None)

    command.upgrade(config, _REVISION)
    command.downgrade(config, _BEFORE)

    assert _recorded_at(engine, unverified) is None
    assert _recorded_at(engine, recorded) == _VERIFIED
    assert _recorded_at(engine, proved) is None
