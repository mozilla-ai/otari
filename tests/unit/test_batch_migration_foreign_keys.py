"""Batch migrations on a SQLite build that enforces foreign keys.

A batch rebuild copies the table it alters into a new one, drops the original and renames.
Where the migration connection enforces foreign keys, dropping a parent takes every child row
with it, and the upgrade still reports success, so the loss is silent. Each case seeds a parent
and a child at the revision before, upgrades with the pragma on, and reads the child back.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event, text

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"


@pytest.fixture
def enforcing_foreign_keys() -> Iterator[None]:
    """Every engine opened while this is active enforces foreign keys, as such a build would."""

    @event.listens_for(Engine, "connect")
    def _enforce(dbapi_connection: Any, _: Any) -> None:  # noqa: ANN401
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    try:
        yield
    finally:
        event.remove(Engine, "connect", _enforce)


def _config(database_url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", database_url)
    config.attributes["configure_logger"] = False
    return config


def _organization(engine: Engine, slug: str) -> str:
    organization_id = uuid.uuid4().hex
    with engine.begin() as connection:
        connection.execute(
            text("INSERT INTO organization (id, name, slug) VALUES (:id, :name, :slug)"),
            {"id": organization_id, "name": slug.title(), "slug": slug},
        )
    return organization_id


def test_adding_the_enforcement_columns_keeps_an_organization_s_domain_claims(
    enforcing_foreign_keys: None, tmp_path: Path
) -> None:
    url = f"sqlite:///{tmp_path / 'enforcement.db'}"
    config = _config(url)
    command.upgrade(config, "b7c3e9d1a4f8")
    engine = create_engine(url)
    try:
        organization_id = _organization(engine, "acme")
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO organization_domain "
                    "(id, organization_id, domain, default_role, enabled, verification_token) "
                    "VALUES (:id, :organization_id, :domain, 'member', 1, :token)"
                ),
                {
                    "id": uuid.uuid4().hex,
                    "organization_id": organization_id,
                    "domain": "acme.example",
                    "token": "a-verification-token",
                },
            )

        command.upgrade(config, "10727bf67704")

        with engine.connect() as connection:
            claims = connection.execute(
                text("SELECT COUNT(*) FROM organization_domain WHERE organization_id = :id"),
                {"id": organization_id},
            ).scalar()
        assert claims == 1
    finally:
        engine.dispose()


def test_letting_an_identity_point_at_no_organization_keeps_its_memberships(
    enforcing_foreign_keys: None, tmp_path: Path
) -> None:
    url = f"sqlite:///{tmp_path / 'identity.db'}"
    config = _config(url)
    command.upgrade(config, "10727bf67704")
    engine = create_engine(url)
    try:
        organization_id = _organization(engine, "acme")
        user_id = uuid.uuid4().hex
        with engine.begin() as connection:
            connection.execute(
                text(
                    'INSERT INTO "user" (id, is_active, is_superuser, active_organization_id) '
                    "VALUES (:id, 1, 0, :organization_id)"
                ),
                {"id": user_id, "organization_id": organization_id},
            )
            connection.execute(
                text(
                    "INSERT INTO organization_member (id, organization_id, user_id, role, status) "
                    "VALUES (:id, :organization_id, :user_id, 'member', 'active')"
                ),
                {"id": uuid.uuid4().hex, "organization_id": organization_id, "user_id": user_id},
            )

        command.upgrade(config, "4eee3a7ee90f")

        with engine.connect() as connection:
            memberships = connection.execute(
                text("SELECT COUNT(*) FROM organization_member WHERE user_id = :id"), {"id": user_id}
            ).scalar()
        assert memberships == 1
    finally:
        engine.dispose()
