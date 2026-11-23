"""The revision that seals ``organization_guardrails.validate_kwargs``, on SQLite."""

import json
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text

from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    decrypt_mapping,
    forget_local_secret_key,
    generate_secret_key,
)

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"
_REVISION = "f6b8d0a2c4e7"
_BEFORE = "e5a7c9b1d3f6"
_AT = datetime(2026, 11, 21, tzinfo=UTC).strftime("%Y-%m-%d %H:%M:%S.%f")
_SECRET = "patronus-key-nobody-should-read"


@pytest.fixture(autouse=True)
def _no_key_left_loaded() -> Iterator[None]:
    """Keep a key one test loads from a file out of the next one."""
    yield
    forget_local_secret_key()


@pytest.fixture
def before(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Config, Engine]]:
    """A SQLite database at the revision before this one, with no key configured."""
    monkeypatch.delenv("OTARI_SECRET_KEY", raising=False)
    monkeypatch.delenv("GATEWAY_SECRET_KEY", raising=False)
    url = f"sqlite:///{tmp_path / 'guardrails.db'}"
    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    config.set_main_option("sqlalchemy.url", url)
    config.attributes["configure_logger"] = False
    command.upgrade(config, _BEFORE)
    engine = create_engine(url)
    try:
        yield config, engine
    finally:
        engine.dispose()


def _mandate(engine: Engine, profile: str, validate_kwargs: dict[str, object] | None) -> uuid.UUID:
    organization, mandate = uuid.uuid4(), uuid.uuid4()
    with engine.begin() as connection:
        connection.execute(
            text("INSERT INTO organization (id, name, slug) VALUES (:id, 'Acme', :slug)"),
            {"id": organization.hex, "slug": f"acme-{organization.hex[:6]}"},
        )
        connection.execute(
            text(
                "INSERT INTO organization_guardrails (id, organization_id, profile, mode, on_unavailable, "
                "directions, validate_kwargs, enabled, applies_to_all_workspaces, created_at, updated_at) "
                "VALUES (:id, :org, :profile, 'block', 'block', '[\"input\"]', :kwargs, 1, 1, :at, :at)"
            ),
            {
                "id": mandate.hex,
                "org": organization.hex,
                "profile": profile,
                "kwargs": None if validate_kwargs is None else json.dumps(validate_kwargs),
                "at": _AT,
            },
        )
    return mandate


def _sealed(engine: Engine, mandate: uuid.UUID) -> str | None:
    with engine.connect() as connection:
        sealed: str | None = connection.execute(
            text("SELECT encrypted_validate_kwargs FROM organization_guardrails WHERE id = :id"),
            {"id": mandate.hex},
        ).scalar_one()
    return sealed


def _columns(engine: Engine) -> set[str]:
    with engine.connect() as connection:
        return {row[1] for row in connection.execute(text("PRAGMA table_info(organization_guardrails)"))}


def test_stored_arguments_are_sealed_and_the_clear_column_is_gone(
    before: tuple[Config, Engine], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, engine = before
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    mandate = _mandate(engine, "patronus", {"threshold": 0.8, "patronus_api_key": _SECRET})

    command.upgrade(config, _REVISION)

    sealed = _sealed(engine, mandate)
    assert sealed is not None
    assert _SECRET not in sealed
    assert decrypt_mapping(sealed) == {"threshold": 0.8, "patronus_api_key": _SECRET}
    assert "validate_kwargs" not in _columns(engine)


def test_a_mandate_with_no_arguments_seals_nothing(
    before: tuple[Config, Engine], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, engine = before
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    missing = _mandate(engine, "plain", None)
    empty = _mandate(engine, "empty", {})

    command.upgrade(config, _REVISION)

    assert _sealed(engine, missing) is None
    assert _sealed(engine, empty) is None


def test_arguments_with_no_key_configured_refuse_the_upgrade(before: tuple[Config, Engine]) -> None:
    config, engine = before
    _mandate(engine, "patronus", {"patronus_api_key": _SECRET})

    with pytest.raises((RuntimeError, SecretBoxUnavailableError), match="OTARI_SECRET_KEY"):
        command.upgrade(config, _REVISION)


def test_the_downgrade_puts_the_arguments_back_in_clear(
    before: tuple[Config, Engine], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, engine = before
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    mandate = _mandate(engine, "patronus", {"threshold": 0.8})
    command.upgrade(config, _REVISION)

    command.downgrade(config, _BEFORE)

    with engine.connect() as connection:
        stored = connection.execute(
            text("SELECT validate_kwargs FROM organization_guardrails WHERE id = :id"), {"id": mandate.hex}
        ).scalar_one()
    assert json.loads(stored) == {"threshold": 0.8}
    assert "encrypted_validate_kwargs" not in _columns(engine)
