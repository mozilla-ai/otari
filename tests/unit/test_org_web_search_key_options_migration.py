"""The revision that seals ``org_web_search_keys.options``, on SQLite."""

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
_REVISION = "e5a7c9b1d3f6"
_BEFORE = "d4f6a8c0e2b5"
_AT = datetime(2026, 11, 21, tzinfo=UTC).strftime("%Y-%m-%d %H:%M:%S.%f")
_SECRET = "tavily-extra-nobody-should-read"


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
    url = f"sqlite:///{tmp_path / 'websearch.db'}"
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


def _key(engine: Engine, name: str, options: dict[str, object] | None) -> uuid.UUID:
    organization, key_id = uuid.uuid4(), uuid.uuid4()
    with engine.begin() as connection:
        connection.execute(
            text("INSERT INTO organization (id, name, slug) VALUES (:id, 'Acme', :slug)"),
            {"id": organization.hex, "slug": f"acme-{organization.hex[:6]}"},
        )
        connection.execute(
            text(
                "INSERT INTO org_web_search_keys (id, organization_id, kind, provider, name, options, "
                "is_org_default, created_at, updated_at) "
                "VALUES (:id, :org, 'search', 'tavily', :name, :options, 0, :at, :at)"
            ),
            {
                "id": key_id.hex,
                "org": organization.hex,
                "name": name,
                "options": None if options is None else json.dumps(options),
                "at": _AT,
            },
        )
    return key_id


def _sealed(engine: Engine, key_id: uuid.UUID) -> str | None:
    with engine.connect() as connection:
        sealed: str | None = connection.execute(
            text("SELECT encrypted_options FROM org_web_search_keys WHERE id = :id"), {"id": key_id.hex}
        ).scalar_one()
    return sealed


def _columns(engine: Engine) -> set[str]:
    with engine.connect() as connection:
        return {row[1] for row in connection.execute(text("PRAGMA table_info(org_web_search_keys)"))}


def test_stored_options_are_sealed_and_the_clear_column_is_gone(
    before: tuple[Config, Engine], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, engine = before
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    key_id = _key(engine, "ours", {"search_depth": "advanced", "extra_credential": _SECRET})

    command.upgrade(config, _REVISION)

    sealed = _sealed(engine, key_id)
    assert sealed is not None
    assert _SECRET not in sealed
    assert decrypt_mapping(sealed) == {"search_depth": "advanced", "extra_credential": _SECRET}
    assert "options" not in _columns(engine)


def test_a_key_with_no_options_seals_nothing(before: tuple[Config, Engine], monkeypatch: pytest.MonkeyPatch) -> None:
    config, engine = before
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    missing = _key(engine, "plain", None)
    empty = _key(engine, "empty", {})

    command.upgrade(config, _REVISION)

    assert _sealed(engine, missing) is None
    assert _sealed(engine, empty) is None


def test_options_with_no_key_configured_refuse_the_upgrade(before: tuple[Config, Engine]) -> None:
    config, engine = before
    _key(engine, "ours", {"extra_credential": _SECRET})

    with pytest.raises((RuntimeError, SecretBoxUnavailableError), match="OTARI_SECRET_KEY"):
        command.upgrade(config, _REVISION)


def test_the_downgrade_puts_the_options_back_in_clear(
    before: tuple[Config, Engine], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, engine = before
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    key_id = _key(engine, "ours", {"search_depth": "advanced"})
    command.upgrade(config, _REVISION)

    command.downgrade(config, _BEFORE)

    with engine.connect() as connection:
        stored = connection.execute(
            text("SELECT options FROM org_web_search_keys WHERE id = :id"), {"id": key_id.hex}
        ).scalar_one()
    assert json.loads(stored) == {"search_depth": "advanced"}
    assert "encrypted_options" not in _columns(engine)
